"""Fine-tune M1/M4 from the fixed XLM-R checkpoint with resumable training.

Examples:
  python train_student.py --lang zh --run-dir checkpoints/m1
  python train_student.py --lang en --run-dir checkpoints/m4
  python train_student.py --lang zh --limit-train 200 --max-steps 40 --save-steps 5 \
      --eval-steps 10 --run-dir checkpoints/pilot_zh
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification

from segment import M0, M0_REV, expand_labels, load_tokenizer


def token_loss(logits, labels, reduction="mean"):
    return torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction=reduction,
    )


def load_examples(path, tokenizer, max_length, limit=None):
    examples = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            row = json.loads(line)
            body_ids = row["input_ids"]
            token_labels = expand_labels(row["word_ids"], row["labels"])
            if len(body_ids) != len(token_labels):
                raise ValueError(f"{path}:{line_no}: token/label length mismatch")
            if len(body_ids) + 2 > max_length:
                raise ValueError(f"{path}:{line_no}: sequence exceeds max_length={max_length}")
            if not body_ids:
                raise ValueError(f"{path}:{line_no}: empty token sequence")

            if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
                raise ValueError(f"{path}:{line_no}: tokenizer is missing XLM-R boundary token IDs")
            input_ids = [tokenizer.cls_token_id, *body_ids, tokenizer.sep_token_id]
            special_count = len(input_ids) - len(body_ids)
            if special_count != 2:
                raise ValueError(f"{path}:{line_no}: expected two XLM-R special tokens, got {special_count}")
            labels = [-100] + token_labels + [-100]
            examples.append({"input_ids": input_ids, "labels": labels})
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"No training examples found in {path}")
    return examples


def collate_examples(examples, pad_token_id):
    max_length = max(len(row["input_ids"]) for row in examples)
    input_ids, attention_mask, labels = [], [], []
    for row in examples:
        padding = max_length - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_token_id] * padding)
        attention_mask.append([1] * len(row["input_ids"]) + [0] * padding)
        labels.append(row["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    tp = fp = fn = correct = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            mask = batch["labels"] != -100
            labels = batch["labels"][mask]
            predictions = logits.argmax(dim=-1)[mask]
            count = labels.numel()
            total_loss += token_loss(logits, batch["labels"], reduction="sum").item()
            total_tokens += count
            tp += ((predictions == 1) & (labels == 1)).sum().item()
            fp += ((predictions == 1) & (labels == 0)).sum().item()
            fn += ((predictions == 0) & (labels == 1)).sum().item()
            correct += (predictions == labels).sum().item()
    model.train()
    if total_tokens == 0:
        raise ValueError("Development split has no supervised tokens")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "loss": total_loss / total_tokens,
        "accuracy": correct / total_tokens,
        "keep_precision": precision,
        "keep_recall": recall,
        "keep_f1": f1,
        "supervised_tokens": total_tokens,
    }


def write_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def save_checkpoint(path, model, optimizer, scheduler, scaler, state, config):
    temp_path = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "state": state,
        "config": config,
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, temp_path)
    os.replace(temp_path, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, choices=("zh", "en"))
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--dev-file", default=None)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=196)
    parser.add_argument("--seed", type=int, default=713)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("epochs, batch-size, and gradient-accumulation-steps must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max-steps must be positive")
    if args.save_steps < 1 or args.eval_steps < 1:
        parser.error("save-steps and eval-steps must be positive")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_file = Path(args.train_file or f"data/labels/{args.lang}_train.jsonl")
    dev_file = Path(args.dev_file or f"data/labels/{args.lang}_dev.jsonl")
    tokenizer = load_tokenizer()
    train_data = load_examples(train_file, tokenizer, args.max_length, args.limit_train)
    dev_data = load_examples(dev_file, tokenizer, args.max_length)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("The XLM-R tokenizer has no pad_token_id")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"
    optimizer_name = "bitsandbytes.AdamW8bit" if use_fp16 else "torch.AdamW"
    gradient_scale = 128.0 if use_fp16 else 1.0
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_batches = math.ceil(len(train_data) / args.batch_size)
    steps_per_epoch = math.ceil(train_batches / args.gradient_accumulation_steps)
    planned_steps = args.max_steps or steps_per_epoch * args.epochs
    config = {
        "lang": args.lang,
        "model": M0,
        "revision": M0_REV,
        "train_file": str(train_file),
        "dev_file": str(dev_file),
        "train_examples": len(train_data),
        "dev_examples": len(dev_data),
        "limit_train": args.limit_train,
        "epochs": args.epochs,
        "max_steps": planned_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "seed": args.seed,
        "special_and_padding_labels": -100,
        "precision": "fp16" if use_fp16 else "fp32",
        "optimizer": optimizer_name,
        "gradient_checkpointing": use_fp16,
        "gradient_scale": gradient_scale,
    }
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError(f"Run configuration differs from existing {config_path}; use a new run directory")
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    model = AutoModelForTokenClassification.from_pretrained(M0, revision=M0_REV).to(
        device, dtype=torch.float16 if use_fp16 else torch.float32
    )
    if use_fp16:
        import bitsandbytes as bnb

        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup_steps = int(planned_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: step / max(1, warmup_steps) if step < warmup_steps
        else max(0.0, (planned_steps - step) / max(1, planned_steps - warmup_steps)),
    )
    scaler = None

    checkpoint_path = run_dir / "checkpoint.pt"
    state = {"global_step": 0, "epoch": 0, "next_batch": 0, "best_dev_loss": None}
    if args.resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["config"] != config:
            raise ValueError("Checkpoint configuration mismatch; refusing an unsafe resume")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if scaler is not None and saved["scaler"] is not None:
            scaler.load_state_dict(saved["scaler"])
        state = saved["state"]
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        if saved["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        print(f"Resuming {args.lang} from step {state['global_step']}", flush=True)
    elif checkpoint_path.exists():
        raise ValueError(f"{checkpoint_path} already exists; use --resume or choose a new run directory")

    dev_loader = DataLoader(
        dev_data, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda rows: collate_examples(rows, pad_token_id),
    )
    if state["global_step"] == 0:
        dev_metrics = evaluate(model, dev_loader, device)
        state["best_dev_loss"] = dev_metrics["loss"]
        model.save_pretrained(run_dir / "best")
        tokenizer.save_pretrained(run_dir / "best")
        write_jsonl(run_dir / "metrics.jsonl", {"event": "eval", "step": 0, **dev_metrics})

    optimizer.zero_grad(set_to_none=True)
    model.train()
    for epoch in range(state["epoch"], args.epochs):
        if state["global_step"] >= planned_steps:
            break
        generator = torch.Generator()
        generator.manual_seed(args.seed + epoch)
        order = torch.randperm(len(train_data), generator=generator).tolist()
        epoch_data = [train_data[index] for index in order]
        epoch_loader = DataLoader(
            epoch_data, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda rows: collate_examples(rows, pad_token_id),
        )
        start_batch = state["next_batch"] if epoch == state["epoch"] else 0
        running_loss = 0.0
        accumulated = 0
        accumulated_tokens = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(epoch_loader):
            if batch_index < start_batch:
                continue
            batch_group_size = min(
                args.gradient_accumulation_steps,
                len(epoch_loader) - (batch_index // args.gradient_accumulation_steps)
                * args.gradient_accumulation_steps,
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast(device_type="cuda", enabled=use_fp16):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                loss = token_loss(logits, batch["labels"])
            (loss * gradient_scale / batch_group_size).backward()
            running_loss += loss.item()
            accumulated += 1
            accumulated_tokens += int((batch["labels"] != -100).sum())
            is_update = accumulated == batch_group_size or batch_index + 1 == len(epoch_loader)
            if not is_update:
                continue

            if use_fp16:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(gradient_scale)
                        if not torch.isfinite(parameter.grad).all():
                            raise FloatingPointError(
                                f"Non-finite gradient at step {state['global_step'] + 1}; "
                                "reduce loss scale or use FP32 training"
                            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            state["global_step"] += 1
            state["epoch"] = epoch
            state["next_batch"] = batch_index + 1
            if state["next_batch"] == len(epoch_loader):
                state["epoch"], state["next_batch"] = epoch + 1, 0

            write_jsonl(run_dir / "metrics.jsonl", {
                "event": "train", "step": state["global_step"], "epoch": epoch + 1,
                "batch": batch_index + 1, "loss": running_loss / accumulated,
                "learning_rate": scheduler.get_last_lr()[0],
                "supervised_tokens": accumulated_tokens,
            })
            running_loss, accumulated, accumulated_tokens = 0.0, 0, 0

            did_eval = state["global_step"] % args.eval_steps == 0 or state["global_step"] == planned_steps
            if did_eval:
                dev_metrics = evaluate(model, dev_loader, device)
                write_jsonl(run_dir / "metrics.jsonl", {
                    "event": "eval", "step": state["global_step"], **dev_metrics,
                })
                if dev_metrics["loss"] < state["best_dev_loss"]:
                    state["best_dev_loss"] = dev_metrics["loss"]
                    model.save_pretrained(run_dir / "best")
                    tokenizer.save_pretrained(run_dir / "best")
                model.train()

            if did_eval or state["global_step"] % args.save_steps == 0 or state["global_step"] == planned_steps:
                save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, state, config)
            if state["global_step"] >= planned_steps:
                break
        if state["global_step"] >= planned_steps:
            break
        state["epoch"], state["next_batch"] = epoch + 1, 0

    save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, state, config)
    summary = {
        **config,
        "global_step": state["global_step"],
        "selected_checkpoint": str(run_dir / "best"),
        "best_dev_loss": state["best_dev_loss"],
        "effective_supervised_train_tokens": sum(len(row["labels"]) - 2 for row in train_data),
        "effective_supervised_dev_tokens": sum(len(row["labels"]) - 2 for row in dev_data),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
