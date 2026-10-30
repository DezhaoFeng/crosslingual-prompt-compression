"""Run a local teacher on pilot chunks, then segment + align + stats (DECISIONS.md B3/B4/B6).

On a CSIL 4090:
    pip install -r requirements.lock  (then a CUDA torch build instead of +cpu)
    huggingface-cli login             (gated model: accept its terms on the HF page first)
    python run_teacher.py --n 3              # 3 train chunks per language, prompt v2
    python run_teacher.py --n 10             # resumes: already-done ids are skipped
    python run_teacher.py --prompt v1 --n 3  # official prompt verbatim

Output: results/teacher/{model}__{prompt}.jsonl (one file per prompt version).

Appends one JSON line per chunk to --out, so a reboot loses at most the current chunk.
"""
import argparse
import hashlib
import importlib.metadata
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

import constrained
from align import align, teacher_stats
from constrained import DeletionConstraint, TokenTable
from segment import SegmentError, load_tokenizer, segment

CONSTRAINED_VERSION = "deletion-grammar-v1"  # constrained.py: whole original words, original order

MODELS = {
    "CohereLabs/tiny-aya-global": "af89d219b53ed9b13b8a4645f9c8028973510324",
    "CohereLabs/tiny-aya-earth": "0624fafbe200f930b89566fc4ec4da314ba48d89",
    "CohereLabs/aya-101": "e7dad472b9de8a30a00cf08a05c19003bf59028d",  # 12.9B mT5 encoder-decoder
    "Qwen/Qwen3-14B": "40c069824f4251a91eefaf281ebe4c544efd3e18",  # 14.8B dense, byte-level BPE
}
# chat models: token that closes the reply (followed by EOS), and extra chat-template arguments
CHAT_END = {"Qwen/Qwen3-14B": "<|im_end|>"}  # default "<|END_RESPONSE|>" (Tiny Aya)
CHAT_KWARGS = {"Qwen/Qwen3-14B": {"enable_thinking": False}}  # no <think> reasoning, answer directly
# encoder-decoder models: no chat template, prompt = SYSTEM + blank line + user prompt as plain text;
# weights are fp32 (48 GB), so they are loaded quantised (bitsandbytes) with bf16 compute
SEQ2SEQ = {"CohereLabs/aya-101"}
QUANTIZED = SEQ2SEQ | {"Qwen/Qwen3-14B"}  # do not fit in 24 GB at bf16
PLAIN_SEP = chr(10) * 2  # blank line between SYSTEM and the user prompt
# official compression_instructions.json #4 at LLMLingua commit 5a4c78a, verbatim
PROMPT_VERSION = "llmlingua2-instr4-verbatim"
SYSTEM = ("You are an excellent linguist and very good at compressing passages into short expressions "
          "by removing unimportant words, while retaining as much information as possible.")
USER = ("Compress some text to short expressions, and such that you (GPT-4) can reconstruct it as close as possible "
        "to the original. Unlike the usual text compression, I need you to comply with the 5 conditions below: "
        "1. You can ONLY remove unimportant words. 2. Do not change the order of words. 3. Do not change the original "
        "words, e.g. 'asking'->'ask' is NOT OK, 'current'->'now' is NOT OK. 4. Do not use abbreviations or emojis, "
        "e.g. 'without'->'w/o' is NOT OK, 'as soon as possible'->'ASAP' is NOT OK. 5. Do not add new words or "
        "symbols, this is very important. For example, 'dedicate 3 hours to each chapter'->'3 hours/chapter' is NOT "
        "OK because you add new token '/', just compress it into '3 hours each chapter'. '30 eggs plus 20 eggs equals "
        "50 eggs'->'30+20=50' is also NOT OK becuase you add new symbols + and =, just compress it into '30 plus 20 "
        "equals 50'. \nCompress the origin aggressively by removing words only. Compress the origin as short as you "
        "can, while retaining as much information as possible. \nIf you understand, please compress the following "
        "text: \n{text_to_compress}\nThe compressed text is: ")

# v2 = v1's instruction and 5 conditions + explicit output rules + one deletion-only example in the
# input's language. Examples are our own parallel sentences (not Belebele, not pilot); test_run_teacher.py
# checks that each is a legal deletion under segment() + align().
PROMPT_VERSIONS = {"v1": PROMPT_VERSION, "v2": "llmlingua2-instr4-outputonly-1shot",
                   "v3": "llmlingua2-instr4-outputonly-1shot-target33",
                   "v4": "llmlingua2-instr4-outputonly-1shot-target33-ex33"}
# v3 = v2 + this one sentence (the v2 example is unchanged and keeps ~65% of its words: known confound)
TARGET_RULE = "Keep about one third (33%) of the original words. \n"
OUTPUT_RULES = ("Output ONLY the compressed text: no explanation, no heading, no Markdown (such as **), "
                "no quotes, and no reconstructed version. Keep the original language. \n")
EXAMPLES = {
    "en": ("The old library, which was built in 1890 by the city council, was closed for repairs during the "
           "summer of 2015.",
           "old library built in 1890 by city council, closed for repairs summer 2015."),
    "zh": ("这座老图书馆是在1890年由市议会建造的，并在2015年夏天因为维修而关闭。",
           "老图书馆1890年由市议会建造，2015年夏天因为维修关闭。"),
    "sw": ("Maktaba ya zamani, ambayo ilijengwa mwaka 1890 na halmashauri ya jiji, ilifungwa kwa ajili ya "
           "matengenezo wakati wa kiangazi cha mwaka 2015.",
           "Maktaba ya zamani ilijengwa mwaka 1890 na halmashauri ya jiji, ilifungwa kwa matengenezo kiangazi 2015."),
}
# v4 = v3 with the examples' compressions changed to keep about a third of the words (v2/v3 examples
# keep 0.61-0.68, contradicting the 33% target). Same originals; sw not checked by a native speaker.
EXAMPLES_V4 = {
    "en": "library built in 1890, closed 2015.",          # keep 0.30 of content words
    "zh": "图书馆1890年建造，2015年关闭",                     # 0.36
    "sw": "Maktaba ilijengwa mwaka 1890, ilifungwa mwaka 2015.",  # 0.33
}
V2_TAIL = ("Example: \nOriginal text: {ex_orig}\nCompressed text: {ex_comp}\n\n"
           "Now compress the following text: \n{text_to_compress}\nThe compressed text is: ")


def user_prompt(version, lang, text):
    if version == "v1":
        return USER.format(text_to_compress=text)
    conditions = USER.split("If you understand")[0]  # v1 up to its closing line
    ex_orig, ex_comp = EXAMPLES[lang]
    if version == "v4":
        ex_comp = EXAMPLES_V4[lang]
    target = TARGET_RULE if version in ("v3", "v4") else ""
    return conditions + OUTPUT_RULES + target + V2_TAIL.format(ex_orig=ex_orig, ex_comp=ex_comp,
                                                               text_to_compress=text)


def prompt_sha256(version):
    if version == "v1":
        return sha256(SYSTEM + "\x00" + USER)
    parts = [SYSTEM, USER, OUTPUT_RULES, V2_TAIL, json.dumps(EXAMPLES, sort_keys=True)]
    if version in ("v3", "v4"):
        parts.append(TARGET_RULE)
    if version == "v4":
        parts.append(json.dumps(EXAMPLES_V4, sort_keys=True))
    return sha256("\x00".join(parts))


def load_chunks(pilot, langs, split, n, start=0):
    rows = []
    for lang in langs:
        lang_rows = [json.loads(l) for l in open(pilot / f"{lang}.jsonl", encoding="utf-8")]
        rows += [r for r in lang_rows if r["split"] == split][start:start + n]
    return rows


def generate(model, tok, prompt, row, max_new_tokens, constraint=None, chat_kwargs=None):
    user = user_prompt(prompt, row["lang"], row["text"])
    if model.config.is_encoder_decoder:
        ids = tok(SYSTEM + PLAIN_SEP + user, return_tensors="pt").input_ids.to(model.device)
        out_start = 1  # generate() returns decoder_start_token + new tokens
    else:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True, **(chat_kwargs or {}))["input_ids"].to(model.device)
        out_start = ids.shape[1]
    extra = {}
    if constraint is not None:
        # prefix_allowed_tokens_fn sees decoder ids for encoder-decoder models
        extra["prefix_allowed_tokens_fn"] = constraint.prefix_allowed_tokens_fn(out_start)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, do_sample=False, max_new_tokens=max_new_tokens, **extra)
    secs = time.time() - t0
    new = out[0, out_start:].tolist()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    finish = "eos" if new and new[-1] in eos else "length"
    if constraint is not None and constraint.no_legal_path:
        finish = "no_legal_path"
    decode_kw = {"clean_up_tokenization_spaces": False} if model.config.is_encoder_decoder else {}
    res = {
        "raw_output": tok.decode(new, skip_special_tokens=True, **decode_kw),
        "finish": finish,
        "input_tokens": ids.shape[1],
        "output_tokens": len(new),
        "seconds": round(secs, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    if constraint is not None:
        res["unreachable_words"] = constraint.unreachable_words
    return res


def label(row, raw_output, xlmr):
    try:
        seg = segment(row["text"], row["lang"], xlmr)
    except SegmentError as e:
        return {"status": "segment_error", "reason": str(e)}
    r = align(row["text"], seg["word_spans"], raw_output)
    return {**seg, **r, "stats": teacher_stats(row["text"], seg["word_spans"], raw_output, r["labels"])}


NEAR_COPY = 0.95  # word_keep_rate at or above this is reported as a near-copy of the input


def is_usable(status, finish):
    # only a legal deletion from a complete (not truncated) generation may be trained on
    return status == "ok" and finish == "eos"


def sha256(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def run_config(args):
    """Everything that changes the teacher output. Results made under another config are not reused."""
    cfg = {"model": args.model, "revision": MODELS[args.model], "prompt_version": PROMPT_VERSIONS[args.prompt],
           "prompt_sha256": prompt_sha256(args.prompt), "dtype": "bfloat16",
           "params": {"do_sample": False, "max_new_tokens": args.max_new_tokens}}
    if args.model in QUANTIZED:  # Tiny Aya keeps its old fingerprint
        cfg["dtype"] = f"bnb-{getattr(args, 'quant', 'int8')}+bf16-compute"
    if args.model in SEQ2SEQ:
        cfg["prompt_format"] = "plain: SYSTEM + blank line + user prompt"
    if args.model in CHAT_KWARGS:
        cfg["chat_template_kwargs"] = CHAT_KWARGS[args.model]
    if getattr(args, "decoding", "free") == "constrained":  # free decoding keeps the old fingerprint
        cfg["decoding"] = CONSTRAINED_VERSION
        cfg["constraint_sha256"] = hashlib.sha256(Path(constrained.__file__).read_bytes()).hexdigest()
    return cfg


def read_results(path):
    """Read the JSONL results file. A half-written last line (crash/reboot mid-write) is cut off
    with a warning; a broken line anywhere else is an error."""
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.split(b"\n")
    recs, offset = [], 0
    for i, line in enumerate(lines):
        if line.strip():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines) - 1:  # last line, no trailing newline: unfinished write
                    print(f"warning: dropping half-written last line of {path} ({len(line)} bytes)")
                    with open(path, "r+b") as f:
                        f.truncate(offset)
                    break
                raise ValueError(f"{path}: corrupt JSON on line {i + 1}") from None
        offset += len(line) + 1
    else:
        if data and not data.endswith(b"\n"):  # complete last record but its newline was lost
            with open(path, "ab") as f:
                f.write(b"\n")
    return recs


def check_resume(recs, cfg, rows):
    """Refuse to mix results from another config or from different input text."""
    by_id = {r["id"]: r for r in rows}
    for rec in recs:
        if rec["teacher"]["config"] != cfg:
            raise SystemExit(f"{rec['id']} in the output file was made with another config:\n"
                             f"  file: {rec['teacher']['config']}\n  now:  {cfg}\nuse a different --out")
        row = by_id.get(rec["id"])
        if row is not None and rec["text_sha256"] != sha256(row["text"]):
            raise SystemExit(f"{rec['id']}: input text changed since it was run; use a different --out")


def summarize(path):
    by = defaultdict(list)
    for r in read_results(path):
        by[r["lang"]].append(r)
    for lang, rs in by.items():
        ok = [r for r in rs if r["status"] == "ok"]
        mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
        tps = sum(r["teacher"]["output_tokens"] for r in rs) / max(1e-9, sum(r["teacher"]["seconds"] for r in rs))
        # an unchanged copy is a legal "deletion" too; count near-copies separately
        near_copy = sum(r["stats"]["word_keep_rate"] >= NEAR_COPY for r in ok)
        print(f"{lang}: n={len(rs)} usable={sum(r['usable'] for r in rs)} ok={len(ok)} near_copy(ok)={near_copy} "
              f"status={dict(Counter(r['status'] for r in rs))} "
              f"truncated={sum(r['teacher']['finish'] == 'length' for r in rs)} "
              f"char_comp_rate(all)={mean([r['stats']['char_comp_rate'] for r in rs if 'stats' in r])} "
              f"word_keep_rate(ok)={mean([r['stats']['word_keep_rate'] for r in ok])} "
              f"tok/s={tps:.1f} peak_vram_gb={max(r['teacher']['peak_vram_gb'] for r in rs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global", choices=list(MODELS))
    ap.add_argument("--langs", default="en,zh,sw")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=3, help="chunks per language")
    ap.add_argument("--start", type=int, default=0, help="skip this many chunks per language first "
                    "(the first 3 train chunks are development samples)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--pilot", default="data/pilot")
    ap.add_argument("--prompt", default="v2", choices=list(PROMPT_VERSIONS))
    ap.add_argument("--decoding", default="free", choices=["free", "constrained"])
    ap.add_argument("--quant", default="int8", choices=["int8", "nf4"], help="models in QUANTIZED only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    suffix = "" if args.decoding == "free" else "__constrained"
    out = Path(args.out or f"results/teacher/{args.model.split('/')[-1]}__{args.prompt}{suffix}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = run_config(args)
    rows = load_chunks(Path(args.pilot), args.langs.split(","), args.split, args.n, args.start)
    recs = read_results(out)
    check_resume(recs, cfg, rows)
    done = {r["id"] for r in recs}
    todo = [r for r in rows if r["id"] not in done]

    if todo:
        rev = MODELS[args.model]
        tok = AutoTokenizer.from_pretrained(args.model, revision=rev)
        if args.model in QUANTIZED:
            q = (BitsAndBytesConfig(load_in_8bit=True) if args.quant == "int8" else
                 BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16))
            cls = AutoModelForSeq2SeqLM if args.model in SEQ2SEQ else AutoModelForCausalLM
            model = cls.from_pretrained(args.model, revision=rev, dtype=torch.bfloat16,
                                        quantization_config=q, device_map={"": 0}).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model, revision=rev,
                                                         dtype=torch.bfloat16).to("cuda").eval()
        xlmr = load_tokenizer()
        meta = {"config": cfg, "transformers": importlib.metadata.version("transformers"),
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
        if args.decoding == "constrained":
            table = TokenTable.from_hf(tok)
            eos_id = model.generation_config.eos_token_id
            eos_id = eos_id[0] if isinstance(eos_id, list) else eos_id
            # chat models close with <|END_RESPONSE|> then EOS; T5 just emits </s>
            end_response = (eos_id if model.config.is_encoder_decoder
                            else tok.convert_tokens_to_ids(CHAT_END.get(args.model, "<|END_RESPONSE|>")))
        for i, row in enumerate(todo, 1):
            constraint = None
            if args.decoding == "constrained":
                try:
                    spans = segment(row["text"], row["lang"], xlmr)["word_spans"]
                except SegmentError as e:  # no grammar without words; record and skip, never patch
                    rec = {"id": row["id"], "lang": row["lang"], "split": row["split"], "text": row["text"],
                           "text_sha256": sha256(row["text"]),
                           "teacher": {**meta, "raw_output": None, "finish": "not_run", "input_tokens": 0,
                                       "output_tokens": 0, "seconds": 0.0, "peak_vram_gb": 0.0},
                           "status": "segment_error", "reason": str(e), "usable": False}
                    with open(out, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue
                constraint = DeletionConstraint(row["text"], spans, table, end_response, eos_id,
                                                allow_leading_space=table.kind == "sentencepiece")
            g = generate(model, tok, args.prompt, row, args.max_new_tokens, constraint, CHAT_KWARGS.get(args.model))
            lab = label(row, g["raw_output"], xlmr)
            rec = {"id": row["id"], "lang": row["lang"], "split": row["split"], "text": row["text"],
                   "text_sha256": sha256(row["text"]), "teacher": {**meta, **g}, **lab,
                   "usable": is_usable(lab["status"], g["finish"])}
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(todo)}] {row['id']} {rec['status']} finish={g['finish']} "
                  f"{g['output_tokens']} tok {g['seconds']}s")
    summarize(out)


if __name__ == "__main__":
    main()
