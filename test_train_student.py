"""Focused tests for label expansion, special-token masking, and padding."""
import train_student as ts


class DummyTokenizer:
    pad_token_id = 1
    cls_token_id = 0
    sep_token_id = 2


def test_load_examples_expands_word_labels_and_masks_special_tokens(tmp_path):
    import json

    path = tmp_path / "labels.jsonl"
    row = {
        "input_ids": [11, 12, 13],
        "word_ids": [0, 0, 1],
        "labels": [1, 0],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    examples = ts.load_examples(path, DummyTokenizer(), max_length=8)
    assert examples == [{"input_ids": [0, 11, 12, 13, 2], "labels": [-100, 1, 1, 0, -100]}]


def test_collator_masks_padding_in_loss():
    import torch

    batch = ts.collate_examples([
        {"input_ids": [0, 10, 2], "labels": [-100, 1, -100]},
        {"input_ids": [0, 11, 12, 2], "labels": [-100, 0, 1, -100]},
    ], pad_token_id=1)
    assert batch["input_ids"].tolist() == [[0, 10, 2, 1], [0, 11, 12, 2]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]
    assert batch["labels"].tolist() == [[-100, 1, -100, -100], [-100, 0, 1, -100]]
    assert int((batch["labels"] != -100).sum()) == 3
