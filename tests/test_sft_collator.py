from src.data.sft_collator import build_sft_feature, SFTDataCollator


class DummyTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(ch) for ch in text]


def test_build_sft_feature_masks_prompt_tokens_when_enabled():
    tok = DummyTokenizer()
    feat = build_sft_feature(
        tokenizer=tok,
        prompt_text="USER:",
        target_text="OK",
        eos_text="<eos>",
        max_length=64,
        assistant_only_loss=True,
    )
    prompt_len = len("USER:")
    assert feat["labels"][:prompt_len] == [-100] * prompt_len
    assert feat["labels"][prompt_len:] == feat["input_ids"][prompt_len:]


def test_build_sft_feature_keeps_full_labels_when_disabled():
    tok = DummyTokenizer()
    feat = build_sft_feature(
        tokenizer=tok,
        prompt_text="USER:",
        target_text="OK",
        eos_text="<eos>",
        max_length=64,
        assistant_only_loss=False,
    )
    assert feat["labels"] == feat["input_ids"]


def test_sft_collator_pads_labels_with_ignore_index():
    tok = DummyTokenizer()
    collator = SFTDataCollator(tokenizer=tok)
    batch = collator(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [3], "attention_mask": [1], "labels": [3]},
        ]
    )
    assert batch["input_ids"].tolist() == [[1, 2], [3, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert batch["labels"].tolist() == [[-100, 2], [3, -100]]
