from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_name: str, trust_remote_code: bool = True):
    # NOTE:
    # Some tokenizers (notably Mistral-family / derivative tokenizers) have shipped with an
    # incorrect regex pattern in the fast-tokenizer pre-tokenizer config, which can lead to
    # incorrect tokenization. Recent versions of `transformers` recommend passing
    # `fix_mistral_regex=True` when loading such tokenizers.
    #
    # We pass it opportunistically and fall back gracefully for older transformers versions.
    try:
        tok = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_fast=True,
            fix_mistral_regex=True,
        )
    except TypeError:
        tok = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
    # Qwen-style tokenizers often prefer right padding for generation; adjust if needed.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model(
    model_name: str,
    dtype: torch.dtype,
    trust_remote_code: bool = True,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map="auto",
    )
    return model
