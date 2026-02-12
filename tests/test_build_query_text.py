from src.data.import_hf_structured_sft import build_query_text


def test_build_query_text_u10bei_concats_system_and_user():
    q = build_query_text(
        "u-10bei/structured_data_with_cot_dataset_512_v2",
        "You are an expert in TOML format.",
        "Produce a TOML document for a prescription.",
    )
    assert q == "You are an expert in TOML format.\n\nProduce a TOML document for a prescription."


def test_build_query_text_other_datasets_keep_user_only():
    q = build_query_text(
        "daichira/structured-3k-mix-sft",
        "SYSTEM IGNORED",
        "User prompt",
    )
    assert q == "User prompt"
