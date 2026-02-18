import tomllib

from src.data.toml_jsonlike import convert_json_payload_to_toml, looks_like_json_payload


def test_looks_like_json_payload():
    assert looks_like_json_payload('{"a": 1}')
    assert looks_like_json_payload('   [1,2,3]')
    assert not looks_like_json_payload('a = 1')


def test_convert_json_object_to_toml_root():
    ok, toml_text, err = convert_json_payload_to_toml('{"a": 1, "b": true, "c": null}')
    assert ok, err
    obj = tomllib.loads(toml_text)
    assert obj["a"] == 1
    assert obj["b"] is True
    # null is mapped to empty string
    assert obj["c"] == ""


def test_convert_json_array_wraps_root_key():
    ok, toml_text, err = convert_json_payload_to_toml('[{"a": 1}, {"b": 2}]')
    assert ok, err
    obj = tomllib.loads(toml_text)
    assert "root" in obj
    assert obj["root"][0]["a"] == 1
    assert obj["root"][1]["b"] == 2
