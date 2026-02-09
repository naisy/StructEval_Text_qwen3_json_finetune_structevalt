from src.data.validators import parse_json, is_json_only, check_task_constraints_article_meta


def test_parse_json_ok():
    ok, obj, err = parse_json('{"a": 1}')
    assert ok and obj["a"] == 1 and err is None


def test_is_json_only():
    assert is_json_only('{"a": 1}')
    # markdown fences are not allowed (even if the JSON inside is valid)
    assert not is_json_only('```json\n{"a":1}\n```')
    assert not is_json_only('```json\n{"a":1}\n``` extra')


def test_constraints():
    obj = {
        "title": "X",
        "authors": [{"name": "A", "affiliation": "U"}, {"name": "B", "affiliation": "I"}],
        "publication": {"year": 2024},
        "keywords": ["k1", "k2"],
    }
    c = check_task_constraints_article_meta(obj)
    assert c["authors_len_is_2"]
