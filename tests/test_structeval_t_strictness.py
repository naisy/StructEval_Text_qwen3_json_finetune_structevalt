from src.eval.structeval_scoring import structeval_t_score


def test_yaml_scalar_is_not_counted_as_valid_syntax():
    task = {"raw_output_metric": [], "output_type": "YAML"}
    s = structeval_t_score(task, "Sure! Here is the answer.")
    assert s["render_score"] == 0.0


def test_yaml_mapping_is_valid_syntax():
    task = {"raw_output_metric": [], "output_type": "YAML"}
    s = structeval_t_score(task, "a: 1\nb: 2\n")
    assert s["render_score"] == 1.0


def test_yaml_fenced_block_is_invalid_syntax():
    task = {"raw_output_metric": [], "output_type": "YAML"}
    s = structeval_t_score(task, "```yaml\na: 1\n```\n")
    assert s["render_score"] == 0.0


def test_csv_single_column_is_not_counted_as_valid_syntax():
    task = {"raw_output_metric": [], "output_type": "CSV"}
    s = structeval_t_score(task, "just some text")
    assert s["render_score"] == 0.0


def test_csv_two_columns_is_valid_syntax():
    task = {"raw_output_metric": [], "output_type": "CSV"}
    s = structeval_t_score(task, "a,b\n1,2\n")
    assert s["render_score"] == 1.0
