from src.rl.rewards import compute_reward_components, combine_reward


def _cfg():
    return {
        "reward": {
            "w_parse_json": 1.0,
            "w_only_json": 0.5,
            "w_parse_yaml": 1.0,
            "w_only_yaml": 0.5,
            "w_parse_toml": 1.0,
            "w_only_toml": 0.5,
            "w_parse_xml": 1.0,
            "w_only_xml": 0.5,
            "w_parse_csv": 1.0,
            "w_only_csv": 0.5,
            "p_parse_fail": -3.0,
            "p_only": -1.0,
        }
    }


def test_reward_json():
    c = '{"a": 1}'
    comps = compute_reward_components(c, output_type="JSON")
    assert comps["parse"] == 1.0
    assert comps["only"] == 1.0
    r = combine_reward(comps, _cfg(), output_type="JSON")
    assert r > 0


def test_reward_yaml_only_vs_extra_text():
    good = """a: 1\nb: 2\n"""
    # markdown fences are considered invalid outputs for this project.
    bad = """```yaml\na: 1\n```\n"""
    comps_g = compute_reward_components(good, output_type="YAML")
    assert comps_g["parse"] == 1.0
    assert comps_g["only"] == 1.0

    comps_b = compute_reward_components(bad, output_type="YAML")
    # Reward code extracts the payload for parse scoring, but fenced wrappers
    # must be penalized via `only=0` and `extraneous=1`.
    assert comps_b["parse"] == 1.0
    assert comps_b["only"] == 0.0


def test_reward_other_formats_parse():
    toml = """a = 1\n"""
    xml = """<root><a>1</a></root>"""
    csv = """a,b\n1,2\n"""

    assert compute_reward_components(toml, output_type="TOML")["parse"] == 1.0
    # Canonical TOML should give a perfect soft score.
    comps_t = compute_reward_components(toml, output_type="TOML")
    assert comps_t["toml_canonical"] == 1.0
    assert comps_t["toml_canonical_soft"] == 1.0

    # Non-canonical but parseable TOML should still get a dense (0,1) score.
    non_canon = """b = 1\na = 2\n"""
    comps_nc = compute_reward_components(non_canon, output_type="TOML")
    assert comps_nc["parse"] == 1.0
    assert comps_nc["toml_canonical"] == 0.0
    assert 0.0 < comps_nc["toml_canonical_soft"] < 1.0
    assert compute_reward_components(xml, output_type="XML")["parse"] == 1.0
    assert compute_reward_components(csv, output_type="CSV")["parse"] == 1.0
