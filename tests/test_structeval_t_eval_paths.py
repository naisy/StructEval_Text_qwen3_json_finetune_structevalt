from src.eval.structeval_scoring import structeval_t_score


def test_structeval_t_score_supports_wildcards():
    # Minimal JSON containing an array of objects.
    generation = '{"museum": {"galleries": [{"artworks": [{"title": "A"}, {"title": "B"}]}]}}'
    task = {
        "raw_output_metric": [
            "museum.galleries[0].artworks.*.title",  # dot-wildcard
            "museum.galleries[0].artworks[*].title",  # bracket-wildcard
        ]
    }
    s = structeval_t_score(task, generation)
    assert s["render_score"] == 1.0
    assert s["key_validation_score"] == 1.0
    assert s["final_eval_score"] == 1.0
    assert s["raw_output_eval"] == [True, True]
