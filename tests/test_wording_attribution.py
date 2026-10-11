"""Gold integrity, proper scores, and separation of evidence from explanations."""
import copy
import json
import math

import numpy as np
import pytest

from wording_attribution import (CLASSES, audit_metrics, candidate_distribution,
                                 digest, fixture, posthoc_metrics, prompt_for,
                                 run_experiment)


def test_gold_cannot_leak_into_either_probe():
    case = fixture()["cases"][0]
    poisoned = copy.deepcopy(case)
    poisoned.update(gold_origin="POISON", gold_evidence=[{"span": "POISON"}], quotes=["POISON"])
    for probe in ("wording", "origin"):
        assert prompt_for(case, probe) == prompt_for(poisoned, probe)
        assert "POISON" not in prompt_for(poisoned, probe)


def test_quote_and_paraphrase_gold_are_not_message_role_or_string_matching():
    cases = {c["id"]: c for c in fixture()["cases"]}
    quoted = cases["user_quote"]
    assert quoted["messages"][-1]["role"] == "user"
    assert quoted["gold_origin"] == "assistant"
    assert quoted["quotes"][0]["quoted_author"] == "assistant"
    assert quoted["gold_evidence"][0]["message_id"] == "m2"
    assert "adversarial concealment" not in cases["paraphrase"]["messages"][1]["text"]
    assert cases["paraphrase"]["gold_origin"] == "assistant"
    assert cases["incomplete"]["gold_origin"] == "unknown"
    assert cases["deleted"]["gold_origin"] == "neither"
    assert not cases["incomplete"]["complete"]


def test_edits_change_only_declared_fields():
    cases = fixture()["cases"]
    base = cases[0]["messages"]
    expected = {"deleted": [(1, "text")], "user_origin": [(1, "role")],
                "paraphrase": [(1, "text")], "irrelevant_edit": [(0, "text")],
                "user_quote": [(2, "text")], "incomplete": []}
    for case in cases[1:]:
        changed = [(i, k) for i in range(len(base)) for k in base[i]
                   if base[i][k] != case["messages"][i][k]]
        assert changed == expected[case["id"]]


def test_brier_logloss_and_reliability_known_answers():
    p = dict(zip(CLASSES, [0.7, 0.1, 0.1, 0.1]))
    result = audit_metrics(["assistant"], [p])
    assert result["brier_valid_only"] == pytest.approx(0.12)
    assert result["log_loss_valid_only"] == pytest.approx(-math.log(0.7))
    assert result["ece_valid_only_descriptive"] == pytest.approx(0.3)
    assert sum(b["n"] for b in result["reliability_bins"]) == 1
    assert result["selective"][-1]["risk"] is None
    assert result["selective"][-1]["coverage"] == 0


@pytest.mark.parametrize("bad", [None, {}, {"assistant": 1},
    dict(zip(CLASSES, [math.nan, 0, 0, 1])), dict(zip(CLASSES, [math.inf, 0, 0, 0])),
    dict(zip(CLASSES, [-0.1, 1.1, 0, 0])), dict(zip(CLASSES, [0.1] * 4)),
    dict(zip(CLASSES, [True, 0, 0, 0])), dict(zip(CLASSES, ["1", 0, 0, 0])),
    dict(zip(CLASSES, [[0.7], [0.1], [0.1], [0.1]])), dict(zip(CLASSES, [10**1000, 0, 0, 0]))])
def test_invalid_probabilities_remain_visible_failures(bad):
    result = audit_metrics(["assistant"], [bad])
    assert result["invalid_n"] == 1
    assert result["confusion"]["assistant"]["invalid"] == 1
    assert result["brier_valid_only"] is None
    assert result["accuracy_including_invalid_as_errors"] == 0
    assert result["assistant_origin_invalid_n"] == 1
    assert result["selective"][0]["coverage"] == 0
    json.dumps(result, allow_nan=False)


def test_zero_gold_probability_is_infinite_loss_not_silent_clipping():
    result = audit_metrics(["assistant"], [dict(zip(CLASSES, [0, 1, 0, 0]))])
    assert result["log_loss_is_infinite"]
    assert result["log_loss_valid_only"] is None
    assert result["false_user_attribution_rate"] == 1
    assert result["brier_valid_only"] == 2
    json.dumps(result, allow_nan=False)


def test_invalid_rows_reduce_coverage_and_accuracy():
    result = audit_metrics(["assistant", "assistant"], [dict(zip(CLASSES, [1, 0, 0, 0])), None])
    assert result["accuracy_including_invalid_as_errors"] == 0.5
    assert result["selective"][0]["coverage"] == 0.5
    assert result["valid_n"] == 1


def test_generator_thresholds_are_not_consumed_by_validation():
    result = audit_metrics(["assistant"], [None], thresholds=iter([0, 0.5]))
    assert len(result["selective"]) == 2


@pytest.mark.parametrize("golds,probabilities", [([], []), (["assistant"], []), (["bad"], [None])])
def test_bad_gold_cannot_silently_truncate(golds, probabilities):
    with pytest.raises(ValueError):
        audit_metrics(golds, probabilities)


def test_stable_candidate_normalization():
    p = candidate_distribution({"x": {"logprob": -10000}, "y": {"logprob": -10001}})
    assert p["x"] == pytest.approx(1 / (1 + math.exp(-1)))
    with pytest.raises(ValueError):
        candidate_distribution({"x": {"logprob": math.nan}})


class RecordingBackend:
    """A labeled test double, never a semantic-model result."""
    n_layers = 3

    def __init__(self):
        self.patches = []

    def score(self, prompt, continuations, patch=None):
        if patch:
            self.patches.append(patch)
        return {k: {"logprob": -2.0 - i} for i, k in enumerate(continuations)}

    def state(self, prompt, layer):
        return np.array([1.0, 2.0]) if '"role": "assistant"' in prompt else np.array([2.0, 1.0])

    def self_score(self, prompt, continuations, layer):
        return self.score(prompt, continuations)

    def metadata(self):
        return {"kind": "test_double"}


def test_report_keeps_causal_levels_separate_and_hashes_evidence():
    report = run_experiment(RecordingBackend())
    assert report["fixture"]["kind"] == "synthetic_controlled_case_study"
    assert report["internal_interventions"]["status"] == "not_run"
    assert report["posthoc"]["status"] == "not_collected"
    assert report["report_sha256"] == digest({k: v for k, v in report.items() if k != "report_sha256"})
    assert len(report["prompt_effects"]) == 6
    json.dumps(report, allow_nan=False)


def test_both_patch_directions_have_self_and_norm_matched_controls():
    backend = RecordingBackend()
    report = run_experiment(backend, patch_layer=1)
    trials = report["internal_interventions"]["trials"]
    assert len(trials) == 14
    for recipient in ("assistant_origin", "user_origin"):
        rows = [r for r in trials if r["recipient"] == recipient]
        assert rows[0]["kind"] == "self"
        assert rows[0]["delta_norm"] == 0
        donor = next(r for r in rows if r["kind"] == "donor")
        assert donor["delta_norm"] > 0
        for row in rows[3:]:
            assert row["delta_norm"] == pytest.approx(donor["delta_norm"])
    assert all(p["layer"] == 1 for p in backend.patches)
    assert report["internal_interventions"]["self_control_passed"]


def test_explanation_cannot_rewrite_origin_or_frozen_results():
    report = run_experiment(RecordingBackend())
    untouched = copy.deepcopy(report)
    evidence = report["fixture"]["cases"][0]["gold_evidence"][0]
    record = {"report_sha256": report["report_sha256"], "case_id": "assistant_origin",
              "claimed_origin": "assistant", "message_id": evidence["message_id"],
              "span": evidence["span"], "comparison_id": report["prompt_effects"][0]["comparison_id"],
              "probability_positive": 0.9}
    correct = posthoc_metrics(report, [record])["rows"][0]
    assert correct["origin_correct"] and correct["evidence_valid"]
    assert correct["sign_brier"] is None  # the test double has an exactly zero effect
    wrong = posthoc_metrics(report, [{**record, "claimed_origin": "user"}])["rows"][0]
    assert not wrong["origin_correct"] and not wrong["evidence_valid"]
    invalid = posthoc_metrics(report, [{**record, "report_sha256": "invented"}])
    assert invalid["invalid_n"] == 1
    assert report == untouched
    report["results"][0]["user_minus_assistant_logprob"] = 123
    with pytest.raises(ValueError, match="changed"):
        posthoc_metrics(report, [record])
