"""Metrics, checked against the worked examples in the guide's Metrics reference tab."""

import pytest

from app.services.metrics import (
    cohen_kappa, consistency, jaccard, kendall_tau_distance, micro_prf, pass_at_k,
    pass_hat_k, precision_recall_f1, wilson_interval,
)


def test_precision_recall_worked_example():
    # Gold gaps {Statistics, Power BI, Pandas, Excel}; predicted {Statistics, Power BI, Pandas, SQL}
    p, r, f1 = precision_recall_f1({"Statistics", "Power BI", "Pandas", "SQL"},
                                   {"Statistics", "Power BI", "Pandas", "Excel"})
    assert (p, r, f1) == (0.75, 0.75, 0.75)


def test_empty_sets_are_perfect():
    assert precision_recall_f1(set(), set()) == (1.0, 1.0, 1.0)


def test_micro_pools_counts():
    p, r, _ = micro_prf([({"a", "b"}, {"a"}), ({"c"}, {"c", "d"})])
    assert p == 2 / 3 and r == 2 / 3


def test_kappa_worked_example():
    # 20 roadmaps: you pass 14, judge passes 15, agree on 13 passes and 4 fails -> 0.625
    human = ["pass"] * 13 + ["pass"] * 1 + ["fail"] * 2 + ["fail"] * 4
    judge = ["pass"] * 13 + ["fail"] * 1 + ["pass"] * 2 + ["fail"] * 4
    assert cohen_kappa(human, judge) == pytest.approx(0.625)


def test_kappa_of_an_always_pass_judge_is_zero():
    human = ["pass"] * 18 + ["fail"] * 2
    judge = ["pass"] * 20
    assert cohen_kappa(human, judge) == pytest.approx(0.0)


def test_weighted_kappa_punishes_big_disagreements_more():
    human = [1, 2, 3, 1, 2, 3]
    near = [1, 2, 2, 1, 2, 3]
    far = [3, 2, 3, 1, 2, 1]
    assert cohen_kappa(human, near, "quadratic") > cohen_kappa(human, far, "quadratic")


def test_pass_at_k_and_pass_hat_k_worked_example():
    # 5 runs, 4 successes
    assert pass_at_k(5, 4, 1) == pytest.approx(0.8)
    assert pass_hat_k(5, 4, 2) == pytest.approx(0.6)
    assert pass_hat_k(5, 4, 5) == 0.0


def test_wilson_worked_example():
    low, high = wilson_interval(14, 20)
    assert low == pytest.approx(0.48, abs=0.01) and high == pytest.approx(0.85, abs=0.01)


def test_jaccard_and_consistency():
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert consistency([{"a"}, {"a"}, {"a"}]) == 1.0
    assert consistency([{"a"}, {"b"}]) == 0.0


def test_kendall_tau_distance():
    assert kendall_tau_distance(["a", "b", "c"], ["a", "b", "c"]) == 0.0
    assert kendall_tau_distance(["a", "b", "c"], ["c", "b", "a"]) == 1.0
    assert kendall_tau_distance(["a", "b", "c"], ["b", "a", "c"]) == pytest.approx(1 / 3)
