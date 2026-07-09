"""Correctness tests for the statistical core of the pipeline.

Each test pins a function against an INDEPENDENTLY known answer:
- the two-proportion z-test against a hand-worked textbook example,
- Welch's t-test against ``scipy.stats.ttest_ind(equal_var=False)``,
- Benjamini-Hochberg against ``scipy.stats.false_discovery_control``,
- the verdict rules against their documented decision branches.
"""

import math
import os
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze import benjamini_hochberg, two_proportion_test, welch_t_test  # noqa: E402
from verdict import VerdictConfig, decide  # noqa: E402


# --------------------------------------------------------------------------- #
# Two-proportion z-test                                                        #
# --------------------------------------------------------------------------- #
def test_two_proportion_known_answer():
    """n=100/arm, p1=0.10, p2=0.20.

    Pooled p = 0.15, pooled SE = sqrt(0.15*0.85*(1/100+1/100)) = 0.0504975,
    z = 0.10 / 0.0504975 = 1.98030, two-sided p = 0.047669.
    Unpooled (Wald) SE = sqrt(0.1*0.9/100 + 0.2*0.8/100) = 0.05,
    95% CI = 0.10 +/- 1.959964*0.05 = [0.0020, 0.1980].
    """
    r = two_proportion_test(n1=100, p1=0.10, n2=100, p2=0.20, alpha=0.05)
    assert r["stat"] == pytest.approx(1.98030, abs=1e-3)
    assert r["pval"] == pytest.approx(0.04767, abs=1e-3)
    assert r["diff"] == pytest.approx(0.10, abs=1e-9)
    assert r["ci"][0] == pytest.approx(0.0020, abs=1e-3)
    assert r["ci"][1] == pytest.approx(0.1980, abs=1e-3)


def test_two_proportion_symmetric_pvalue():
    """Swapping the arms flips the sign of z but leaves the p-value intact."""
    a = two_proportion_test(100, 0.10, 100, 0.20, 0.05)
    b = two_proportion_test(100, 0.20, 100, 0.10, 0.05)
    assert a["stat"] == pytest.approx(-b["stat"], abs=1e-12)
    assert a["pval"] == pytest.approx(b["pval"], abs=1e-12)


# --------------------------------------------------------------------------- #
# Welch's t-test vs. scipy ground truth                                        #
# --------------------------------------------------------------------------- #
def _summary(sample):
    arr = np.asarray(sample, dtype=float)
    return len(arr), arr.mean(), arr.var(ddof=1)


def test_welch_matches_scipy_on_samples():
    """Feed our summary-stat implementation the moments of two fixed samples
    and require agreement with scipy's Welch t-test on the raw samples."""
    control = [4.1, 5.2, 3.8, 6.0, 5.5, 4.9, 5.1, 6.3, 4.4, 5.7, 5.0, 4.6]
    treatment = [5.9, 6.4, 5.1, 7.2, 6.8, 5.5, 6.9, 7.0, 6.1, 5.8]

    n1, m1, v1 = _summary(control)
    n2, m2, v2 = _summary(treatment)
    r = welch_t_test(n1, m1, v1, n2, m2, v2, alpha=0.05)

    sp = stats.ttest_ind(treatment, control, equal_var=False)
    assert r["stat"] == pytest.approx(sp.statistic, abs=1e-6)
    assert r["pval"] == pytest.approx(sp.pvalue, abs=1e-6)
    assert r["df"] == pytest.approx(sp.df, abs=1e-6)


def test_welch_matches_scipy_second_case():
    control = [10.0, 12.0, 9.5, 11.2, 10.8, 13.1, 9.9, 10.4]
    treatment = [11.5, 13.2, 12.8, 14.0, 11.1, 12.6]

    n1, m1, v1 = _summary(control)
    n2, m2, v2 = _summary(treatment)
    r = welch_t_test(n1, m1, v1, n2, m2, v2, alpha=0.05)

    sp = stats.ttest_ind(treatment, control, equal_var=False)
    assert r["stat"] == pytest.approx(sp.statistic, abs=1e-6)
    assert r["pval"] == pytest.approx(sp.pvalue, abs=1e-6)
    assert r["df"] == pytest.approx(sp.df, abs=1e-6)


# --------------------------------------------------------------------------- #
# Benjamini-Hochberg vs. scipy ground truth                                    #
# --------------------------------------------------------------------------- #
def test_bh_matches_scipy_false_discovery_control():
    pvals = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.3, 0.7]
    adj, _ = benjamini_hochberg(pvals, fdr=0.05)
    expected = stats.false_discovery_control(pvals, method="bh")
    for got, exp in zip(adj, expected):
        assert abs(got - exp) < 1e-9


def test_bh_rejection_flags():
    """Adjusted p <= fdr is exactly the rejection set."""
    pvals = [0.001, 0.02, 0.03, 0.5, 0.9]
    fdr = 0.05
    adj, rejected = benjamini_hochberg(pvals, fdr=fdr)
    assert rejected == [a <= fdr for a in adj]


def test_bh_is_monotone_in_rank_order():
    pvals = [0.9, 0.001, 0.3, 0.02, 0.05]
    adj, _ = benjamini_hochberg(pvals, fdr=0.05)
    by_p = [adj[i] for i in sorted(range(len(pvals)), key=lambda k: pvals[k])]
    assert by_p == sorted(by_p)  # non-decreasing along increasing raw p


# --------------------------------------------------------------------------- #
# Verdict decision tree                                                        #
# --------------------------------------------------------------------------- #
def _result(**overrides):
    base = {
        "rel_uplift": 0.05,
        "direction": 1,
        "significant_bh": True,
        "mde": 0.02,
        "control_n": 5000,
        "treatment_n": 5000,
        "peeks": 0,
        "p_adj": 0.001,
    }
    base.update(overrides)
    return base


def test_verdict_ship():
    cfg = VerdictConfig()
    v = decide(_result(), cfg)
    assert v["verdict"] == "ship"


def test_verdict_significant_but_below_floor_is_inconclusive():
    cfg = VerdictConfig()
    v = decide(_result(rel_uplift=0.002, mde=0.001), cfg)  # below 0.005 floor
    assert v["verdict"] == "inconclusive"


def test_verdict_underpowered_needs_more_data():
    cfg = VerdictConfig()
    v = decide(
        _result(significant_bh=False, control_n=100, treatment_n=100,
                rel_uplift=0.03, mde=0.10, p_adj=0.4),
        cfg,
    )
    assert v["verdict"] == "needs_more_data"


def test_verdict_wrong_direction_is_no_ship():
    cfg = VerdictConfig()
    v = decide(_result(rel_uplift=-0.04), cfg)  # significant regression
    assert v["verdict"] == "no_ship"
