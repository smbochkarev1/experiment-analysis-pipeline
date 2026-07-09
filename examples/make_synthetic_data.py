#!/usr/bin/env python3
"""Generate a synthetic experiments.csv for the demo.

All numbers here are FICTIONAL. We simulate summary statistics for ~30 A/B
experiments spanning the full spectrum of outcomes an analyst actually sees:
clear winners, regressions, underpowered tests, dead-flat nulls, tiny-but-
"significant" effects, and borderline results that only look significant until
the multiple-comparison correction is applied.

Run:  python examples/make_synthetic_data.py
"""
import csv
import os

import numpy as np

RNG = np.random.default_rng(2026)
OUT = os.path.join(os.path.dirname(__file__), "experiments.csv")


def prop_row(exp_id, name, base, lift_rel, n, direction="increase", peeks=0):
    """A proportion (conversion-rate) experiment.

    base    : true control rate; lift_rel: true relative treatment lift.
    We draw observed conversions from a Binomial so values look like real data.
    """
    p_c = base
    p_t = base * (1 + lift_rel)
    c_conv = RNG.binomial(n, p_c)
    t_conv = RNG.binomial(n, min(max(p_t, 0), 1))
    return {
        "experiment_id": exp_id, "name": name, "metric_type": "proportion",
        "control_n": n, "control_value": round(c_conv / n, 6),
        "treatment_n": n, "treatment_value": round(t_conv / n, 6),
        "control_var": "", "treatment_var": "",
        "direction": direction, "peeks": peeks,
    }


def mean_row(exp_id, name, mu_c, effect_rel, sd, n, direction="increase", peeks=0):
    """A continuous-metric experiment (e.g. revenue per user, session length).

    Reports observed means and sample variances drawn from Normal(mu, sd).
    """
    mu_t = mu_c * (1 + effect_rel)
    c = RNG.normal(mu_c, sd, n)
    t = RNG.normal(mu_t, sd, n)
    return {
        "experiment_id": exp_id, "name": name, "metric_type": "mean",
        "control_n": n, "control_value": round(float(c.mean()), 6),
        "treatment_n": n, "treatment_value": round(float(t.mean()), 6),
        "control_var": round(float(c.var(ddof=1)), 6),
        "treatment_var": round(float(t.var(ddof=1)), 6),
        "direction": direction, "peeks": peeks,
    }


rows = []
# --- clear winners --------------------------------------------------------
rows.append(prop_row("EXP-001", "Checkout button color",        0.120, +0.08,  40000))
rows.append(prop_row("EXP-002", "One-click reorder",            0.045, +0.15,  60000))
rows.append(mean_row("EXP-003", "Revenue per user - new plan",  52.0,  +0.06, 9.0, 30000))
rows.append(prop_row("EXP-004", "Simplified signup form",       0.220, +0.05,  50000))
rows.append(mean_row("EXP-005", "Search ranking v2",            3.10,  +0.09, 1.4, 45000))

# --- clear regressions (no-ship) -----------------------------------------
rows.append(prop_row("EXP-006", "Aggressive upsell modal",      0.180, -0.09,  50000))
rows.append(mean_row("EXP-007", "Ads density +2 slots",         41.0,  -0.05, 8.0, 35000))
rows.append(prop_row("EXP-008", "Mandatory account gate",       0.300, -0.06,  40000))

# --- underpowered (needs more data) --------------------------------------
rows.append(prop_row("EXP-009", "Onboarding tooltip",           0.150, +0.10,    600))
rows.append(mean_row("EXP-010", "Premium banner copy",          25.0,  +0.08, 12.0,  450))
rows.append(prop_row("EXP-011", "Referral nudge (pilot)",       0.070, +0.20,    800))
rows.append(prop_row("EXP-012", "Dark mode toggle placement",   0.090, +0.12,    500))

# --- flat / inconclusive (well powered, truly null) ----------------------
rows.append(prop_row("EXP-013", "Footer link reorder",          0.110, +0.00,  60000))
rows.append(mean_row("EXP-014", "Font size +1px",               30.0,  +0.00, 6.0, 40000))
rows.append(prop_row("EXP-015", "Icon set refresh",             0.200, +0.002, 55000))
rows.append(mean_row("EXP-016", "Loading spinner style",        18.0,  -0.003, 5.0, 38000))

# --- tiny but statistically significant (below practical floor) ----------
rows.append(prop_row("EXP-017", "Micro copy on CTA",            0.250, +0.004, 400000))
rows.append(mean_row("EXP-018", "Rounding of prices .99",       12.0,  +0.003, 3.0, 300000))

# --- borderline: raw p just under 0.05, killed by BH correction ----------
rows.append(prop_row("EXP-019", "Email subject A",              0.130, +0.018, 22000))
rows.append(prop_row("EXP-020", "Email subject B",              0.130, +0.017, 22000))
rows.append(prop_row("EXP-021", "Push timing shift",            0.060, +0.03,  30000))

# --- moderate winners with realistic noise -------------------------------
rows.append(prop_row("EXP-022", "Guest checkout",               0.160, +0.04,  70000))
rows.append(mean_row("EXP-023", "Bundle recommendation",        48.0,  +0.035, 10.0, 55000))
rows.append(prop_row("EXP-024", "Progress bar in cart",         0.140, +0.045, 48000))

# --- guardrail / peeking flagged -----------------------------------------
rows.append(prop_row("EXP-025", "Autoplay video (peeked x5)",   0.100, +0.06,  20000, peeks=5))
rows.append(mean_row("EXP-026", "Notif frequency (peeked x8)",  22.0,  +0.05, 7.0, 15000, peeks=8))

# --- decrease-is-good metrics --------------------------------------------
rows.append(prop_row("EXP-027", "Faster page (bounce rate)",    0.350, -0.06,  45000, direction="decrease"))
rows.append(mean_row("EXP-028", "Reduce checkout steps (time)", 95.0,  -0.07, 20.0, 30000, direction="decrease"))
rows.append(prop_row("EXP-029", "Error-state redesign (errors)",0.040, +0.04, 40000, direction="decrease"))  # regression: errors up

# --- large sample, small real win ----------------------------------------
rows.append(mean_row("EXP-030", "Homepage hero rework",         60.0,  +0.02, 11.0, 120000))


def main():
    cols = ["experiment_id", "name", "metric_type", "control_n", "control_value",
            "control_var", "treatment_n", "treatment_value", "treatment_var",
            "direction", "peeks"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"wrote {len(rows)} synthetic experiments -> {OUT}")


if __name__ == "__main__":
    main()
