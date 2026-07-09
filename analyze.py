#!/usr/bin/env python3
"""Batch A/B experiment analysis pipeline.

Reads a table of experiments (control vs treatment summary statistics),
runs the appropriate significance test for each, applies a Benjamini-Hochberg
false-discovery-rate correction across the whole batch, assigns a rule-based
verdict to each experiment, and renders a single self-contained interactive
HTML report plus a machine-readable summary CSV.

Usage:
    python analyze.py --input examples/experiments.csv
    python analyze.py --input experiments.csv --outdir output --config config.yaml

Input schema (CSV header):
    experiment_id, metric_type, control_n, control_value,
    treatment_n, treatment_value, direction,
    [control_var], [treatment_var], [peeks], [name]

  metric_type : "proportion"  -> control_value / treatment_value are rates in [0,1]
                "mean"        -> control_value / treatment_value are means;
                                 control_var / treatment_var required (variances)
    direction : "increase" | "decrease" | 1 | -1  (desired direction of effect)
        peeks : optional int, number of interim looks (sequential testing guard)
"""

import argparse
import csv
import math
import os
import sys
from datetime import datetime, timezone

from scipy import stats

from verdict import VerdictConfig, decide

Z_POWER_DEFAULT = 0.80  # target power for MDE reference


# --------------------------------------------------------------------------- #
# Config loading                                                              #
# --------------------------------------------------------------------------- #
def load_config(path):
    defaults = {
        "alpha": 0.05,
        "fdr_level": 0.05,
        "min_practical_uplift": 0.005,
        "min_sample_per_arm": 1000,
        "target_power": 0.80,
        "underpowered_if_mde_gt_uplift": True,
        "peeking_guard": True,
        "max_peeks_without_correction": 1,
        "use_multiple_comparison": True,
    }
    if path and os.path.exists(path):
        try:
            import yaml
            with open(path) as fh:
                loaded = yaml.safe_load(fh) or {}
            defaults.update({k: v for k, v in loaded.items() if v is not None})
        except ImportError:
            print("warning: PyYAML not installed, using default config", file=sys.stderr)
    return defaults


# --------------------------------------------------------------------------- #
# Parsing                                                                     #
# --------------------------------------------------------------------------- #
def parse_direction(raw):
    s = str(raw).strip().lower()
    if s in ("increase", "up", "+1", "1", "higher", "positive"):
        return 1
    if s in ("decrease", "down", "-1", "lower", "negative"):
        return -1
    return 1  # default: bigger is better


def read_experiments(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def _f(row, key, default=None):
    v = row.get(key)
    if v is None or str(v).strip() == "":
        return default
    return float(v)


def _i(row, key, default=None):
    v = _f(row, key, default)
    return int(v) if v is not None else default


# --------------------------------------------------------------------------- #
# Statistics                                                                  #
# --------------------------------------------------------------------------- #
def two_proportion_test(n1, p1, n2, p2, alpha):
    """Two-sided two-proportion z-test.

    Test statistic uses the POOLED proportion (standard for H0: p1 == p2).
    The confidence interval for the difference uses the UNPOOLED (Wald) SE,
    which is the conventional choice for reporting effect size.
    """
    x1, x2 = p1 * n1, p2 * n2
    pooled = (x1 + x2) / (n1 + n2)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    diff = p2 - p1
    if se_pooled == 0:
        z, pval = 0.0, 1.0
    else:
        z = diff / se_pooled
        pval = 2 * stats.norm.sf(abs(z))

    se_wald = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zc = stats.norm.ppf(1 - alpha / 2)
    ci = (diff - zc * se_wald, diff + zc * se_wald)

    # MDE (absolute) at target power, using pooled variance under baseline p1
    z_beta = stats.norm.ppf(Z_POWER_DEFAULT)
    se_design = math.sqrt(p1 * (1 - p1) * (1 / n1 + 1 / n2)) if 0 < p1 < 1 else se_pooled
    mde_abs = (zc + z_beta) * se_design
    return {"stat": z, "pval": pval, "diff": diff, "ci": ci,
            "se": se_wald, "mde_abs": mde_abs, "test": "two-proportion z-test"}


def welch_t_test(n1, m1, v1, n2, m2, v2, alpha):
    """Welch's two-sample t-test (unequal variances) from summary stats."""
    if v1 is None or v2 is None:
        raise ValueError("metric_type 'mean' requires control_var and treatment_var")
    se = math.sqrt(v1 / n1 + v2 / n2)
    diff = m2 - m1
    if se == 0:
        t, pval, df = 0.0, 1.0, n1 + n2 - 2
    else:
        t = diff / se
        # Welch-Satterthwaite degrees of freedom
        df = (v1 / n1 + v2 / n2) ** 2 / (
            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        )
        pval = 2 * stats.t.sf(abs(t), df)
    tc = stats.t.ppf(1 - alpha / 2, df)
    ci = (diff - tc * se, diff + tc * se)

    z_beta = stats.norm.ppf(Z_POWER_DEFAULT)
    zc = stats.norm.ppf(1 - alpha / 2)
    mde_abs = (zc + z_beta) * se
    return {"stat": t, "pval": pval, "diff": diff, "ci": ci, "df": df,
            "se": se, "mde_abs": mde_abs, "test": "Welch's t-test"}


def compute_stats(row, alpha):
    metric_type = str(row["metric_type"]).strip().lower()
    exp_id = row["experiment_id"].strip()
    name = row.get("name", "").strip() or exp_id
    direction = parse_direction(row.get("direction", "increase"))
    n1, n2 = _i(row, "control_n"), _i(row, "treatment_n")
    c_val, t_val = _f(row, "control_value"), _f(row, "treatment_value")
    peeks = _i(row, "peeks", 0)

    if metric_type == "proportion":
        r = two_proportion_test(n1, c_val, n2, t_val, alpha)
    elif metric_type == "mean":
        r = welch_t_test(n1, c_val, _f(row, "control_var"),
                         n2, t_val, _f(row, "treatment_var"), alpha)
    else:
        raise ValueError(f"{exp_id}: unknown metric_type '{metric_type}'")

    rel_uplift = (t_val - c_val) / c_val if c_val != 0 else float("nan")
    mde_rel = r["mde_abs"] / abs(c_val) if c_val != 0 else float("nan")

    return {
        "experiment_id": exp_id,
        "name": name,
        "metric_type": metric_type,
        "test": r["test"],
        "control_n": n1,
        "treatment_n": n2,
        "control_value": c_val,
        "treatment_value": t_val,
        "direction": direction,
        "abs_diff": r["diff"],
        "rel_uplift": rel_uplift,
        "ci_low": r["ci"][0],
        "ci_high": r["ci"][1],
        "stat": r["stat"],
        "p_value": r["pval"],
        "mde": mde_rel,
        "peeks": peeks,
    }


# --------------------------------------------------------------------------- #
# Multiple-comparison correction                                             #
# --------------------------------------------------------------------------- #
def benjamini_hochberg(pvals, fdr):
    """Return (adjusted_pvalues, rejected_flags) via Benjamini-Hochberg.

    Adjusted p-values are the monotone step-up BH values; an experiment is
    "rejected" (significant) when its adjusted p-value <= fdr.
    """
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    prev = 1.0
    # walk from largest to smallest p-value enforcing monotonicity
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        val = pvals[i] * n / rank
        prev = min(prev, val)
        adj[i] = min(prev, 1.0)
    rejected = [adj[i] <= fdr for i in range(n)]
    return adj, rejected


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def render_html(results, cfg, meta, template_path):
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(template_path) or "."),
        autoescape=select_autoescape(["html", "j2"]),
    )
    env.filters["pct"] = lambda x, d=2: ("n/a" if x is None or (isinstance(x, float) and math.isnan(x))
                                         else f"{x*100:.{d}f}%")
    env.filters["sig"] = lambda x, d=4: ("n/a" if x is None or (isinstance(x, float) and math.isnan(x))
                                         else f"{x:.{d}g}")
    template = env.get_template(os.path.basename(template_path))
    return template.render(results=results, cfg=cfg, meta=meta)


def write_summary_csv(results, path):
    cols = ["experiment_id", "name", "metric_type", "test", "control_n", "treatment_n",
            "control_value", "treatment_value", "abs_diff", "rel_uplift",
            "ci_low", "ci_high", "stat", "p_value", "p_adj", "significant_bh",
            "mde", "verdict", "reason", "flags"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = dict(r)
            row["flags"] = "|".join(r.get("flags", []))
            w.writerow(row)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def run(input_path, outdir, config_path, template_path):
    cfg = load_config(config_path)
    rows = read_experiments(input_path)

    results = [compute_stats(r, cfg["alpha"]) for r in rows]

    # Batch-wide multiple-comparison correction
    pvals = [r["p_value"] for r in results]
    if cfg.get("use_multiple_comparison", True):
        adj, rejected = benjamini_hochberg(pvals, cfg["fdr_level"])
    else:
        adj = pvals[:]
        rejected = [p <= cfg["alpha"] for p in pvals]
    for r, a, rej in zip(results, adj, rejected):
        r["p_adj"] = a
        r["significant_bh"] = rej

    # Verdicts
    vcfg = VerdictConfig.from_dict(cfg)
    for r in results:
        v = decide(r, vcfg)
        r.update(v)

    # Order: winners first, then regressions, then the rest by adjusted p-value
    order_key = {"ship": 0, "no_ship": 1, "needs_more_data": 2, "inconclusive": 3}
    results.sort(key=lambda r: (order_key.get(r["verdict"], 9), r["p_adj"]))

    counts = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    os.makedirs(outdir, exist_ok=True)
    meta = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "input": os.path.basename(input_path),
        "n_experiments": len(results),
        "counts": counts,
    }

    html = render_html(results, cfg, meta, template_path)
    html_path = os.path.join(outdir, "report.html")
    with open(html_path, "w") as fh:
        fh.write(html)

    csv_path = os.path.join(outdir, "summary.csv")
    write_summary_csv(results, csv_path)

    print(f"Analyzed {len(results)} experiments.")
    for k in ("ship", "no_ship", "needs_more_data", "inconclusive"):
        if k in counts:
            print(f"  {k:16s}: {counts[k]}")
    print(f"Report:  {html_path}")
    print(f"Summary: {csv_path}")
    return html_path, csv_path


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Batch A/B experiment analysis pipeline")
    ap.add_argument("--input", required=True, help="path to experiments.csv")
    ap.add_argument("--outdir", default="output", help="output directory (default: output)")
    ap.add_argument("--config", default=os.path.join(here, "config.yaml"),
                    help="verdict config YAML (default: config.yaml)")
    ap.add_argument("--template", default=os.path.join(here, "templates", "report.html.j2"),
                    help="Jinja2 HTML template")
    args = ap.parse_args(argv)
    run(args.input, args.outdir, args.config, args.template)


if __name__ == "__main__":
    main()
