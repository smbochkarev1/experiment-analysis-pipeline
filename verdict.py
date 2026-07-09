"""Verdict rules for the experiment analysis pipeline.

A verdict turns a raw statistical result into a decision. The rules are
deliberately explicit and configurable (see config.yaml) so that a reviewer
can see *why* an experiment was shipped or held, not just a p-value.

Verdict taxonomy
----------------
- ship            : significant AFTER correction, effect in the desired
                    direction, and practically meaningful.
- no_ship         : significant AFTER correction, but the effect is in the
                    wrong direction (a regression / guardrail breach).
- needs_more_data : underpowered - not enough sample to detect the observed
                    effect, so absence of significance is uninformative.
- inconclusive    : adequately powered but no significant effect, or a
                    significant-but-tiny effect below the practical floor.
"""

from dataclasses import dataclass


@dataclass
class VerdictConfig:
    alpha: float = 0.05
    min_practical_uplift: float = 0.005
    min_sample_per_arm: int = 1000
    underpowered_if_mde_gt_uplift: bool = True
    peeking_guard: bool = True
    max_peeks_without_correction: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "VerdictConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


def decide(result: dict, cfg: VerdictConfig) -> dict:
    """Return {'verdict': str, 'reason': str, 'flags': [str]}.

    `result` is the per-experiment stats dict produced by analyze.compute_stats,
    already carrying the corrected significance flag `significant_bh`.
    """
    flags = []

    uplift = result["rel_uplift"]              # signed relative uplift
    direction = result["direction"]            # +1 want increase, -1 want decrease
    significant = result["significant_bh"]     # significant after BH correction
    mde = result["mde"]                        # minimum detectable effect (relative)
    n_min = min(result["control_n"], result["treatment_n"])
    peeks = result.get("peeks", 0) or 0

    # Effect points in the direction we wanted (desired improvement)?
    desired = (uplift * direction) > 0
    practically_meaningful = abs(uplift) >= cfg.min_practical_uplift

    # --- Guardrails / flags -------------------------------------------------
    # "Underpowered" is a statement about the DESIGN, not about observed noise:
    # the sample is too small to reliably detect the effect sizes we care about.
    # Driving it purely off sample size keeps the rule defensible; the MDE vs
    # observed-effect comparison is surfaced as context only.
    underpowered = n_min < cfg.min_sample_per_arm
    if underpowered:
        flags.append("sample_below_min")
    if cfg.underpowered_if_mde_gt_uplift and mde is not None and abs(uplift) < mde \
            and not significant:
        flags.append("observed_effect_below_mde")
    if cfg.peeking_guard and peeks > cfg.max_peeks_without_correction:
        flags.append("peeking_no_alpha_spending")

    # --- Decision tree ------------------------------------------------------
    if significant:
        if desired and practically_meaningful:
            return _v("ship",
                      f"Significant after BH correction (p_adj={result['p_adj']:.4g}), "
                      f"{uplift:+.2%} in the desired direction, above the "
                      f"{cfg.min_practical_uplift:.1%} practical floor.", flags)
        if not desired:
            return _v("no_ship",
                      f"Significant after correction but effect is a regression "
                      f"({uplift:+.2%}, wrong direction).", flags)
        # significant but tiny
        return _v("inconclusive",
                  f"Statistically significant but only {uplift:+.2%} - below the "
                  f"{cfg.min_practical_uplift:.1%} practical floor; not worth shipping.",
                  flags)

    # Not significant after correction
    if underpowered:
        return _v("needs_more_data",
                  f"Not significant, but underpowered: n_min={n_min:,} and observed "
                  f"effect ({uplift:+.2%}) is within noise (MDE={mde:.2%}). "
                  f"Absence of signal is uninformative.", flags)

    return _v("inconclusive",
              f"Adequately powered (n_min={n_min:,}, MDE={mde:.2%}) but no significant "
              f"effect after correction (p_adj={result['p_adj']:.4g}). Flat result.",
              flags)


def _v(verdict: str, reason: str, flags: list) -> dict:
    return {"verdict": verdict, "reason": reason, "flags": sorted(set(flags))}
