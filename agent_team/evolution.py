"""Automatic reviewer evolution (idea #6): let the council improve over time.

Models improve; a seat wired to last year's model is leaving detection on the
table. Because the judgment log records which *model* produced each seat's
verdict, we can compare models head-to-head on the same labeled cases and
recommend a swap when a challenger measurably beats the incumbent — turning
"should we upgrade the security seat to GPT-6?" from a vibe into a decision with
a confidence interval behind it.

Per (seat, model) we estimate sensitivity (defect recall) and specificity from
the labeled log, rank by balanced accuracy, and recommend a swap only when the
challenger's advantage clears a margin *and* rests on enough evidence — never
promoting a model that merely got lucky on three cases. The recommendation is
advisory: a human still changes the config.

Deterministic; reuses the reliability posteriors; unit-tested.
"""

from collections import defaultdict
from dataclasses import dataclass, field

from .reliability import beta_ppf


@dataclass
class ModelStanding:
    seat: str
    model: str
    sens_mean: float
    sens_lo: float
    sens_hi: float
    spec_mean: float
    bal_acc: float           # (sensitivity + specificity) / 2
    n_defect: int
    n_clean: int


@dataclass
class SeatEvolution:
    seat: str
    incumbent: str | None
    recommended: str | None
    swap: bool
    reason: str
    standings: list[ModelStanding] = field(default_factory=list)


def _standings(judgments, prior_a: float, prior_b: float, ci: float) -> dict[str, dict[str, ModelStanding]]:
    agg: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fn": 0, "tn": 0, "fp": 0})
    for j in judgments:
        seat, model, verdict, label = (j.get("seat"), j.get("model"),
                                       j.get("verdict"), j.get("label"))
        if not seat or not model or verdict not in ("approve", "reject") \
                or label not in ("defect", "clean"):
            continue
        a = agg[(seat, model)]
        if label == "defect":
            a["tp" if verdict == "reject" else "fn"] += 1
        else:
            a["tn" if verdict == "approve" else "fp"] += 1

    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    out: dict[str, dict[str, ModelStanding]] = defaultdict(dict)
    for (seat, model), a in agg.items():
        sa, sb = prior_a + a["tp"], prior_b + a["fn"]
        ta, tb = prior_a + a["tn"], prior_b + a["fp"]
        sens = sa / (sa + sb)
        spec = ta / (ta + tb)
        out[seat][model] = ModelStanding(
            seat=seat, model=model,
            sens_mean=sens, sens_lo=beta_ppf(lo_q, sa, sb), sens_hi=beta_ppf(hi_q, sa, sb),
            spec_mean=spec, bal_acc=(sens + spec) / 2,
            n_defect=a["tp"] + a["fn"], n_clean=a["tn"] + a["fp"])
    return out


def recommend_evolution(judgments, min_n: int = 10, margin: float = 0.1,
                        prior_a: float = 0.5, prior_b: float = 0.5,
                        ci: float = 0.90) -> dict[str, SeatEvolution]:
    """Per seat, recommend the best-supported model and whether to swap.

    `min_n` is the minimum labeled defects *and* cleans a challenger needs before
    it is eligible; `margin` is how much balanced accuracy it must beat the
    incumbent by. Incumbent = the model with the most judgments on that seat.
    """
    per_seat = _standings(judgments, prior_a, prior_b, ci)
    result: dict[str, SeatEvolution] = {}
    for seat, models in per_seat.items():
        standings = sorted(models.values(), key=lambda s: s.bal_acc, reverse=True)
        incumbent = max(models.values(),
                        key=lambda s: s.n_defect + s.n_clean).model
        eligible = [s for s in standings
                    if s.n_defect >= min_n and s.n_clean >= min_n]
        inc = models[incumbent]
        if not eligible:
            reason = f"insufficient evidence (need n>={min_n} defect & clean per model)"
            result[seat] = SeatEvolution(seat, incumbent, incumbent, False, reason, standings)
            continue
        best = eligible[0]
        swap = best.model != incumbent and (best.bal_acc - inc.bal_acc) >= margin
        if swap:
            reason = (f"{best.model} balanced-acc {best.bal_acc:.0%} beats incumbent "
                      f"{incumbent} {inc.bal_acc:.0%} by >= {margin:.0%}")
        elif best.model == incumbent:
            reason = f"incumbent {incumbent} is already the best-supported model"
        else:
            reason = (f"challenger {best.model} leads but by < {margin:.0%} — hold")
        result[seat] = SeatEvolution(seat, incumbent, best.model if swap else incumbent,
                                     swap, reason, standings)
    return result


def format_evolution(result: dict[str, SeatEvolution]) -> str:
    if not result:
        return "\n=== Reviewer evolution ===\n(no model-tagged judgments yet)"
    lines = ["", "=== Reviewer evolution (per-seat model comparison) ==="]
    for seat in sorted(result):
        ev = result[seat]
        flag = "SWAP ->" if ev.swap else "keep"
        lines.append(f"  {seat:<12} {flag} {ev.recommended}   ({ev.reason})")
        for s in ev.standings:
            mark = "*" if s.model == ev.incumbent else " "
            lines.append(f"    {mark} {s.model:<22} bal_acc={s.bal_acc:>4.0%} "
                         f"sens={s.sens_mean:>4.0%}[{s.sens_lo:.0%}-{s.sens_hi:.0%}] "
                         f"spec={s.spec_mean:>4.0%} (n={s.n_defect}+{s.n_clean})")
    return "\n".join(lines)
