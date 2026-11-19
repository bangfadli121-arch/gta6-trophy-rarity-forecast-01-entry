"""GTA VI trophy rarity forecaster (program/v1, Python 3.13, stdlib only).

Idea: an Exophase earn rate is a ratio, earners / players-tracked, and both
sides move.  A "rate stays put" forecast is wrong whenever Exophase refreshes:
on launch night the page sat unchanged at 638 tracked players, then refreshed
to 3,868 and every rate but the 100% one fell, because newly tracked players
hold fewer trophies than the early crowd.  So this program forecasts the two
sides separately and divides:

  * players tracked P(t) and per-trophy earners E_i(t) = rate_i(t) * P(t) are
    each fitted in log space against x = ln(1 + hours since the release day
    began, 2026-11-19T00:00:00Z).  Log-time growth saturates naturally, which
    is what a launch surge does: the player base keeps growing, but each
    extra day adds proportionally less.
  * Each trophy's earner slope is shrunk toward the median slope across
    trophies, so one noisy trophy cannot extrapolate wildly; slopes and the
    implied rate move are capped.
  * The rate forecast is E_i(target) / P(target), turned into five quantiles
    on the logit scale with a spread that combines a random-walk term (the
    house baseline's 0.5 * sqrt(days)), the fit residual, and a floor for
    model error.

When the history is flat (too few distinct snapshots to fit a slope -- the
practice cases, where every read is the same launch-night page), the program
falls back on a prior calibrated on the one refresh actually observed:
players 638 -> 3,868 (x6.06) and trophies earned 943 -> 4,276 between the
launch-night page and the 04:30Z read.  The headline ratio (4.53/6.06 =
0.748) is misleading: it is dominated by trophy #2, which sits at 100% in
both reads.  Excluding #2, earners grew only 305 -> 408 (x1.34) while
players grew x6.06, so a typical trophy's rate fell to x0.22.  That is a
log-time dilution slope of ln(0.22) / ln(1 + 4.5h) ~= -0.89 per unit of
ln(1 + hours), applied from the snapshot's time to the target and capped.
The live page later the same morning sums to the same totals (10.53% of
rates ex-#2 vs 10.55% at 04:30), i.e. it is the same refresh, which is why
the proxy scoring in the build notes agrees with this prior.  A wider
spread is added in that regime because a refresh step is lumpy, not
continuous.

Deterministic: no randomness, no wall-clock reads, no network.  Two runs on
the same case write byte-identical output.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release-day anchor for the log-time axis.
ANCHOR = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# First observed Exophase refresh (launch-night page -> 04:30Z read, 19 Nov):
# players 638 -> 3,868 (x6.06); trophies earned ex-#2 305 -> 408 (x1.34),
# so a typical (non-100%) rate fell to x0.22 over ln(1 + 4.5h) of log-time.
REFRESH_RATE_RATIO = 0.22           # typical rate ratio across one refresh
REFRESH_X_SPAN = math.log1p(4.5)   # ln(1 + hours) covered by that refresh
DILUTION_SLOPE = math.log(REFRESH_RATE_RATIO) / REFRESH_X_SPAN  # ~ -0.89
FLAT_SHIFT_CAP = -2.5               # never dilute a flat-history rate further
FLAT_SPREAD = 0.55               # extra logit spread while history is flat
BASE_SPREAD = 0.15               # floor for model error in the fitted regime
RW_SIGMA = 0.5                   # logit random-walk sigma per sqrt(day)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up, so output is stable and short."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def x_of(moment):
    """ln(1 + hours since the release-day anchor)."""
    hours = max(0.0, (moment - ANCHOR).total_seconds() / 3600.0)
    return math.log1p(hours)


def ols(xs, ys):
    """Least squares line y = a + b*x; returns (a, b, residual_sd)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / max(1, n - 2))
    return a, b, resid


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def last_rates(case):
    """Last non-null rate per trophy, scanning history backwards."""
    n = len(case["trophies"])
    out = [None] * n
    for read in reversed(case["history"]):
        rates = read.get("rates") or []
        for i in range(n):
            if out[i] is None and i < len(rates) and rates[i] is not None:
                out[i] = rates[i]
        if all(v is not None for v in out):
            break
    return [0.01 if v is None else v for v in out]


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    x_target = x_of(target)
    base = last_rates(case)
    rw = RW_SIGMA * math.sqrt(max(0.0, horizon_days))

    # Distinct snapshots only: Exophase sits unchanged for hours, and a run
    # of identical reads carries no more information than one of them.
    seen = set()
    snaps = []
    for read in history:
        rates = tuple(read.get("rates") or [])
        key = (read.get("players_tracked"), rates)
        if key in seen:
            continue
        seen.add(key)
        snaps.append(read)
    xs = [x_of(parse_time(r["at"])) for r in snaps]
    can_fit = len(snaps) >= 3 and (max(xs) - min(xs)) > 0.05

    medians = [None] * len(trophies)
    spreads = [None] * len(trophies)

    if can_fit:
        # --- dilution model: fit ln(players) and ln(earners) vs ln-time ---
        ln_p = [math.log(max(1, r["players_tracked"])) for r in snaps]
        a_p, b_p, _ = ols(xs, ln_p)
        b_p = max(-1.5, min(1.5, b_p))
        ln_p_t = a_p + b_p * x_target

        slopes = []
        fits = []
        for i in range(len(trophies)):
            px, py = [], []
            for r, x in zip(snaps, xs):
                rates = r.get("rates") or []
                if i < len(rates) and rates[i] is not None and r.get("players_tracked"):
                    earners = max(0.5, rates[i] / 100.0 * r["players_tracked"])
                    px.append(x)
                    py.append(math.log(earners))
            if len(px) >= 3 and (max(px) - min(px)) > 0.05:
                a_i, b_i, resid = ols(px, py)
                slopes.append(b_i)
                fits.append((a_i, b_i, resid, len(px)))
            else:
                fits.append(None)
        b_med = median(slopes) if slopes else 0.0
        for i in range(len(trophies)):
            fit = fits[i]
            if fit is None:
                medians[i] = base[i]
                spreads[i] = math.sqrt(rw * rw + 0.45 * 0.45)
                continue
            a_i, b_i, resid, n_i = fit
            # Shrink the trophy's slope toward the cross-trophy median.
            w = n_i / (n_i + 3.0)
            b_i = w * b_i + (1.0 - w) * b_med
            b_i = max(-2.0, min(2.0, b_i))
            ln_e_t = a_i + b_i * x_target
            rate = 100.0 * math.exp(ln_e_t - ln_p_t)
            # Cap the implied move away from the last rate.
            shift = logit(rate) - logit(base[i])
            shift = max(-3.5, min(2.5, shift))
            medians[i] = inv_logit(logit(base[i]) + shift)
            resid_c = max(0.10, min(0.80, resid))
            spreads[i] = math.sqrt(rw * rw + resid_c * resid_c + BASE_SPREAD ** 2)
    else:
        # --- flat history: dilute along the observed refresh slope, from
        # the snapshot's log-time to the target's; the spread admits the
        # step is lumpy. ---
        x_first = xs[0] if xs else x_of(parse_time(case["at"]))
        shift = max(FLAT_SHIFT_CAP, DILUTION_SLOPE * (x_target - x_first))
        spread = math.sqrt(rw * rw + FLAT_SPREAD * FLAT_SPREAD)
        for i in range(len(trophies)):
            medians[i] = inv_logit(logit(base[i]) + shift)
            spreads[i] = spread

    rows = []
    for i, trophy in enumerate(trophies):
        med = medians[i]
        sig = spreads[i]
        if base[i] >= 99.99:
            # The opening-mission trophy: everyone tracked holds it, and it
            # stayed at 100% through every refresh seen.  Keep it pinned
            # near the top with a narrow band, not a wide logit walk.
            med, sig = 99.9, 0.35
        lg = logit(med)
        qs = []
        for z in Z:
            q = inv_logit(lg + z * sig)
            qs.append(round4(min(100.0, max(0.01, q))))
        # Rounding can only break monotonicity by a hair; enforce it.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
