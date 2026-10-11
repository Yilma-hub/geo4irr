"""
Soil moisture sensor network design: how many, what kind, and where.

Activity 7 of the project narrative. The question a Delaware grower actually
asks is narrower than the research framing: "I can afford three probes on this
pivot. Where do they go so the number they give me means something?"

The method here is maximum-entropy / minimum-redundancy selection over a
candidate grid. Candidate locations are scored by how much they reduce
uncertainty about field-average root zone depletion, given the spatial structure
of the soil map units and an exponential covariance model of soil water. The
selection is greedy, which is within a constant factor of optimal for this
submodular objective and runs fast enough to do interactively on a phone.

Placement depth then follows from the crop's root distribution rather than a
fixed rule, because putting the deep sensor at 36 inches on a soil with a
restrictive layer at 24 inches produces a reading that never changes and quietly
destroys the grower's trust in the whole system.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from .bayesian import SENSOR_CLASSES, SensorClass
from .fao56 import SoilProfile
from .soils import get_soil


@dataclass(slots=True)
class Candidate:
    """A possible sensor station within the field."""

    id: str
    x: float                 # local coordinates, feet from field origin
    y: float
    soil_key: str
    elevation_ft: float = 0.0
    zone: str = "default"
    yield_index: float = 1.0   # relative productivity from yield map history

    @property
    def soil(self) -> SoilProfile:
        return get_soil(self.soil_key)


def _covariance(a: Candidate, b: Candidate, range_ft: float = 400.0,
                nugget: float = 0.15) -> float:
    """Exponential spatial covariance of root zone water content.

    Two stations on different soil series are treated as much less correlated
    than distance alone would imply, since the map unit boundary is the real
    discontinuity in available water capacity.
    """
    d = math.hypot(a.x - b.x, a.y - b.y)
    base = (1.0 - nugget) * math.exp(-d / range_ft)
    if a.soil_key != b.soil_key:
        base *= 0.35
    if abs(a.elevation_ft - b.elevation_ft) > 8:
        base *= 0.7
    return base + (nugget if a.id == b.id else 0.0)


def _entropy_gain(chosen: list[Candidate], cand: Candidate,
                  range_ft: float) -> float:
    """Conditional variance of a candidate given the already-chosen stations.

    This is the greedy maximum-entropy criterion: pick the station whose reading
    is least predictable from what we already measure, because that is the one
    carrying new information.
    """
    if not chosen:
        return 1.0
    # Conditional variance under a Gaussian process, computed with a small
    # Gauss-Jordan solve. Field-scale designs stay well under 10 stations, so
    # the cubic cost is irrelevant.
    k = len(chosen)
    K = [[_covariance(chosen[i], chosen[j], range_ft) for j in range(k)]
         for i in range(k)]
    kv = [_covariance(cand, chosen[i], range_ft) for i in range(k)]
    for i in range(k):
        K[i][i] += 1e-6
    # Solve K w = kv
    A = [row[:] + [kv[i]] for i, row in enumerate(K)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) < 1e-12:
            continue
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        A[col] = [v / pv for v in A[col]]
        for r in range(k):
            if r != col and abs(A[r][col]) > 1e-12:
                f = A[r][col]
                A[r] = [a - f * b for a, b in zip(A[r], A[col])]
    w = [A[i][k] for i in range(k)]
    reduction = sum(w[i] * kv[i] for i in range(k))
    return max(0.0, 1.0 - reduction)


@dataclass(slots=True)
class PlacementPlan:
    stations: list[dict]
    n_stations: int
    total_cost_usd: float
    expected_depletion_sd_in: float
    sd_without_sensors_in: float
    coverage_score: float
    rationale: list[str] = field(default_factory=list)


def design_network(
    candidates: Sequence[Candidate],
    budget_usd: float,
    root_depth_in: float,
    sensor_key: str = "capacitance_pro",
    max_stations: int = 6,
    correlation_range_ft: float = 400.0,
    baseline_sd_in: float = 0.55,
    restrictive_layer_in: float | None = None,
) -> PlacementPlan:
    """Greedy maximum-entropy sensor siting under a budget constraint."""
    sc = SENSOR_CLASSES[sensor_key]
    affordable = min(max_stations, int(budget_usd // max(1.0, sc.cost_usd)))
    if affordable < 1:
        cheapest = min(SENSOR_CLASSES.values(), key=lambda s: s.cost_usd)
        return PlacementPlan(
            stations=[], n_stations=0, total_cost_usd=0.0,
            expected_depletion_sd_in=baseline_sd_in,
            sd_without_sensors_in=baseline_sd_in, coverage_score=0.0,
            rationale=[
                f"A {sc.label} runs about ${sc.cost_usd:,.0f} installed, which is "
                f"over the ${budget_usd:,.0f} budget. The {cheapest.label} at "
                f"${cheapest.cost_usd:,.0f} would fit and still cuts model "
                "uncertainty meaningfully once it is field calibrated."
            ],
        )

    chosen: list[Candidate] = []
    remaining = list(candidates)
    rationale: list[str] = []

    # Seed at the location closest to the field's productivity centroid, weighted
    # by yield index. That station represents the acres that pay the bills.
    tw = sum(c.yield_index for c in remaining) or 1.0
    cx = sum(c.x * c.yield_index for c in remaining) / tw
    cy = sum(c.y * c.yield_index for c in remaining) / tw
    seed = min(remaining, key=lambda c: math.hypot(c.x - cx, c.y - cy))
    chosen.append(seed)
    remaining.remove(seed)
    rationale.append(
        f"Station 1 sits on {seed.soil.series} near the productivity centre of "
        "the field. It is the reference for scheduling the whole set.")

    while len(chosen) < affordable and remaining:
        scored = [(c, _entropy_gain(chosen, c, correlation_range_ft)) for c in remaining]
        best, gain = max(scored, key=lambda t: t[1])
        if gain < 0.18 and len(chosen) >= 2:
            rationale.append(
                f"Stopped at {len(chosen)} stations. Another probe would repeat "
                "what the existing ones already tell you.")
            break
        chosen.append(best)
        remaining.remove(best)
        series_note = (
            f"different soil ({best.soil.series})"
            if best.soil_key != chosen[0].soil_key
            else "a separate part of the field"
        )
        rationale.append(
            f"Station {len(chosen)} covers {series_note}; it adds "
            f"{gain * 100:.0f}% new information.")

    # Depth placement
    depth_limit = min(root_depth_in, restrictive_layer_in or root_depth_in)
    stations = []
    for i, c in enumerate(chosen, start=1):
        depths = [d for d in sc.depths_in if d <= depth_limit + 2]
        if not depths:
            depths = [max(4, int(depth_limit * 0.4))]
        stations.append({
            "station": i,
            "x_ft": round(c.x),
            "y_ft": round(c.y),
            "zone": c.zone,
            "soil": c.soil.series,
            "soil_key": c.soil_key,
            "awc_in_per_in": c.soil.awc_in_per_in,
            "depths_in": depths,
            "sensor": sc.label,
            "telemetry": sc.telemetry,
            "cost_usd": sc.cost_usd,
            "note": (
                "Field calibrate against a gravimetric sample before trusting "
                "absolute values."
                if sc.salinity_sensitive else
                "Log the install depth; a two-inch error is a real error in the balance."
            ),
        })

    n = len(chosen)
    # Uncertainty reduction: sensor error floor plus a spatial representation term
    # that decays with the number of well-separated stations.
    sigma_sensor = sc.sigma_depletion_in(depth_limit)
    spatial = baseline_sd_in * (0.75 / math.sqrt(n))
    expected_sd = math.sqrt(sigma_sensor ** 2 + spatial ** 2)

    if depth_limit < root_depth_in:
        rationale.append(
            f"Deepest sensor set at {depths[-1]} inches because the profile is "
            f"restricted at {restrictive_layer_in:.0f} inches. Anything deeper "
            "would read a layer the roots never reach.")

    rationale.append(
        f"Expected uncertainty in field depletion drops from about "
        f"{baseline_sd_in:.2f} in modelled to {expected_sd:.2f} in with sensors. "
        "That is roughly half a day of corn water use in July, which is the "
        "difference between starting the pivot on the right day and the day after.")

    return PlacementPlan(
        stations=stations,
        n_stations=n,
        total_cost_usd=round(n * sc.cost_usd, 2),
        expected_depletion_sd_in=round(expected_sd, 3),
        sd_without_sensors_in=baseline_sd_in,
        coverage_score=round(min(1.0, 0.35 + 0.16 * n), 2),
        rationale=rationale,
    )


def grid_candidates(
    width_ft: float, height_ft: float, spacing_ft: float,
    soil_map: dict[str, list[tuple[float, float, float, float]]] | None = None,
    default_soil: str = "sassafras",
    yield_map: dict[tuple[int, int], float] | None = None,
) -> list[Candidate]:
    """Build a candidate grid over a field boundary.

    soil_map maps a soil key to a list of (x0, y0, x1, y1) rectangles. In
    production this comes from the intersection of the field polygon with the
    SSURGO map units; the rectangle form keeps the module testable without a
    geometry dependency.
    """
    out: list[Candidate] = []
    n = 0
    y = spacing_ft / 2
    while y < height_ft:
        x = spacing_ft / 2
        while x < width_ft:
            soil = default_soil
            if soil_map:
                for key, boxes in soil_map.items():
                    if any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in boxes):
                        soil = key
                        break
            yi = 1.0
            if yield_map:
                yi = yield_map.get((int(x // spacing_ft), int(y // spacing_ft)), 1.0)
            n += 1
            out.append(Candidate(id=f"c{n}", x=x, y=y, soil_key=soil, yield_index=yi))
            x += spacing_ft
        y += spacing_ft
    return out


def compare_sensor_tiers(
    candidates: Sequence[Candidate], budget_usd: float, root_depth_in: float,
) -> list[dict]:
    """What does the budget buy: a few good probes or many cheap ones?

    This is the hybrid-deployment question the narrative raises. On Delaware
    sand the answer is usually not the obvious one: spatial variability in
    available water capacity across a single pivot is often larger than the
    measurement error of a mid-tier probe, so more stations beat better stations
    until you run out of distinct soil map units.
    """
    rows = []
    for key, sc in SENSOR_CLASSES.items():
        if sc.cost_usd <= 0:
            continue
        plan = design_network(candidates, budget_usd, root_depth_in,
                              sensor_key=key, max_stations=10)
        rows.append({
            "sensor": sc.label,
            "sensor_key": key,
            "unit_cost": sc.cost_usd,
            "stations_affordable": plan.n_stations,
            "total_cost": plan.total_cost_usd,
            "expected_sd_in": plan.expected_depletion_sd_in,
            "telemetry": sc.telemetry,
        })
    rows.sort(key=lambda r: r["expected_sd_in"])
    if rows:
        rows[0]["recommended"] = True
    return rows


# ---------------------------------------------------------------------------
# Sensor placement depth (Activity 7)
# ---------------------------------------------------------------------------
# Activity 7 splits the problem in two, and the second half is easy to miss:
#
#   "Once the optimal number and locations of sensors are identified, the
#    placement depth of sensors in the root zone is obtained from the universal
#    Multiple Linear Regression approach."
#
# Horizontal placement is a maximum-entropy problem — where do probes tell you
# most about the field. Depth is a different question with a different answer:
# given that you can only afford three sensors on a profile, at which depths do
# their readings best predict total root-zone storage?
#
# The universal MLR approach answers it directly. Model profile storage as a
# linear combination of the candidate depths, fit against a record of full
# profile measurements, and choose the subset of depths maximising adjusted R².
# "Universal" here means the same fitted relation transfers across fields with
# similar profiles rather than being refit per installation — which matters,
# because no grower has a season of neutron probe data before deciding where to
# put their sensors.


def _ols(design: list[list[float]], y: list[float],
         ridge: float = 1e-8) -> tuple[list[float], float]:
    """Least squares by normal equations with Gaussian elimination.

    Small problems — at most half a dozen depths — so the normal equations are
    fine and avoid a numerical dependency.

    A small ridge term is not optional here. Soil moisture at neighbouring
    depths in the same profile is very nearly collinear, which is exactly the
    redundancy this whole calculation exists to detect, and it makes X'X
    singular. Without regularisation the solve fails on precisely the candidate
    subsets that matter most, and the algorithm silently prefers the
    poorly-correlated depths it could actually invert. The ridge is scaled to
    the magnitude of X'X so it never distorts a well-conditioned fit.
    """
    n, p = len(design), len(design[0])
    xtx = [[sum(design[i][a] * design[i][b] for i in range(n)) for b in range(p)]
           for a in range(p)]
    xty = [sum(design[i][a] * y[i] for i in range(n)) for a in range(p)]

    scale = max((abs(xtx[i][i]) for i in range(p)), default=1.0) or 1.0
    for i in range(1, p):                    # never penalise the intercept
        xtx[i][i] += ridge * scale

    for c in range(p):
        pivot = max(range(c, p), key=lambda r: abs(xtx[r][c]))
        if abs(xtx[pivot][c]) < 1e-14:
            return [0.0] * p, 0.0
        xtx[c], xtx[pivot] = xtx[pivot], xtx[c]
        xty[c], xty[pivot] = xty[pivot], xty[c]
        for r in range(p):
            if r == c:
                continue
            f = xtx[r][c] / xtx[c][c]
            for k in range(c, p):
                xtx[r][k] -= f * xtx[c][k]
            xty[r] -= f * xty[c]
    beta = [xty[i] / xtx[i][i] for i in range(p)]

    mean_y = sum(y) / n
    ss_tot = sum((v - mean_y) ** 2 for v in y)
    ss_res = sum((y[i] - sum(beta[a] * design[i][a] for a in range(p))) ** 2
                 for i in range(n))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2


def _combinations(items: list, k: int):
    if k == 0:
        yield []
        return
    for i in range(len(items) - k + 1):
        for rest in _combinations(items[i + 1:], k - 1):
            yield [items[i]] + rest


def optimal_sensor_depths(
    profile_observations: list[dict],
    candidate_depths_in: list[int] | None = None,
    n_sensors: int = 3,
    root_depth_in: float = 36.0,
    restrictive_layer_in: float | None = None,
) -> dict:
    """Choose sensor depths by universal multiple linear regression.

    `profile_observations` is a list of dicts mapping depth in inches to
    volumetric water content, one per profile measurement occasion — neutron
    probe readings, gravimetric samples, or a research-grade multi-depth probe
    used to site the cheaper ones.

    The target is total root-zone storage, which is what the water balance
    actually needs. Every subset of candidate depths of the requested size is
    fitted and scored on adjusted R², which penalises adding a depth that only
    fits noise.

    Depths below a restrictive layer are dropped before fitting. A probe under a
    fragipan measures water the crop cannot reach, and it will read wet while the
    crop above it goes short — a failure mode that looks like a working sensor.
    """
    if not profile_observations:
        raise ValueError("need profile observations to fit the depth model")

    limit = min(root_depth_in, restrictive_layer_in or root_depth_in)
    depths = sorted({int(d) for obs in profile_observations for d in obs})
    if candidate_depths_in:
        depths = [d for d in candidate_depths_in if d in depths]
    usable = [d for d in depths if d <= limit]
    dropped = [d for d in depths if d > limit]
    if len(usable) < n_sensors:
        n_sensors = max(1, len(usable))

    # Target: depth-weighted storage over the root zone, inches.
    targets = []
    for obs in profile_observations:
        present = sorted(d for d in obs if d <= limit)
        total = 0.0
        prev = 0.0
        for d in present:
            total += obs[d] * (d - prev)
            prev = d
        if prev < limit and present:
            total += obs[present[-1]] * (limit - prev)
        targets.append(total)

    best = None
    scored = []
    for subset in _combinations(usable, n_sensors):
        design = [[1.0] + [obs[d] for d in subset] for obs in profile_observations]
        beta, r2 = _ols(design, targets)
        n, p = len(targets), len(subset) + 1
        adj = 1 - (1 - r2) * (n - 1) / max(1, n - p) if n > p else r2
        scored.append({"depths_in": subset, "r2": round(r2, 4),
                       "adjusted_r2": round(adj, 4)})
        if best is None or adj > best["adjusted_r2"]:
            best = {"depths_in": subset, "r2": round(r2, 4),
                    "adjusted_r2": round(adj, 4),
                    "coefficients": [round(b, 4) for b in beta]}

    scored.sort(key=lambda s: -s["adjusted_r2"])
    single = {}
    for d in usable:
        design = [[1.0, obs[d]] for obs in profile_observations]
        _, r2 = _ols(design, targets)
        single[d] = round(r2, 4)

    return {
        "recommended_depths_in": best["depths_in"],
        "adjusted_r2": best["adjusted_r2"],
        "r2": best["r2"],
        "coefficients": best["coefficients"],
        "single_depth_r2": single,
        "alternatives": scored[:6],
        "depths_dropped_below_restrictive_layer": dropped,
        "n_observations": len(profile_observations),
        "note": _depth_note(best, single, dropped, restrictive_layer_in),
    }


def _depth_note(best: dict, single: dict, dropped: list, restrictive: float | None) -> str:
    parts = []
    depths = ", ".join(f'{d}"' for d in best["depths_in"])
    parts.append(f"Put the sensors at {depths}. Together they explain "
                 f"{best['r2'] * 100:.0f} percent of the variation in root zone "
                 "storage.")
    if single:
        top = max(single, key=lambda d: single[d])
        parts.append(f'A single probe at {top}" would get '
                     f"{single[top] * 100:.0f} percent on its own — that is the "
                     "one to install first if the budget only stretches to one.")
    if dropped:
        where = (f"the restrictive layer at {restrictive:.0f} inches"
                 if restrictive else "the bottom of the root zone")
        parts.append(
            f"Depths of {', '.join(str(d) for d in dropped)} inches were dropped: "
            f"they sit below {where}, where a probe reads water the crop cannot "
            "reach and will show wet while the crop above it goes short.")
    return " ".join(parts)
