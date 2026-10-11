"""
Tradeoffs and multi-objective optimization — Activities 8 and 9.

Activity 9 names the method: "the NSGA-II multi-objective optimization technique
will be used to obtain non-dominated (i.e., optimal) irrigation and nutrient
management solutions. Strategies identified by the user via the scenario analysis
module of Geo4Irr are compared with the Pareto optimal solutions to quantify
tradeoffs in yield, costs, and conservation effects."

So this module has to do three things, and the third is the one that matters
most:

  1. NSGA-II proper — fast non-dominated sorting, crowding distance, binary
     tournament on rank then crowding, simulated binary crossover and polynomial
     mutation, elitist survival. Not a weighted sum. A weighted sum cannot find
     points on a concave Pareto front and, worse, it forces the person to state
     an exchange rate between bushels and nitrate before they have seen what the
     exchange rate is. The entire point of showing a front is that they get to
     see it first.

  2. Evaluate any candidate management strategy — irrigation regime, application
     depth, trigger threshold, nitrogen rate, number of splits — through the same
     water balance, nitrogen and hydrology engines the daily advisory uses, so
     the optimizer and the recommendation cannot disagree.

  3. Locate the grower's own current practice on that front and say what it would
     cost to move. "Your plan is 14 bushels below the frontier at the same water"
     is a usable sentence. A Pareto front on its own is a picture of someone
     else's field.

Objectives, following Activity 8's three categories — crop production and farm
profitability, irrigation and fertilization costs, environmental sustainability:

    maximize   net return, dollars per acre
    minimize   irrigation applied, acre-inches
    minimize   nitrate leached, pounds per acre

All three are converted to minimization internally, which is the convention
NSGA-II is written in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Sequence

from .crops import get_crop
from .fao56 import SoilProfile, WeatherDay, run_season
from .nitrogen import build_nitrogen_plan, nitrate_leaching_estimate, water_nutrient_index
from .soils import curve_number_for
from .swatplus import (
    FieldProcessModel, REGIMES, SYSTEMS, get_regime, get_system,
    AquiferState,
)


# ---------------------------------------------------------------------------
# Decision variables
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DecisionSpace:
    """What the optimizer is allowed to change.

    Bounds are the grower's real constraints, not abstractions. `max_application_in`
    is what the system can physically put out in a pass, `allocation_ac_in` is the
    well permit, and the trigger fraction cannot exceed 1.0 because irrigating
    past total available water is not a strategy, it is drainage.
    """
    trigger_fraction_min: float = 0.30     # of TAW at which to irrigate
    trigger_fraction_max: float = 0.95
    depth_min_in: float = 0.25
    depth_max_in: float = 1.20
    n_rate_min: float = 0.0
    n_rate_max: float = 260.0
    splits_min: int = 1
    splits_max: int = 5
    allocation_ac_in: float | None = None

    def clamp(self, genome: list[float]) -> list[float]:
        lo = [self.trigger_fraction_min, self.depth_min_in,
              self.n_rate_min, float(self.splits_min)]
        hi = [self.trigger_fraction_max, self.depth_max_in,
              self.n_rate_max, float(self.splits_max)]
        return [max(l, min(h, g)) for g, l, h in zip(genome, lo, hi)]


@dataclass(slots=True)
class Strategy:
    """One management strategy — a point in decision space."""
    trigger_fraction: float
    depth_in: float
    n_rate_lb: float
    splits: int
    regime_key: str = "full"
    system_key: str = "sprinkler_pivot"
    label: str = ""

    @classmethod
    def from_genome(cls, g: Sequence[float], regime_key: str,
                    system_key: str) -> "Strategy":
        return cls(trigger_fraction=g[0], depth_in=g[1], n_rate_lb=g[2],
                   splits=int(round(g[3])), regime_key=regime_key,
                   system_key=system_key)

    def genome(self) -> list[float]:
        return [self.trigger_fraction, self.depth_in, self.n_rate_lb, float(self.splits)]


@dataclass(slots=True)
class Outcome:
    """What a strategy produced. Objectives plus everything worth reporting."""
    strategy: Strategy
    yield_bu: float
    yield_loss_pct: float
    irrigation_in: float
    passes: int
    nitrate_leached_lb: float
    n_applied_lb: float
    water_cost: float
    nitrogen_cost: float
    gross_revenue: float
    net_return: float
    deep_percolation_in: float
    stress_days: int
    wnipi: float
    feasible: bool = True
    violation: str = ""

    def objectives(self) -> tuple[float, float, float]:
        """Minimization form: (-net return, irrigation, nitrate leached)."""
        return (-self.net_return, self.irrigation_in, self.nitrate_leached_lb)


# ---------------------------------------------------------------------------
# Evaluating a strategy through the real engines
# ---------------------------------------------------------------------------

@dataclass
class FieldContext:
    """Everything a strategy needs to be simulated against."""
    weather: Sequence[WeatherDay]
    soil: SoilProfile
    crop_key: str
    planting: date
    lat_deg: float
    acres: float = 130.0
    capacity_gpm: float = 800.0
    yield_goal_bu: float = 200.0
    crop_price: float = 4.60          # dollars per bushel
    n_price: float = 0.62             # dollars per pound of N
    water_cost_per_ac_in: float = 9.50
    previous_crop: str = "soybean_full"
    cover_crop: str = "none"
    manure: dict | None = None
    psnt_ppm: float | None = None
    irrigation_nitrate_ppm: float = 0.0
    water_table_depth_in: float | None = None


def evaluate(strategy: Strategy, ctx: FieldContext,
             space: DecisionSpace | None = None) -> Outcome:
    """Run one strategy through the water balance, nitrogen and hydrology engines.

    Deliberately not a response surface or a fitted emulator. The optimizer sees
    exactly the model the daily advisory sees, so a strategy that looks good on
    the Pareto front behaves the same way when a grower actually adopts it.
    """
    crop = get_crop(ctx.crop_key)
    regime = get_regime(strategy.regime_key)
    system = get_system(strategy.system_key)
    cn = curve_number_for_soil(ctx.soil)

    schedule, results = _simulate_schedule(strategy, ctx, crop, regime, system, cn)

    irrigation_in = sum(schedule.values())
    passes = len(schedule)
    stress_days = sum(1 for r in results if r.stress)

    # Activity 6: the irrigation schedule goes into the hydrology model, which
    # returns the Eq 1 flux terms. This is not decoration for the optimizer — a
    # water balance alone only sees percolation caused by irrigation it applied
    # itself, so it reports near-zero leaching for every strategy and the
    # environmental objective goes flat. Most Delaware nitrate leaves on rain
    # falling into a wet profile, which is a runoff-and-percolation calculation,
    # not an irrigation one.
    # Splitting is the agronomic core of the Delaware nitrogen recommendation —
    # on sand it matters more than the total rate. What drives leaching is not
    # the season's rate but how much mineral nitrogen is sitting in the profile
    # when a storm arrives, and that is what splitting reduces. Set the pool the
    # hydrology model starts with accordingly, and do not penalise splitting a
    # second time afterwards.
    split_efficiency = min(0.94, 0.62 + 0.080 * strategy.splits)
    exposed_n = strategy.n_rate_lb * (1.0 - split_efficiency)

    hydro = FieldProcessModel(
        curve_number=cn, ksat_in_hr=ctx.soil.ksat_in_hr,
        aquifer=AquiferState(water_table_depth_in=ctx.water_table_depth_in),
        irrigation_nitrate_ppm=ctx.irrigation_nitrate_ppm)
    # Background mineralisation from organic matter, plus the fertiliser
    # actually exposed at any one time.
    hydro.soil_nitrate_lb = 24.0 + exposed_n
    for r in results:
        hydro.step(r.day, rain_in=r.rain_in,
                   irrigation_in=schedule.get(r.day, 0.0),
                   depletion_in=r.depletion_in, taw_in=r.taw_in,
                   root_depth_in=r.root_depth_in, system=system, acres=ctx.acres)
    fluxes = hydro.season_totals()
    deep_perc = fluxes.get("percolation_in", 0.0)

    # Yield from the FAO-33 style stress accumulation already in the advisory
    # engine, with the critical window weighted.
    rel_et = _relative_et(results)
    yield_loss_pct = min(85.0, max(0.0, crop.yield_response_ky * (1 - rel_et) * 100))
    yield_bu = ctx.yield_goal_bu * (1 - yield_loss_pct / 100.0)

    # Nitrogen response: yield is also limited by N supply, using a Mitscherlich
    # curve rather than a linear one, because the last thirty pounds of nitrogen
    # buy far less than the first thirty and a linear model would push the
    # optimizer to absurd rates.
    n_plan = build_nitrogen_plan(
        crop_key=ctx.crop_key, yield_goal=ctx.yield_goal_bu,
        previous_crop=ctx.previous_crop, cover_crop=ctx.cover_crop,
        manure=ctx.manure, psnt_ppm=ctx.psnt_ppm,
        soil_awc_in_per_in=ctx.soil.awc_in_per_in, planting=ctx.planting)
    n_required = n_plan.total_requirement_lb
    n_available = strategy.n_rate_lb + n_plan.total_credits_lb
    n_factor = 1.0 - math.exp(-2.6 * n_available / max(1.0, n_required))
    n_factor = min(1.0, n_factor / (1.0 - math.exp(-2.6)))
    yield_bu *= n_factor

    nitrate_lb = fluxes.get("nitrate_leached_lb_per_ac", 0.0)

    water_cost = irrigation_in * ctx.water_cost_per_ac_in
    n_cost = strategy.n_rate_lb * ctx.n_price
    revenue = yield_bu * ctx.crop_price
    net = revenue - water_cost - n_cost

    feasible, violation = True, ""
    if space and space.allocation_ac_in is not None and irrigation_in > space.allocation_ac_in:
        feasible = False
        violation = (f"Applies {irrigation_in:.1f} in against a "
                     f"{space.allocation_ac_in:.1f} in allocation.")
    if strategy.depth_in > system.max_application_in:
        feasible = False
        violation = (f"{strategy.depth_in:.2f} in exceeds what a "
                     f"{system.label.lower()} can apply in one pass.")

    wnipi = water_nutrient_index(
        yield_farm=yield_bu, yield_baseline=max(1.0, ctx.yield_goal_bu * 0.82),
        irrigation_in=irrigation_in,
        eto_in=max(1.0, sum(r.eto_in for r in results)),
        n_applied=strategy.n_rate_lb,
        n_uptake_baseline=max(1.0, n_required * 0.80),
        p_applied=n_plan.credits.get("p2o5_applied", 60.0) if n_plan.credits else 60.0,
        p_uptake_baseline=55.0)

    return Outcome(
        strategy=strategy, yield_bu=round(yield_bu, 1),
        yield_loss_pct=round(yield_loss_pct, 1),
        irrigation_in=round(irrigation_in, 2), passes=passes,
        nitrate_leached_lb=round(nitrate_lb, 1),
        n_applied_lb=round(strategy.n_rate_lb, 1),
        water_cost=round(water_cost, 2), nitrogen_cost=round(n_cost, 2),
        gross_revenue=round(revenue, 2), net_return=round(net, 2),
        deep_percolation_in=round(deep_perc, 2), stress_days=stress_days,
        wnipi=wnipi, feasible=feasible, violation=violation)


def curve_number_for_soil(soil: SoilProfile) -> float:
    """Curve number from the soil's own hydrologic behaviour."""
    if soil.ksat_in_hr >= 4.0:
        return 68.0
    if soil.ksat_in_hr >= 1.0:
        return 74.0
    if soil.ksat_in_hr >= 0.4:
        return 80.0
    return 86.0


def _simulate_schedule(strategy: Strategy, ctx: FieldContext, crop, regime,
                       system, cn: float):
    """Trigger-based scheduling, replayed day by day.

    The strategy is a rule, not a fixed calendar: irrigate when depletion passes
    a fraction of TAW, apply a set depth, subject to the system's minimum and
    maximum and to the deficit regime's target. Replaying the rule against real
    weather is what makes two strategies comparable — a fixed calendar would just
    be comparing two guesses about the weather.
    """
    schedule: dict[date, float] = {}
    # Two passes: the balance needs the schedule, the schedule needs the balance.
    # Iterating twice converges for a monotone trigger rule and costs almost
    # nothing at this problem size.
    for _ in range(2):
        results = run_season(ctx.weather, crop, ctx.soil, ctx.planting,
                             ctx.lat_deg, irrigations=schedule, curve_number=cn)
        new: dict[date, float] = {}
        last: date | None = None
        for r in results:
            if r.taw_in <= 0:
                continue
            frac = r.depletion_in / r.taw_in
            mad_adj = strategy.trigger_fraction * regime.mad_multiplier
            if frac < min(0.98, mad_adj):
                continue
            if last is not None and (r.day - last).days < 2:
                continue
            depth = min(strategy.depth_in * regime.et_fraction,
                        r.depletion_in,
                        0.90 * r.taw_in,
                        system.max_application_in)
            if depth < system.min_application_in:
                continue
            new[r.day] = round(depth, 2)
            last = r.day
        if new == schedule:
            break
        schedule = new
    results = run_season(ctx.weather, crop, ctx.soil, ctx.planting,
                         ctx.lat_deg, irrigations=schedule, curve_number=cn)
    return schedule, results


def _relative_et(results) -> float:
    """Season relative ET with the yield-critical window weighted more heavily."""
    if not results:
        return 1.0
    num = den = 0.0
    n = len(results)
    for i, r in enumerate(results):
        # The critical window sits around the middle of the season for the
        # determinate summer crops that dominate Delmarva irrigation.
        pos = i / max(1, n - 1)
        w = 1.6 if 0.42 <= pos <= 0.68 else 1.0
        num += w * r.etc_adj_in
        den += w * r.etc_in if r.etc_in > 0 else 0.0
    return min(1.0, num / den) if den > 0 else 1.0


# ---------------------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------------------

def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """Pareto dominance in minimization form."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def fast_non_dominated_sort(objectives: list[tuple[float, ...]]) -> list[list[int]]:
    """Deb's fast non-dominated sort, O(M N^2).

    Returns fronts as lists of indices, front 0 being the non-dominated set.
    """
    n = len(objectives)
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    domination_count = [0] * n
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                dominated_by[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt: list[int] = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return [f for f in fronts if f]


def crowding_distance(objectives: list[tuple[float, ...]],
                      front: list[int]) -> dict[int, float]:
    """Crowding distance — the diversity mechanism that keeps the front spread.

    Without it NSGA-II converges into a clump, and a clumped Pareto front is
    useless for the thing this is for: showing a grower the whole range of
    available exchange rates, not three neighbouring points.
    """
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            dist[i] = math.inf
        return dist
    m = len(objectives[front[0]])
    for k in range(m):
        ordered = sorted(front, key=lambda i: objectives[i][k])
        dist[ordered[0]] = math.inf
        dist[ordered[-1]] = math.inf
        lo, hi = objectives[ordered[0]][k], objectives[ordered[-1]][k]
        span = hi - lo
        if span <= 0:
            continue
        for j in range(1, len(ordered) - 1):
            prev_v = objectives[ordered[j - 1]][k]
            next_v = objectives[ordered[j + 1]][k]
            dist[ordered[j]] += (next_v - prev_v) / span
    return dist


class _Rng:
    """Deterministic generator.

    A Pareto front that shifts between runs is not a decision aid. Two people
    looking at the same field must see the same frontier.
    """

    def __init__(self, seed: int = 20260415) -> None:
        self.s = seed

    def random(self) -> float:
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return ((self.s >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    def randint(self, lo: int, hi: int) -> int:
        return lo + int(self.random() * (hi - lo + 1)) % (hi - lo + 1)


def _sbx(p1: list[float], p2: list[float], lo: list[float], hi: list[float],
         rng: _Rng, eta: float = 15.0, prob: float = 0.9
         ) -> tuple[list[float], list[float]]:
    """Simulated binary crossover — the real-coded analogue of one-point crossover."""
    c1, c2 = p1[:], p2[:]
    if rng.random() > prob:
        return c1, c2
    for i in range(len(p1)):
        if rng.random() > 0.5 or abs(p1[i] - p2[i]) < 1e-12:
            continue
        x1, x2 = min(p1[i], p2[i]), max(p1[i], p2[i])
        rand = rng.random()
        beta = 1.0 + 2.0 * (x1 - lo[i]) / max(1e-12, x2 - x1)
        alpha = 2.0 - beta ** -(eta + 1)
        betaq = ((rand * alpha) ** (1.0 / (eta + 1)) if rand <= 1.0 / alpha
                 else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1)))
        c1[i] = 0.5 * ((x1 + x2) - betaq * (x2 - x1))
        beta = 1.0 + 2.0 * (hi[i] - x2) / max(1e-12, x2 - x1)
        alpha = 2.0 - beta ** -(eta + 1)
        betaq = ((rand * alpha) ** (1.0 / (eta + 1)) if rand <= 1.0 / alpha
                 else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1)))
        c2[i] = 0.5 * ((x1 + x2) + betaq * (x2 - x1))
        c1[i] = max(lo[i], min(hi[i], c1[i]))
        c2[i] = max(lo[i], min(hi[i], c2[i]))
    return c1, c2


def _polynomial_mutation(g: list[float], lo: list[float], hi: list[float],
                         rng: _Rng, eta: float = 20.0) -> list[float]:
    out = g[:]
    prob = 1.0 / len(g)
    for i in range(len(g)):
        if rng.random() > prob:
            continue
        span = hi[i] - lo[i]
        if span <= 0:
            continue
        delta1 = (out[i] - lo[i]) / span
        delta2 = (hi[i] - out[i]) / span
        rand = rng.random()
        mut_pow = 1.0 / (eta + 1.0)
        if rand < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
            deltaq = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
            deltaq = 1.0 - val ** mut_pow
        out[i] = max(lo[i], min(hi[i], out[i] + deltaq * span))
    return out


@dataclass
class ParetoResult:
    front: list[Outcome]
    all_evaluated: list[Outcome]
    generations: int
    evaluations: int
    convergence: list[dict]


def optimize(ctx: FieldContext, space: DecisionSpace | None = None,
             regime_key: str = "full", system_key: str = "sprinkler_pivot",
             population: int = 48, generations: int = 24,
             seed: int = 20260415,
             progress: Callable[[int, int], None] | None = None) -> ParetoResult:
    """NSGA-II over irrigation and nitrogen strategies.

    Population and generation counts are modest on purpose. Four decision
    variables and three objectives is a small problem, each evaluation replays a
    full season through the water balance, and the front stabilises well before
    a thousand evaluations. Running ten times longer would produce a slightly
    smoother curve and no different decision.
    """
    space = space or DecisionSpace()
    lo = [space.trigger_fraction_min, space.depth_min_in,
          space.n_rate_min, float(space.splits_min)]
    hi = [space.trigger_fraction_max, space.depth_max_in,
          space.n_rate_max, float(space.splits_max)]
    rng = _Rng(seed)

    def make_genome() -> list[float]:
        return [l + rng.random() * (h - l) for l, h in zip(lo, hi)]

    def evaluate_genome(g: list[float]) -> Outcome:
        return evaluate(Strategy.from_genome(space.clamp(g), regime_key, system_key),
                        ctx, space)

    pop = [make_genome() for _ in range(population)]
    outcomes = [evaluate_genome(g) for g in pop]
    evaluations = len(outcomes)
    convergence: list[dict] = []

    for gen in range(generations):
        objs = [_penalized(o) for o in outcomes]
        fronts = fast_non_dominated_sort(objs)
        rank = {}
        crowd = {}
        for r, f in enumerate(fronts):
            d = crowding_distance(objs, f)
            for i in f:
                rank[i], crowd[i] = r, d[i]

        # Binary tournament on rank, then crowding distance.
        def tournament() -> list[float]:
            a, b = rng.randint(0, population - 1), rng.randint(0, population - 1)
            if rank[a] < rank[b]:
                return pop[a]
            if rank[b] < rank[a]:
                return pop[b]
            return pop[a] if crowd[a] >= crowd[b] else pop[b]

        offspring: list[list[float]] = []
        while len(offspring) < population:
            c1, c2 = _sbx(tournament(), tournament(), lo, hi, rng)
            offspring.append(_polynomial_mutation(c1, lo, hi, rng))
            if len(offspring) < population:
                offspring.append(_polynomial_mutation(c2, lo, hi, rng))

        off_outcomes = [evaluate_genome(g) for g in offspring]
        evaluations += len(off_outcomes)

        # Elitist survival across parents and offspring together.
        combined = pop + offspring
        combined_out = outcomes + off_outcomes
        combined_objs = [_penalized(o) for o in combined_out]
        fronts = fast_non_dominated_sort(combined_objs)

        new_pop, new_out = [], []
        for f in fronts:
            if len(new_pop) + len(f) <= population:
                for i in f:
                    new_pop.append(combined[i])
                    new_out.append(combined_out[i])
            else:
                d = crowding_distance(combined_objs, f)
                for i in sorted(f, key=lambda i: -d[i])[:population - len(new_pop)]:
                    new_pop.append(combined[i])
                    new_out.append(combined_out[i])
                break
        pop, outcomes = new_pop, new_out

        best = [o for o in outcomes if o.feasible]
        convergence.append({
            "generation": gen + 1,
            "front_size": len(fast_non_dominated_sort([_penalized(o) for o in outcomes])[0]),
            "best_net_return": round(max((o.net_return for o in best), default=0.0), 2),
            "least_leaching_lb": round(min((o.nitrate_leached_lb for o in best),
                                           default=0.0), 1),
        })
        if progress:
            progress(gen + 1, generations)

    objs = [_penalized(o) for o in outcomes]
    front_idx = fast_non_dominated_sort(objs)[0]
    front = [outcomes[i] for i in front_idx if outcomes[i].feasible]
    front.sort(key=lambda o: o.irrigation_in)
    return ParetoResult(front=front, all_evaluated=outcomes,
                        generations=generations, evaluations=evaluations,
                        convergence=convergence)


def _penalized(o: Outcome) -> tuple[float, ...]:
    """Constraint handling: infeasible solutions are pushed behind every feasible one.

    Deb's constrained-domination, in the simple form appropriate here. An
    over-allocation strategy is not a tradeoff a grower gets to consider; it is
    a permit violation.
    """
    obj = o.objectives()
    if o.feasible:
        return obj
    return tuple(v + 1e6 for v in obj)


# ---------------------------------------------------------------------------
# Locating the grower on the front
# ---------------------------------------------------------------------------

def compare_to_front(current: Outcome, front: Sequence[Outcome]) -> dict:
    """Where the grower's own practice sits relative to the Pareto set.

    This is what Activity 9 actually asks for — the scenario the user defines,
    compared against the non-dominated solutions, expressed as tradeoffs. Three
    specific comparisons, because "you are dominated" tells nobody what to do:

      same water, more money — how much yield and return is available without
      applying another drop;
      same return, less water — how much water is being spent for nothing;
      same return, less nitrate — the environmental slack at no cost.
    """
    if not front:
        return {"on_front": False, "note": "No feasible Pareto solutions found."}

    dominating = [o for o in front
                  if dominates(o.objectives(), current.objectives())]
    on_front = not dominating

    def nearest(pred, key) -> Outcome | None:
        candidates = [o for o in front if pred(o)]
        return min(candidates, key=key) if candidates else None

    same_water = nearest(lambda o: o.irrigation_in <= current.irrigation_in + 0.5,
                         lambda o: -o.net_return)
    same_return = nearest(lambda o: o.net_return >= current.net_return - 1.0,
                          lambda o: o.irrigation_in)
    cleaner = nearest(lambda o: o.net_return >= current.net_return - 1.0,
                      lambda o: o.nitrate_leached_lb)

    out: dict = {
        "on_front": on_front,
        "dominated_by": len(dominating),
        "current": _brief(current),
        "opportunities": [],
    }
    if same_water and same_water.net_return > current.net_return + 2:
        out["opportunities"].append({
            "kind": "more_return_same_water",
            "gain_per_acre": round(same_water.net_return - current.net_return, 2),
            "strategy": _brief(same_water),
            "note": (f"About ${same_water.net_return - current.net_return:.0f} per acre "
                     "is available without applying another drop — it comes from "
                     "when and how much per pass, not from more water."),
        })
    if same_return and same_return.irrigation_in < current.irrigation_in - 0.5:
        saved = current.irrigation_in - same_return.irrigation_in
        out["opportunities"].append({
            "kind": "less_water_same_return",
            "water_saved_in": round(saved, 2),
            "strategy": _brief(same_return),
            "note": (f"{saved:.1f} acre-inches are being applied for no return. "
                     "On a 130-acre field that is roughly "
                     f"{saved * 130 * 27154 / 1000:.0f} thousand gallons and the "
                     "pumping cost that goes with it."),
        })
    if cleaner and cleaner.nitrate_leached_lb < current.nitrate_leached_lb - 2:
        cut = current.nitrate_leached_lb - cleaner.nitrate_leached_lb
        out["opportunities"].append({
            "kind": "less_nitrate_same_return",
            "nitrate_cut_lb": round(cut, 1),
            "strategy": _brief(cleaner),
            "note": (f"{cut:.0f} lb N/ac of leaching can come out at the same "
                     "return, mostly by splitting the rate further rather than "
                     "by cutting it."),
        })
    if on_front and not out["opportunities"]:
        out["note"] = ("This strategy is already non-dominated: nothing on the "
                       "frontier does better on return, water and nitrate at once.")
    return out


def _brief(o: Outcome) -> dict:
    return {
        "trigger_fraction": round(o.strategy.trigger_fraction, 2),
        "depth_in": round(o.strategy.depth_in, 2),
        "n_rate_lb": round(o.strategy.n_rate_lb, 0),
        "splits": o.strategy.splits,
        "yield_bu": o.yield_bu,
        "irrigation_in": o.irrigation_in,
        "passes": o.passes,
        "nitrate_leached_lb": o.nitrate_leached_lb,
        "net_return": o.net_return,
        "wnipi": o.wnipi,
    }


def front_summary(result: ParetoResult) -> dict:
    """The front reduced to the handful of points worth putting in front of a person.

    A forty-point Pareto set is a research output. A grower needs the corners and
    the knee: the most profitable, the least thirsty, the cleanest, and the one
    where the exchange rate turns.
    """
    front = result.front
    if not front:
        return {"n": 0, "note": "No feasible solutions."}

    most_profit = max(front, key=lambda o: o.net_return)
    least_water = min(front, key=lambda o: o.irrigation_in)
    cleanest = min(front, key=lambda o: o.nitrate_leached_lb)
    knee = _knee_point(front)

    return {
        "n": len(front),
        "evaluations": result.evaluations,
        "most_profitable": _brief(most_profit),
        "least_water": _brief(least_water),
        "least_nitrate": _brief(cleanest),
        "balanced": _brief(knee),
        "water_range_in": [least_water.irrigation_in,
                           max(o.irrigation_in for o in front)],
        "return_range": [round(min(o.net_return for o in front), 2),
                         round(most_profit.net_return, 2)],
        "exchange_rate": _exchange_rate(front),
        "front": [_brief(o) for o in front],
    }


def _knee_point(front: Sequence[Outcome]) -> Outcome:
    """The knee: the point furthest from the line joining the extremes.

    Where the curve bends is where the exchange rate changes fastest, and it is
    almost always the point a person picks once they can see the shape.
    """
    if len(front) <= 2:
        return front[0]
    xs = [o.irrigation_in for o in front]
    ys = [o.net_return for o in front]
    x0, y0, x1, y1 = xs[0], ys[0], xs[-1], ys[-1]
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy) or 1.0
    best, best_d = front[0], -1.0
    for o, x, y in zip(front, xs, ys):
        d = abs(dy * x - dx * y + x1 * y0 - y1 * x0) / norm
        if d > best_d:
            best, best_d = o, d
    return best


def _exchange_rate(front: Sequence[Outcome]) -> dict:
    """What the last acre-inch is actually buying, along the front."""
    pts = sorted(front, key=lambda o: o.irrigation_in)
    if len(pts) < 2:
        return {}
    segments = []
    for a, b in zip(pts, pts[1:]):
        dw = b.irrigation_in - a.irrigation_in
        if dw <= 0.05:
            continue
        segments.append({
            "from_in": a.irrigation_in, "to_in": b.irrigation_in,
            "dollars_per_acre_inch": round((b.net_return - a.net_return) / dw, 2),
            "lb_nitrate_per_acre_inch": round(
                (b.nitrate_leached_lb - a.nitrate_leached_lb) / dw, 2),
        })
    return {"segments": segments,
            "note": ("Read down the list: where dollars per acre-inch goes "
                     "negative, the next pass costs more than it returns.")}
