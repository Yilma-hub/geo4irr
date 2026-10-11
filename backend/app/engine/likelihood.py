"""
The Bayesian machinery the narrative specifies, written as the narrative states it.

Activity 5 does not say "use a Kalman filter." It says to use a statistically
correct likelihood function, gives it as Eq 4, gives its logarithmic form as
Eq 5, requires that correlated residuals be handled by an appropriate
transformation, and asks for a generative filtering technique for streaming
data. Activity 7 then reuses the same Eq 5 to represent measurement errors when
choosing where to put sensors. So the likelihood is not an implementation detail
buried inside a filter — it is a shared object that two separate activities
depend on, and it is written that way here.

    Eq 4    L(x|y) = p(y|x) = p(x|y) p(y) / p(x)

    Eq 5    log L(x|y) = -(n/2) log(2pi) - sum log(sigma_i)
                         - (1/2) sum ((y_i - y~_i) / sigma_i)^2

where y and y~ are observed and modelled responses, n is the number of
observations, and sigma is the standard deviation of measurement error obtained
from Activity 4.

The sign convention in the narrative's Eq 5 as typeset drops the leading minus
signs; the standard Gaussian log-likelihood is used here, which is what every
citation behind it intends and what makes the numbers behave.

Correlated residuals matter more here than they look. Soil moisture from one
probe on consecutive days is strongly autocorrelated — the profile does not
forget overnight — and treating fourteen daily readings as fourteen independent
observations makes the filter roughly four times more confident than the data
justifies. Activity 5 calls for "appropriate transformations"; the ones
implemented are AR(1) whitening in time and a full covariance Cholesky solve for
the general case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Eq 5, the independent case
# ---------------------------------------------------------------------------

def log_likelihood(observed: Sequence[float], modeled: Sequence[float],
                   sigma: Sequence[float] | float) -> float:
    """Gaussian log-likelihood, Eq 5, assuming independent residuals.

    sigma comes from Activity 4 — the empirical error characterisation of each
    sensing device — and is per-observation, because the whole point of the
    hybrid sensing strategy is that a research TDR probe and a cheap capacitance
    probe do not deserve equal weight.
    """
    n = len(observed)
    if n == 0:
        return 0.0
    if len(modeled) != n:
        raise ValueError("observed and modeled must be the same length")
    sig = [float(sigma)] * n if isinstance(sigma, (int, float)) else list(sigma)
    if len(sig) != n:
        raise ValueError("sigma must be scalar or the same length as observed")
    if any(s <= 0 for s in sig):
        raise ValueError("measurement error standard deviations must be positive")

    total = -0.5 * n * math.log(TWO_PI)
    for y, ym, s in zip(observed, modeled, sig):
        total -= math.log(s)
        total -= 0.5 * ((y - ym) / s) ** 2
    return total


# ---------------------------------------------------------------------------
# Correlated residuals
# ---------------------------------------------------------------------------

def ar1_whiten(observed: Sequence[float], modeled: Sequence[float],
               sigma: Sequence[float] | float, rho: float
               ) -> tuple[list[float], list[float], list[float]]:
    """Whitening transformation for AR(1) residuals.

    Daily soil moisture residuals from one probe are autocorrelated: the profile
    carries yesterday's error into today. The standard transformation divides the
    first residual by sqrt(1 - rho^2) and replaces each subsequent residual with
    e_t - rho e_{t-1}, after which the sequence is independent and Eq 5 applies
    unchanged.

    Returns transformed (observed, modeled, sigma) triples so the caller can
    hand them straight to log_likelihood.
    """
    n = len(observed)
    if n == 0:
        return [], [], []
    if not -0.999 < rho < 0.999:
        raise ValueError("rho must lie strictly between -1 and 1")
    sig = [float(sigma)] * n if isinstance(sigma, (int, float)) else list(sigma)

    resid = [y - m for y, m in zip(observed, modeled)]
    out_r = [resid[0] * math.sqrt(1.0 - rho * rho)]
    out_s = [sig[0]]
    for t in range(1, n):
        out_r.append(resid[t] - rho * resid[t - 1])
        out_s.append(sig[t])
    # Express as (observed, modeled=0) so the residual is preserved exactly.
    return out_r, [0.0] * n, out_s


def estimate_ar1_rho(residuals: Sequence[float]) -> float:
    """Lag-1 autocorrelation of a residual series.

    Estimated rather than assumed, because it differs by sensor and by soil: a
    capacitance probe in Evesboro sand tracks a wetting front within hours and
    decorrelates fast, while the same probe in Matapeake silt loam does not.
    """
    n = len(residuals)
    if n < 3:
        return 0.0
    mean = sum(residuals) / n
    num = sum((residuals[t] - mean) * (residuals[t - 1] - mean) for t in range(1, n))
    den = sum((r - mean) ** 2 for r in residuals)
    if den <= 0:
        return 0.0
    return max(-0.95, min(0.95, num / den))


def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
    """Lower-triangular Cholesky factor. Raises if the matrix is not positive definite."""
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                d = matrix[i][i] - s
                if d <= 1e-12:
                    raise ValueError("covariance matrix is not positive definite")
                L[i][j] = math.sqrt(d)
            else:
                L[i][j] = (matrix[i][j] - s) / L[j][j]
    return L


def _forward_substitute(L: list[list[float]], b: Sequence[float]) -> list[float]:
    n = len(b)
    x = [0.0] * n
    for i in range(n):
        x[i] = (b[i] - sum(L[i][k] * x[k] for k in range(i))) / L[i][i]
    return x


def log_likelihood_correlated(observed: Sequence[float], modeled: Sequence[float],
                              covariance: list[list[float]]) -> float:
    """Multivariate Gaussian log-likelihood with a full error covariance.

    The general form of the "appropriate transformation" Activity 5 asks for.
    Used when residuals are correlated in space rather than only in time — three
    probes fifty feet apart in the same map unit are not three independent
    observations, and the sensor network design in Activity 7 depends on saying
    so honestly.

        log L = -(n/2) log(2pi) - sum log(diag(L)) - (1/2) z'z,  z = L^-1 (y - y~)
    """
    n = len(observed)
    if n == 0:
        return 0.0
    L = _cholesky(covariance)
    resid = [y - m for y, m in zip(observed, modeled)]
    z = _forward_substitute(L, resid)
    log_det_half = sum(math.log(L[i][i]) for i in range(n))
    return -0.5 * n * math.log(TWO_PI) - log_det_half - 0.5 * sum(v * v for v in z)


def exponential_covariance(distances_ft: list[list[float]], sigma: Sequence[float],
                           range_ft: float = 400.0, nugget: float = 0.15
                           ) -> list[list[float]]:
    """Build an error covariance from a separation matrix.

    Exponential model with a nugget: two probes at zero separation still
    disagree, because sensor-to-sensor variation and installation contact are
    real. The nugget is what keeps the matrix invertible and keeps the design
    algorithm from believing that co-located sensors are perfectly redundant.
    """
    n = len(sigma)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                out[i][j] = sigma[i] ** 2
            else:
                rho = (1.0 - nugget) * math.exp(-distances_ft[i][j] / range_ft)
                out[i][j] = rho * sigma[i] * sigma[j]
    return out


# ---------------------------------------------------------------------------
# Eq 4: prior, likelihood, posterior
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Gaussian:
    """A Gaussian belief about one scalar state variable."""
    mean: float
    var: float

    @property
    def sd(self) -> float:
        return math.sqrt(max(0.0, self.var))

    def interval(self, lower: float = 0.05, upper: float = 0.95) -> tuple[float, float]:
        """Prediction interval. Activity 5 asks specifically for 5th and 95th."""
        return (self.mean + _probit(lower) * self.sd,
                self.mean + _probit(upper) * self.sd)

    def logpdf(self, x: float) -> float:
        return (-0.5 * math.log(TWO_PI * self.var)
                - 0.5 * (x - self.mean) ** 2 / self.var)


def _probit(p: float) -> float:
    """Inverse standard normal CDF, Acklam's rational approximation.

    Accurate to about 1.15e-9 in absolute value, which is far beyond what a soil
    moisture prediction interval needs and cheap enough to run per field per day.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def bayes_update(prior: Gaussian, observation: float, obs_sd: float) -> Gaussian:
    """Conjugate Gaussian posterior — Eq 4 with a Gaussian prior and Eq 5 likelihood.

    For a Gaussian prior and a Gaussian likelihood the posterior is available in
    closed form, so no sampling is needed. This is the fast path taken for the
    nightly batch across every field. Where the posterior is not Gaussian —
    a bimodal wetting front, a state constrained at zero depletion — the
    generative filter below takes over.
    """
    if obs_sd <= 0:
        raise ValueError("observation standard deviation must be positive")
    var_obs = obs_sd ** 2
    var_post = 1.0 / (1.0 / prior.var + 1.0 / var_obs)
    mean_post = var_post * (prior.mean / prior.var + observation / var_obs)
    return Gaussian(mean_post, var_post)


def evidence(prior: Gaussian, observation: float, obs_sd: float) -> float:
    """The marginal p(y) in Eq 4 — the denominator, the model evidence.

    Worth computing rather than dropping. A run of low evidence means the
    observations keep surprising the model, which is the signal that the crop
    curve or the soil parameters are wrong for this field and that no amount of
    filtering will fix it.
    """
    var = prior.var + obs_sd ** 2
    return math.exp(-0.5 * math.log(TWO_PI * var)
                    - 0.5 * (observation - prior.mean) ** 2 / var)


# ---------------------------------------------------------------------------
# Generative filtering
# ---------------------------------------------------------------------------

@dataclass
class GenerativeFilter:
    """Sequential Monte Carlo filter for streaming smart-sensing data.

    Activity 5 asks for "algorithms for data streaming using a generative
    filtering technique." This is that: a particle representation of the joint
    posterior over the state and the parameters that are being estimated
    alongside it, updated by Eq 5 weights as each observation arrives.

    Why it is worth having next to the closed-form update. The soil water
    deficit is bounded below at zero and the posterior piles up against that
    bound after a big rain, where a Gaussian would put mass on negative
    depletion. Kcb bias and Ks are also bounded. And the narrative explicitly
    wants the parameters updated, not only the states — a particle filter
    estimates both in one pass without linearising anything.

    Kept to a few hundred particles per field: enough for stable 5th and 95th
    percentiles, cheap enough that a research field can run it every night.
    """
    n_particles: int = 400
    seed: int = 20260401
    state: list[dict] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    _rng_state: int = 0

    def _rand(self) -> float:
        """Deterministic uniform generator.

        Deliberately not the standard library's: two people looking at the same
        field on the same day must see the same prediction interval, and a
        recommendation that changes because a process restarted is a
        recommendation nobody trusts twice.
        """
        self._rng_state = (self._rng_state * 6364136223846793005 + 1442695040888963407) \
            % (1 << 64)
        return ((self._rng_state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    def _randn(self) -> float:
        u1 = max(1e-12, self._rand())
        u2 = self._rand()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(TWO_PI * u2)

    def initialize(self, dr_in: float, dr_sd: float,
                   kcb_bias: float = 1.0, kcb_bias_sd: float = 0.10,
                   ks: float = 1.0, ks_sd: float = 0.05,
                   taw_in: float = 3.0) -> None:
        self._rng_state = self.seed
        self.state = []
        for _ in range(self.n_particles):
            self.state.append({
                "dr": min(taw_in, max(0.0, dr_in + dr_sd * self._randn())),
                "kcb_bias": max(0.4, min(1.8, kcb_bias + kcb_bias_sd * self._randn())),
                "ks": max(0.0, min(1.0, ks + ks_sd * self._randn())),
            })
        self.weights = [1.0 / self.n_particles] * self.n_particles

    def propagate(self, advance, process_sd: float,
                  kcb_walk_sd: float = 0.02, taw_in: float = 3.0) -> None:
        """Push every particle forward one day through the water balance.

        `advance` is a callable taking a particle dict and returning the new
        depletion, so the filter never needs to know what model is underneath it.
        That is what lets the same filter sit on top of pyfao56 or the built-in
        step without modification.
        """
        for p in self.state:
            p["dr"] = advance(p)
            p["dr"] = min(taw_in, max(0.0, p["dr"] + process_sd * self._randn()))
            p["kcb_bias"] = max(0.4, min(1.8, p["kcb_bias"] + kcb_walk_sd * self._randn()))

    def update(self, observations: Sequence[tuple[str, float, float]]) -> dict:
        """Reweight by Eq 5 across all observations arriving on one day.

        Each observation is (variable, value, sigma). Variables are 'dr' for a
        soil moisture probe converted to depletion, 'kcb_bias' for a canopy or
        NDVI derived coefficient ratio, and 'ks' for a canopy-temperature stress
        index. Weights multiply because the log-likelihoods add, which is Eq 5
        with n greater than one and exactly the coherence the narrative is after:
        a probe and a UAV pass on the same day are used together, through the
        physics that links them, rather than in silos.
        """
        if not observations:
            return {"n_obs": 0, "ess": self.effective_sample_size()}

        logw = [math.log(max(w, 1e-300)) for w in self.weights]
        for i, p in enumerate(self.state):
            for var, value, sigma in observations:
                if var not in p or sigma <= 0:
                    continue
                logw[i] += log_likelihood([value], [p[var]], [sigma])

        m = max(logw)
        w = [math.exp(v - m) for v in logw]
        total = sum(w)
        if total <= 0:                       # every particle rejected
            self.weights = [1.0 / self.n_particles] * self.n_particles
            return {"n_obs": len(observations), "ess": float(self.n_particles),
                    "degenerate": True}
        self.weights = [v / total for v in w]

        ess = self.effective_sample_size()
        resampled = False
        if ess < self.n_particles / 2:
            self._systematic_resample()
            resampled = True
        return {"n_obs": len(observations), "ess": round(ess, 1),
                "resampled": resampled, "log_evidence": round(m + math.log(total), 3)}

    def effective_sample_size(self) -> float:
        s = sum(w * w for w in self.weights)
        return 1.0 / s if s > 0 else float(self.n_particles)

    def _systematic_resample(self) -> None:
        """Systematic resampling: lower variance than multinomial, same cost."""
        n = self.n_particles
        positions = [(i + self._rand()) / n for i in range(n)]
        cumulative, total = [], 0.0
        for w in self.weights:
            total += w
            cumulative.append(total)
        new_state, i, j = [], 0, 0
        while i < n and j < n:
            if positions[i] < cumulative[j]:
                new_state.append(dict(self.state[j]))
                i += 1
            else:
                j += 1
        while len(new_state) < n:
            new_state.append(dict(self.state[-1]))
        self.state = new_state
        self.weights = [1.0 / n] * n

    # -- summaries ---------------------------------------------------------

    def mean(self, var: str = "dr") -> float:
        return sum(w * p[var] for w, p in zip(self.weights, self.state))

    def sd(self, var: str = "dr") -> float:
        mu = self.mean(var)
        return math.sqrt(max(0.0, sum(w * (p[var] - mu) ** 2
                                      for w, p in zip(self.weights, self.state))))

    def quantile(self, q: float, var: str = "dr") -> float:
        pairs = sorted(zip((p[var] for p in self.state), self.weights))
        acc = 0.0
        for value, w in pairs:
            acc += w
            if acc >= q:
                return value
        return pairs[-1][0]

    def interval(self, var: str = "dr", lower: float = 0.05,
                 upper: float = 0.95) -> tuple[float, float]:
        """The 5th and 95th percentiles Activity 5 names explicitly."""
        return self.quantile(lower, var), self.quantile(upper, var)

    def summary(self, var: str = "dr") -> dict:
        lo, hi = self.interval(var)
        return {"mean": round(self.mean(var), 3), "sd": round(self.sd(var), 3),
                "p05": round(lo, 3), "p50": round(self.quantile(0.5, var), 3),
                "p95": round(hi, 3), "ess": round(self.effective_sample_size(), 1)}


# ---------------------------------------------------------------------------
# Activity 4: characterising measurement error from paired data
# ---------------------------------------------------------------------------

def characterize_error(reference: Sequence[float], device: Sequence[float]) -> dict:
    """Empirical error statistics for a sensing device against a reference.

    Activity 4 calls for characterising the stochastic properties of measurement
    error by both empirical and parametric methods, using paired field data. This
    is the calculation that turns a decade of LIRF gravimetric-versus-probe
    records — or a season of Delaware neutron probe checks — into the sigma that
    Eq 5 needs.

    Bias and precision are reported separately because they are different
    problems with different fixes. A probe reading consistently 0.02 low is
    calibrated out in an afternoon; a probe with a 0.05 standard deviation around
    the truth cannot be fixed at all and simply deserves less weight.
    """
    n = len(reference)
    if n < 3 or len(device) != n:
        raise ValueError("need at least three paired observations")
    resid = [d - r for r, d in zip(reference, device)]
    bias = sum(resid) / n
    var = sum((e - bias) ** 2 for e in resid) / (n - 1)
    sd = math.sqrt(var)
    rmse = math.sqrt(sum(e * e for e in resid) / n)
    rho = estimate_ar1_rho(resid)

    ref_mean = sum(reference) / n
    ss_tot = sum((r - ref_mean) ** 2 for r in reference)
    ss_res = sum(e * e for e in resid)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Effective independent sample size once autocorrelation is accounted for.
    n_eff = n * (1 - rho) / (1 + rho) if rho > -0.999 else n

    return {
        "n": n,
        "bias": round(bias, 4),
        "sd": round(sd, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "ar1_rho": round(rho, 3),
        "effective_n": round(n_eff, 1),
        "sigma_for_eq5": round(sd, 4),
        "note": _error_note(bias, sd, rho),
    }


def _error_note(bias: float, sd: float, rho: float) -> str:
    parts = []
    if abs(bias) > 2 * sd / math.sqrt(3):
        parts.append(
            f"A systematic offset of {bias:+.3f} dominates the random error. "
            "Calibrate this device before weighting it; a bias is not noise and "
            "the filter cannot average it away.")
    if rho > 0.5:
        parts.append(
            f"Residuals are strongly autocorrelated (rho = {rho:.2f}). Consecutive "
            "readings are not independent observations — whiten before applying "
            "Eq 5 or the filter will be roughly "
            f"{math.sqrt((1 + rho) / max(1e-6, 1 - rho)):.1f} times more confident "
            "than the data supports.")
    if not parts:
        parts.append("Residuals look independent and unbiased; use the standard "
                     "deviation directly as sigma in Eq 5.")
    return " ".join(parts)
