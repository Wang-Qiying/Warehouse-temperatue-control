from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog, minimize_scalar


@dataclass
class DPSolution:
    q_star: float
    e_req: float
    e_pv: float
    e_grid: float


@dataclass
class DPDebugResult:
    solution: DPSolution
    x_grid: np.ndarray
    F: List[np.ndarray]
    f: List[np.ndarray]
    x_path: np.ndarray
    q_path: np.ndarray
    e_req_path: np.ndarray
    e_pv_path: np.ndarray
    e_grid_path: np.ndarray
    stage_checks: List[Dict[str, Any]]
    lp_comparison: Optional[Dict[str, float]] = None


@dataclass
class _ProblemParams:
    fore_len: int
    A: float
    B: float
    x0: float
    w_fore: np.ndarray
    h: float
    E_free: np.ndarray
    lam: np.ndarray
    eta_c: float = 1.0
    eta_h: float = 1.0
    Tmax: float = 6.0
    Tmin: float = 2.0
    qmin: float = -1.0
    qmax: float = 1.0

    def __post_init__(self) -> None:
        self.w_fore = np.asarray(self.w_fore, dtype=float).reshape(-1)
        self.E_free = np.asarray(self.E_free, dtype=float).reshape(-1)
        self.lam = np.asarray(self.lam, dtype=float).reshape(-1)
        if self.fore_len <= 0:
            raise ValueError("fore_len must be a positive integer.")
        if len(self.w_fore) != self.fore_len:
            raise ValueError("len(w_fore) must equal fore_len.")
        if len(self.E_free) != self.fore_len:
            raise ValueError("len(E_free) must equal fore_len.")
        if len(self.lam) != self.fore_len:
            raise ValueError("len(lam) must equal fore_len.")
        if self.A <= 0:
            raise ValueError("A must be positive for the five-interval recursion.")
        if self.B <= 0:
            raise ValueError("B must be positive for the current DP formulation.")
        if self.qmin > self.qmax:
            raise ValueError("qmin must not exceed qmax.")
        if self.Tmin > self.Tmax:
            raise ValueError("Tmin must not exceed Tmax.")
        if self.Tmin + self.h > self.Tmax - self.h:
            raise ValueError("Exact five-interval recursion requires L <= U, i.e. h <= (Tmax - Tmin)/2.")

    @property
    def L(self) -> float:
        return self.Tmin + self.h

    @property
    def U(self) -> float:
        return self.Tmax - self.h

    @property
    def Tref(self) -> float:
        return 0.5 * (self.L + self.U)


@dataclass
class _PiecewiseLinearConvex:
    """Convex piecewise linear function defined on the whole real line.

    breaks: finite sorted breakpoints.
    slopes: open-interval slopes; length len(breaks)+1.
    The representation is exact once a single anchor value is fixed.
    """

    breaks: np.ndarray
    slopes: np.ndarray
    anchor_x: float
    anchor_val: float

    def __post_init__(self) -> None:
        self.breaks = np.asarray(self.breaks, dtype=float).reshape(-1)
        self.slopes = np.asarray(self.slopes, dtype=float).reshape(-1)
        if len(self.slopes) != len(self.breaks) + 1:
            raise ValueError("slopes length must equal len(breaks)+1")
        self._build_intercepts()

    def _build_intercepts(self) -> None:
        m = len(self.breaks)
        idx = np.searchsorted(self.breaks, self.anchor_x, side="right")
        intercepts = np.empty(m + 1, dtype=float)
        intercepts[idx] = self.anchor_val - self.slopes[idx] * self.anchor_x
        for j in range(idx + 1, m + 1):
            b = self.breaks[j - 1]
            val_left = self.slopes[j - 1] * b + intercepts[j - 1]
            intercepts[j] = val_left - self.slopes[j] * b
        for j in range(idx - 1, -1, -1):
            b = self.breaks[j]
            val_right = self.slopes[j + 1] * b + intercepts[j + 1]
            intercepts[j] = val_right - self.slopes[j] * b
        self.intercepts = intercepts

    def value(self, x: float | np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        idx = np.searchsorted(self.breaks, arr, side="right")
        return self.slopes[idx] * arr + self.intercepts[idx]

    def slope_left(self, x: float) -> float:
        idx = np.searchsorted(self.breaks, x, side="left")
        return float(self.slopes[idx])

    def slope_right(self, x: float) -> float:
        idx = np.searchsorted(self.breaks, x, side="right")
        return float(self.slopes[idx])

    def slope_open(self, x: float | np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        idx = np.searchsorted(self.breaks, arr, side="right")
        return self.slopes[idx]

    def crossing_set(self, c: float, L: float, U: float, tol: float = 1e-12) -> tuple[float, float] | None:
        intervals: list[tuple[float, float]] = []
        left = -np.inf
        for j, s in enumerate(self.slopes):
            right = self.breaks[j] if j < len(self.breaks) else np.inf
            lo = max(L, left)
            hi = min(U, right)
            if abs(s - c) <= tol and lo < hi:
                intervals.append((lo, hi))
            left = right
        for b in self.breaks:
            if L - tol <= b <= U + tol:
                fl = self.slope_left(float(b))
                fr = self.slope_right(float(b))
                if fl - tol <= c <= fr + tol:
                    intervals.append((float(b), float(b)))
        if not intervals:
            return None
        lo = min(a for a, _ in intervals)
        hi = max(b for _, b in intervals)
        return (max(lo, L), min(hi, U))


@dataclass
class _StageRecursion:
    t: int
    p: _ProblemParams
    next_fun: _PiecewiseLinearConvex
    fun: _PiecewiseLinearConvex
    xH: float
    x0c: float
    xC: float
    beta1: float
    beta2: float
    beta3: float
    beta4: float
    a_plus: float
    a_minus: float

    def xi_plus(self, x: float) -> float:
        return self.p.A * x + self.p.w_fore[self.t] + self.p.B * self.p.eta_h * self.p.E_free[self.t]

    def xi_minus(self, x: float) -> float:
        return self.p.A * x + self.p.w_fore[self.t] - self.p.B * self.p.eta_c * self.p.E_free[self.t]

    def r_minus(self, x: float) -> float:
        return self.p.A * x + self.p.w_fore[self.t] + self.p.B * self.p.qmin

    def r_plus(self, x: float) -> float:
        return self.p.A * x + self.p.w_fore[self.t] + self.p.B * self.p.qmax

    def structural_optimizer(self, x: float) -> tuple[float, float, str]:
        if x <= self.beta1:
            return self.xH, -self.p.A * self.a_plus, "I"
        if x <= self.beta2:
            xp = self.xi_plus(x)
            return xp, self.p.A * float(self.next_fun.slope_open(xp)), "II"
        if x <= self.beta3:
            return self.x0c, 0.0, "III"
        if x <= self.beta4:
            xm = self.xi_minus(x)
            return xm, self.p.A * float(self.next_fun.slope_open(xm)), "IV"
        return self.xC, self.p.A * self.a_minus, "V"

    def actual_optimizer(self, x: float) -> tuple[float, float, str, str]:
        xhat, hatf, structural_case = self.structural_optimizer(x)
        rmin = self.r_minus(x)
        rplus = self.r_plus(x)
        ell = max(self.p.L, rmin)
        uu = min(self.p.U, rplus)
        tol = 1e-12

        if ell <= uu:
            if xhat < ell:
                xstar = ell
            elif xhat > uu:
                xstar = uu
            else:
                xstar = xhat
        else:
            xstar = rplus if rplus < self.p.L else rmin

        if ell <= uu and ell - tol <= xhat <= uu + tol:
            slope = hatf
            proj_case = "P1"
        elif abs(xstar - rmin) <= 1e-10:
            slope = self.p.A * float(self.next_fun.slope_open(rmin))
            proj_case = "P2"
        elif abs(xstar - rplus) <= 1e-10:
            slope = self.p.A * float(self.next_fun.slope_open(rplus))
            proj_case = "P3"
        elif abs(xstar - self.p.L) <= 1e-10:
            xi_m = self.xi_minus(x)
            xi_p = self.xi_plus(x)
            if self.p.L < xi_m - tol:
                slope = self.p.A * self.a_minus
            elif xi_m - tol <= self.p.L <= xi_p + tol:
                slope = 0.0
            else:
                slope = -self.p.A * self.a_plus
            proj_case = "P4"
        elif abs(xstar - self.p.U) <= 1e-10:
            xi_m = self.xi_minus(x)
            xi_p = self.xi_plus(x)
            if self.p.U < xi_m - tol:
                slope = self.p.A * self.a_minus
            elif xi_m - tol <= self.p.U <= xi_p + tol:
                slope = 0.0
            else:
                slope = -self.p.A * self.a_plus
            proj_case = "P5"
        else:
            raise RuntimeError("Failed to classify projection case in DP recursion.")
        return float(xstar), float(slope), structural_case, proj_case

    def stage_cost(self, x: float, x_next: float) -> float:
        q = (x_next - self.p.A * x - self.p.w_fore[self.t]) / self.p.B
        _, _, e_grid = _stage_energy_from_q(q, self.p.E_free[self.t], self.p.eta_c, self.p.eta_h)
        return float(self.p.lam[self.t] * e_grid)


def _stage_energy_from_q(q: np.ndarray | float, E_free: np.ndarray | float, eta_c: float, eta_h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_arr = np.asarray(q, dtype=float)
    E_arr = np.asarray(E_free, dtype=float)
    e_req = np.maximum(q_arr / eta_h, 0.0) + np.maximum(-q_arr / eta_c, 0.0)
    e_pv = np.minimum(e_req, E_arr)
    e_grid = np.maximum(e_req - E_arr, 0.0)
    return e_req, e_pv, e_grid


def _build_plot_grid(p: _ProblemParams, num_points: int = 4001, margin: float = 4.0) -> np.ndarray:
    lo = hi = p.x0
    lows = [lo]
    highs = [hi]
    for wt in p.w_fore:
        lo = p.A * lo + p.B * p.qmin + wt
        hi = p.A * hi + p.B * p.qmax + wt
        lows.append(lo)
        highs.append(hi)
    x_low = min(min(lows), p.L) - margin
    x_high = max(max(highs), p.U) + margin
    return np.linspace(x_low, x_high, num_points)


def _compress_breaks_slopes(breaks: list[float], slopes: list[float], tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    if not breaks:
        return np.asarray([], dtype=float), np.asarray([slopes[0]], dtype=float)
    new_breaks: list[float] = []
    new_slopes: list[float] = [slopes[0]]
    for k, bk in enumerate(breaks):
        if abs(slopes[k + 1] - new_slopes[-1]) <= tol:
            continue
        new_breaks.append(float(bk))
        new_slopes.append(float(slopes[k + 1]))
    return np.asarray(new_breaks, dtype=float), np.asarray(new_slopes, dtype=float)


def _project_to_interval(interval: tuple[float, float], xref: float) -> float:
    a, b = interval
    if xref < a:
        return float(a)
    if xref > b:
        return float(b)
    return float(xref)


def _extended_critical_point(fun: _PiecewiseLinearConvex, c: float, L: float, U: float, xref: float, tol: float = 1e-12) -> float:
    if fun.slope_left(L) > c + tol:
        return -np.inf
    if fun.slope_right(U) < c - tol:
        return np.inf
    crossing = fun.crossing_set(c, L, U, tol=tol)
    if crossing is None:
        raise RuntimeError("Crossing set should not be empty after range checks.")
    return _project_to_interval(crossing, xref)


def _stage_breakpoints(next_fun: _PiecewiseLinearConvex, p: _ProblemParams, t: int, xH: float, x0c: float, xC: float, beta1: float, beta2: float, beta3: float, beta4: float) -> np.ndarray:
    w = p.w_fore[t]
    E = p.E_free[t]
    vals: list[float] = []

    def add(v: float) -> None:
        if np.isfinite(v):
            vals.append(float(v))

    for v in [beta1, beta2, beta3, beta4]:
        add(v)

    # feasible-set switching points: xhat hits ell or uu
    add((p.L - w - p.B * p.qmin) / p.A)  # r_minus = L
    add((p.U - w - p.B * p.qmax) / p.A)  # r_plus = U

    # P4/P5 internal switching caused by xi^- and xi^+
    add((p.L - w + p.B * p.eta_c * E) / p.A)  # xi^- = L
    add((p.L - w - p.B * p.eta_h * E) / p.A)  # xi^+ = L
    add((p.U - w + p.B * p.eta_c * E) / p.A)  # xi^- = U
    add((p.U - w - p.B * p.eta_h * E) / p.A)  # xi^+ = U

    # if structural optimizer is constant, projection switches when it meets reachable bounds
    for xc in [xH, x0c, xC]:
        if np.isfinite(xc):
            add((xc - w - p.B * p.qmin) / p.A)  # xc = r_minus
            add((xc - w - p.B * p.qmax) / p.A)  # xc = r_plus

    # preimages of next-stage breakpoints under xi^+, xi^-, r^+, r^-
    for b in next_fun.breaks:
        add((b - w - p.B * p.eta_h * E) / p.A)
        add((b - w + p.B * p.eta_c * E) / p.A)
        add((b - w - p.B * p.qmin) / p.A)
        add((b - w - p.B * p.qmax) / p.A)

    if not vals:
        return np.asarray([], dtype=float)
    return np.sort(np.unique(np.round(np.asarray(vals, dtype=float), 12)))


def _sample_interval_representatives(breaks: np.ndarray) -> list[float]:
    if len(breaks) == 0:
        return [0.0]
    reps = [float(breaks[0] - 1.0)]
    for k in range(len(breaks) - 1):
        reps.append(float(0.5 * (breaks[k] + breaks[k + 1])))
    reps.append(float(breaks[-1] + 1.0))
    return reps


def _build_stage_recursion(next_fun: _PiecewiseLinearConvex, p: _ProblemParams, t: int) -> _StageRecursion:
    a_plus = p.lam[t] / (p.B * p.eta_h)
    a_minus = p.lam[t] / (p.B * p.eta_c)
    xref = p.Tref

    xH = _extended_critical_point(next_fun, -a_plus, p.L, p.U, xref)
    x0c = _extended_critical_point(next_fun, 0.0, p.L, p.U, xref)
    xC = _extended_critical_point(next_fun, a_minus, p.L, p.U, xref)

    beta1 = (xH - p.w_fore[t] - p.B * p.eta_h * p.E_free[t]) / p.A
    beta2 = (x0c - p.w_fore[t] - p.B * p.eta_h * p.E_free[t]) / p.A
    beta3 = (x0c - p.w_fore[t] + p.B * p.eta_c * p.E_free[t]) / p.A
    beta4 = (xC - p.w_fore[t] + p.B * p.eta_c * p.E_free[t]) / p.A

    stage = _StageRecursion(
        t=t,
        p=p,
        next_fun=next_fun,
        fun=next_fun,  # placeholder, overwritten below
        xH=float(xH),
        x0c=float(x0c),
        xC=float(xC),
        beta1=float(beta1),
        beta2=float(beta2),
        beta3=float(beta3),
        beta4=float(beta4),
        a_plus=float(a_plus),
        a_minus=float(a_minus),
    )

    raw_breaks = _stage_breakpoints(next_fun, p, t, xH, x0c, xC, beta1, beta2, beta3, beta4)
    reps = _sample_interval_representatives(raw_breaks)
    slopes = []
    for rep in reps:
        _, slope, _, _ = stage.actual_optimizer(rep)
        slopes.append(float(slope))
    breaks, piece_slopes = _compress_breaks_slopes(list(raw_breaks), slopes)

    x_anchor = p.Tref
    x_star_anchor, _, _, _ = stage.actual_optimizer(x_anchor)
    F_anchor = stage.stage_cost(x_anchor, x_star_anchor) + float(next_fun.value(x_star_anchor))
    fun = _PiecewiseLinearConvex(breaks, piece_slopes, x_anchor, F_anchor)

    return _StageRecursion(
        t=t,
        p=p,
        next_fun=next_fun,
        fun=fun,
        xH=float(xH),
        x0c=float(x0c),
        xC=float(xC),
        beta1=float(beta1),
        beta2=float(beta2),
        beta3=float(beta3),
        beta4=float(beta4),
        a_plus=float(a_plus),
        a_minus=float(a_minus),
    )


def _stage_check(stage: _StageRecursion, tol: float = 1e-10) -> Dict[str, Any]:
    breaks = stage.fun.breaks
    reps = _sample_interval_representatives(breaks)
    records = []
    # Check monotonicity only on open intervals intersecting [L,U], because the
    # analytical threshold-crossing construction is defined on that interval.
    internal = [b for b in stage.fun.breaks if stage.p.L + tol < b < stage.p.U - tol]
    lu_edges = [stage.p.L] + internal + [stage.p.U]
    lu_slopes = []
    for a, b in zip(lu_edges[:-1], lu_edges[1:]):
        mid = 0.5 * (a + b)
        lu_slopes.append(float(stage.fun.slope_open(mid)))
    monotone_ok = bool(np.all(np.diff(lu_slopes) >= -1e-10))
    case_ok = True
    for x in reps:
        xhat, hatf, scase = stage.structural_optimizer(x)
        xstar, actual_slope, scase2, pcase = stage.actual_optimizer(x)
        if scase != scase2:
            case_ok = False
        if scase == "I":
            cond_ok = x <= stage.beta1 + tol
        elif scase == "II":
            cond_ok = stage.beta1 - tol <= x <= stage.beta2 + tol
        elif scase == "III":
            cond_ok = stage.beta2 - tol <= x <= stage.beta3 + tol
        elif scase == "IV":
            cond_ok = stage.beta3 - tol <= x <= stage.beta4 + tol
        else:
            cond_ok = x >= stage.beta4 - tol
        case_ok = case_ok and cond_ok
        records.append({
            "x": float(x),
            "structural_case": scase,
            "projection_case": pcase,
            "xhat": float(xhat),
            "xstar": float(xstar),
            "hatf": float(hatf),
            "actual_slope": float(actual_slope),
        })
    return {
        "stage": int(stage.t + 1),
        "xH": float(stage.xH),
        "x0": float(stage.x0c),
        "xC": float(stage.xC),
        "beta1": float(stage.beta1),
        "beta2": float(stage.beta2),
        "beta3": float(stage.beta3),
        "beta4": float(stage.beta4),
        "monotone_slope_ok": monotone_ok,
        "five_interval_case_ok": case_ok,
        "records": records,
    }


# -----------------------------
# Exact five-interval DP solver
# -----------------------------
def solve_dp_debug(
    fore_len: int,
    A: float,
    B: float,
    x0: float,
    w_fore: np.ndarray,
    h: float,
    E_free: np.ndarray,
    lam: np.ndarray,
    eta_c: float = 1.0,
    eta_h: float = 1.0,
    Tmax: float = 6.0,
    Tmin: float = 2.0,
    qmin: float = -1.0,
    qmax: float = 1.0,
    num_grid: int = 4001,
    compare_with_lp: bool = False,
) -> DPDebugResult:
    p = _ProblemParams(
        fore_len=fore_len,
        A=A,
        B=B,
        x0=x0,
        w_fore=w_fore,
        h=h,
        E_free=E_free,
        lam=lam,
        eta_c=eta_c,
        eta_h=eta_h,
        Tmax=Tmax,
        Tmin=Tmin,
        qmin=qmin,
        qmax=qmax,
    )

    N = p.fore_len
    terminal = _PiecewiseLinearConvex(np.asarray([], dtype=float), np.asarray([0.0], dtype=float), p.Tref, 0.0)
    stage_data: list[_StageRecursion | None] = [None] * N
    funs: list[_PiecewiseLinearConvex] = [terminal for _ in range(N + 1)]
    funs[N] = terminal

    for t in range(N - 1, -1, -1):
        stage_data[t] = _build_stage_recursion(funs[t + 1], p, t)
        funs[t] = stage_data[t].fun

    x_grid = _build_plot_grid(p, num_points=num_grid)
    F = [np.asarray(funs[t].value(x_grid), dtype=float) for t in range(N + 1)]
    f = [np.asarray(funs[t].slope_open(x_grid), dtype=float) for t in range(N + 1)]

    x_path = [p.x0]
    q_path: list[float] = []
    e_req_path: list[float] = []
    e_pv_path: list[float] = []
    e_grid_path: list[float] = []
    for t in range(N):
        st = stage_data[t]
        assert st is not None
        x = x_path[-1]
        x_next, _, _, _ = st.actual_optimizer(x)
        q = (x_next - p.A * x - p.w_fore[t]) / p.B
        e_req, e_pv, e_grid = _stage_energy_from_q(q, p.E_free[t], p.eta_c, p.eta_h)
        x_path.append(float(x_next))
        q_path.append(float(q))
        e_req_path.append(float(e_req))
        e_pv_path.append(float(e_pv))
        e_grid_path.append(float(e_grid))

    sol = DPSolution(
        q_star=float(q_path[0]),
        e_req=float(e_req_path[0]),
        e_pv=float(e_pv_path[0]),
        e_grid=float(e_grid_path[0]),
    )

    stage_checks = [_stage_check(st) for st in stage_data if st is not None]
    lp_comp = None
    if compare_with_lp:
        lp_sol = solve_lp(
            fore_len=fore_len,
            A=A,
            B=B,
            x0=x0,
            w_fore=w_fore,
            h=h,
            E_free=E_free,
            lam=lam,
            eta_c=eta_c,
            eta_h=eta_h,
            Tmax=Tmax,
            Tmin=Tmin,
            qmin=qmin,
            qmax=qmax,
        )
        lp_comp = {
            "abs_diff_q": abs(sol.q_star - lp_sol.q_star),
            "abs_diff_e_req": abs(sol.e_req - lp_sol.e_req),
            "abs_diff_e_pv": abs(sol.e_pv - lp_sol.e_pv),
            "abs_diff_e_grid": abs(sol.e_grid - lp_sol.e_grid),
        }

    return DPDebugResult(
        solution=sol,
        x_grid=x_grid,
        F=F,
        f=f,
        x_path=np.asarray(x_path, dtype=float),
        q_path=np.asarray(q_path, dtype=float),
        e_req_path=np.asarray(e_req_path, dtype=float),
        e_pv_path=np.asarray(e_pv_path, dtype=float),
        e_grid_path=np.asarray(e_grid_path, dtype=float),
        stage_checks=stage_checks,
        lp_comparison=lp_comp,
    )


def solve_dp(
    fore_len: int,
    A: float,
    B: float,
    x0: float,
    w_fore: np.ndarray,
    h: float,
    E_free: np.ndarray,
    lam: np.ndarray,
    eta_c: float = 2.0,
    eta_h: float = 1.0,
    Tmax: float = 6.0,
    Tmin: float = 2.0,
    qmin: float = -1.0,
    qmax: float = 1.0,
) -> DPSolution:
    return solve_dp_debug(
        fore_len=fore_len,
        A=A,
        B=B,
        x0=x0,
        w_fore=w_fore,
        h=h,
        E_free=E_free,
        lam=lam,
        eta_c=eta_c,
        eta_h=eta_h,
        Tmax=Tmax,
        Tmin=Tmin,
        qmin=qmin,
        qmax=qmax,
    ).solution


# -----------------------------
# LP baseline solver (unchanged)
# -----------------------------
def solve_lp(
    fore_len: int,
    A: float,
    B: float,
    x0: float,
    w_fore: np.ndarray,
    h: float,
    E_free: np.ndarray,
    lam: np.ndarray,
    eta_c: float = 2.0,
    eta_h: float = 1.0,
    Tmax: float = 6.0,
    Tmin: float = 2.0,
    qmin: float = -1.0,
    qmax: float = 1.0,
    penalty_violation: float = 1e5,
    primary_tol: float = 1e-9,
) -> DPSolution:
    p = _ProblemParams(
        fore_len=fore_len,
        A=A,
        B=B,
        x0=x0,
        w_fore=w_fore,
        h=h,
        E_free=E_free,
        lam=lam,
        eta_c=eta_c,
        eta_h=eta_h,
        Tmax=Tmax,
        Tmin=Tmin,
        qmin=qmin,
        qmax=qmax,
    )
    N = p.fore_len

    n = 6 * N
    iq, ix, ig, isl, isu, idr = 0, N, 2 * N, 3 * N, 4 * N, 5 * N
    A_eq, b_eq, A_ub, b_ub = [], [], [], []

    for k in range(N):
        row = np.zeros(n)
        row[ix + k] = 1.0
        row[iq + k] = -p.B
        rhs = p.w_fore[k]
        if k == 0:
            rhs += p.A * p.x0
        else:
            row[ix + k - 1] = -p.A
        A_eq.append(row)
        b_eq.append(rhs)

    for k in range(N):
        row = np.zeros(n)
        row[iq + k] = 1.0 / p.eta_h
        row[ig + k] = -1.0
        A_ub.append(row)
        b_ub.append(p.E_free[k])

        row = np.zeros(n)
        row[iq + k] = -1.0 / p.eta_c
        row[ig + k] = -1.0
        A_ub.append(row)
        b_ub.append(p.E_free[k])

        row = np.zeros(n)
        row[ix + k] = -1.0
        row[isl + k] = -1.0
        A_ub.append(row)
        b_ub.append(-p.L)

        row = np.zeros(n)
        row[ix + k] = 1.0
        row[isu + k] = -1.0
        A_ub.append(row)
        b_ub.append(p.U)

        row = np.zeros(n)
        row[ix + k] = 1.0
        row[idr + k] = -1.0
        A_ub.append(row)
        b_ub.append(p.Tref)

        row = np.zeros(n)
        row[ix + k] = -1.0
        row[idr + k] = -1.0
        A_ub.append(row)
        b_ub.append(-p.Tref)

    A_eq = np.asarray(A_eq, dtype=float)
    b_eq = np.asarray(b_eq, dtype=float)
    A_ub = np.asarray(A_ub, dtype=float)
    b_ub = np.asarray(b_ub, dtype=float)

    bounds = []
    bounds.extend([(p.qmin, p.qmax)] * N)
    bounds.extend([(None, None)] * N)
    bounds.extend([(0.0, None)] * N)
    bounds.extend([(0.0, None)] * N)
    bounds.extend([(0.0, None)] * N)
    bounds.extend([(0.0, None)] * N)

    c_primary = np.zeros(n)
    c_primary[ig:ig + N] = p.lam
    c_primary[isl:isl + N] = penalty_violation
    c_primary[isu:isu + N] = penalty_violation

    res_primary = linprog(c_primary, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res_primary.status != 0:
        raise RuntimeError(f"Primary LP solve failed: {res_primary.message}")
    primary_opt = float(res_primary.fun)

    tie_weights = np.arange(N, 0, -1, dtype=float)
    c_secondary = np.zeros(n)
    c_secondary[idr:idr + N] = tie_weights

    A_ub_2 = np.vstack([A_ub, c_primary])
    b_ub_2 = np.concatenate([b_ub, [primary_opt + primary_tol * max(1.0, abs(primary_opt))]])

    res = linprog(c_secondary, A_ub=A_ub_2, b_ub=b_ub_2, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"Secondary LP solve failed: {res.message}")

    q_star = float(res.x[iq])
    e_req, e_pv, e_grid = _stage_energy_from_q(q_star, p.E_free[0], p.eta_c, p.eta_h)
    return DPSolution(q_star=q_star, e_req=float(e_req), e_pv=float(e_pv), e_grid=float(e_grid))


# -----------------------------
# Validation and plotting helpers
# -----------------------------
def validate_exact_dp_against_lp(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        dp = solve_dp(**case)
        lp = solve_lp(**case)
        dbg = solve_dp_debug(**case)
        rows.append({
            "case_id": idx,
            "abs_diff_q": abs(dp.q_star - lp.q_star),
            "abs_diff_e_req": abs(dp.e_req - lp.e_req),
            "abs_diff_e_pv": abs(dp.e_pv - lp.e_pv),
            "abs_diff_e_grid": abs(dp.e_grid - lp.e_grid),
            "five_interval_case_ok": all(sc["five_interval_case_ok"] for sc in dbg.stage_checks),
            "monotone_slope_ok": all(sc["monotone_slope_ok"] for sc in dbg.stage_checks),
        })
    return rows


def plot_dp_stage_slopes(debug_result: DPDebugResult, save_prefix: str) -> None:
    for t in range(len(debug_result.f) - 1):
        plt.figure(figsize=(8, 5))
        plt.step(debug_result.x_grid, debug_result.f[t], where="post")
        plt.xlabel(r"State $x_t$")
        plt.ylabel(rf"$f_{{{t+1}}}(x)$")
        plt.title(f"Exact five-interval DP slope at stage {t+1}")
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_slope_t{t+1}.png", dpi=220)
        plt.close()


# -----------------------------
# Additional baseline controllers
# -----------------------------
def solve_lyapunov_hvac(
    fore_len: int,
    A: float,
    B: float,
    x0: float,
    w_fore: np.ndarray,
    h: float,
    E_free: np.ndarray,
    lam: np.ndarray,
    V_lyap: float = 5.0,
    comfort_weight: float = 0.0,
    energy_weight: float = 1.0,
    pv_awareness: float = 0.5,
    comfort_halfband: float = 0.4,
    Tref: float | None = None,
    Gamma: float | None = None,
    eta_c: float = 2.0,
    eta_h: float = 1.0,
    Tmax: float = 6.0,
    Tmin: float = 2.0,
    qmin: float = -1.0,
    qmax: float = 1.0,
) -> DPSolution:
    """
    Literature-style but tunable Lyapunov baseline for the warehouse.

    Core:
        H_t = T_t + Gamma
        min_q  drift(q) + V * [ energy_weight * energy_cost(q)
                              + comfort_weight * discomfort(q) ]

    Notes:
        - Only the first-step information w_fore[0], E_free[0], lam[0] is used.
        - h is ignored and kept only for interface compatibility.
    """
    _ = fore_len
    _ = h

    w0 = float(np.asarray(w_fore, dtype=float).reshape(-1)[0])
    e0 = float(np.asarray(E_free, dtype=float).reshape(-1)[0])
    lam0 = float(np.asarray(lam, dtype=float).reshape(-1)[0])

    pv_awareness = float(np.clip(pv_awareness, 0.0, 1.0))

    if Tref is None:
        Tref = 0.5 * (Tmin + Tmax)
    if Gamma is None:
        Gamma = -Tref  # H_t = T_t - Tref

    H_t = x0 + Gamma

    def discomfort_of_T(T_next: float) -> float:
        dev = abs(T_next - Tref) - comfort_halfband
        return max(dev, 0.0) ** 2

    def energy_cost_of_q(q: float) -> tuple[float, float, float, float]:
        e_req, e_pv, e_grid = _stage_energy_from_q(q, e0, eta_c, eta_h)
        cost = lam0 * ((1.0 - pv_awareness) * e_req + pv_awareness * e_grid)
        return cost, e_req, e_pv, e_grid

    def objective(q: float) -> float:
        q = float(q)
        T_next = A * x0 + B * q + w0
        H_next = T_next + Gamma
        drift = 0.5 * (H_next**2 - H_t**2)

        dis = discomfort_of_T(T_next)
        e_cost, _, _, _ = energy_cost_of_q(q)

        return drift + V_lyap * (energy_weight * e_cost + comfort_weight * dis)

    # Breakpoints so bounded minimization is only applied on smooth intervals.
    pts = [qmin, qmax, 0.0, -eta_c * e0, eta_h * e0]
    if abs(B) > 1e-12:
        q_b1 = (Tref - comfort_halfband - (A * x0 + w0)) / B
        q_b2 = (Tref + comfort_halfband - (A * x0 + w0)) / B
        pts.extend([q_b1, q_b2])

    pts = [float(np.clip(p, qmin, qmax)) for p in pts]
    pts = sorted(set([round(p, 12) for p in pts]))

    cand_q = []
    cand_val = []
    for q in pts:
        cand_q.append(float(q))
        cand_val.append(objective(float(q)))

    for a, b in zip(pts[:-1], pts[1:]):
        if b - a <= 1e-10:
            continue
        res = minimize_scalar(objective, bounds=(a, b), method='bounded')
        if res.success:
            q_loc = float(np.clip(res.x, qmin, qmax))
            cand_q.append(q_loc)
            cand_val.append(objective(q_loc))

    idx = int(np.argmin(np.array(cand_val)))
    q_star = float(np.clip(cand_q[idx], qmin, qmax))
    _, e_req, e_pv, e_grid = energy_cost_of_q(q_star)

    return DPSolution(
        q_star=q_star,
        e_req=float(e_req),
        e_pv=float(e_pv),
        e_grid=float(e_grid),
    )


def solve_rule_base(
    fore_len: int,
    A: float,
    B: float,
    x0: float,
    w_fore: np.ndarray,
    h: float,
    E_free: np.ndarray,
    lam: np.ndarray,
    eta_c: float = 2.0,
    eta_h: float = 1.0,
    Tmax: float = 6.0,
    Tmin: float = 2.0,
    qmin: float = -1.0,
    qmax: float = 1.0,
    deadband: float = 0.50,
    safety_margin: float = 0.05,
    pre_margin: float = 0.40,
    risk_tol: float = 0.15,
    pv_ratio_trigger: float = 0.60,
) -> DPSolution:
    """
    Model-based proportional rule-based baseline.

    Core idea:
        1) Predict the one-step free response: x_free = A*x0 + w0
        2) Choose q so that the nominal next-step temperature is pulled back toward Tref:
              q_nom = (Tref - x_free) / B
        3) Add deadband and hard-bound protection

    q > 0 : heating
    q < 0 : cooling
    """
    _ = fore_len
    _ = h
    _ = lam
    _ = pre_margin
    _ = risk_tol
    _ = pv_ratio_trigger

    w_fore = np.asarray(w_fore, dtype=float).reshape(-1)
    E_free = np.asarray(E_free, dtype=float).reshape(-1)

    if len(w_fore) < 1:
        raise ValueError("w_fore must be non-empty")
    if len(E_free) < 1:
        raise ValueError("E_free must be non-empty")

    Tref = 0.5 * (Tmin + Tmax)
    w0 = float(w_fore[0])

    # x_free_now = x0

    # Hard-bound protection
    # if x0 >= Tmax - safety_margin:
    #     q_star = float(qmin)
    # elif x0 <= Tmin + safety_margin:
    #     q_star = float(qmax)
    # else:
    #     err = Tref - x_free_now
    #     if abs(err) <= deadband:
    #         q_star = 0.0
    #     else:
    #         if abs(B) <= 1e-12:
    #             q_star = 0.0
    #         else:
    #             q_star = err / B

    q_star = (Tref-x0)/10

    q_star = float(np.clip(q_star, qmin, qmax))

    e_req, e_pv, e_grid = _stage_energy_from_q(q_star, E_free[0], eta_c, eta_h)
    return DPSolution(
        q_star=float(q_star),
        e_req=float(e_req),
        e_pv=float(e_pv),
        e_grid=float(e_grid),
    )
