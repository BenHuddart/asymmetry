"""
Prototype: Bayesian(-ish) experimental design for "suggest next measurement point".

Validates the math for:
  - D-optimal utility (info gain) for a candidate new point x
  - c-optimal utility for a single parameter of interest (rank-one posterior update)
  - counting-time solve to reach a target post-fit variance on a parameter
  - rank-one prediction vs Monte-Carlo "reality" check by refitting

Uses iminuit.LeastSquares + Migrad for all fits, mirroring production usage.

This is a standalone script; it does not modify or import anything from the
asymmetry repo itself.
"""

from __future__ import annotations

import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares

RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------
# Generic sensitivity / D-optimal / c-optimal machinery
# ----------------------------------------------------------------------------


def numeric_gradient(f, theta, x, extra_args=()):
    """Central finite-difference gradient of f(x, *theta, *extra_args) wrt theta.

    Step size per component: max(1e-6, 1e-6*|theta_j|).
    Returns an array of shape (n_params,).
    """
    theta = np.asarray(theta, dtype=float)
    n = theta.size
    grad = np.empty(n)
    for j in range(n):
        h = max(1e-6, 1e-6 * abs(theta[j]))
        tp = theta.copy()
        tm = theta.copy()
        tp[j] += h
        tm[j] -= h
        yp = f(x, *tp, *extra_args)
        ym = f(x, *tm, *extra_args)
        grad[j] = (yp - ym) / (2.0 * h)
    return grad


def d_optimal_utility(g, cov, sigma_new):
    """IG(x) = 0.5 * log(1 + g^T Cov g / sigma_new^2)."""
    gcg = g @ cov @ g
    return 0.5 * np.log1p(gcg / sigma_new**2)


def c_optimal_delta_var(g, cov, sigma_new, k):
    """Rank-one posterior variance reduction for parameter index k.

    ΔVar_k(x) = (Cov g)_k^2 / (sigma_new^2 + g^T Cov g)
    post variance = Cov_kk - ΔVar_k
    Returns (delta_var_k, post_var_k).
    """
    cg = cov @ g
    gcg = g @ cov @ g
    delta = cg[k] ** 2 / (sigma_new**2 + gcg)
    post = cov[k, k] - delta
    return delta, post


def solve_counting_time_for_target(g, cov, k, sigma_emp, t_ref, target_var):
    """Solve for counting time t such that post variance on parameter k hits target_var,
    at fixed x, with sigma_new(x, t)^2 = sigma_emp^2 * t_ref / t.

    post_var(t) = Cov_kk - (Cov g)_k^2 / (sigma_new(t)^2 + g^T Cov g)

    As t -> infinity, sigma_new^2 -> 0, so post_var -> Cov_kk - (Cov g)_k^2 / (g^T Cov g).
    This is the floor achievable with a *single* added point at this x (infinite exposure).
    If target_var < floor, it is unreachable; report the floor and t=inf.

    Otherwise solve analytically:
      target = Cov_kk - (Cov g)_k^2 / (sigma_new^2 + gcg)
      => (Cov g)_k^2 / (sigma_new^2 + gcg) = Cov_kk - target
      => sigma_new^2 = (Cov g)_k^2 / (Cov_kk - target) - gcg
      => t = sigma_emp^2 * t_ref / sigma_new^2
    """
    cg = cov @ g
    gcg = g @ cov @ g
    ck2 = cg[k] ** 2
    ckk = cov[k, k]

    floor_var = ckk - ck2 / gcg  # t -> infinity limit

    if target_var <= floor_var:
        return dict(reachable=False, t_over_tref=np.inf, floor_var=floor_var)

    denom = ckk - target_var
    sigma_new2 = ck2 / denom - gcg
    if sigma_new2 <= 0:
        # Shouldn't happen given target_var > floor_var, but guard anyway.
        return dict(reachable=False, t_over_tref=np.inf, floor_var=floor_var)

    t_over_tref = sigma_emp**2 / sigma_new2
    return dict(reachable=True, t_over_tref=t_over_tref, floor_var=floor_var)


def fit_least_squares(model, x, y, sigma, start, param_names, limits=None):
    """Fit model(x, *params) to (x, y, sigma) via iminuit LeastSquares + Migrad.

    Returns the Minuit object (post-migrad).
    """
    cost = LeastSquares(x, y, sigma, model)
    m = Minuit(cost, *start, name=param_names)
    if limits:
        for name, lim in limits.items():
            m.limits[name] = lim
    m.migrad()
    return m


def safe_model_eval(f, x, params, floor=None):
    """Evaluate f, replacing non-finite results with `floor` (for flat/undefined regions)."""
    val = f(x, *params)
    if not np.isfinite(val):
        return floor if floor is not None else 0.0
    return val


# ----------------------------------------------------------------------------
# Case 1: straight line
# ----------------------------------------------------------------------------


def line_model(x, m, b):
    return m * x + b


def case1_line():
    print("=" * 78)
    print("CASE 1: Straight line  y = m x + b   true (m,b) = (0.5, 1.0)")
    print("=" * 78)

    true_m, true_b = 0.5, 1.0
    x_data = np.linspace(1, 9, 8)
    sigma = 0.1
    y_true = line_model(x_data, true_m, true_b)
    y_data = y_true + RNG.normal(0, sigma, size=x_data.size)

    m = fit_least_squares(
        line_model, x_data, y_data, np.full_like(x_data, sigma),
        start=(1.0, 0.0), param_names=("m", "b"),
    )
    theta_hat = np.array(m.values)
    cov = np.array(m.covariance)
    print(f"Fitted theta_hat = {theta_hat}, valid={m.valid}")
    print(f"Covariance:\n{cov}")

    x_grid = np.linspace(0, 10, 201)
    sigma_new = 0.1

    ig = np.empty_like(x_grid)
    dvar_m = np.empty_like(x_grid)
    dvar_b = np.empty_like(x_grid)
    for i, xc in enumerate(x_grid):
        g = numeric_gradient(line_model, theta_hat, xc)
        ig[i] = d_optimal_utility(g, cov, sigma_new)
        dvar_m[i], _ = c_optimal_delta_var(g, cov, sigma_new, k=0)
        dvar_b[i], _ = c_optimal_delta_var(g, cov, sigma_new, k=1)

    i_d = np.argmax(ig)
    i_m = np.argmax(dvar_m)
    i_b = np.argmax(dvar_b)

    print(f"\nD-optimal argmax: x* = {x_grid[i_d]:.3f}  (IG = {ig[i_d]:.6f})")
    print(f"  boundary check: x*(D) at 0 or 10? -> "
          f"{'YES (x=0)' if np.isclose(x_grid[i_d], 0) else ('YES (x=10)' if np.isclose(x_grid[i_d], 10) else 'NO')}")

    print(f"\nc-optimal for m: argmax x* = {x_grid[i_m]:.3f}  (DeltaVar_m = {dvar_m[i_m]:.6f})")
    print(f"  boundary check: x*(m) at 0 or 10? -> "
          f"{'YES (x=0)' if np.isclose(x_grid[i_m], 0) else ('YES (x=10)' if np.isclose(x_grid[i_m], 10) else 'NO')}")

    print(f"\nc-optimal for b: argmax x* = {x_grid[i_b]:.3f}  (DeltaVar_b = {dvar_b[i_b]:.6f})")
    print(f"  near x=0 check: |x*(b)| <= 0.5? -> {abs(x_grid[i_b]) <= 0.5}")

    # Sanity: IG and DeltaVar_m, DeltaVar_b at the two boundaries
    print("\nUtility at boundaries:")
    print(f"  x=0:  IG={ig[0]:.6f}  DVar_m={dvar_m[0]:.6f}  DVar_b={dvar_b[0]:.6f}")
    print(f"  x=10: IG={ig[-1]:.6f}  DVar_m={dvar_m[-1]:.6f}  DVar_b={dvar_b[-1]:.6f}")
    print()
    return dict(x_grid=x_grid, ig=ig, dvar_m=dvar_m, dvar_b=dvar_b, theta_hat=theta_hat, cov=cov)


# ----------------------------------------------------------------------------
# Case 2: Arrhenius
# ----------------------------------------------------------------------------

K_B = 0.08617  # meV/K


def arrhenius_model(T, a, Ea):
    return a * np.exp(-Ea / (K_B * T))


def case2_arrhenius():
    print("=" * 78)
    print("CASE 2: Arrhenius  y = a * exp(-Ea / (kB T))   true (a, Ea) = (10, 20 meV)")
    print("=" * 78)

    true_a, true_Ea = 10.0, 20.0
    T_data = np.linspace(100, 300, 8)
    y_true = arrhenius_model(T_data, true_a, true_Ea)
    sigma_data = 0.03 * y_true  # heteroscedastic, 3% of y
    y_data = y_true + RNG.normal(0, sigma_data)

    m = fit_least_squares(
        arrhenius_model, T_data, y_data, sigma_data,
        start=(8.0, 15.0), param_names=("a", "Ea"),
    )
    theta_hat = np.array(m.values)
    cov = np.array(m.covariance)
    print(f"Fitted theta_hat = {theta_hat}, valid={m.valid}")
    print(f"Covariance:\n{cov}")

    T_grid = np.linspace(80, 320, 201)
    # sigma_new(T): interpolate/extrapolate the *relative* error model (3% of true y)
    # from the data errors -- since sigma_data = 0.03*y_true, and we know the fractional
    # law, extrapolate using the fitted model's y value with the same fractional 3%.
    # This mirrors "interpolated/extrapolated from the data errors": we interpolate the
    # fractional error (sigma_data/y_true) linearly across T_data, then extrapolate flatly
    # (clamped to endpoint) outside the data range, and apply it to the fitted-curve y(T).
    frac_err = sigma_data / y_true  # ~0.03 constant here, but treat generally
    frac_err_grid = np.interp(T_grid, T_data, frac_err)  # np.interp clamps outside range
    y_fit_grid = arrhenius_model(T_grid, *theta_hat)
    sigma_new_grid = np.abs(frac_err_grid * y_fit_grid)
    # guard against zero/negative sigma_new
    sigma_new_grid = np.clip(sigma_new_grid, 1e-8, None)

    dvar_Ea = np.empty_like(T_grid)
    ig = np.empty_like(T_grid)
    for i, Tc in enumerate(T_grid):
        g = numeric_gradient(arrhenius_model, theta_hat, Tc)
        ig[i] = d_optimal_utility(g, cov, sigma_new_grid[i])
        dvar_Ea[i], _ = c_optimal_delta_var(g, cov, sigma_new_grid[i], k=1)

    i_star = np.argmax(dvar_Ea)
    T_star = T_grid[i_star]
    lo_val = dvar_Ea[0]
    hi_val = dvar_Ea[-1]
    ratio = hi_val / lo_val if lo_val > 0 else np.inf

    print(f"\nc-optimal for Ea: argmax T* = {T_star:.2f} K  (DeltaVar_Ea = {dvar_Ea[i_star]:.6e})")
    which_end = "low-T end (80 K)" if T_star < (T_grid[0] + T_grid[-1]) / 2 else "high-T end (320 K)"
    print(f"  T* is at the {which_end} of the candidate range")
    print(f"  DeltaVar_Ea(80 K)  = {lo_val:.6e}")
    print(f"  DeltaVar_Ea(320 K) = {hi_val:.6e}")
    print(f"  utility ratio (high/low) = {ratio:.4f}")
    print()
    return dict(T_grid=T_grid, dvar_Ea=dvar_Ea, ig=ig, theta_hat=theta_hat, cov=cov,
                sigma_new_grid=sigma_new_grid, T_data=T_data, sigma_data=sigma_data)


# ----------------------------------------------------------------------------
# Case 3: order parameter
# ----------------------------------------------------------------------------


def order_param_model_scalar(T, y0, Tc, alpha, beta):
    """Scalar version: y0*(1-(T/Tc)^alpha)^beta for T<Tc else 0.
    Used inside numeric_gradient (needs scalar in/out)."""
    if T >= Tc:
        return 0.0
    ratio = T / Tc
    base = 1.0 - ratio**alpha
    if base <= 0:
        return 0.0
    return y0 * base**beta


def order_param_model_vec(T, y0, Tc, alpha, beta):
    """Vectorized version for fitting (array T)."""
    T = np.asarray(T, dtype=float)
    out = np.zeros_like(T)
    mask = T < Tc
    ratio = np.where(mask, T / Tc, 0.0)
    base = 1.0 - ratio**alpha
    base = np.where(mask, base, 0.0)
    base_pos = np.clip(base, 0.0, None)
    out = np.where(mask, y0 * base_pos**beta, 0.0)
    return out


def case3_order_parameter():
    print("=" * 78)
    print("CASE 3: Order parameter  y = y0*(1-(T/Tc)^alpha)^beta, T<Tc else 0")
    print("        true (y0, Tc, alpha, beta) = (1.0, 100.0, 2.0, 0.35)")
    print("=" * 78)

    true_params = (1.0, 100.0, 2.0, 0.35)
    T_data = np.linspace(10, 95, 10)
    sigma = 0.02
    y_true = order_param_model_vec(T_data, *true_params)
    y_data = y_true + RNG.normal(0, sigma, size=T_data.size)

    m = fit_least_squares(
        order_param_model_vec, T_data, y_data, np.full_like(T_data, sigma),
        start=(0.9, 105.0, 1.5, 0.4), param_names=("y0", "Tc", "alpha", "beta"),
        limits=dict(y0=(0, None), Tc=(50, 200), alpha=(0.01, None), beta=(0.01, None)),
    )
    theta_hat = np.array(m.values)
    cov = np.array(m.covariance)
    print(f"Fitted theta_hat = {theta_hat}, valid={m.valid}")
    print(f"Covariance:\n{cov}")

    T_grid = np.linspace(5, 120, 201)
    sigma_new = 0.02
    dvar_Tc = np.empty_like(T_grid)
    ig = np.empty_like(T_grid)
    n_nonfinite = 0
    for i, Tc_cand in enumerate(T_grid):
        g = numeric_gradient(order_param_model_scalar, theta_hat, Tc_cand)
        if not np.all(np.isfinite(g)):
            n_nonfinite += 1
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        ig[i] = d_optimal_utility(g, cov, sigma_new)
        dvar_Tc[i], _ = c_optimal_delta_var(g, cov, sigma_new, k=1)

    i_star = np.argmax(dvar_Tc)
    T_star = T_grid[i_star]
    print(f"\nNon-finite gradients encountered & zeroed: {n_nonfinite} / {T_grid.size} grid points")
    print(f"c-optimal for Tc: argmax T* = {T_star:.3f} K  (DeltaVar_Tc = {dvar_Tc[i_star]:.6e})")
    print(f"  expected range ~95-100 K: {'YES' if 95 <= T_star <= 100 else 'NO'}")

    # Secondary peak check: find local maxima away from the global peak.
    # simple local-max detection with a minimum separation
    is_localmax = (
        (dvar_Tc[1:-1] > dvar_Tc[:-2]) & (dvar_Tc[1:-1] > dvar_Tc[2:])
    )
    local_max_idx = np.where(is_localmax)[0] + 1
    local_max_idx = local_max_idx[np.argsort(dvar_Tc[local_max_idx])[::-1]]
    print(f"  local maxima (T, DeltaVar_Tc), top 5 by height:")
    for idx in local_max_idx[:5]:
        print(f"    T={T_grid[idx]:7.2f}  DeltaVar_Tc={dvar_Tc[idx]:.6e}")
    secondary = "YES" if len(local_max_idx) > 1 else "NO"
    print(f"  secondary peak present: {secondary}")
    print()
    return dict(T_grid=T_grid, dvar_Tc=dvar_Tc, ig=ig, theta_hat=theta_hat, cov=cov,
                T_star=T_star, T_data=T_data, sigma=sigma, true_params=true_params)


# ----------------------------------------------------------------------------
# Case 4: rank-one prediction vs Monte-Carlo reality
# ----------------------------------------------------------------------------


def case4_rank_one_vs_reality(case3_result):
    print("=" * 78)
    print("CASE 4: Rank-one prediction vs Monte-Carlo reality (order-parameter Tc)")
    print("=" * 78)

    theta_hat = case3_result["theta_hat"]
    cov = case3_result["cov"]
    T_data = case3_result["T_data"]
    sigma = case3_result["sigma"]
    true_params = case3_result["true_params"]
    T_star = case3_result["T_star"]
    T_bad = 20.0  # flat region well below Tc, deliberately uninformative-ish... but check

    n_mc = 200
    param_names = ("y0", "Tc", "alpha", "beta")
    limits = dict(y0=(0, None), Tc=(50, 200), alpha=(0.01, None), beta=(0.01, None))

    def realized_post_var(x_new, label):
        g = numeric_gradient(order_param_model_scalar, theta_hat, x_new)
        _, pred_post_var = c_optimal_delta_var(g, cov, sigma, k=1)

        realized_vars = []
        n_fail = 0
        for trial in range(n_mc):
            rng_trial = np.random.default_rng(1000 + trial)
            y_true_data = order_param_model_vec(T_data, *true_params)
            y_data = y_true_data + rng_trial.normal(0, sigma, size=T_data.size)

            y_true_new = order_param_model_scalar(x_new, *true_params)
            y_new = y_true_new + rng_trial.normal(0, sigma)

            T_all = np.append(T_data, x_new)
            y_all = np.append(y_data, y_new)
            sig_all = np.full_like(T_all, sigma)

            mm = fit_least_squares(
                order_param_model_vec, T_all, y_all, sig_all,
                start=(0.9, 105.0, 1.5, 0.4), param_names=param_names, limits=limits,
            )
            if not mm.valid:
                n_fail += 1
                continue
            err_Tc = mm.errors["Tc"]
            realized_vars.append(err_Tc**2)

        realized_vars = np.array(realized_vars)
        mean_realized = realized_vars.mean()
        ratio = pred_post_var / mean_realized if mean_realized > 0 else np.nan

        print(f"\n-- {label}: x_new = {x_new:.2f} K --")
        print(f"  rank-one predicted post-Var(Tc) = {pred_post_var:.6e}")
        print(f"  MC mean realized post-Var(Tc)   = {mean_realized:.6e}  "
              f"(n_valid={realized_vars.size}/{n_mc}, n_fail={n_fail})")
        print(f"  predicted / realized ratio      = {ratio:.4f}")
        print(f"  prior Var(Tc) (no new point)     = {cov[1, 1]:.6e}")
        return pred_post_var, mean_realized

    realized_post_var(T_star, "suggested x* (near Tc)")
    realized_post_var(T_bad, "deliberately bad x (flat region, T=20 K)")
    print()


# ----------------------------------------------------------------------------
# Case 5: counting-time solve
# ----------------------------------------------------------------------------


def case5_counting_time(case3_result):
    print("=" * 78)
    print("CASE 5: Counting-time solve (order-parameter Tc)")
    print("=" * 78)

    theta_hat = case3_result["theta_hat"]
    cov = case3_result["cov"]
    T_star = case3_result["T_star"]
    sigma_emp = case3_result["sigma"]  # sigma at reference counting time t_ref
    t_ref = 1.0

    g = numeric_gradient(order_param_model_scalar, theta_hat, T_star)

    print(f"x* = {T_star:.3f} K, sigma_emp = {sigma_emp}, t_ref = {t_ref}")
    print(f"Prior Var(Tc) = {cov[1,1]:.6e}  (sigma = {np.sqrt(cov[1,1]):.4f} K)")

    for target_sigma in (0.5, 0.01):
        target_var = target_sigma**2
        result = solve_counting_time_for_target(g, cov, k=1, sigma_emp=sigma_emp,
                                                  t_ref=t_ref, target_var=target_var)
        print(f"\nTarget: sigma(Tc) <= {target_sigma} K  (Var target = {target_var:.3e})")
        print(f"  floor Var(Tc) as t->inf at this x* = {result['floor_var']:.6e}  "
              f"(floor sigma = {np.sqrt(max(result['floor_var'],0)):.4f} K)")
        if result["reachable"]:
            print(f"  REACHABLE: required t / t_ref = {result['t_over_tref']:.4f}")
        else:
            print(f"  UNREACHABLE with a single added point at this x "
                  f"(target below the t->inf floor)")
    print()


# ----------------------------------------------------------------------------


def main():
    r1 = case1_line()
    r2 = case2_arrhenius()
    r3 = case3_order_parameter()
    case4_rank_one_vs_reality(r3)
    case5_counting_time(r3)


if __name__ == "__main__":
    main()
