# fixed_point.py
import warnings
import numpy as np
from scipy.optimize import root, least_squares
import math

# Robust LinAlgWarning import across SciPy versions
try:
    from scipy.linalg import LinAlgWarning
except Exception:
    class LinAlgWarning(Warning):
        pass

# Keep output clean
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Global Settings ---
EH_MODE = False  # True: EH test (β=0). False: full (g, λ, β).

# --- Constants / scales ---
PI = np.pi
LAM_VAR, BET_VAR = 0.05, 0.005           # variable scales (solver-space → physical)
SX, SL, SB = 1.0, 1.0, 5.0               # residual weights
X_MAX   = 30.0
EPS_M   = 1e-6
EPS_D   = 1e-8
PENALTY = 1e6

# --- helpers ---
def inv1(z, eps): return z / (z*z + eps*eps)
def inv2(z, eps): return 1.0 / (z*z + eps*eps)

def to_phys(vars_):
    if not np.all(np.isfinite(vars_)): return (None, None, None)
    x, lt, bt = vars_
    if abs(x) > X_MAX: return (None, None, None)
    g = np.exp(x); lam = lt * LAM_VAR; beta = bt * BET_VAR
    if not np.all(np.isfinite([g, lam, beta])): return (None, None, None)
    return g, lam, beta

def get_aux_vars(g, lam, beta):
    if g is None or lam is None or beta is None: return (None, None)
    M = 1.0 - 2.0*lam
    U = 32.0 * PI * g * beta
    Delta = U - M/3.0
    if not np.all(np.isfinite([M, Delta])): return (None, None)
    return M, Delta

# --- Coefficients ---
def get_A_coeffs(g, lam, beta, M, Delta):
    invM, invM2 = inv1(M, EPS_M), inv2(M, EPS_M)
    invD, invD2 = inv1(Delta, EPS_D), inv2(Delta, EPS_D)
    A1 = -2*lam + (g/(2*PI))*(4*invM - 4 + 64*PI*g*beta*(invM2*invD) + (2/3.)*invM2 + 32*PI*g*beta*(invM2*invD2))
    A2 =  lam - (g/(4*PI))*(4*invM + (2/3.)*invM2 + 32*PI*g*beta*(invM2*invD2))
    return A1, A2

def get_B_coeffs(g, lam, beta, M, Delta):
    invM, invM2 = inv1(M, EPS_M), inv2(M, EPS_M)
    invD, invD2 = inv1(Delta, EPS_D), inv2(Delta, EPS_D)
    B1 = (1.0/(4.0*PI))*(-5/6 - (25/36)*invM - (1/6)*(invM*invD) + (32/3.)*PI*g*beta*(invM*invD) + (16/3.)*PI*g*beta*(invM*invD2))
    B2 = -(1.0/(8.0*PI))*((10/3.)*invM + (1/6.)*invM2 + (16/3.)*PI*g*beta*(invM*invD2))
    return B1, B2

def get_C_coeffs(g, lam, beta, M, Delta):
    invM, invM2 = inv1(M, EPS_M), inv2(M, EPS_M)
    invD, invD2 = inv1(Delta, EPS_D), inv2(Delta, EPS_D)
    invM3 = invM*invM2; invMD = invM*invD
    C1 = (1.0/(16.0*PI*PI))*(-13/360 + (3/4.)*invM - (1/90.)*invM3 - (23/60.)*invM2 + (1/18.)*invMD + (1/6.)*(M*invMD*invD2))
    C2sum = (89/180.)*invM2 - 0.5*invMD - (1/6.)*(M*invMD*invD2) + (1/60.)*invM3
    C2 = -(1.0/(32.0*PI*PI))*C2sum
    return C1, C2

# --- Beta systems ---
def beta_system(vars_):
    g, lam, beta = to_phys(vars_)
    M, Delta = get_aux_vars(g, lam, beta)
    if M is None or Delta is None: return np.full(3, PENALTY, float)
    B1, B2 = get_B_coeffs(g, lam, beta, M, Delta)
    denom = 1.0 - g*B2
    if (not np.isfinite(denom)) or abs(denom) < 1e-12: return np.full(3, PENALTY, float)
    eta_N = (g*B1)/denom
    A1, A2 = get_A_coeffs(g, lam, beta, M, Delta)
    C1, C2 = get_C_coeffs(g, lam, beta, M, Delta)
    beta_x    = 2.0 + eta_N
    beta_lam  = (eta_N - 2.0)*lam + (g/(8.0*PI))*(A1 + eta_N*A2)
    beta_beta = 2.0*eta_N*beta + (g/(4.0*PI))*(C1 + eta_N*C2)
    out = np.array([SX*beta_x, SL*beta_lam, SB*beta_beta], float)
    return out if np.all(np.isfinite(out)) else np.full(3, PENALTY, float)

def beta_system_eh(vars_):
    x, lam = vars_
    if not np.isfinite(x) or not np.isfinite(lam) or abs(x) > X_MAX: return np.array([PENALTY, PENALTY], float)
    g = np.exp(x); M = 1.0 - 2.0*lam; beta = 0.0; Delta = -M/3.0
    B1, B2 = get_B_coeffs(g, lam, beta, M, Delta)
    denom = 1.0 - g*B2
    if (not np.isfinite(denom)) or abs(denom) < 1e-12: return np.array([PENALTY, PENALTY], float)
    eta_N = (g*B1)/denom
    A1, A2 = get_A_coeffs(g, lam, beta, M, Delta)
    beta_x   = 2.0 + eta_N
    beta_lam = (eta_N - 2.0)*lam + (g/(8.0*PI))*(A1 + eta_N*A2)
    out = np.array([SX*beta_x, SL*beta_lam], float)
    return out if np.all(np.isfinite(out)) else np.array([PENALTY, PENALTY], float)

def beta_system_safe(v):
    try:
        r = beta_system(v)
        return np.clip(r, -PENALTY, PENALTY).astype(float) if np.all(np.isfinite(r)) else np.full_like(v, PENALTY, float)
    except Exception:
        return np.full_like(v, PENALTY, float)

def beta_system_clamped(v, beta_target):
    g, lam, beta = to_phys(v)
    M, Delta = get_aux_vars(g, lam, beta)
    if M is None or Delta is None: return np.full(3, PENALTY, float)
    B1, B2 = get_B_coeffs(g, lam, beta, M, Delta)
    denom = 1.0 - g*B2
    if (not np.isfinite(denom)) or abs(denom) < 1e-12: return np.full(3, PENALTY, float)
    eta_N = (g*B1)/denom
    A1, A2 = get_A_coeffs(g, lam, beta, M, Delta)
    beta_x   = 2.0 + eta_N
    beta_lam = (eta_N - 2.0)*lam + (g/(8.0*PI))*(A1 + eta_N*A2)
    clamp = SB*(beta - beta_target)
    out = np.array([SX*beta_x, SL*beta_lam, clamp], float)
    return out if np.all(np.isfinite(out)) else np.full(3, PENALTY, float)

def beta_system_clamped_safe(v, t):
    try:
        r = beta_system_clamped(v, t)
        return np.clip(r, -PENALTY, PENALTY).astype(float) if np.all(np.isfinite(r)) else np.full_like(v, PENALTY, float)
    except Exception:
        return np.full_like(v, PENALTY, float)

def beta_phys(x, lam, beta):
    g = np.exp(x); M, Delta = get_aux_vars(g, lam, beta)
    if M is None or Delta is None: return np.array([np.inf, np.inf, np.inf], float)
    B1, B2 = get_B_coeffs(g, lam, beta, M, Delta)
    denom = 1.0 - g*B2
    if (not np.isfinite(denom)) or abs(denom) < 1e-12: return np.array([np.inf, np.inf, np.inf], float)
    eta_N = (g*B1)/denom
    A1, A2 = get_A_coeffs(g, lam, beta, M, Delta)
    C1, C2 = get_C_coeffs(g, lam, beta, M, Delta)
    beta_x    = 2.0 + eta_N
    beta_lam  = (eta_N - 2.0)*lam + (g/(8.0*PI))*(A1 + eta_N*A2)
    beta_beta = 2.0*eta_N*beta + (g/(4.0*PI))*(C1 + eta_N*C2)
    return np.array([beta_x, beta_lam, beta_beta], float)

# --- seeding & solvers ---
def _generate_wobble_seeds_eh(init, pcts=(0.10, 0.20)):
    x0, l0 = init; seeds = [init.copy()]
    for p in pcts:
        for dx in (0.0, +p, -p):
            for ml in (1.0, 1.0+p, 1.0-p):
                s = np.array([x0 + dx, l0*ml], float)
                if not np.allclose(s, seeds[-1]): seeds.append(s)
    rng = np.random.default_rng(42)
    for _ in range(10): seeds.append((init + rng.normal(scale=[0.05, 0.10], size=2)).astype(float))
    return seeds

def _solve_with_wobbles(system_fn, seeds, tol=1e-8, method='hybr'):
    best = {"res": np.inf, "sol": None, "seed": None}
    for seed in seeds:
        sol = root(system_fn, seed, method=method)
        try: maxres = float(np.max(np.abs(system_fn(sol.x))))
        except Exception: maxres = np.inf
        if sol.success and np.isfinite(maxres) and maxres < tol: return sol, maxres, seed
        if np.isfinite(maxres) and maxres < best["res"]: best = {"res": maxres, "sol": sol, "seed": seed}
    return None, best["res"], best["seed"]

def self_test_EH_limit(g=1.0, lam=0.1):
    print("--- Running Self-Test for EH Limit ---")
    M = 1.0 - 2.0*lam; beta = 0.0; Delta = -M/3.0
    B1, B2 = get_B_coeffs(g, lam, beta, M, Delta)
    etaN = (g*B1)/(1.0 - g*B2)
    A1, A2 = get_A_coeffs(g, lam, beta, M, Delta)
    beta_lam_full = (etaN - 2.0)*lam + (g/(8.0*PI))*(A1 + etaN*A2)
    x = np.log(g)
    _, beta_lam_eh = beta_system_eh(np.array([x, lam], float))
    diff = beta_lam_full - beta_lam_eh
    print(f"Δβ_λ (full@β=0 minus EH) = {diff:.2e}")
    print("SUCCESS: EH mode matches the β=0 slice of the full system." if abs(diff) < 1e-12
          else "WARNING: Still mismatched; recheck A/B transcription.")
    print("-"*40)

def eh_fixed_point_seed(tol=1e-8):
    init = np.array([np.log(0.7), 0.2], float)
    seeds = _generate_wobble_seeds_eh(init)
    for m in ("krylov", "anderson", "hybr"):
        for s in seeds:
            sol = root(beta_system_eh, s, method=m)
            rmax = float(np.max(np.abs(beta_system_eh(sol.x))))
            if sol.success and rmax < tol: return sol.x
    # fallback: smallest residual
    best = min((float(np.max(np.abs(beta_system_eh(root(beta_system_eh, s, method="hybr").x)))), s) for s in seeds)[1]
    return best

def _print_state(t, rmax, vec):
    g, lam, beta = to_phys(vec)
    if g is None:
        x, lt, bt = vec
        print(f"  β→{t:.4f} : rmax={rmax:.2e} | invalid state: x={x:.3e}, λ_t={lt:.3e}, β_t={bt:.3e}")
    else:
        print(f"  β→{t:.4f} : rmax={rmax:.2e} | g={g:.6f}, λ={lam:.6f}, β={beta:.6f}")

def _lsq_step(fun, x0):
    lb = np.array([-X_MAX, -10.0/LAM_VAR, -10.0/BET_VAR], float)
    ub = np.array([ +X_MAX,  +10.0/LAM_VAR, +10.0/BET_VAR], float)
    return least_squares(fun, x0, method="trf", bounds=(lb, ub),
                         ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000)

def run_full_continuation(beta_target=0.01, steps=40, tol=1e-8):
    print(f">>> Continuation: β : 0 → {beta_target} in {steps} steps")
    x0, lam0 = eh_fixed_point_seed(tol=1e-8)
    vars_ = np.array([x0, lam0/LAM_VAR, 0.0/BET_VAR], float)
    for t in np.linspace(0.0, beta_target, steps+1)[1:]:
        F = lambda v: beta_system_clamped_safe(v, t)
        res = _lsq_step(F, vars_)
        r = F(res.x); rmax = float(np.max(np.abs(r)))
        _print_state(t, rmax, res.x)
        if rmax >= 1e-6: print("  (warning: step not fully tight; proceeding with best state)")
        vars_ = res.x
    print(">>> Verifying final point with full system …")
    res_final = _lsq_step(beta_system_safe, vars_)
    res_final = _lsq_step(beta_system_safe, res_final.x)  # extra refine
    R_scaled  = beta_system_safe(res_final.x)
    Rmax_scaled = float(np.max(np.abs(R_scaled)))
    g_star, lam_star, beta_star = to_phys(res_final.x)
    Rx, Rl, Rb = beta_phys(np.log(g_star), lam_star, beta_star)
    print("--- Verification ---")
    print(f"  g*={g_star:.6f}, λ*={lam_star:.6f}, β*={beta_star:.6f}")
    print(f"  |β_x|={abs(Rx):.2e}, |β_λ|={abs(Rl):.2e}, |β_β|={abs(Rb):.2e}")
    print(f"  (solver max|scaled β|={Rmax_scaled:.2e})")
    return res_final, np.array([Rx, Rl, Rb]), Rmax_scaled

# --- stability / robustness / reproducibility ---
def jacobian_beta_phys(x, lam, beta, hx=1e-6, hl=1e-6, hb=1e-6):
    f0 = beta_phys(x, lam, beta); J = np.zeros((3,3), float)
    J[:,0] = (beta_phys(x+hx, lam, beta) - beta_phys(x-hx, lam, beta)) / (2*hx)
    J[:,1] = (beta_phys(x, lam+hl, beta) - beta_phys(x, lam-hl, beta)) / (2*hl)
    J[:,2] = (beta_phys(x, lam, beta+hb) - beta_phys(x, lam, beta-hb)) / (2*hb)
    return J

def compute_stability_from_result(res_final):
    g_star, lam_star, beta_star = to_phys(res_final.x)
    x_star = np.log(g_star)
    J = jacobian_beta_phys(x_star, lam_star, beta_star)
    evals = np.linalg.eigvals(J)
    thetas = -np.sort_complex(evals)
    return (g_star, lam_star, beta_star, J, evals, thetas)

def robustness_sweep(beta_target=0.01, epsD_list=(1e-7, 1e-8), weight_list=((1,1,1),(1,1,5)), steps=30):
    epsD0 = globals()['EPS_D']; sx0, sl0, sb0 = globals()['SX'], globals()['SL'], globals()['SB']
    rows = []
    try:
        for epsD in epsD_list:
            globals()['EPS_D'] = epsD
            for (sx, sl, sb) in weight_list:
                globals()['SX'], globals()['SL'], globals()['SB'] = sx, sl, sb
                res_final, Rphys, Rmax = run_full_continuation(beta_target=beta_target, steps=steps)
                g_star, lam_star, beta_star = to_phys(res_final.x)
                SX_old, SL_old, SB_old = globals()['SX'], globals()['SL'], globals()['SB']
                globals()['SX'], globals()['SL'], globals()['SB'] = 1.0, 1.0, 1.0
                _, _, _, J, evals, thetas = compute_stability_from_result(res_final)
                globals()['SX'], globals()['SL'], globals()['SB'] = SX_old, SL_old, SB_old
                rows.append({"EPS_D": epsD, "weights": (sx, sl, sb), "g*": g_star, "lam*": lam_star,
                             "beta*": beta_star, "max|scaled β|": Rmax, "eig(J)": evals, "theta": thetas})
    finally:
        globals()['EPS_D'] = epsD0
        globals()['SX'], globals()['SL'], globals()['SB'] = sx0, sl0, sb0
    print("\n=== Robustness sweep ===")
    for r in rows:
        print(f"EPS_D={r['EPS_D']:.0e}, w={r['weights']} -> g*={r['g*']:.6f}, λ*={r['lam*']:.6f}, β*={r['beta*']:.6f}, max|scaled β|={r['max|scaled β|']:.2e}")
        print(f"    eig(J) = {np.array2string(r['eig(J)'], precision=4)}")
        print(f"    θ (=-eig) = {np.array2string(r['theta'], precision=4)}")
    return rows

def print_reproducibility():
    import sys, platform, numpy, scipy
    print("\n=== Reproducibility ===")
    print(f"Python : {sys.version.split()[0]}  ({platform.system()} {platform.release()})")
    print(f"NumPy  : {numpy.__version__}")
    print(f"SciPy  : {scipy.__version__}")
    print(f"Seeds  : NumPy default_rng(42) used in seeding functions")
    print("Tols   : least_squares ftol=xtol=gtol=1e-12; solver bounds x∈[-30,30], "
          f"λ_t∈[{-10.0/LAM_VAR:.1f},{10.0/LAM_VAR:.1f}], β_t∈[{-10.0/BET_VAR:.1f},{10.0/BET_VAR:.1f}]")
    print(f"Reg    : EPS_M={EPS_M:.0e}, EPS_D={EPS_D:.0e}")

# --- main ---
if __name__ == '__main__':
    self_test_EH_limit()
    if EH_MODE:
        # optional EH-only solve
        init = np.array([np.log(0.7), 0.2], float)
        seeds = _generate_wobble_seeds_eh(init, pcts=(0.10, 0.20))
        sol, maxres, seed_used = _solve_with_wobbles(beta_system_eh, seeds, tol=1e-8)
        if sol is not None:
            x_star, lam_star = sol.x; g_star = np.exp(x_star)
            print("\n--- Fixed point (EH) Found ---")
            print(f"  g*={g_star:.6f}, λ*={lam_star:.6f}, max|beta|={maxres:.2e}")
        else:
            print("\n--- EH solve did not converge ---")
    else:
        res_final, Rphys, Rmax = run_full_continuation(beta_target=0.01, steps=40)
        g_star, lam_star, beta_star, J, evals, thetas = compute_stability_from_result(res_final)
        print("\n=== Stability (Jacobian at FP) ===")
        print(f"g*={g_star:.6f}, λ*={lam_star:.6f}, β*={beta_star:.6f}")
        print("eig(J) =", evals)
        print("θ (=-eig) =", thetas)
        robustness_sweep(beta_target=0.01, epsD_list=(1e-7,1e-8), weight_list=((1,1,1),(1,1,5)), steps=20)
        print_reproducibility()
