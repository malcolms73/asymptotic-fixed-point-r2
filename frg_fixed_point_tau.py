#!/usr/bin/env python3
import argparse, math, warnings
import numpy as np
from scipy.optimize import least_squares

# --- Clean noisy warnings across SciPy versions ---
try:
    from scipy.linalg import LinAlgWarning
except Exception:
    class LinAlgWarning(Warning): pass

warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------------
# Global knobs (safe defaults)
# -------------------------------
PI = np.pi
# Regularizers for near-singular denominators
EPS_M, EPS_D = 1e-6, 1e-8

# Residual weights (x≡ln g, λ, τ)
WX, WL, WT = 1.0, 1.0, 5.0

# Bounds in solver space: x∈[-30,30], λ_t, τ_t are the unscaled variables we optimize
X_MAX = 30.0
LAM_VAR, TAU_VAR = 0.05, 0.005  # scale physical ↔ solver vars

# -------------------------------
# Argument parsing
# -------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="FRG fixed-point solver (Option B: fixed m^2)")
    ap.add_argument("--m2", type=float, default=5.0, help="Dimensionless mass m^2=M_k^2/k^2 (fixed)")
    ap.add_argument("--tau-max", type=float, default=0.004, help="Target tau for continuation")
    ap.add_argument("--tol", type=float, default=1e-8, help="Tolerance on max residual per step")
    ap.add_argument("--steps-max", type=int, default=60, help="Max adaptive steps")
    ap.add_argument("--seed-x", type=float, default=math.log(0.7), help="EH seed: x=ln g")
    ap.add_argument("--seed-lam", type=float, default=0.2, help="EH seed: lambda")
    return ap.parse_args()

# -------------------------------
# Helper maps and denominators
# -------------------------------
def inv1(z, eps): return z / (z*z + eps*eps)
def inv2(z, eps): return 1.0 / (z*z + eps*eps)

def to_phys(v):
    """Map solver vars (x, lam_t, tau_t) ↦ physical (g, λ, τ)."""
    x, lt, tt = v
    if not np.isfinite(x) or abs(x) > X_MAX: return (None, None, None)
    g = float(np.exp(x))
    lam = float(lt * LAM_VAR)
    tau = float(tt * TAU_VAR)
    if not np.all(np.isfinite([g, lam, tau])): return (None, None, None)
    return g, lam, tau

def get_aux(g, lam, tau, m2_fixed):
    """M = 1-2λ, Δ = 32π g τ - M/3; keep m2=m2_fixed for threshold functions."""
    if g is None: return (None, None)
    M = 1.0 - 2.0*lam
    U = 32.0 * PI * g * tau
    Delta = U - M/3.0
    if not np.all(np.isfinite([M, Delta])): return (None, None)
    return M, Delta

# -------------------------------
# Coefficients (placeholders → tuned to the projection used)
# -------------------------------
def get_A_coeffs(g, lam, tau, M, Delta):
    invM, invM2 = inv1(M, EPS_M), inv2(M, EPS_M)
    invD, invD2 = inv1(Delta, EPS_D), inv2(Delta, EPS_D)
    A1 = -2*lam + (g/(2*PI))*(4*invM - 4 + 64*PI*g*tau*(invM2*invD) + (2/3.)*invM2 + 32*PI*g*tau*(invM2*invD2))
    A2 =  lam - (g/(4*PI))*(4*invM + (2/3.)*invM2 + 32*PI*g*tau*(invM2*invD2))
    return A1, A2

def get_B_coeffs(g, lam, tau, M, Delta):
    invM, invM2 = inv1(M, EPS_M), inv2(M, EPS_M)
    invD, invD2 = inv1(Delta, EPS_D), inv2(Delta, EPS_D)
    B1 = (1.0/(4.0*PI))*(-5/6 - (25/36)*invM - (1/6)*(invM*invD) + (32/3.)*PI*g*tau*(invM*invD) + (16/3.)*PI*g*tau*(invM*invD2))
    B2 = -(1.0/(8.0*PI))*((10/3.)*invM + (1/6.)*invM2 + (16/3.)*PI*g*tau*(invM*invD2))
    return B1, B2

def get_C_coeffs(g, lam, tau, M, Delta):
    invM, invM2 = inv1(M, EPS_M), inv2(M, EPS_M)
    invD, invD2 = inv1(Delta, EPS_D), inv2(Delta, EPS_D)
    invM3 = invM*invM2; invMD = invM*invD
    C1 = (1.0/(16.0*PI*PI))*(-13/360 + (3/4.)*invM - (1/90.)*invM3 - (23/60.)*invM2 + (1/18.)*invMD + (1/6.)*(M*invMD*invD2))
    C2sum = (89/180.)*invM2 - 0.5*invMD - (1/6.)*(M*invMD*invD2) + (1/60.)*invM3
    C2 = -(1.0/(32.0*PI*PI))*C2sum
    return C1, C2

# -------------------------------
# Beta functions (physical basis)
# -------------------------------
def beta_phys(x, lam, tau, m2_fixed):
    g = float(np.exp(x))
    M, Delta = get_aux(g, lam, tau, m2_fixed)
    if M is None: return np.array([np.inf, np.inf, np.inf], float)
    B1, B2 = get_B_coeffs(g, lam, tau, M, Delta)
    denom = 1.0 - g*B2
    if (not np.isfinite(denom)) or abs(denom) < 1e-12: return np.array([np.inf, np.inf, np.inf], float)
    eta_N = (g*B1)/denom
    A1, A2 = get_A_coeffs(g, lam, tau, M, Delta)
    C1, C2 = get_C_coeffs(g, lam, tau, M, Delta)
    beta_x   = 2.0 + eta_N
    beta_lam = (eta_N - 2.0)*lam + (g/(8.0*PI))*(A1 + eta_N*A2)
    beta_tau = 2.0*eta_N*tau + (g/(4.0*PI))*(C1 + eta_N*C2)
    return np.array([beta_x, beta_lam, beta_tau], float)

def beta_scaled(v, m2_fixed):
    """Scaled residuals in solver variables (x, lam_t, tau_t)."""
    g, lam, tau = to_phys(v)
    if g is None: return np.array([1e6, 1e6, 1e6], float)
    bx, bl, bt = beta_phys(np.log(g), lam, tau, m2_fixed)
    # scale lambdas back to solver space
    return np.array([WX*bx, WL*bl, WT*bt], float)

def beta_clamped(v, tau_target, m2_fixed):
    """Clamp τ to target for continuation (solve {β_x=0, β_λ=0, τ - τ_target = 0})."""
    g, lam, tau = to_phys(v)
    if g is None: return np.array([1e6, 1e6, 1e6], float)
    bx, bl, _ = beta_phys(np.log(g), lam, tau, m2_fixed)
    clamp = tau - tau_target
    return np.array([WX*bx, WL*bl, WT*clamp], float)

# -------------------------------
# Solvers / utilities
# -------------------------------
def lsq(fun, x0):
    lb = np.array([-X_MAX, -10.0/LAM_VAR, -10.0/TAU_VAR], float)
    ub = np.array([ +X_MAX, +10.0/LAM_VAR, +10.0/TAU_VAR], float)
    return least_squares(fun, x0, method="trf", bounds=(lb, ub),
                         ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=4000)

def print_state(t, rmax, v):
    g, lam, tau = to_phys(v)
    if g is None:
        print(f"  τ→{t:.6f} : rmax={rmax:.2e} | invalid")
    else:
        print(f"  τ→{t:.6f} : rmax={rmax:.2e} | g={g:.6f}, λ={lam:.6f}, τ={tau:.6f}")

def self_test_EH():
    # sanity: EH slice equals τ=0 cut of full system
    g, lam, tau = 1.0, 0.1, 0.0
    bx1, bl1, bt1 = beta_phys(np.log(g), lam, tau, m2_fixed=5.0)
    bx2, bl2, bt2 = bx1, bl1, bt1  # here EH = τ=0 slice of the same formulas
    print("--- Running Self-Test for EH Limit (τ=0 slice) ---")
    print(f"Δβ_λ (full@τ=0 minus EH) =  {bl1 - bl2: .2e}")
    print("----------------------------------------")

# -------------------------------
# Continuation in τ (adaptive)
# -------------------------------
def adaptive_continuation(m2_fixed=5.0, tau_max=0.004, tol=1e-8, steps_max=60,
                          seed_x=np.log(0.7), seed_lam=0.2):
    print(f">>> Continuation (adaptive): τ : 0 → {tau_max} (M2_FIXED={m2_fixed}) ")
    # seed at τ=0
    v = np.array([seed_x, seed_lam/LAM_VAR, 0.0/TAU_VAR], float)
    # try to lock onto the good branch quickly
    ts = [tau_max/1000.0, tau_max/600.0, tau_max/450.0]
    t, it = 0.0, 0
    for s in ts:
        it += 1
        F = lambda z: beta_clamped(z, s, m2_fixed)
        res = lsq(F, v)
        r = F(res.x); rmax = float(np.max(np.abs(r)))
        if rmax > 1e-2:
            print("  (warning: stuck on a bad branch; restarting from EH seed)")
            v = np.array([seed_x, seed_lam/LAM_VAR, 0.0/TAU_VAR], float)
        else:
            print_state(s, rmax, res.x)
            v = res.x
            t = s

    # adaptive to tau_max
    n = 0
    while t < tau_max and n < steps_max:
        # propose step that shrinks as we approach target
        dt = min( max( (tau_max - t)/6.0, tau_max/400.0 ), tau_max/20.0 )
        t_next = min(t + dt, tau_max)
        F = lambda z: beta_clamped(z, t_next, m2_fixed)
        res = lsq(F, v)
        r = F(res.x); rmax = float(np.max(np.abs(r)))
        print_state(t_next, rmax, res.x)
        v = res.x
        t = t_next
        n += 1

    return v

# -------------------------------
# Jacobian and eigensystem
# -------------------------------
def jacobian_beta_phys(x, lam, tau, m2_fixed, hx=1e-6, hl=1e-6, ht=1e-6):
    f0 = beta_phys(x, lam, tau, m2_fixed); J = np.zeros((3,3), float)
    J[:,0] = (beta_phys(x+hx, lam, tau, m2_fixed) - beta_phys(x-hx, lam, tau, m2_fixed)) / (2*hx)
    J[:,1] = (beta_phys(x, lam+hl, tau, m2_fixed) - beta_phys(x, lam-hl, tau, m2_fixed)) / (2*hl)
    J[:,2] = (beta_phys(x, lam, tau+ht, m2_fixed) - beta_phys(x, lam, tau-ht, m2_fixed)) / (2*ht)
    return J

def map_evecs_to_gltau(evecs_xltau, g_star):
    """Columns are eigenvectors in (x,λ,τ). Map to (g,λ,τ) via δg = g* δx and unit-normalize."""
    M = np.diag([g_star, 1.0, 1.0])  # (x,λ,τ) → (g,λ,τ)
    E = M @ evecs_xltau
    # unit-norm columns
    for j in range(E.shape[1]):
        n = np.linalg.norm(E[:,j])
        if n > 0: E[:,j] /= n
    return E

# -------------------------------
# Main
# -------------------------------
def main():
    args = parse_args()
    m2 = float(args.m2)
    self_test_EH()

    # 1) continuation in τ with fixed m^2
    v = adaptive_continuation(m2_fixed=m2, tau_max=args.tau_max, tol=args.tol, steps_max=args.steps_max,
                              seed_x=args.seed_x, seed_lam=args.seed_lam)

    # 2) verify final point against full system (unclamped)
    res_final = lsq(lambda z: beta_scaled(z, m2), v)
    # refine once more
    res_final = lsq(lambda z: beta_scaled(z, m2), res_final.x)

    g_star, lam_star, tau_star = to_phys(res_final.x)
    bx, bl, bt = beta_phys(np.log(g_star), lam_star, tau_star, m2)

    print(">>> Verifying final point with full system …")
    print("--- Verification ---")
    print(f"  g*={g_star:.6f}, λ*={lam_star:.6f}, τ*={tau_star:.6f}")
    print(f"  |β_x|={abs(bx):.2e}, |β_λ|={abs(bl):.2e}, |β_τ|={abs(bt):.2e}")
    r_scaled = beta_scaled(res_final.x, m2)
    print(f"  (solver max|scaled β|={float(np.max(np.abs(r_scaled))):.2e})\n")

    # 3) stability matrix and eigen-system at FP
    J = jacobian_beta_phys(np.log(g_star), lam_star, tau_star, m2)
    evals, evecs = np.linalg.eig(J)
    thetas = -np.real(evals)

    print("=== Stability (Jacobian at FP) ===")
    print(f"g*={g_star:.6f}, λ*={lam_star:.6f}, τ*={tau_star:.6f}")
    print("eig(J) =", np.real(evals))
    print("θ (=-eig) =", thetas, "\n")

    # map eigenvectors to (g,λ,τ)
    E_gltau = map_evecs_to_gltau(evecs, g_star)
    labels = ["g","λ","τ"]
    dom = []
    print("--- Eigenvectors (columns) mapped to (g, λ, τ), unit-norm columns ---")
    np.set_printoptions(precision=6, suppress=True)
    print(E_gltau)
    for j in range(E_gltau.shape[1]):
        dom.append(labels[int(np.argmax(np.abs(E_gltau[:,j])) )])
    print("\nDominant component per eigenvector:", dom)

if __name__ == "__main__":
    main()
