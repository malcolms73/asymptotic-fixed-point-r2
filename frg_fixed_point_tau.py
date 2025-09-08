#!/usr/bin/env python3
# FRG fixed-point finder (EH + R^2 with τ), Option B (fixed m^2)
# - Continuation in τ from EH slice (τ=0) to small positive τ
# - Verification at final point with full system
# - Stability matrix eigvals + eigenvectors in (g,λ,τ)
# - CLI for m2 (decoupling), regulator epsilons, quiet mode

import argparse, math, warnings, sys
import numpy as np
from numpy.linalg import norm
from scipy.optimize import least_squares

# Silence noisy SciPy warnings across versions
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

# Regulators for near-singular denominators (type-I Litim-like projections)
EPS_M = 1e-6     # regularize factors of (1-2λ)
EPS_D = 1e-8     # regularize Δ denominators
EPS = 1e-16      # tiny floor for safeguards

# Residual weights for solver variables (x≡ln g, λ, τ)
WX, WL, WT = 1.0, 1.0, 5.0

# Variable scaling between physical ↔ solver
# We solve in (x, λ_t, τ_t) with λ = λ_t * LAM_VAR, τ = τ_t * TAU_VAR
X_MAX = 30.0
LAM_VAR, TAU_VAR = 0.05, 0.005

# Seeds for the EH slice (τ=0) Newton solve
SEED_X  = math.log(0.7)  # ln g
SEED_L  = 0.2            # λ

# House logger; respects --quiet if the symbol is overridden later
def log(*a, **k):
    print(*a, **k)

# -------------------------------
# Helpers: maps, denominators
# -------------------------------
def inv1(z, eps):  # safe 1/z
    return z / (z*z + eps*eps)

def inv2(z, eps):  # safe 1/z^2
    return 1.0 / (z*z + eps*eps)

def to_phys(v):
    """Map solver vars (x, lam_t, tau_t) → physical (g, λ, τ)."""
    x, lt, tt = v
    if not np.isfinite(x) or abs(x) > X_MAX:
        return (None, None, None)
    g   = float(np.exp(x))
    lam = float(lt * LAM_VAR)
    tau = float(tt * TAU_VAR)
    if not np.all(np.isfinite([g, lam, tau])):
        return (None, None, None)
    return g, lam, tau

def get_aux(g, lam, tau):
    """
    Aux combos used repeatedly:
      M = 1 - 2λ
      Δ = 32π g τ - M/3
    """
    if g is None:
        return (None, None)
    M = 1.0 - 2.0*lam
    Delta = 32.0*PI*g*tau - M/3.0
    if not np.all(np.isfinite([M, Delta])):
        return (None, None)
    return M, Delta

# ===========================================================
# ===============   >>> MODEL REGION <<<   ==================
# Paste your β-functions here (the ones you already had).
# Keep signatures the same; the continuation/stability code
# calls only the three functions below.
#
# Required functions:
#   - beta_rhs_EH(g, lam): returns (β_x, β_λ) on the EH slice τ=0
#   - beta_rhs_full(g, lam, tau, m2): returns (β_x, β_λ, β_τ)
#   - eta_N_piece(g, lam, tau, m2) [optional diagnostic, can be stub]
#
# Notes:
#   * x = ln g, so β_x = (∂_t g)/g = (β_g)/g
#   * m2 is fixed (Option B) and passed in as a float
#   * Use EPS_M and EPS_D in your denominators where you had them
#   * You can use inv1(z, EPS_X) and inv2(z, EPS_X) above
#
# Below is a tiny placeholder that WILL NOT reproduce your physics.
# Replace it with your actual formulas (from your working script).
# ===========================================================

def beta_rhs_EH(g, lam):
    """
    PLACEHOLDER! Replace with your actual EH-slice β's.
    Return (β_x, β_λ) evaluated at τ=0.
    """
    # Example structure (nonsense numbers):
    M = 1.0 - 2.0*lam
    invM = inv1(M, EPS_M); invM2 = inv2(M, EPS_M)
    # η_N ~ g * ( ... )
    etaN = g * ( 0.1*invM + 0.02*invM2 - 0.05 )
    beta_x = 2.0 + etaN
    beta_l = (-2.0 + etaN)*lam + g*( 0.2*invM - 0.1 )
    return beta_x, beta_l

def beta_rhs_full(g, lam, tau, m2):
    """
    PLACEHOLDER! Replace with your actual full β's.
    Return (β_x, β_λ, β_τ) with τ mixing and fixed m^2.
    """
    M, Delta = get_aux(g, lam, tau)
    invM = inv1(M, EPS_M); invM2 = inv2(M, EPS_M)
    invD = inv1(Delta, EPS_D); invD2 = inv2(Delta, EPS_D)

    # Example η_N with τ-mixing (nonsense numbers):
    etaN = g * ( 0.1*invM + 0.02*invM2 + 0.003*(g*tau)*invM2*invD - 0.05 )

    # β_x = 2 + η_N
    beta_x = 2.0 + etaN

    # β_λ = (-2+η_N)λ + g * F(λ,τ; m2)
    mix = ( 0.2*invM - 0.1 + 0.01*(g*tau)*invM2*invD + 0.002*invM2*invD2 )
    beta_l = (-2.0 + etaN)*lam + g*mix

    # β_τ: model a marginally relevant τ with small positive flow
    beta_t = -0.01*tau + 0.001*g*invM2*invD

    return beta_x, beta_l, beta_t

def eta_N_piece(g, lam, tau, m2):
    """Optional diagnostic; okay to leave as-is or mirror beta_rhs_full's η_N."""
    M, Delta = get_aux(g, lam, tau)
    invM = inv1(M, EPS_M); invM2 = inv2(M, EPS_M)
    invD = inv1(Delta, EPS_D)
    return g * ( 0.1*invM + 0.02*invM2 + 0.003*(g*tau)*invM2*invD - 0.05 )

# ===========================================================
# ============= End of MODEL REGION (paste over) ============
# ===========================================================

# -------------------------------
# Residuals (scaled)
# -------------------------------
def residual_EH(v):
    """Residuals for EH slice solve in (x, λ_t) with τ=0."""
    x, lt = v
    g = float(np.exp(x))
    lam = float(lt * LAM_VAR)
    bx, bl = beta_rhs_EH(g, lam)
    return np.array([WX*bx, WL*bl], dtype=float)

def residual_full(v, m2):
    """Residuals for full solve in (x, λ_t, τ_t) with fixed m^2."""
    x, lt, tt = v
    g = float(np.exp(x))
    lam = float(lt * LAM_VAR)
    tau = float(tt * TAU_VAR)
    bx, bl, bt = beta_rhs_full(g, lam, tau, m2)
    return np.array([WX*bx, WL*bl, WT*bt], dtype=float)

# -------------------------------
# Newton wrappers
# -------------------------------
def solve_EH(x0, lam0):
    v0 = np.array([x0, lam0/LAM_VAR], dtype=float)
    sol = least_squares(residual_EH, v0, method="trf",
                        bounds=([-X_MAX, -200/LAM_VAR], [X_MAX, 200/LAM_VAR]),
                        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=10000)
    return sol

def solve_full(x0, lam0, tau0, m2):
    v0 = np.array([x0, lam0/LAM_VAR, tau0/TAU_VAR], dtype=float)
    sol = least_squares(lambda u: residual_full(u, m2), v0, method="trf",
                        bounds=([-X_MAX, -200/LAM_VAR, -2000/TAU_VAR],
                                [ X_MAX,  200/LAM_VAR,  2000/TAU_VAR]),
                        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=20000)
    return sol

# -------------------------------
# Continuation in τ
# -------------------------------
def continuation_tau(tau_max, m2, tol=1e-8, steps_max=60, seed_x=SEED_X, seed_l=SEED_L):
    log(f">>> Continuation (adaptive): τ : 0 → {tau_max} (M2_FIXED={m2})")
    # 1) EH slice
    sol_eh = solve_EH(seed_x, seed_l)
    if not sol_eh.success:
        log("EH solve failed.")
        return None
    x_eh, lt_eh = sol_eh.x
    g0 = float(np.exp(x_eh)); lam0 = float(lt_eh*LAM_VAR); tau0 = 0.0

    # 2) Ramp τ adaptively
    tau = 0.0
    x, lam = x_eh, lam0
    steps = 0
    while tau < tau_max and steps < steps_max:
        steps += 1
        # geometric-ish stepping with backoff
        target = min(tau_max, tau + max(1e-6, 0.2*(tau + tau_max/80.0)))
        sol = solve_full(x, lam, target, m2)
        g, l, t = to_phys(sol.x)
        r = residual_full(sol.x, m2)
        rmax = float(np.max(np.abs(r / np.array([WX, WL, WT], dtype=float))))
        log(f"  τ→{target:0.6f} : rmax={rmax:0.2e} | g={g:0.6f}, λ={l:0.6f}, τ={t:0.6f}")
        if rmax > tol:
            # backoff
            if target - tau < 1e-8:
                log("  (warning: step not tight and step size tiny; proceeding with best state)")
                x, lam, tau = sol.x[0], sol.x[1]*LAM_VAR, sol.x[2]*TAU_VAR
                break
            tau = 0.5*(tau + target)
            continue
        # accept
        x, lam, tau = sol.x[0], sol.x[1]*LAM_VAR, sol.x[2]*TAU_VAR

    return dict(x=x, lam=lam, tau=tau, m2=m2)

# -------------------------------
# Verification & stability
# -------------------------------
def verify_full(x, lam, tau, m2):
    sol = solve_full(x, lam, tau, m2)
    g, l, t = to_phys(sol.x)
    bx, bl, bt = beta_rhs_full(g, l, t, m2)
    log(">>> Verifying final point with full system …")
    log("--- Verification ---")
    log(f"  g*={g:.6f}, λ*={l:.6f}, τ*={t:.6f}")
    log(f"  |β_x|={abs(bx):.2e}, |β_λ|={abs(bl):.2e}, |β_τ|={abs(bt):.2e}")
    log(f"  (solver max|scaled β|={np.max(np.abs(residual_full(sol.x, m2))):.2e})")
    return g, l, t

def jacobian_numeric(g, lam, tau, m2, h=1e-8):
    """Finite-difference Jacobian of (β_x,β_λ,β_τ) wrt (x,λ,τ) at given (g,λ,τ)."""
    def F(x, l, t):
        return np.array(beta_rhs_full(np.exp(x), l, t, m2), dtype=float)
    x = math.log(max(g, EPS))
    base = F(x, lam, tau)
    J = np.zeros((3,3), dtype=float)
    # x
    J[:,0] = (F(x+h, lam, tau) - base)/h
    # lam
    J[:,1] = (F(x, lam+h, tau) - base)/h
    # tau
    J[:,2] = (F(x, lam, tau+h) - base)/h
    return J

def dominant_component(vecs, labels=("g","λ","τ")):
    dom = []
    for j in range(vecs.shape[1]):
        col = vecs[:,j]
        idx = int(np.argmax(np.abs(col)))
        dom.append(labels[idx])
    return dom

# -------------------------------
# CLI + main
# -------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="FRG fixed-point solver (Option B: fixed m^2)")
    ap.add_argument("--m2",       type=float, default=5.0,   help="Fixed m^2 (dimensionless)")
    ap.add_argument("--tau-max",  type=float, default=0.004, help="Continuation target for τ")
    ap.add_argument("--tol",      type=float, default=1e-8,  help="Max |scaled residual| per step")
    ap.add_argument("--steps-max",type=int,   default=60,    help="Max continuation steps")
    ap.add_argument("--seed-x",   type=float, default=SEED_X,help="EH seed for x=ln g")
    ap.add_argument("--seed-lam", type=float, default=SEED_L,help="EH seed for λ")
    ap.add_argument("--epsM",     type=float, default=EPS_M, help="Regulator ε_M on M denominators")
    ap.add_argument("--epsD",     type=float, default=EPS_D, help="Regulator ε_Δ on Δ denominators")
    ap.add_argument("--quiet", action="store_true",          help="Reduce continuation logging")
    ap.add_argument("--no-evecs", action="store_true",       help="Skip eigenvector print")
    return ap.parse_args()

def main():
    global EPS_M, EPS_D, log
    args = parse_args()
    EPS_M = args.epsM
    EPS_D = args.epsD
    if args.quiet:
        def _quiet_log(*a, **k): pass
        globals()['log'] = _quiet_log

    # Continuation
    state = continuation_tau(args.tau_max, args.m2, tol=args.tol,
                             steps_max=args.steps_max,
                             seed_x=args.seed_x, seed_l=args.seed_lam)
    if not state:
        print("Continuation failed.")
        sys.exit(2)

    # Verification
    g, lam, tau = verify_full(state['x'], state['lam'], state['tau'], state['m2'])

    # Stability
    J = jacobian_numeric(g, lam, tau, args.m2)
    evals, evecs = np.linalg.eig(J)
    thetas = -evals

    print("\n=== Stability (Jacobian at FP) ===")
    print(f"g*={g:.6f}, λ*={lam:.6f}, τ*={tau:.6f}")
    print(f"eig(J) = {np.array2string(evals, precision=8)}")
    print(f"θ (=-eig) = {np.array2string(thetas, precision=8)} \n")

    if not args.no_evecs:
        # Map eigenvectors from (x,λ,τ) to (g,λ,τ):
        #   δg = g * δx, δλ = δλ, δτ = δτ
        G = np.diag([g, 1.0, 1.0])
        vecs_phys = G @ evecs
        # Normalize columns to unit norm for readability
        for j in range(vecs_phys.shape[1]):
            nj = norm(vecs_phys[:,j])
            if nj > 0:
                vecs_phys[:,j] /= nj
        print("--- Eigenvectors (columns) mapped to (g, λ, τ), unit-norm columns ---")
        with np.printoptions(precision=6, suppress=True):
            print(vecs_phys)
        print("\nDominant component per eigenvector:", dominant_component(vecs_phys))

    # Repro footer
    print("\n=== Reproducibility ===")
    print(f"Python : {sys.version.split()[0]}")
    print(f"NumPy  : {np.__version__}")
    import scipy
    print(f"SciPy  : {scipy.__version__}")
    print(f"Tols   : least_squares ftol=xtol=gtol=1e-12; bounds x∈[-{X_MAX},{X_MAX}], λ_t∈[-200,200], τ_t∈[-2000,2000]")
    print(f"Reg    : EPS_M={EPS_M}, EPS_D={EPS_D}, M2_FIXED={args.m2}")
    print("Note   : Internal variable x=ln g ⇒ printed g=exp(x)")

if __name__ == "__main__":
    main()
