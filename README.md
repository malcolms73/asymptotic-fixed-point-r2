# Asymptotic Fixed Point Finder (EH + \(R^2\) with \(\tau\))

Small, self-contained NumPy/SciPy program to locate and validate a UV fixed point in the Einstein–Hilbert + \(R^2\) truncation.  
Verifies the **EH slice \(\tau=0\)**, continues into small positive \(\tau\), refines the full fixed point, prints physical residuals, computes the stability matrix/eigenpairs, and reports reproducibility. Includes quick sensitivity checks (regulator and decoupling in \(m^2\)).

> **Notation:** internally we solve for \(x=\ln g\) and map back via \(g=\exp x\). Variables shown in the output are the physical \((g,\lambda,\tau)\).

---

## Install & run

~~~bash
# Recommended modern stack (aligns with Colab / Py3.12)
pip install "numpy>=2.0" "scipy>=1.12"

# Run the main driver
python frg_fixed_point_tau.py
~~~

### Optional CLI flags

~~~bash
# All flags are optional; defaults shown in parentheses
python frg_fixed_point_tau.py \
  --tau-target 0.004   # continuation target for τ (0.004)
  --m2 5.0             # fixed m^2 = M2_FIXED (decoupling test; 5.0)
  --epsM 1e-6          # regulator ε_M on M denominators (1e-6)
  --epsD 1e-8          # regulator ε_Δ on Δ denominators (1e-8)
  --no-evecs           # skip eigenvector print (eigenvalues only)
  --quiet              # reduce continuation logging
~~~

### Examples

~~~bash
# Default run
python frg_fixed_point_tau.py

# Decoupling check (lighter heavy sector)
python frg_fixed_point_tau.py --m2 3.0

# Regulator sensitivity (bump ε_Δ)
python frg_fixed_point_tau.py --m2 10.0 --epsD 1e-7
~~~

### Example output (excerpt)

> Numbers may shift slightly across NumPy/SciPy/Python versions; the structure should persist.

~~~text
>>> Continuation (adaptive): τ : 0 → 0.004 (M2_FIXED=5.0)
  ...
>>> Verifying final point with full system …
--- Verification ---
  g*=48.412516, λ*=-3.876570, τ*=0.000212
  |β_x|=4.44e-16, |β_λ|=7.11e-15, |β_τ|=2.17e-19

=== Stability (Jacobian at FP) ===
eig(J) = [-9.88503304 -2.12261046 -3.13484774]
θ (=-eig) = [ 9.88503304  2.12261046  3.13484774]

--- Eigenvectors (columns) mapped to (g, λ, τ), unit-norm columns ---
[[ 0.617467 -0.995131 -0.997043]
 [-0.786597  0.098558  0.076841]
 [ 0.000069  0.000044  0.000240]]

Dominant component per eigenvector: ['λ', 'g', 'g']
~~~

### Reproducibility

~~~text
Python : 3.12.11
NumPy  : 2.0.2
SciPy  : 1.16.1
Seeds  : NumPy default_rng(42)
Solver : least_squares (TRF), bounds x∈[-30,30], λ_t∈[-200,200], τ_t∈[-2000,2000]
Tols   : ftol=xtol=gtol=1e-12
Reg    : EPS_M=1e-6, EPS_D=1e-8, M2_FIXED=5.0
Note   : Internal variable x = ln g ⇒ g = exp(x) in printed results
~~~

### Cite

~~~bibtex
@misc{Scott_AFPR2_2025,
  title  = {Asymptotic Fixed Point Finder (EH + R^2)},
  author = {Malcolm Scott},
  year   = {2025},
  doi    = {10.5281/zenodo.16924371},
  url    = {https://doi.org/10.5281/zenodo.16924371}
}
~~~

::contentReference[oaicite:0]{index=0}

