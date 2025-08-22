\# Asymptotic Fixed Point Finder (EH + R^2)



Small, self-contained NumPy/SciPy program to locate and validate a UV fixed point in the Einstein–Hilbert + R^2 truncation. Verifies the β=0 EH slice, continues in β, refines the full fixed point, prints physical residuals, computes stability matrix/eigenvalues, runs a minimal robustness sweep, and reports reproducibility.



\## Install \& run

```bash

pip install -r requirements.txt

python fixed\_point.py

```



\## Example output (excerpt)

```

--- Verification ---

g\*=80.418321, λ\*=-8.351335, β\*=0.000678

|β\_x|≈4.4e-15, |β\_λ|≈2.1e-13, |β\_β|≈2.0e-16



=== Robustness sweep (excerpt) ===

EPS\_D=1e-08, w=(1,1,5) -> g\*=80.418321, λ\*=-8.351335, β\*=0.000678

eig(J) ≈ \[-2.3487e+03, -1.0754e-01, -1.4483e+01]

θ = -eig(J) ≈ \[2.3487e+03, 1.4483e+01, 1.0754e-01]

```

\## Files

\- `fixed\_point.py` — solver + stability + robustness + reproducibility  

\- `requirements.txt` — tested versions  

\- `examples/reference\_win\_py39.txt` — saved output from a known working run  

\- `LICENSE` — project license



```

\*\*Reproducibility (tested env)\*\*  

Python 3.9.13 · NumPy 1.26.4 · SciPy 1.10.1  

Seeds: NumPy `default\_rng(42)`  

Solver: `least\_squares` (TRF), bounds x∈\[-30,30], λ\_t∈\[-200,200], β\_t∈\[-2000,2000], tolerances ftol=xtol=gtol=1e−12  

Regulators: ε\_M=1e−6, ε\_Δ=1e−8

```







