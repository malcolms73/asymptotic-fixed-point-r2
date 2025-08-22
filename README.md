# Asymptotic Fixed Point Finder (EH + R^2)

Small, self-contained NumPy/SciPy program to locate and validate a UV fixed point in the Einstein–Hilbert + R^2 truncation. Verifies the β=0 EH slice, continues in β, refines the full fixed point, prints physical residuals, computes stability matrix/eigenvalues, runs a minimal robustness sweep, and reports reproducibility.

## Install & run
```bash
pip install -r requirements.txt
python fixed_point.py
```



## Example output (excerpt)

```
--- Verification ---
g*=80.418321, λ*=-8.351335, β*=0.000678
|β_x|≈4.4e-15, |β_λ|≈2.1e-13, |β_β|≈2.0e-16

=== Robustness sweep (excerpt) ===
EPS_D=1e-08, w=(1,1,5) -> g*=80.418321, λ*=-8.351335, β*=0.000678
eig(J) ≈ [-2.3487e+03, -1.0754e-01, -1.4483e+01]
θ = -eig(J) ≈ [2.3487e+03, 1.4483e+01, 1.0754e-01]

```

## Files

 - `fixed_point.py` — solver + stability + robustness + reproducibility  

 - `requirements.txt` — tested versions  

 - `examples/reference_win_py39.txt` — saved output from a known working run  

 - `LICENSE` — project license



**Reproducibility (tested env)**  

```
Python 3.9.13 · NumPy 1.26.4 · SciPy 1.10.1
Seeds: NumPy default_rng(42)
Solver: least_squares (TRF), bounds x∈[-30,30], λ_t∈[-200,200], β_t∈[-2000,2000], tolerances ftol=xtol=gtol=1e−12
Regulators: ε_M=1e−6, ε_Δ=1e−8
```

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16924371.svg)](https://doi.org/10.5281/zenodo.16924371)

## Cite
If you use this code, please cite the archived release:
```
@misc{Scott_AFPR2_2025,
  title  = {Asymptotic Fixed Point Finder (EH + R^2)},
  author = {Malcolm Scott},
  year   = {2025},
  doi    = {10.5281/zenodo.16924371},
  url    = {https://doi.org/10.5281/zenodo.16924371}
}
```


