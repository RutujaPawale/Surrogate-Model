# FEA Surrogate Model: Predicting Stress Concentration from Geometry

Finite element analysis (FEA) is accurate but slow. This project trains machine learning surrogates on FEA results so that the stress concentration of a plate with a hole can be predicted almost instantly, and checks how far those predictions can be trusted, including outside the range the models were trained on.

## Problem

A rectangular plate (width W, height H) with a central circular hole (radius r) is pulled by a uniform tensile stress. The peak von Mises stress occurs at the edge of the hole. The goal is to predict the **stress concentration factor** `Kt = peak stress / applied stress` from the geometry.

## Approach

1. **FEA data generation** (`data_gen/gen_data.py`)
   - Quarter-plate model using symmetry, linear elastic, plane stress, quadratic triangular elements
   - scikit-fem for the solver, gmsh for meshing (refined near the hole)
   - 1000 geometries sampled with Latin hypercube sampling (seed 42)
   - Inputs: plate width W, hole diameter over width d/W (0.05 to 0.50), aspect ratio H/W (2 to 3), applied stress
2. **FEA verification**
   - Peak stress compared with Peterson's analytical formula: within about 0.6%
   - Mesh convergence study; 32 elements along the hole arc chosen (change under 1%)
3. **Surrogate models** (`models/baselines.py`)
   - Linear regression, cubic polynomial with ridge, gradient boosting, Gaussian process (ARD), MLP
   - Target is `Kt` rather than raw stress, because the problem is linear and load only scales the answer
   - Split by geometry (one row per simulation) to avoid leakage

## Results

Interpolation (random 70/15/15 split):

| Model | Mean rel. error | Max rel. error |
|---|---|---|
| Linear | 2.6% | 6.8% |
| Poly3 + Ridge | 0.064% | 0.22% |
| Gradient boosting | 0.058% | 0.27% |
| MLP | 0.048% | 0.33% |
| **Gaussian process** | **0.015%** | **0.18%** |

Extrapolation (trained on d/W below 0.40, tested on d/W of 0.45 and above):

| Model | Mean rel. error | Max rel. error |
|---|---|---|
| **Gaussian process** | **0.30%** | **0.58%** |
| Poly3 + Ridge | 0.66% | 1.17% |
| MLP | 3.5% | 6.3% |
| Linear | 9.8% | 12.6% |
| Gradient boosting | 10.3% | 14.1% |

Key findings:

- Gradient boosting is accurate in interpolation but fails in extrapolation, because tree models cannot extend beyond the training range. A random split alone would have hidden this.
- The Gaussian process gave the plate width W a length scale at its upper bound, meaning it learned that W does not matter. This matches the physics, since only the ratios d/W and H/W control the stress concentration.
- Surrogate error is below the FEA's own mesh discretization error (about 0.2 to 0.5%).
- Batched inference is roughly 8,000x faster than a single FEA run (about 0.175 s per run here).

![Predicted vs true](results/baselines_pred_vs_true.png)
![Extrapolation error](results/baselines_extrapolation.png)

## Project structure

```
data_gen/gen_data.py     FEA data generation and verification
models/baselines.py      Baseline surrogate models and evaluation
data/plate_hole.csv      Generated dataset (1000 FEA runs)
results/                 Result tables and plots
decisions.md             Design decisions and reasoning
requirements.txt         Python dependencies
```

## How to run

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python data_gen/gen_data.py --verify
python data_gen/gen_data.py --n 1000 --n_hole 32 --out data/plate_hole.csv
python models/baselines.py
```

## Limitations

- Valid only within the sampled parameter ranges and for this one geometry family.
- Inherits any error from the FEA used to create the labels.
- 2D linear FEA is already cheap, so the speedup is real but modest; the payoff grows for 3D, nonlinear or CFD problems.
- The scalar target is nearly one-dimensional, so high accuracy is expected. Full stress-field prediction is the harder problem.
- Does not replace final FEA verification for safety-critical designs.

## Next steps

- Predict the full stress field (CNN or graph neural network)
- Uncertainty estimates and an optimization loop on top of the surrogate
- Extend to other geometries (fillets, brackets) and to airflow
