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

## Results: Scalar Stress Concentration (Kt)

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
- **Batched inference speedup for scalar models:** Batched inference is roughly 8,600x to 32,000x faster than FEA (e.g., Gaussian Process batched latency of 20.3 $\mu\text{s/sample}$ vs. ~175 ms FEA runtime from `results/baseline_results.csv`). Single-sample and batched latencies for the field models are listed separately below.

![Predicted vs true](results/baselines_pred_vs_true.png)
![Extrapolation error](results/baselines_extrapolation.png)

## Stress field prediction

Moving beyond a single scalar stress concentration factor ($K_t$), this stage predicts the continuous 2D normalized von Mises stress field ($\sigma_{\text{vm}} / \sigma_{\text{applied}}$) directly from geometry $(d/W, H/W)$.

### Body-fitted polar grid (primary representation)

#### Grid definition
Rather than imposing an arbitrary Cartesian grid over a changing plate geometry, the stress field is sampled on a body-fitted polar grid focused around the circular notch (`data_gen/gen_fields_polar.py`):
- **Radial coordinate:** $\rho = r + s(W/2 - r)$ with normalized radial parameter $s \in [0, 1]$ (64 points)
- **Angular coordinate:** $\theta \in [0, \pi/2]$ (64 points)
- **Physical coordinates:** $x = \rho \cos\theta, \quad y = \rho \sin\theta$
- **Resolution:** $64 \times 64$ ($4,096$ points), `float32`

**Key geometric properties**:
- **No hole mask needed:** Because the radial coordinate starts at the notch boundary $\rho = r$ ($s=0$) and extends to the ligament edge $\rho = W/2$ ($s=1$), all $4,096$ points lie strictly on solid plate material. No pixels fall inside the void, so no artificial inpainting or boundary masking is needed.
- **Notch root alignment:** The notch root ($x=r, y=0$), where peak stress occurs, is mapped to coordinate $(s=0, \theta=0)$ across every geometry. At this location, the sampled stress matches the FEA peak stress $K_{t,\text{gross}}$ with a mean relative difference of $0.00035\%$ (max $0.022\%$).
- **Domain coverage:** The polar grid covers **only the region within $W/2$ of the hole centre**, rather than the entire plate quarter.

#### Results: Polar Grid Surrogates & Benchmarks
We evaluate Proper Orthogonal Decomposition + Gaussian Process (POD+GP, 6 modes) and a CNN-Deconv Neural Network against two trivial baselines: a Uniform field equal to 1.0, and the pixelwise Mean Training Field.

All numbers below come directly from `results/field_diagnostics_polar.csv`:

| Split | Model / Baseline | Full Rel. $L_2$ Error (Mean) | Full Rel. $L_2$ Error (Max) | Perturbation Rel. $L_2$ Error (Mean) | Perturbation Rel. $L_2$ Error (Max) | Peak Error vs. Grid (Mean) | Peak Error vs. Grid (Max) | Source File |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Interp** | Uniform Field (1.0) | 33.829% | 50.320% | 100.000% | 100.000% | 70.599% | 76.948% | `results/field_diagnostics_polar.csv` |
| **Interp** | Mean Training Field | 15.788% | 30.680% | 53.705% | 182.574% | 9.656% | 21.023% | `results/field_diagnostics_polar.csv` |
| **Interp** | **POD + GP (6 modes)** | **0.281%** | **0.717%** | **0.924%** | **3.738%** | **0.068%** | **0.282%** | `results/field_diagnostics_polar.csv` |
| **Interp** | **CNN-Deconv NN** | 0.613% | 1.405% | 2.093% | 8.361% | 0.546% | 1.590% | `results/field_diagnostics_polar.csv` |
| **Extrap** | Uniform Field (1.0) | 48.409% | 50.320% | 100.000% | 100.000% | 75.988% | 77.018% | `results/field_diagnostics_polar.csv` |
| **Extrap** | Mean Training Field | 32.868% | 35.744% | 67.853% | 71.159% | 23.547% | 26.828% | `results/field_diagnostics_polar.csv` |
| **Extrap** | **POD + GP (6 modes)** | **2.534%** | **4.471%** | **5.198%** | **8.927%** | **1.904%** | **2.959%** | `results/field_diagnostics_polar.csv` |
| **Extrap** | **CNN-Deconv NN** | 9.307% | 11.773% | 19.176% | 23.501% | 4.067% | 5.466% | `results/field_diagnostics_polar.csv` |

*(Note: In `results/field_diagnostics_polar.csv`, peak error evaluated against scalar $K_{t,\text{gross}}$ from the CSV yields identical values: Interp POD+GP mean 0.068%, max 0.282%; NN mean 0.546%, max 1.590%; Extrap POD+GP mean 1.904%, max 2.959%; NN mean 4.067%, max 5.466%.)*

#### Model Timings & Latency Comparison
Single-sample and batched inference timings are listed separately below:
- **POD + GP (6 modes)**:
  - Single-sample latency: **1,973.4 $\mu$s (1.97 ms)** (interpolation), **3,010.3 $\mu$s (3.01 ms)** (extrapolation) (`results/pod_results_polar.csv`).
  - Batched latency: **357.8 $\mu$s/sample (0.36 ms)** (interpolation), **212.2 $\mu$s/sample (0.21 ms)** (extrapolation) (`results/pod_results_polar.csv`).
- **CNN-Deconv NN (1,774,945 trainable parameters)**:
  - Single-sample latency: **2,961.5 $\mu$s (2.96 ms)** (interpolation), **3,309.0 $\mu$s (3.31 ms)** (extrapolation) (`results/nn_results_polar.csv`).
  - Batched latency: **451.3 $\mu$s/sample (0.45 ms)** (interpolation), **470.1 $\mu$s/sample (0.47 ms)** (extrapolation) (`results/nn_results_polar.csv`).
- **Reference FEA runtime:** 174.98 ms ($174,976.2\ \mu\text{s}$) per simulation ($51.80\text{ ms}$ meshing + $123.18\text{ ms}$ solving from `data/plate_hole.csv` / `results/field_comparison.csv`).

**Model comparison:** POD+GP beat the neural network across every error metric on the polar grid. It achieved a mean relative $L_2$ error of $0.281\%$ vs. $0.613\%$ on interpolation and $2.534\%$ vs. $9.307\%$ on extrapolation, while requiring only 6 scalar GP fits and running faster in batched inference ($357.8\ \mu\text{s}$ vs. $451.3\ \mu\text{s}$).

---

### What failed and why: The Cartesian grid

Earlier attempts resampled the finite element solution onto a fixed Cartesian grid: $\xi = x/W \in [0, 0.5]$ ($n_\xi = 64$) and $\eta = y/W \in [0, 1.0]$ ($n_\eta = 128$), using an analytic circular mask ($\xi^2 + \eta^2 \ge (d/(2W))^2$) to zero the hole interior.

All numbers below come directly from `results/field_diagnostics.csv`:

| Split | Model / Baseline | Mean Rel. $L_2$ Error (%) | Max Rel. $L_2$ Error (%) | Perturbation Rel. $L_2$ Error (Mean) | Near-Hole Rel. $L_2$ Error (Mean) | Peak Error (Mean) | Source File |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **Interp** | Uniform Field (1.0) | 20.144% | 38.030% | 100.000% | 42.049% | 68.413% | `results/field_diagnostics.csv` |
| **Interp** | Mean Training Field | 14.517% | 26.026% | 102.537% | 22.693% | 46.734% | `results/field_diagnostics.csv` |
| **Interp** | POD + GP (20 modes) | 5.288% | 10.369% | 41.066% | 15.482% | 19.832% | `results/field_diagnostics.csv` |
| **Interp** | CNN-Deconv NN | 8.780% | 11.649% | 59.248% | 19.561% | 39.565% | `results/field_diagnostics.csv` |
| **Extrap** | Uniform Field (1.0) | 35.893% | 38.051% | 100.000% | 46.924% | 75.131% | `results/field_diagnostics.csv` |
| **Extrap** | Mean Training Field | 30.092% | 32.630% | 83.793% | 39.827% | 67.829% | `results/field_diagnostics.csv` |
| **Extrap** | POD + GP (20 modes) | 28.168% | 30.856% | 78.415% | 37.414% | 66.230% | `results/field_diagnostics.csv` |
| **Extrap** | CNN-Deconv NN | 33.864% | 40.988% | 94.165% | 32.115% | 31.441% | `results/field_diagnostics.csv` |

*(Cartesian latencies: POD+GP single-sample 7,916.1 $\mu$s, batched 839.4 $\mu$s; CNN NN single-sample 6,587.1 $\mu$s, batched 1,781.7 $\mu$s from `results/pod_results.csv` and `results/nn_results.csv`.)*

#### Root causes of failure on the Cartesian grid:
1. **The hole boundary moves across the grid:** As $d/W$ varies from $0.05$ to $0.50$, the circular hole boundary shifts across pixel columns and rows. The spatial discontinuity is not fixed in coordinate space.
2. **Small holes span only 3 to 4 pixels:** At $d/W = 0.057$, the hole radius is only $r/W \approx 0.0286$, spanning roughly $3.6$ pixels. The steep stress drop (from $\approx 3.0$ at the hole rim to $\approx 1.0$ in the far field) is squeezed into 2–3 pixels, causing severe aliasing and blurring.
3. **Hole filling creates artificial edges:** Because PCA requires complete rectangular matrices, inpainting the hole with nearest-neighbor Euclidean distance transform creates geometric seams that dominate spatial covariance. POD required 20 modes just to reach $0.17\%$ reconstruction error on filled fields, yet still yielded $41.07\%$ mean perturbation error ($274.08\%$ max) on valid pixels.
4. **The worst samples were all small holes:** On the Cartesian interpolation test set, all 10 worst samples by perturbation error were small holes ($d/W = 0.057$ to $0.097$, e.g., ID 994 with $d/W = 0.0571$ had $274.08\%$ perturbation error for POD+GP and $203.43\%$ for NN). On the polar grid, ID 994's perturbation error dropped to **$3.738\%$** (POD) and **$8.361\%$** (NN).

#### Domain & comparability notice:
**The two grids' errors are not directly comparable.** The Cartesian grid ($64 \times 128$) covers the full plate quarter up to $\eta = 1.0$, where most pixels are far-field plate material under nearly uniform stress ($\approx 1.0$). The polar grid ($64 \times 64$) covers **only the region within $W/2$ of the hole centre**, focusing directly on the steep stress concentration gradient.

---

## Project structure

```
data_gen/gen_data.py             FEA data generation and verification
data_gen/gen_fields.py           Resample FEA stress fields onto fixed Cartesian grid (128x64)
data_gen/gen_fields_polar.py     Resample FEA stress fields onto body-fitted polar grid (64x64)
models/baselines.py              Baseline scalar surrogate models (Kt prediction)
models/field_pod.py              POD + Gaussian Process field surrogate (--data flag for Cartesian/Polar)
models/field_nn.py               PyTorch CNN-Deconv field surrogate (--data flag for Cartesian/Polar)
models/compare_fields.py         Comparison of Cartesian field surrogates, baselines, and FEA
models/field_diagnostics.py      Diagnostic metrics and trivial baselines for Cartesian grid
models/field_diagnostics_polar.py Diagnostic metrics and trivial baselines for Polar grid
results/plot_field_samples.py    Visualize example masked stress fields
data/plate_hole.csv              Generated tabular FEA dataset (1000 runs)
results/                         Result tables (.csv), prediction archives (.npz), and comparison plots
decisions.md                     Design decisions log and engineering rationale
requirements.txt                 Python dependencies
```

## How to run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 1. Scalar FEA and baseline surrogates
python data_gen/gen_data.py --verify
python data_gen/gen_data.py --n 1000 --n_hole 32 --out data/plate_hole.csv
python models/baselines.py

# 2. Regenerate 2D stress field datasets (data/*.npz are gitignored):
python data_gen/gen_fields_polar.py    # Generates data/fields_polar.npz (Polar 64x64 grid)
python data_gen/gen_fields.py          # Generates data/fields.npz (Cartesian 128x64 grid)

# 3. Train and evaluate body-fitted polar surrogates (recommended):
python models/field_pod.py --data data/fields_polar.npz     # POD+GP (6 modes)
python models/field_nn.py --data data/fields_polar.npz      # CNN-Deconv NN (64x64)
python models/field_diagnostics_polar.py                    # Polar diagnostics & worst-sample list

# 4. Train and evaluate Cartesian surrogates (historical benchmark):
python models/field_pod.py --data data/fields.npz           # POD+GP (20 modes)
python models/field_nn.py --data data/fields.npz            # CNN-Deconv NN (128x64)
python models/compare_fields.py
python models/field_diagnostics.py
```

## Limitations

- Valid only within the sampled parameter ranges and for this one geometry family.
- Inherits any error from the FEA used to create the labels.
- The polar grid covers only the ligament within $W/2$ of the notch, not the far-field plate edges.
- 2D linear FEA is already cheap, so surrogate speedup is modest; the payoff grows for 3D, nonlinear or CFD problems.
- Does not replace final FEA verification for safety-critical designs.

## Next steps

- Physics-informed neural operators (FNO, DeepONet) for mesh-free field prediction
- Uncertainty quantification on full fields via ensemble methods
- Topology optimization and inverse design using surrogate gradients
