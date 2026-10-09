# Decisions Log: FEA Surrogate Model

A record of each design decision and the reason for it. Reword anything so it sounds like you before using it in an interview.

## 1. FEA setup

- **Problem:** rectangular plate with a central circular hole under uniform tension. It has a known analytical answer, so the FEA can be checked before it is trusted.
- **Quarter model with symmetry:** the plate has two symmetry planes, so I model one quarter. Left edge has no x-displacement, bottom edge has no y-displacement. This gives the same result at a quarter of the cost and also stops rigid-body motion.
- **Plane stress, linear elastic:** the plate is thin, loads stay below yield, and deformations are small. Material is steel-like (E = 210 GPa, nu = 0.30), units mm, N, MPa.
- **Quadratic (P2) triangular elements:** they capture the steep stress gradient near the hole much better than linear elements on the same mesh.
- **Output:** peak von Mises stress, smoothed onto the nodes with an L2 projection. Raw quadrature-point values underestimate the peak, which sits on the hole boundary.
- **Tools:** scikit-fem (solver) and gmsh (mesh), because both install with pip on Windows and are fully scriptable for hundreds of runs.

## 2. FEA verification

- Compared the peak stress with Peterson's finite-width plate formula at three hole sizes: all within 0.6%.
- Mesh convergence study (elements along the hole arc: 4, 8, 16, 32, 64 gave peak stress 292.6, 307.2, 313.1, 314.6, 315.1 MPa).
- **Decision: 32 elements along the hole arc.** It is the first level where the change drops below 1% (+0.5%), at about 1500 elements per run.
- The converged FEA sits about 0.3 to 0.5% above Peterson's curve fit. Peterson's fit is only accurate to about 1%, so the correct claim is agreement within about 1%, not an exact match.
- A 50-run test batch before the full dataset: all runs succeeded and error vs. theory stayed between 0.02% and 0.60%.

## 3. Dataset

- **Parameters and ranges:** plate width W 60 to 200 mm, hole diameter over width 0.05 to 0.50, aspect ratio H/W 2 to 3, applied stress 50 to 200 MPa.
- **Why these ranges:** d/W is capped at 0.50 so Peterson's fit stays valid for checking; H/W of at least 2 keeps the loaded ends far from the hole (Saint-Venant's principle); the load range is arbitrary because stress scales linearly with it.
- **Latin hypercube sampling, seed 42:** covers each parameter range evenly with few samples, and the seed makes the dataset reproducible with one command.
- **1000 runs generated** at 32 elements along the arc. Failed runs are logged with an error message instead of being dropped silently.

## 4. Target and split

- **Target is `kt_gross`** (peak stress divided by applied stress), not raw stress. The problem is linear, so doubling the load exactly doubles the stress; the load input carries no information, and Kt isolates the dependence on geometry.
- **Split by geometry:** one row is one simulation, so the same geometry never appears in both train and test (no leakage). Random 70/15/15 split.
- **Extrapolation test:** train on d/W below 0.40, test on d/W of 0.45 and above, with a gap in between. This needs no extra FEA runs and shows how each model behaves outside its training range.

## 5. Models compared and why

| Model | Why it is in the comparison |
|---|---|
| Linear regression | Baseline: shows whether ML is needed at all |
| Cubic polynomial + ridge | Simple smooth model; the physics is a smooth curve, so a low-order polynomial is a strong, cheap baseline |
| Gradient boosting | Standard strong model for tabular data; included to test a tree-based approach |
| Gaussian process (ARD) | Suited to small, smooth datasets, and the per-input length scales show which inputs matter |
| MLP (64, 64, tanh) | Neural network baseline, the starting point for the later full-field models |

Inputs were scaled before every model. The MLP target was also scaled.

## 6. Findings

**Interpolation (random split), mean relative error:** linear 2.6%, polynomial 0.064%, gradient boosting 0.058%, MLP 0.048%, Gaussian process 0.015%.

- Linear regression fails because Kt is a curved function of d/W, which justifies using ML.
- All nonlinear models are within about 0.3% worst case. These errors are below the FEA's own mesh discretization error (0.2 to 0.5%), so further accuracy here would not mean anything.

**Extrapolation (d/W of 0.45 and above), mean relative error:** Gaussian process 0.30%, polynomial 0.66%, MLP 3.5%, linear 9.8%, gradient boosting 10.3%.

- Gradient boosting is excellent in interpolation and fails in extrapolation, because trees predict a constant per region and cannot extend past the training range. A random split alone would have hidden this.
- The MLP degrades because neural networks extrapolate unpredictably.
- The Gaussian process and polynomial hold up because their smooth structure continues past the data.
- R-squared is misleading on the extrapolation set (negative for several models) because the target range is narrow; relative error is the fairer metric.

**Gaussian process length scales:** W hit the upper bound (100,000), which triggers scikit-learn's ConvergenceWarning but is a good sign: the model decided the plate width does not matter. That matches the physics, since only the ratios d/W and H/W matter. d/W (2.9) matters most, H/W (8.75) less.

## 7. Decision: Gaussian process is the best scalar model

It had the lowest error in both tests (0.015% interpolation, 0.30% extrapolation), gives uncertainty estimates, and correctly identified W as irrelevant. Its costs are slower training (about 9 s here) and poor scaling to large datasets, which is fine at 1000 samples.

## 8. Limitations to state upfront

- The measured speedup (about 8,600x) is for batched inference against about 0.175 s per FEA run. A single query has more overhead, and 2D linear FEA is already cheap, so the real payoff grows for 3D, nonlinear or CFD problems.
- The scalar problem is nearly one-dimensional, so high accuracy is expected. The full stress-field prediction is the harder problem.
- Valid only inside the sampled ranges, for this one geometry family, and it inherits any error from the FEA labels.
- It does not replace final FEA verification for safety-critical designs.

## 9. Next steps

1. Look at the saved plots and add them to the README with the results table.
2. Decide between polishing the scalar stage first or moving on to full stress-field prediction (CNN or graph network).
3. Time a single-sample prediction as well as batched, so the speedup claim is complete.

## 10. Stress field prediction decisions

### Grid choice (xi in [0, 0.5], eta in [0, 1.0], 64 x 128)
- **Non-dimensional coordinates:** Physical plate geometries vary in width ($W \in [60, 200]\text{ mm}$) and aspect ratio ($H/W \in [2, 3]$). Resampling onto non-dimensional coordinates $\xi = x/W$ and $\eta = y/W$ maps all quarter plates onto a standardized rectangular spatial domain.
- **Domain extents:** In symmetry quarter coordinates, the plate width runs from $x=0$ to $x=W/2$, corresponding exactly to $\xi \in [0, 0.5]$. For height, since $H/W \ge 2$, the quarter plate has total height $H/(2W) \ge 1.0$. Restricting the grid to $\eta \in [0, 1.0]$ captures the region surrounding the hole and up to one plate width away. By Saint-Venant's principle, stress perturbations decay rapidly away from the hole and become uniform far before $\eta = 1.0$, so this window captures 100% of the non-trivial stress redistribution.
- **Grid resolution (64 x 128):** Using $n_\xi = 64$ points and $n_\eta = 128$ points produces uniform, square pixels ($\Delta \xi = 0.5/63 \approx 0.007937$, $\Delta \eta = 1.0/127 \approx 0.007874$). Furthermore, powers-of-2 spatial dimensions ($64 \times 128$) cleanly interface with standard transposed-convolution upsampling stages in deep learning architectures.
- **Nearest-node fallback:** Valid plate pixels near the curved hole or outer boundaries that fall slightly outside the linear mesh due to boundary discretization rounding are assigned values from the nearest mesh node. Across the 1,000 dataset samples in `data/fields.npz`, exactly zero pixels required fallback after boundary projection.

### Hole-mask handling
- **Physical boundary:** The interior of the hole ($\xi^2 + \eta^2 < (d/(2W))^2$) contains void, not material. Evaluating the finite element solution inside the void is physically meaningless.
- **Inpainting for PCA/POD:** Setting hole pixels to zero or a constant creates an artificial, geometry-dependent jump discontinuity at the circular perimeter. In SVD/PCA, these artificial step edges dominate the spatial covariance matrix and waste principal modes reconstructing the geometric edge instead of physical stress gradients. To solve this, hole pixels are filled prior to PCA using nearest-neighbor Euclidean distance transform inpainting (`scipy.ndimage.distance_transform_edt`). After POD projection and GP prediction, the exact analytical circular mask is reapplied, zeroing the void.
- **Masked loss for PyTorch NN:** In the neural network decoder, the loss is formulated as a masked MSE:
  $$\mathcal{L} = \frac{\sum_{\text{pixels}} M_{i,j} \cdot (\hat{y}_{i,j} - y_{i,j})^2}{\sum_{\text{pixels}} M_{i,j}}$$
  where $M_{i,j} \in \{0, 1\}$ is the boolean plate mask. The network's backpropagation gradients are strictly computed from valid plate pixels; the network is never penalized or trained on void pixels. The analytical mask is also reapplied post-inference.

### Number of PCA modes
- In `models/field_pod.py`, the validation set reconstruction error was evaluated across mode counts from 1 to 20 on the 70/15/15 split.
- Reconstruction error decreased monotonically with mode count: mode 1 had ~16.5% error, mode 5 had ~3.2% error, mode 10 had ~1.1% error, mode 15 had ~0.4% error, and mode 20 achieved 0.17% validation reconstruction error.
- Because no mode count below 20 dropped below the strict 0.10% threshold, $K = 20$ modes was selected as specified by the decision rule. 20 modes retain over 99.9% of the spatial variance while requiring only 20 lightweight Gaussian process models to fit.

### Why each model was included
- **Proper Orthogonal Decomposition + Gaussian Process (POD + GP):** The classical reduced-order modeling (ROM) benchmark in computational mechanics. It projects high-dimensional fields ($8,192$ pixels) onto an optimal low-dimensional linear subspace ($20$ orthogonal spatial modes via SVD), then fits an independent non-linear Gaussian Process with ARD RBF kernel to map geometry inputs $(d/W, H/W)$ to each mode coefficient. It is interpretable, data-efficient, and inherently smooth.
- **Convolutional / Transposed-Convolutional Neural Network (CNN-Deconv NN):** The modern deep learning surrogate benchmark. It expands $(d/W, H/W)$ through fully-connected layers into a coarse spatial feature map ($256 \times 4 \times 8$), then applies 4 transposed-convolution layers with ReLU activations and bilinear interpolation to decode the spatial stress field directly. It tests whether end-to-end gradient descent can discover localized stress gradient features without explicit orthogonal subspace decomposition.
- **Comparison with scalar baselines:** Contrasts field prediction against scalar stress concentration ($K_t$) models: scalar models achieve $<0.05\%$ error and run in $\sim 5\ \mu\text{s}$ ($28,000\times$ faster than FEA), while full-field models predict all $8,192$ spatial locations with $5.3\text{--}8.8\%$ error and run in $\sim 0.8\text{--}1.8\text{ ms}$ ($100\times\text{--}200\times$ faster than FEA).

