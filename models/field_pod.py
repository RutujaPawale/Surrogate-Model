"""
POD + Gaussian Process Surrogate Model for 2D Stress Field Prediction.

Loads data/fields.npz.
Inputs: d_over_W, H_over_W (dimensionless geometry).
Target: 2D normalized stress field (sigma_vm / sigma) on fixed (128, 64) grid.

Splits:
  (A) Interpolation: random 70/15/15 split by geometry (seed 42).
  (B) Extrapolation: train on d_over_W < 0.40, test on d_over_W >= 0.45.

Method:
  1. For PCA, fill circular hole pixels using nearest valid plate pixel value
     (scipy.ndimage.distance_transform_edt with return_indices).
  2. Flatten fields and fit PCA on the training set only.
  3. Evaluate validation reconstruction relative L2 error for 1..20 modes.
     Choose the smallest number of modes with mean error < 0.1% (or 20 if none).
  4. Train one Gaussian process per mode (StandardScaler on inputs, ARD RBF + WhiteKernel,
     normalize_y=True, seed 42) to predict the mode coefficient from (d_over_W, H_over_W).
  5. Predict test fields, reconstruct 2D fields, and apply the analytic hole mask.
  6. Calculate metrics on valid pixels only:
     - Relative L2 error (mean, max)
     - Mean absolute pixel error
     - Peak stress error (%)
  7. Measure single-sample and batched prediction latency (microseconds).
  8. Save:
     - results/pod_predictions.npz
     - results/pod_results.csv
     - results/pod_best_worst.png
"""
import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.ndimage import distance_transform_edt
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fill_hole_pixels(fields, masks):
    """
    Fills hole interior pixels using the nearest valid plate pixel value
    via Euclidean distance transform indices.
    
    Parameters:
        fields (np.ndarray): (N, ny, nx) array of normalized stress fields
        masks (np.ndarray): (N, ny, nx) boolean mask (True = valid plate, False = hole)
        
    Returns:
        filled_fields (np.ndarray): (N, ny, nx) array with hole pixels filled
    """
    N = len(fields)
    filled = np.zeros_like(fields)
    for i in range(N):
        _, ind = distance_transform_edt(~masks[i], return_indices=True)
        filled[i] = fields[i][ind[0], ind[1]]
    return filled


def fit_pca_and_select_modes(flat_train, flat_val, masks_val, fields_val, max_modes=20, threshold_pct=0.1):
    """
    Fits PCA on training data and determines the optimal number of modes
    based on validation reconstruction error on valid plate pixels.
    """
    pca_full = PCA(n_components=max_modes, random_state=42)
    pca_full.fit(flat_train)

    coeffs_val = pca_full.transform(flat_val)
    mean_errors = []

    print("\n--- Validation Reconstruction Error vs. Number of Modes ---")
    print("Mode | Mean Rel L2 [%] | Max Rel L2 [%]")
    print("-----+-----------------+---------------")

    for k in range(1, max_modes + 1):
        recon_flat = pca_full.mean_ + coeffs_val[:, :k] @ pca_full.components_[:k]
        recon = recon_flat.reshape(len(flat_val), fields_val.shape[1], fields_val.shape[2])

        sample_errors = []
        for j in range(len(flat_val)):
            m = masks_val[j]
            y_t = fields_val[j][m]
            y_p = recon[j][m]
            rel_l2 = 100.0 * np.linalg.norm(y_p - y_t) / np.linalg.norm(y_t)
            sample_errors.append(rel_l2)

        mean_err = float(np.mean(sample_errors))
        max_err = float(np.max(sample_errors))
        mean_errors.append(mean_err)
        print(f" {k:2d}  |     {mean_err:6.3f}%     |    {max_err:6.3f}%")

    # Select smallest mode count under threshold, or max_modes if none
    valid_modes = [k for k, err in enumerate(mean_errors, 1) if err < threshold_pct]
    chosen_modes = valid_modes[0] if valid_modes else max_modes

    print(f"\nSelection: {chosen_modes} modes chosen "
          f"(criterion: smallest mode count with validation error < {threshold_pct}% or {max_modes} if none).")

    return pca_full, chosen_modes


def train_single_gp(k, X_train, y_train_k, seed=42):
    """Fits one Gaussian Process pipeline for mode k."""
    gp = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=C(1.0) * RBF([1.0, 1.0]) + WhiteKernel(1e-5, (1e-10, 1e-1)),
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=seed
        )
    )
    gp.fit(X_train, y_train_k)
    return gp


def train_gp_surrogates(X_train, coeffs_train, n_modes, seed=42):
    """Trains n_modes Gaussian Processes in parallel."""
    print(f"Training {n_modes} Gaussian Process surrogates in parallel across available CPU cores...")
    t0 = time.perf_counter()
    gps = Parallel(n_jobs=-1)(
        delayed(train_single_gp)(k, X_train, coeffs_train[:, k], seed=seed)
        for k in range(n_modes)
    )
    elapsed = time.perf_counter() - t0
    print(f"Trained {n_modes} GPs in {elapsed:.2f} s ({elapsed / n_modes:.2f} s/mode).")
    return gps


def predict_fields(gps, pca, X_test, d_over_W_test, xi, eta, n_modes):
    """
    Predicts 2D normalized stress fields from geometric inputs:
      1. Predicts PCA coefficients via trained GPs.
      2. Reconstructs flattened 2D spatial fields.
      3. Applies analytic circular hole mask (xi^2 + eta^2 >= (d_over_W/2)^2).
    """
    M = len(X_test)
    ny = len(eta)
    nx = len(xi)

    # 1. Predict mode coefficients
    pred_coeffs = np.column_stack([gp.predict(X_test) for gp in gps[:n_modes]])

    # 2. Reconstruct full spatial fields
    pred_flat = pca.mean_ + pred_coeffs @ pca.components_[:n_modes]
    pred_fields = pred_flat.reshape(M, ny, nx).astype(np.float32)

    # 3. Apply analytic hole mask
    XI, ETA = np.meshgrid(xi, eta)
    XI_sq_ETA_sq = XI ** 2 + ETA ** 2

    masks = np.zeros((M, ny, nx), dtype=bool)
    for i in range(M):
        r_sq = (float(d_over_W_test[i]) / 2.0) ** 2
        m = XI_sq_ETA_sq >= r_sq
        masks[i] = m
        pred_fields[i][~m] = 0.0

    return pred_fields, masks


def compute_field_metrics(pred_fields, true_fields, masks):
    """
    Computes field-level evaluation metrics strictly on valid plate pixels:
      - Relative L2 error (%): ||yp - yt||_2 / ||yt||_2 * 100
      - Mean Absolute Error (MAE) per pixel
      - Peak stress error (%): |max(yp) - max(yt)| / max(yt) * 100
    """
    M = len(pred_fields)
    rel_l2_errors = np.zeros(M, dtype=np.float64)
    mae_errors = np.zeros(M, dtype=np.float64)
    peak_errors = np.zeros(M, dtype=np.float64)

    for i in range(M):
        m = masks[i]
        yt_valid = true_fields[i][m]
        yp_valid = pred_fields[i][m]

        # Relative L2 error
        l2_diff = np.linalg.norm(yp_valid - yt_valid)
        l2_true = np.linalg.norm(yt_valid)
        rel_l2_errors[i] = 100.0 * l2_diff / l2_true if l2_true > 0 else 0.0

        # Mean absolute pixel error
        mae_errors[i] = np.mean(np.abs(yp_valid - yt_valid))

        # Peak error
        true_peak = np.max(yt_valid)
        pred_peak = np.max(yp_valid)
        peak_errors[i] = 100.0 * np.abs(pred_peak - true_peak) / true_peak if true_peak > 0 else 0.0

    return {
        "mean_rel_l2_%": float(np.mean(rel_l2_errors)),
        "max_rel_l2_%": float(np.max(rel_l2_errors)),
        "mean_abs_pixel_err": float(np.mean(mae_errors)),
        "mean_peak_err_%": float(np.mean(peak_errors)),
        "max_peak_err_%": float(np.max(peak_errors)),
        "per_sample_rel_l2_%": rel_l2_errors,
        "per_sample_peak_err_%": peak_errors,
    }


def time_inference(gps, pca, X_test, d_over_W_test, xi, eta, n_modes, n_repeats=100):
    """Measures single-sample and batched prediction latency in microseconds."""
    sample_x = X_test[:1]
    sample_d = d_over_W_test[:1]

    # Single-sample latency
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        predict_fields(gps, pca, sample_x, sample_d, xi, eta, n_modes)
    single_us = 1e6 * (time.perf_counter() - t0) / n_repeats

    # Batched latency
    t1 = time.perf_counter()
    predict_fields(gps, pca, X_test, d_over_W_test, xi, eta, n_modes)
    batched_us = 1e6 * (time.perf_counter() - t1) / len(X_test)

    return single_us, batched_us


def plot_best_and_worst_samples(true_fields, pred_fields, masks, d_over_W, H_over_W,
                                rel_l2_list, ids, xi, eta, out_path="results/pod_best_worst.png"):
    """
    Visualizes true field, predicted field, and absolute error map
    for the best and worst test samples.
    """
    best_idx = int(np.argmin(rel_l2_list))
    worst_idx = int(np.argmax(rel_l2_list))

    cases = [("Best Sample", best_idx), ("Worst Sample", worst_idx)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)

    for row, (label, idx) in enumerate(cases):
        m = masks[idx]
        yt = np.where(m, true_fields[idx], np.nan)
        yp = np.where(m, pred_fields[idx], np.nan)
        diff = np.where(m, np.abs(pred_fields[idx] - true_fields[idx]), np.nan)

        d_val = d_over_W[idx]
        hw_val = H_over_W[idx]
        l2_err = rel_l2_list[idx]
        sid = ids[idx]

        # Row titles
        axes[row, 0].set_ylabel(f"{label} (ID {sid})\n$\\eta = y/W$", fontsize=11, fontweight="bold")

        # True Field
        vmax_field = np.nanmax(yt) * 1.02
        vmin_field = 0.5
        im0 = axes[row, 0].imshow(yt, origin="lower", extent=[xi[0], xi[-1], eta[0], eta[-1]],
                                 cmap="inferno", vmin=vmin_field, vmax=vmax_field, aspect="equal")
        axes[row, 0].set_title(f"True Field ($d/W={d_val:.2f}, H/W={hw_val:.2f}$)\nPeak = {np.nanmax(yt):.2f}", fontsize=10)
        axes[row, 0].set_xlabel(r"$\xi = x/W$")
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.04, pad=0.04)

        # Predicted Field
        im1 = axes[row, 1].imshow(yp, origin="lower", extent=[xi[0], xi[-1], eta[0], eta[-1]],
                                 cmap="inferno", vmin=vmin_field, vmax=vmax_field, aspect="equal")
        axes[row, 1].set_title(f"POD-GP Predicted Field\nPeak = {np.nanmax(yp):.2f}", fontsize=10)
        axes[row, 1].set_xlabel(r"$\xi = x/W$")
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.04, pad=0.04)

        # Absolute Error Map
        vmax_err = max(0.05, float(np.nanmax(diff) * 1.05))
        im2 = axes[row, 2].imshow(diff, origin="lower", extent=[xi[0], xi[-1], eta[0], eta[-1]],
                                 cmap="viridis", vmin=0.0, vmax=vmax_err, aspect="equal")
        axes[row, 2].set_title(f"Absolute Error Map\nRel $L_2$ Error = {l2_err:.3f}%", fontsize=10)
        axes[row, 2].set_xlabel(r"$\xi = x/W$")
        cbar2 = fig.colorbar(im2, ax=axes[row, 2], fraction=0.04, pad=0.04)
        cbar2.set_label("|Predicted - True|", fontsize=9)

    fig.suptitle("POD + GP 2D Stress Field Reconstruction: Best vs. Worst Test Samples", fontsize=13, fontweight="bold")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved diagnostic best/worst figure to {out_path}.")


def run_pod_pipeline(npz_path="data/fields.npz", seed=42):
    """Executes the complete POD-GP training and evaluation pipeline for both splits."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{npz_path} not found. Run data_gen/gen_fields.py first.")

    print(f"Loading {npz_path}...")
    data = np.load(npz_path)
    fields = data["fields"]       # (N, ny, nx)
    masks = data["masks"]         # (N, ny, nx)
    d_over_W = data["d_over_W"]   # (N,)
    H_over_W = data["H_over_W"]   # (N,)
    kt_gross = data["kt_gross"]   # (N,)
    ids = data["ids"]             # (N,)
    xi = data["xi"]               # (nx,)
    eta = data["eta"]             # (ny,)
    N = len(fields)

    print(f"Loaded {N} fields of resolution {len(eta)}x{len(xi)}.")

    # 1. Fill holes for PCA
    print("Filling circular hole interiors for PCA decomposition...")
    t0 = time.perf_counter()
    filled_fields = fill_hole_pixels(fields, masks)
    print(f"Holes filled in {time.perf_counter() - t0:.2f} s.")
    flat_all = filled_fields.reshape(N, -1)

    # Inputs: d_over_W, H_over_W only
    X_all = np.column_stack([d_over_W, H_over_W])

    # =========================================================================
    # (A) INTERPOLATION SPLIT (70/15/15, seed 42)
    # =========================================================================
    idx = np.arange(N)
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.30, random_state=seed)
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, random_state=seed)
    print(f"\n=======================================================")
    print(f"SPLIT (A): Interpolation (train={len(idx_tr)}, val={len(idx_va)}, test={len(idx_te)})")
    print(f"=======================================================")

    # Mode selection on validation set
    pca_A, chosen_modes = fit_pca_and_select_modes(
        flat_train=flat_all[idx_tr],
        flat_val=flat_all[idx_va],
        masks_val=masks[idx_va],
        fields_val=fields[idx_va],
        max_modes=20,
        threshold_pct=0.1
    )

    # Train GP surrogates on chosen modes
    coeffs_tr_A = pca_A.transform(flat_all[idx_tr])
    gps_A = train_gp_surrogates(X_all[idx_tr], coeffs_tr_A, n_modes=chosen_modes, seed=seed)

    # Predict test fields
    pred_fields_te_A, pred_masks_te_A = predict_fields(
        gps_A, pca_A, X_all[idx_te], d_over_W[idx_te], xi, eta, n_modes=chosen_modes
    )

    # Metrics on test set
    metrics_A = compute_field_metrics(pred_fields_te_A, fields[idx_te], masks[idx_te])
    single_us_A, batch_us_A = time_inference(
        gps_A, pca_A, X_all[idx_te], d_over_W[idx_te], xi, eta, n_modes=chosen_modes
    )

    print("\n--- Interpolation Test Set Results ---")
    print(f"Mean Relative L2 Error: {metrics_A['mean_rel_l2_%']:.3f}%")
    print(f"Max Relative L2 Error:  {metrics_A['max_rel_l2_%']:.3f}%")
    print(f"Mean Abs Pixel Error:   {metrics_A['mean_abs_pixel_err']:.4f}")
    print(f"Mean Peak Error:        {metrics_A['mean_peak_err_%']:.3f}%")
    print(f"Max Peak Error:         {metrics_A['max_peak_err_%']:.3f}%")
    print(f"Single-sample latency:  {single_us_A:,.1f} us ({single_us_A / 1000:.3f} ms)")
    print(f"Batched latency:        {batch_us_A:,.1f} us/sample ({batch_us_A / 1000:.4f} ms/sample)")

    # Diagnostic best/worst figure on interpolation test set
    plot_best_and_worst_samples(
        true_fields=fields[idx_te],
        pred_fields=pred_fields_te_A,
        masks=masks[idx_te],
        d_over_W=d_over_W[idx_te],
        H_over_W=H_over_W[idx_te],
        rel_l2_list=metrics_A["per_sample_rel_l2_%"],
        ids=ids[idx_te],
        xi=xi,
        eta=eta,
        out_path="results/pod_best_worst.png"
    )

    # =========================================================================
    # (B) EXTRAPOLATION SPLIT (train d_over_W < 0.40, test d_over_W >= 0.45)
    # =========================================================================
    idx_ext_tr = np.where(d_over_W < 0.40)[0]
    idx_ext_te = np.where(d_over_W >= 0.45)[0]
    print(f"\n=======================================================")
    print(f"SPLIT (B): Extrapolation (train={len(idx_ext_tr)} [d/W < 0.40], test={len(idx_ext_te)} [d/W >= 0.45])")
    print(f"=======================================================")

    pca_B = PCA(n_components=chosen_modes, random_state=seed)
    pca_B.fit(flat_all[idx_ext_tr])
    coeffs_tr_B = pca_B.transform(flat_all[idx_ext_tr])

    gps_B = train_gp_surrogates(X_all[idx_ext_tr], coeffs_tr_B, n_modes=chosen_modes, seed=seed)

    pred_fields_te_B, pred_masks_te_B = predict_fields(
        gps_B, pca_B, X_all[idx_ext_te], d_over_W[idx_ext_te], xi, eta, n_modes=chosen_modes
    )

    metrics_B = compute_field_metrics(pred_fields_te_B, fields[idx_ext_te], masks[idx_ext_te])
    single_us_B, batch_us_B = time_inference(
        gps_B, pca_B, X_all[idx_ext_te], d_over_W[idx_ext_te], xi, eta, n_modes=chosen_modes
    )

    print("\n--- Extrapolation Test Set Results ---")
    print(f"Mean Relative L2 Error: {metrics_B['mean_rel_l2_%']:.3f}%")
    print(f"Max Relative L2 Error:  {metrics_B['max_rel_l2_%']:.3f}%")
    print(f"Mean Abs Pixel Error:   {metrics_B['mean_abs_pixel_err']:.4f}")
    print(f"Mean Peak Error:        {metrics_B['mean_peak_err_%']:.3f}%")
    print(f"Max Peak Error:         {metrics_B['max_peak_err_%']:.3f}%")
    print(f"Single-sample latency:  {single_us_B:,.1f} us ({single_us_B / 1000:.3f} ms)")
    print(f"Batched latency:        {batch_us_B:,.1f} us/sample ({batch_us_B / 1000:.4f} ms/sample)")

    # =========================================================================
    # SAVE PREDICTIONS AND METRICS
    # =========================================================================
    # 1. Predictions .npz
    out_preds_path = "results/pod_predictions.npz"
    os.makedirs(os.path.dirname(out_preds_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_preds_path,
        interp_ids=ids[idx_te],
        interp_pred_fields=pred_fields_te_A,
        interp_true_fields=fields[idx_te],
        interp_masks=masks[idx_te],
        interp_d_over_W=d_over_W[idx_te],
        interp_H_over_W=H_over_W[idx_te],
        interp_kt_gross=kt_gross[idx_te],
        extrap_ids=ids[idx_ext_te],
        extrap_pred_fields=pred_fields_te_B,
        extrap_true_fields=fields[idx_ext_te],
        extrap_masks=masks[idx_ext_te],
        extrap_d_over_W=d_over_W[idx_ext_te],
        extrap_H_over_W=H_over_W[idx_ext_te],
        extrap_kt_gross=kt_gross[idx_ext_te],
        xi=xi,
        eta=eta
    )
    print(f"\nSaved predictions archive to {out_preds_path} ({os.path.getsize(out_preds_path)/(1024*1024):.2f} MB).")

    # 2. Results CSV
    out_csv_path = "results/pod_results.csv"
    res_df = pd.DataFrame([
        {
            "split": "interp",
            "n_modes": chosen_modes,
            "mean_rel_l2_%": metrics_A["mean_rel_l2_%"],
            "max_rel_l2_%": metrics_A["max_rel_l2_%"],
            "mean_abs_pixel_err": metrics_A["mean_abs_pixel_err"],
            "mean_peak_err_%": metrics_A["mean_peak_err_%"],
            "max_peak_err_%": metrics_A["max_peak_err_%"],
            "single_sample_us": single_us_A,
            "batched_us_per_sample": batch_us_A,
        },
        {
            "split": "extrap",
            "n_modes": chosen_modes,
            "mean_rel_l2_%": metrics_B["mean_rel_l2_%"],
            "max_rel_l2_%": metrics_B["max_rel_l2_%"],
            "mean_abs_pixel_err": metrics_B["mean_abs_pixel_err"],
            "mean_peak_err_%": metrics_B["mean_peak_err_%"],
            "max_peak_err_%": metrics_B["max_peak_err_%"],
            "single_sample_us": single_us_B,
            "batched_us_per_sample": batch_us_B,
        }
    ])
    res_df.to_csv(out_csv_path, index=False)
    print(f"Saved metrics table to {out_csv_path}.")
    print("\nSummary Results Table:")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POD + GP 2D stress field surrogate.")
    parser.add_argument("--npz", default="data/fields.npz", help="Path to fields.npz")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_pod_pipeline(npz_path=args.npz, seed=args.seed)
