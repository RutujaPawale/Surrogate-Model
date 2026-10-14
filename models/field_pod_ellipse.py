"""
POD + Gaussian Process Surrogate Model for 2D Elliptical Stress Field Prediction.

Loads data/fields_ellipse.npz:
  - Inputs: d_over_W, ar, H_over_W
  - Target: 2D normalized stress field (sigma_vm / sigma) on (64, 64) body-fitted grid.
    All pixels represent solid material; no hole mask needed.

Evaluations (all with seed 42):
  A)  Interpolation: random 70/15/15 split by geometry sample.
  B1) Extrapolation in hole size: train d_over_W < 0.40, test d_over_W >= 0.45.
  B2) Extrapolation to sharper notches: train ar >= 0.8, test ar < 0.7.

Method:
  - Flatten fields and fit PCA on training fields only.
  - Mode selection: choose the smallest number of modes whose mean validation
    reconstruction error is under 0.1% (capped at 30).
    For B1, validation set is the top 15% of training samples by d_over_W.
    For B2, validation set is the lowest 15% of training samples by ar.
  - Train one Gaussian process per mode (StandardScaler on inputs, ARD RBF + WhiteKernel,
    normalize_y=True, n_restarts_optimizer=2, seed 42) predicting the PCA coefficient
    from [d_over_W, ar, H_over_W].
  - Evaluate full-field relative L2 error, perturbation relative L2 error (pred-1 vs true-1),
    and notch root peak error (pixel s=0, phi=0 vs true value).
  - Compare against trivial baselines: uniform field of 1.0 and mean training field.

Outputs:
  - results/field_ellipse_results.csv
  - results/field_ellipse_predictions.npz
  - results/field_ellipse_best_worst.png
"""
import argparse
import os
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_pca_and_select_modes(flat_train, flat_val, max_modes=30, threshold_pct=0.1, print_table=False):
    """
    Fits PCA on training data (or sub-training data) and determines the optimal
    number of modes based on mean validation reconstruction relative L2 error.
    """
    pca_full = PCA(n_components=max_modes, random_state=42)
    pca_full.fit(flat_train)

    coeffs_val = pca_full.transform(flat_val)
    mean_errors = []
    max_errors = []

    if print_table:
        print("\n--- Validation Reconstruction Error vs. Number of Modes (1 to 30) ---")
        print(" Mode | Mean Rel L2 [%] | Max Rel L2 [%]")
        print("------+-----------------+---------------")

    for k in range(1, max_modes + 1):
        recon = pca_full.mean_ + coeffs_val[:, :k] @ pca_full.components_[:k]
        # Relative L2 error per validation sample
        sample_errors = [
            100.0 * np.linalg.norm(recon[j] - flat_val[j]) / np.linalg.norm(flat_val[j])
            for j in range(len(flat_val))
        ]
        m_err = float(np.mean(sample_errors))
        mx_err = float(np.max(sample_errors))
        mean_errors.append(m_err)
        max_errors.append(mx_err)

        if print_table:
            print(f"  {k:2d}  |     {m_err:7.4f}%    |    {mx_err:7.4f}%")

    # Select smallest mode count under threshold, or max_modes if none
    valid_modes = [k for k, err in enumerate(mean_errors, 1) if err < threshold_pct]
    chosen_modes = valid_modes[0] if valid_modes else max_modes

    print(f"\nMode selection: {chosen_modes} modes chosen "
          f"(criterion: smallest mode count with mean validation error < {threshold_pct}% or cap {max_modes}).")

    return pca_full, chosen_modes, mean_errors, max_errors


def train_single_gp(k, X_train, y_train_k, seed=42):
    """Fits one Gaussian Process pipeline for mode k."""
    gp = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=C(1.0) * RBF([1.0, 1.0, 1.0]) + WhiteKernel(1e-5, (1e-10, 1e-1)),
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=seed,
        ),
    )
    gp.fit(X_train, y_train_k)
    return gp


def train_gp_surrogates(X_train, coeffs_train, n_modes, seed=42):
    """Trains n_modes Gaussian Processes in parallel across CPU cores."""
    print(f"Training {n_modes} Gaussian Process surrogates in parallel...")
    t0 = time.perf_counter()
    gps = Parallel(n_jobs=-1)(
        delayed(train_single_gp)(k, X_train, coeffs_train[:, k], seed=seed)
        for k in range(n_modes)
    )
    elapsed = time.perf_counter() - t0
    print(f"Trained {n_modes} GPs in {elapsed:.2f} s ({elapsed / n_modes:.2f} s/mode).")
    return gps


def predict_fields(gps, pca, X_test, n_modes, shape=(64, 64)):
    """
    Predicts full 2D spatial stress fields:
      1. Predicts PCA coefficients using trained GPs.
      2. Inverts PCA projection to reconstruct full spatial fields.
    """
    M = len(X_test)
    pred_coeffs = np.column_stack([gp.predict(X_test) for gp in gps[:n_modes]])
    pred_flat = pca.mean_ + pred_coeffs @ pca.components_[:n_modes]
    pred_fields = pred_flat.reshape(M, shape[0], shape[1]).astype(np.float32)
    return pred_fields


def compute_metrics(pred_fields, true_fields):
    """
    Computes per-sample and summary metrics:
      1. Full-field relative L2 error (%)
      2. Perturbation relative L2 error (%): ||(pred - 1) - (true - 1)|| / ||true - 1||
      3. Peak error (%): at notch root (s=0, phi=0), i.e. pixel [0, 0]
    """
    M = len(pred_fields)
    full_l2 = np.zeros(M, dtype=np.float64)
    pert_l2 = np.zeros(M, dtype=np.float64)
    peak_err = np.zeros(M, dtype=np.float64)

    for i in range(M):
        yp = pred_fields[i]
        yt = true_fields[i]

        diff = yp - yt
        diff_norm = np.linalg.norm(diff)
        true_norm = np.linalg.norm(yt)
        pert_true_norm = np.linalg.norm(yt - 1.0)

        # 1. Full-field relative L2
        full_l2[i] = 100.0 * diff_norm / true_norm if true_norm > 0 else 0.0

        # 2. Perturbation relative L2
        pert_l2[i] = 100.0 * diff_norm / pert_true_norm if pert_true_norm > 0 else 0.0

        # 3. Peak error at notch root (s=0, phi=0) -> pixel [0, 0]
        yp_root = yp[0, 0]
        yt_root = yt[0, 0]
        peak_err[i] = 100.0 * abs(yp_root - yt_root) / yt_root if yt_root > 0 else 0.0

    return {
        "full_rel_l2_mean_%": float(np.mean(full_l2)),
        "full_rel_l2_max_%": float(np.max(full_l2)),
        "pert_rel_l2_mean_%": float(np.mean(pert_l2)),
        "pert_rel_l2_max_%": float(np.max(pert_l2)),
        "peak_err_mean_%": float(np.mean(peak_err)),
        "peak_err_max_%": float(np.max(peak_err)),
        "per_sample_full_l2": full_l2,
        "per_sample_pert_l2": pert_l2,
        "per_sample_peak_err": peak_err,
    }


def compute_baseline_metrics(true_fields, baseline_field):
    """
    Computes metrics for a baseline field (e.g., uniform 1.0 or training mean field).
    """
    M = len(true_fields)
    pred_fields = np.broadcast_to(baseline_field, true_fields.shape)
    return compute_metrics(pred_fields, true_fields)


def time_inference(gps, pca, X_test, n_modes, n_repeats=100, shape=(64, 64)):
    """Measures single-sample and batched inference latency in microseconds."""
    sample_x = X_test[:1]

    # Single-sample latency
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        predict_fields(gps, pca, sample_x, n_modes, shape=shape)
    single_us = 1e6 * (time.perf_counter() - t0) / n_repeats

    # Batched latency
    t1 = time.perf_counter()
    predict_fields(gps, pca, X_test, n_modes, shape=shape)
    batched_us = 1e6 * (time.perf_counter() - t1) / len(X_test)

    return single_us, batched_us


def plot_best_and_worst_samples(true_fields, pred_fields, d_over_W, ar, H_over_W,
                                pert_l2_list, ids, s, phi, out_path="results/field_ellipse_best_worst.png"):
    """
    Visualizes true field, predicted field, and absolute error map
    for the best and worst test samples of Split A in the normalized (s, phi) grid.
    """
    best_idx = int(np.argmin(pert_l2_list))
    worst_idx = int(np.argmax(pert_l2_list))

    cases = [("Best Test Sample", best_idx), ("Worst Test Sample", worst_idx)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    extent = [phi[0], phi[-1], s[0], s[-1]]

    for row, (label, idx) in enumerate(cases):
        yt = true_fields[idx]
        yp = pred_fields[idx]
        diff = np.abs(yp - yt)

        dw_val = d_over_W[idx]
        ar_val = ar[idx]
        hw_val = H_over_W[idx]
        p_err = pert_l2_list[idx]
        sid = ids[idx]

        vmax_field = max(np.max(yt), np.max(yp)) * 1.02
        vmin_field = 0.8

        # 1. True Field
        im0 = axes[row, 0].imshow(yt, origin="lower", extent=extent,
                                  cmap="inferno", vmin=vmin_field, vmax=vmax_field, aspect="auto")
        axes[row, 0].set_title(
            f"True Field ({label}: ID {sid})\n$d/W={dw_val:.2f}, ar={ar_val:.2f}, H/W={hw_val:.2f}$ | Peak: {yt[0,0]:.2f}",
            fontsize=10,
        )
        axes[row, 0].set_xlabel(r"Angle $\phi$ [rad]")
        axes[row, 0].set_ylabel(r"Radial coordinate $s$")
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        # 2. Predicted Field
        im1 = axes[row, 1].imshow(yp, origin="lower", extent=extent,
                                  cmap="inferno", vmin=vmin_field, vmax=vmax_field, aspect="auto")
        axes[row, 1].set_title(
            f"POD+GP Predicted Field\nPeak: {yp[0,0]:.2f}",
            fontsize=10,
        )
        axes[row, 1].set_xlabel(r"Angle $\phi$ [rad]")
        axes[row, 1].set_ylabel(r"Radial coordinate $s$")
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # 3. Absolute Error Map
        vmax_err = max(0.02, float(np.max(diff) * 1.05))
        im2 = axes[row, 2].imshow(diff, origin="lower", extent=extent,
                                  cmap="viridis", vmin=0.0, vmax=vmax_err, aspect="auto")
        axes[row, 2].set_title(
            f"Absolute Error Map ($|\\mathrm{{Pred}} - \\mathrm{{True}}|$\nPerturbation Rel $L_2$ Error = {p_err:.3f}%",
            fontsize=10,
        )
        axes[row, 2].set_xlabel(r"Angle $\phi$ [rad]")
        axes[row, 2].set_ylabel(r"Radial coordinate $s$")
        cbar2 = fig.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)
        cbar2.set_label(r"$|\sigma_{\mathrm{pred}} - \sigma_{\mathrm{true}}| / \sigma$", fontsize=9)

    fig.suptitle(
        "POD + GP Body-Fitted Stress Field Reconstruction (Split A Interpolation): Best vs. Worst Test Samples",
        fontsize=13,
        fontweight="bold",
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    alt_path = "results/pod_best_worst_ellipse.png"
    shutil.copyfile(out_path, alt_path)
    print(f"Saved best/worst test sample diagnostic figure to {out_path} and {alt_path}.")


def run_pipeline(npz_path="data/fields_ellipse.npz", seed=42):
    """
    Executes the complete POD+GP training, mode selection, and evaluation pipeline
    across interpolation and both extrapolation splits.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{npz_path} not found.")

    print(f"Loading {npz_path}...")
    data = np.load(npz_path)
    fields = data["fields"]       # (N, 64, 64)
    d_over_W = data["d_over_W"]   # (N,)
    ar = data["ar"]               # (N,)
    H_over_W = data["H_over_W"]   # (N,)
    kt_gross = data["kt_gross"]   # (N,)
    ids = data["ids"]             # (N,)
    s = data["s"]                 # (64,)
    phi = data["phi"]             # (64,)

    N = len(fields)
    flat_all = fields.reshape(N, -1)
    X_all = np.column_stack([d_over_W, ar, H_over_W])

    print(f"Loaded {N} fields of shape {fields.shape[1]}x{fields.shape[2]}.")
    print(f"Inputs: ['d_over_W', 'ar', 'H_over_W'].")

    all_results = []
    preds_dict = {}

    # =========================================================================
    # SPLIT A: INTERPOLATION (Random 70/15/15 by sample)
    # =========================================================================
    idx = np.arange(N)
    idx_tr_A, idx_tmp_A = train_test_split(idx, test_size=0.30, random_state=seed)
    idx_va_A, idx_te_A = train_test_split(idx_tmp_A, test_size=0.50, random_state=seed)

    print("\n" + "=" * 90)
    print(f"SPLIT A: Interpolation (train={len(idx_tr_A)}, val={len(idx_va_A)}, test={len(idx_te_A)})")
    print("=" * 90)

    # Mode selection on validation set with table printed 1 to 30
    pca_A_full, n_modes_A, mean_errs_A, max_errs_A = fit_pca_and_select_modes(
        flat_train=flat_all[idx_tr_A],
        flat_val=flat_all[idx_va_A],
        max_modes=30,
        threshold_pct=0.1,
        print_table=True,
    )

    # Fit PCA with chosen modes on full training set
    pca_A = PCA(n_components=n_modes_A, random_state=seed)
    pca_A.fit(flat_all[idx_tr_A])
    coeffs_tr_A = pca_A.transform(flat_all[idx_tr_A])

    # Train GP surrogates
    gps_A = train_gp_surrogates(X_all[idx_tr_A], coeffs_tr_A, n_modes=n_modes_A, seed=seed)

    # Predict test fields
    pred_fields_te_A = predict_fields(gps_A, pca_A, X_all[idx_te_A], n_modes=n_modes_A)

    # Timing
    single_us_A, batch_us_A = time_inference(gps_A, pca_A, X_all[idx_te_A], n_modes=n_modes_A)

    # Metrics
    m_pod_A = compute_metrics(pred_fields_te_A, fields[idx_te_A])

    # Baselines
    uniform_field = np.ones((64, 64), dtype=np.float32)
    m_uni_A = compute_baseline_metrics(fields[idx_te_A], uniform_field)

    mean_tr_field_A = np.mean(fields[idx_tr_A], axis=0).astype(np.float32)
    m_mean_A = compute_baseline_metrics(fields[idx_te_A], mean_tr_field_A)

    all_results.extend([
        {
            "split": "interp",
            "model": "POD+GP",
            "n_modes": n_modes_A,
            "full_rel_l2_mean_%": m_pod_A["full_rel_l2_mean_%"],
            "full_rel_l2_max_%": m_pod_A["full_rel_l2_max_%"],
            "pert_rel_l2_mean_%": m_pod_A["pert_rel_l2_mean_%"],
            "pert_rel_l2_max_%": m_pod_A["pert_rel_l2_max_%"],
            "peak_err_mean_%": m_pod_A["peak_err_mean_%"],
            "peak_err_max_%": m_pod_A["peak_err_max_%"],
            "single_sample_us": single_us_A,
            "batched_us_per_sample": batch_us_A,
        },
        {
            "split": "interp",
            "model": "Uniform Field (1.0)",
            "n_modes": np.nan,
            "full_rel_l2_mean_%": m_uni_A["full_rel_l2_mean_%"],
            "full_rel_l2_max_%": m_uni_A["full_rel_l2_max_%"],
            "pert_rel_l2_mean_%": m_uni_A["pert_rel_l2_mean_%"],
            "pert_rel_l2_max_%": m_uni_A["pert_rel_l2_max_%"],
            "peak_err_mean_%": m_uni_A["peak_err_mean_%"],
            "peak_err_max_%": m_uni_A["peak_err_max_%"],
            "single_sample_us": np.nan,
            "batched_us_per_sample": np.nan,
        },
        {
            "split": "interp",
            "model": "Mean Training Field",
            "n_modes": np.nan,
            "full_rel_l2_mean_%": m_mean_A["full_rel_l2_mean_%"],
            "full_rel_l2_max_%": m_mean_A["full_rel_l2_max_%"],
            "pert_rel_l2_mean_%": m_mean_A["pert_rel_l2_mean_%"],
            "pert_rel_l2_max_%": m_mean_A["pert_rel_l2_max_%"],
            "peak_err_mean_%": m_mean_A["peak_err_mean_%"],
            "peak_err_max_%": m_mean_A["peak_err_max_%"],
            "single_sample_us": np.nan,
            "batched_us_per_sample": np.nan,
        },
    ])

    print(f"\n--- Split A (Interpolation) Test Metrics ---")
    print(f"POD+GP ({n_modes_A} modes): Full Rel L2 = {m_pod_A['full_rel_l2_mean_%']:.4f}% (max {m_pod_A['full_rel_l2_max_%']:.4f}%) | "
          f"Pert Rel L2 = {m_pod_A['pert_rel_l2_mean_%']:.4f}% (max {m_pod_A['pert_rel_l2_max_%']:.4f}%) | "
          f"Peak Err = {m_pod_A['peak_err_mean_%']:.4f}% (max {m_pod_A['peak_err_max_%']:.4f}%)")
    print(f"Uniform Field (1.0): Full Rel L2 = {m_uni_A['full_rel_l2_mean_%']:.4f}% | "
          f"Pert Rel L2 = {m_uni_A['pert_rel_l2_mean_%']:.4f}% | Peak Err = {m_uni_A['peak_err_mean_%']:.4f}%")
    print(f"Mean Training Field: Full Rel L2 = {m_mean_A['full_rel_l2_mean_%']:.4f}% | "
          f"Pert Rel L2 = {m_mean_A['pert_rel_l2_mean_%']:.4f}% | Peak Err = {m_mean_A['peak_err_mean_%']:.4f}%")
    print(f"Latency: Single-sample = {single_us_A:,.1f} us ({single_us_A/1000:.3f} ms) | "
          f"Batched = {batch_us_A:,.1f} us/sample ({batch_us_A/1000:.4f} ms/sample)")

    # 10 worst test samples by perturbation error for Split A
    worst_order_A = np.argsort(m_pod_A["per_sample_pert_l2"])[::-1][:10]
    print("\n--- Split A: 10 Worst Test Samples by Perturbation Error ---")
    print(f" {'Rank':>4} | {'ID':>4} | {'d/W':>6} | {'ar':>5} | {'H/W':>5} | {'Pert Rel L2 [%]':>17} | {'Full Rel L2 [%]':>17} | {'Peak Err [%]':>13}")
    print("-" * 85)
    for rank, w_idx in enumerate(worst_order_A, 1):
        sid = ids[idx_te_A][w_idx]
        dw = d_over_W[idx_te_A][w_idx]
        ar_v = ar[idx_te_A][w_idx]
        hw = H_over_W[idx_te_A][w_idx]
        p_err = m_pod_A["per_sample_pert_l2"][w_idx]
        f_err = m_pod_A["per_sample_full_l2"][w_idx]
        pk_err = m_pod_A["per_sample_peak_err"][w_idx]
        print(f" {rank:4d} | {sid:4d} | {dw:6.3f} | {ar_v:5.2f} | {hw:5.2f} | {p_err:17.4f}% | {f_err:17.4f}% | {pk_err:13.4f}%")

    # Diagnostic best/worst figure for Split A
    fig_best_worst_path = "results/field_ellipse_best_worst.png"
    plot_best_and_worst_samples(
        true_fields=fields[idx_te_A],
        pred_fields=pred_fields_te_A,
        d_over_W=d_over_W[idx_te_A],
        ar=ar[idx_te_A],
        H_over_W=H_over_W[idx_te_A],
        pert_l2_list=m_pod_A["per_sample_pert_l2"],
        ids=ids[idx_te_A],
        s=s,
        phi=phi,
        out_path=fig_best_worst_path,
    )

    # Store predictions for NPZ
    preds_dict["interp_ids"] = ids[idx_te_A]
    preds_dict["interp_pred_fields"] = pred_fields_te_A
    preds_dict["interp_true_fields"] = fields[idx_te_A]
    preds_dict["interp_d_over_W"] = d_over_W[idx_te_A]
    preds_dict["interp_ar"] = ar[idx_te_A]
    preds_dict["interp_H_over_W"] = H_over_W[idx_te_A]
    preds_dict["interp_kt_gross"] = kt_gross[idx_te_A]

    # =========================================================================
    # SPLIT B1: EXTRAPOLATION IN HOLE SIZE (train d/W < 0.40, test d/W >= 0.45)
    # =========================================================================
    idx_tr_B1 = np.where(d_over_W < 0.40)[0]
    idx_te_B1 = np.where(d_over_W >= 0.45)[0]

    # Hold out top 15% of training samples by d/W as validation
    order_b1 = idx_tr_B1[np.argsort(d_over_W[idx_tr_B1])]
    n_val_b1 = int(round(0.15 * len(idx_tr_B1)))
    b1_sub_idx = order_b1[:-n_val_b1]
    b1_val_idx = order_b1[-n_val_b1:]

    print("\n" + "=" * 90)
    print(f"SPLIT B1: Extrapolation in Hole Size (train={len(idx_tr_B1)} [d/W < 0.40], test={len(idx_te_B1)} [d/W >= 0.45])")
    print(f"Validation mode selection set: top {len(b1_val_idx)} training samples by d/W")
    print("=" * 90)

    pca_B1_full, n_modes_B1, _, _ = fit_pca_and_select_modes(
        flat_train=flat_all[b1_sub_idx],
        flat_val=flat_all[b1_val_idx],
        max_modes=30,
        threshold_pct=0.1,
        print_table=False,
    )

    # Fit PCA with chosen modes on full B1 training set
    pca_B1 = PCA(n_components=n_modes_B1, random_state=seed)
    pca_B1.fit(flat_all[idx_tr_B1])
    coeffs_tr_B1 = pca_B1.transform(flat_all[idx_tr_B1])

    # Train GPs
    gps_B1 = train_gp_surrogates(X_all[idx_tr_B1], coeffs_tr_B1, n_modes=n_modes_B1, seed=seed)

    # Predict test fields
    pred_fields_te_B1 = predict_fields(gps_B1, pca_B1, X_all[idx_te_B1], n_modes=n_modes_B1)

    # Timing
    single_us_B1, batch_us_B1 = time_inference(gps_B1, pca_B1, X_all[idx_te_B1], n_modes=n_modes_B1)

    # Metrics
    m_pod_B1 = compute_metrics(pred_fields_te_B1, fields[idx_te_B1])
    m_uni_B1 = compute_baseline_metrics(fields[idx_te_B1], uniform_field)
    mean_tr_field_B1 = np.mean(fields[idx_tr_B1], axis=0).astype(np.float32)
    m_mean_B1 = compute_baseline_metrics(fields[idx_te_B1], mean_tr_field_B1)

    all_results.extend([
        {
            "split": "extrap_hole",
            "model": "POD+GP",
            "n_modes": n_modes_B1,
            "full_rel_l2_mean_%": m_pod_B1["full_rel_l2_mean_%"],
            "full_rel_l2_max_%": m_pod_B1["full_rel_l2_max_%"],
            "pert_rel_l2_mean_%": m_pod_B1["pert_rel_l2_mean_%"],
            "pert_rel_l2_max_%": m_pod_B1["pert_rel_l2_max_%"],
            "peak_err_mean_%": m_pod_B1["peak_err_mean_%"],
            "peak_err_max_%": m_pod_B1["peak_err_max_%"],
            "single_sample_us": single_us_B1,
            "batched_us_per_sample": batch_us_B1,
        },
        {
            "split": "extrap_hole",
            "model": "Uniform Field (1.0)",
            "n_modes": np.nan,
            "full_rel_l2_mean_%": m_uni_B1["full_rel_l2_mean_%"],
            "full_rel_l2_max_%": m_uni_B1["full_rel_l2_max_%"],
            "pert_rel_l2_mean_%": m_uni_B1["pert_rel_l2_mean_%"],
            "pert_rel_l2_max_%": m_uni_B1["pert_rel_l2_max_%"],
            "peak_err_mean_%": m_uni_B1["peak_err_mean_%"],
            "peak_err_max_%": m_uni_B1["peak_err_max_%"],
            "single_sample_us": np.nan,
            "batched_us_per_sample": np.nan,
        },
        {
            "split": "extrap_hole",
            "model": "Mean Training Field",
            "n_modes": np.nan,
            "full_rel_l2_mean_%": m_mean_B1["full_rel_l2_mean_%"],
            "full_rel_l2_max_%": m_mean_B1["full_rel_l2_max_%"],
            "pert_rel_l2_mean_%": m_mean_B1["pert_rel_l2_mean_%"],
            "pert_rel_l2_max_%": m_mean_B1["pert_rel_l2_max_%"],
            "peak_err_mean_%": m_mean_B1["peak_err_mean_%"],
            "peak_err_max_%": m_mean_B1["peak_err_max_%"],
            "single_sample_us": np.nan,
            "batched_us_per_sample": np.nan,
        },
    ])

    print(f"\n--- Split B1 (Hole Size) Test Metrics ---")
    print(f"POD+GP ({n_modes_B1} modes): Full Rel L2 = {m_pod_B1['full_rel_l2_mean_%']:.4f}% (max {m_pod_B1['full_rel_l2_max_%']:.4f}%) | "
          f"Pert Rel L2 = {m_pod_B1['pert_rel_l2_mean_%']:.4f}% (max {m_pod_B1['pert_rel_l2_max_%']:.4f}%) | "
          f"Peak Err = {m_pod_B1['peak_err_mean_%']:.4f}% (max {m_pod_B1['peak_err_max_%']:.4f}%)")
    print(f"Uniform Field (1.0): Full Rel L2 = {m_uni_B1['full_rel_l2_mean_%']:.4f}% | "
          f"Pert Rel L2 = {m_uni_B1['pert_rel_l2_mean_%']:.4f}% | Peak Err = {m_uni_B1['peak_err_mean_%']:.4f}%")
    print(f"Mean Training Field: Full Rel L2 = {m_mean_B1['full_rel_l2_mean_%']:.4f}% | "
          f"Pert Rel L2 = {m_mean_B1['pert_rel_l2_mean_%']:.4f}% | Peak Err = {m_mean_B1['peak_err_mean_%']:.4f}%")

    preds_dict["extrap_hole_ids"] = ids[idx_te_B1]
    preds_dict["extrap_hole_pred_fields"] = pred_fields_te_B1
    preds_dict["extrap_hole_true_fields"] = fields[idx_te_B1]
    preds_dict["extrap_hole_d_over_W"] = d_over_W[idx_te_B1]
    preds_dict["extrap_hole_ar"] = ar[idx_te_B1]
    preds_dict["extrap_hole_H_over_W"] = H_over_W[idx_te_B1]
    preds_dict["extrap_hole_kt_gross"] = kt_gross[idx_te_B1]

    # =========================================================================
    # SPLIT B2: EXTRAPOLATION TO SHARPER NOTCHES (train ar >= 0.8, test ar < 0.7)
    # =========================================================================
    idx_tr_B2 = np.where(ar >= 0.8)[0]
    idx_te_B2 = np.where(ar < 0.7)[0]

    # Hold out lowest 15% of training samples by ar as validation
    order_b2 = idx_tr_B2[np.argsort(ar[idx_tr_B2])]
    n_val_b2 = int(round(0.15 * len(idx_tr_B2)))
    b2_val_idx = order_b2[:n_val_b2]
    b2_sub_idx = order_b2[n_val_b2:]

    print("\n" + "=" * 90)
    print(f"SPLIT B2: Extrapolation to Sharper Notches (train={len(idx_tr_B2)} [ar >= 0.8], test={len(idx_te_B2)} [ar < 0.7])")
    print(f"Validation mode selection set: lowest {len(b2_val_idx)} training samples by ar")
    print("=" * 90)

    pca_B2_full, n_modes_B2, _, _ = fit_pca_and_select_modes(
        flat_train=flat_all[b2_sub_idx],
        flat_val=flat_all[b2_val_idx],
        max_modes=30,
        threshold_pct=0.1,
        print_table=False,
    )

    # Fit PCA with chosen modes on full B2 training set
    pca_B2 = PCA(n_components=n_modes_B2, random_state=seed)
    pca_B2.fit(flat_all[idx_tr_B2])
    coeffs_tr_B2 = pca_B2.transform(flat_all[idx_tr_B2])

    # Train GPs
    gps_B2 = train_gp_surrogates(X_all[idx_tr_B2], coeffs_tr_B2, n_modes=n_modes_B2, seed=seed)

    # Predict test fields
    pred_fields_te_B2 = predict_fields(gps_B2, pca_B2, X_all[idx_te_B2], n_modes=n_modes_B2)

    # Timing
    single_us_B2, batch_us_B2 = time_inference(gps_B2, pca_B2, X_all[idx_te_B2], n_modes=n_modes_B2)

    # Metrics
    m_pod_B2 = compute_metrics(pred_fields_te_B2, fields[idx_te_B2])
    m_uni_B2 = compute_baseline_metrics(fields[idx_te_B2], uniform_field)
    mean_tr_field_B2 = np.mean(fields[idx_tr_B2], axis=0).astype(np.float32)
    m_mean_B2 = compute_baseline_metrics(fields[idx_te_B2], mean_tr_field_B2)

    all_results.extend([
        {
            "split": "extrap_notch",
            "model": "POD+GP",
            "n_modes": n_modes_B2,
            "full_rel_l2_mean_%": m_pod_B2["full_rel_l2_mean_%"],
            "full_rel_l2_max_%": m_pod_B2["full_rel_l2_max_%"],
            "pert_rel_l2_mean_%": m_pod_B2["pert_rel_l2_mean_%"],
            "pert_rel_l2_max_%": m_pod_B2["pert_rel_l2_max_%"],
            "peak_err_mean_%": m_pod_B2["peak_err_mean_%"],
            "peak_err_max_%": m_pod_B2["peak_err_max_%"],
            "single_sample_us": single_us_B2,
            "batched_us_per_sample": batch_us_B2,
        },
        {
            "split": "extrap_notch",
            "model": "Uniform Field (1.0)",
            "n_modes": np.nan,
            "full_rel_l2_mean_%": m_uni_B2["full_rel_l2_mean_%"],
            "full_rel_l2_max_%": m_uni_B2["full_rel_l2_max_%"],
            "pert_rel_l2_mean_%": m_uni_B2["pert_rel_l2_mean_%"],
            "pert_rel_l2_max_%": m_uni_B2["pert_rel_l2_max_%"],
            "peak_err_mean_%": m_uni_B2["peak_err_mean_%"],
            "peak_err_max_%": m_uni_B2["peak_err_max_%"],
            "single_sample_us": np.nan,
            "batched_us_per_sample": np.nan,
        },
        {
            "split": "extrap_notch",
            "model": "Mean Training Field",
            "n_modes": np.nan,
            "full_rel_l2_mean_%": m_mean_B2["full_rel_l2_mean_%"],
            "full_rel_l2_max_%": m_mean_B2["full_rel_l2_max_%"],
            "pert_rel_l2_mean_%": m_mean_B2["pert_rel_l2_mean_%"],
            "pert_rel_l2_max_%": m_mean_B2["pert_rel_l2_max_%"],
            "peak_err_mean_%": m_mean_B2["peak_err_mean_%"],
            "peak_err_max_%": m_mean_B2["peak_err_max_%"],
            "single_sample_us": np.nan,
            "batched_us_per_sample": np.nan,
        },
    ])

    print(f"\n--- Split B2 (Sharper Notches) Test Metrics ---")
    print(f"POD+GP ({n_modes_B2} modes): Full Rel L2 = {m_pod_B2['full_rel_l2_mean_%']:.4f}% (max {m_pod_B2['full_rel_l2_max_%']:.4f}%) | "
          f"Pert Rel L2 = {m_pod_B2['pert_rel_l2_mean_%']:.4f}% (max {m_pod_B2['pert_rel_l2_max_%']:.4f}%) | "
          f"Peak Err = {m_pod_B2['peak_err_mean_%']:.4f}% (max {m_pod_B2['peak_err_max_%']:.4f}%)")
    print(f"Uniform Field (1.0): Full Rel L2 = {m_uni_B2['full_rel_l2_mean_%']:.4f}% | "
          f"Pert Rel L2 = {m_uni_B2['pert_rel_l2_mean_%']:.4f}% | Peak Err = {m_uni_B2['peak_err_mean_%']:.4f}%")
    print(f"Mean Training Field: Full Rel L2 = {m_mean_B2['full_rel_l2_mean_%']:.4f}% | "
          f"Pert Rel L2 = {m_mean_B2['pert_rel_l2_mean_%']:.4f}% | Peak Err = {m_mean_B2['peak_err_mean_%']:.4f}%")

    preds_dict["extrap_notch_ids"] = ids[idx_te_B2]
    preds_dict["extrap_notch_pred_fields"] = pred_fields_te_B2
    preds_dict["extrap_notch_true_fields"] = fields[idx_te_B2]
    preds_dict["extrap_notch_d_over_W"] = d_over_W[idx_te_B2]
    preds_dict["extrap_notch_ar"] = ar[idx_te_B2]
    preds_dict["extrap_notch_H_over_W"] = H_over_W[idx_te_B2]
    preds_dict["extrap_notch_kt_gross"] = kt_gross[idx_te_B2]

    # Save coordinate arrays
    preds_dict["s"] = s
    preds_dict["phi"] = phi

    # =========================================================================
    # SAVE CSV AND NPZ
    # =========================================================================
    # 1. Results CSV
    res_df = pd.DataFrame(all_results)
    csv_out_path = "results/field_ellipse_results.csv"
    os.makedirs(os.path.dirname(csv_out_path) or ".", exist_ok=True)
    res_df.to_csv(csv_out_path, index=False)

    print("\n" + "=" * 105)
    print("=== SUMMARY METRICS TABLE: 2D STRESS FIELD PREDICTION (ELLIPTICAL HOLE) ===")
    print("=" * 105)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    print(res_df.to_string(index=False))
    print("-" * 105)

    # 2. Predictions NPZ
    npz_out_path = "results/field_ellipse_predictions.npz"
    np.savez_compressed(npz_out_path, **preds_dict)
    print(f"Saved predictions archive to {npz_out_path} ({os.path.getsize(npz_out_path)/(1024*1024):.2f} MB).")
    print(f"Saved metrics table to {csv_out_path}.")
    print("=" * 105 + "\n")

    return res_df


def main():
    parser = argparse.ArgumentParser(
        description="POD + GP surrogate for 2D body-fitted elliptical stress fields."
    )
    parser.add_argument("--data", default="data/fields_ellipse.npz", help="Path to fields_ellipse.npz")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_pipeline(npz_path=args.data, seed=args.seed)


if __name__ == "__main__":
    main()
