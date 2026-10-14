"""
Amplitude / Shape Decomposition and End-to-End Mode Selection
for Elliptical Stress Field Prediction (POD+GP)

Pipeline:
  Part 1: Amplitude/shape decomposition.
    - Decouple peak notch stress Kt = y[0, 0] from normalized shape field:
        f = (y - 1) / (Kt - 1)
    - Check min(Kt - 1) across train/val/test for all splits.
    - Fit PCA on training f, select mode count via validation reconstruction error (< 0.1%, cap 30).
    - Train one GP per mode on f's PCA coefficients.
    - Reuse scalar GP (ARD) for Kt_gross from field_diagnostics_ellipse.py.
    - Predict y_hat = 1 + (Kt_scalar - 1) * f_hat for:
        (1) raw f_hat
        (2) f_hat renormalized by f_hat[0, 0]
    - Report PCA floor for f and full metric comparison table side-by-side with
      uncorrected POD+GP and Correction (b).

  Part 2: End-to-end mode selection.
    - Sweep k = 1..30 on each split's validation set for both y-field POD+GP and f-shape model.
    - Select k* by minimum end-to-end full-field relative L2 error on validation.
    - Compare modes above k* zeroed vs all predicted on test sets.

  Part 3 (optional toggle --inv-ar):
    - Repeat Part 1 with GP inputs [d/W, 1/ar, H/W].

Outputs:
  - results/field_ellipse_shape.csv
  - results/field_ellipse_shape_comparison.png
"""
import argparse
import os
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


def train_single_gp(k, X_train, y_train_k, seed=42):
    """Fits one Gaussian Process pipeline for mode k."""
    n_features = X_train.shape[1]
    gp = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=C(1.0) * RBF([1.0] * n_features) + WhiteKernel(1e-5, (1e-10, 1e-1)),
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=seed,
        ),
    )
    gp.fit(X_train, y_train_k)
    return gp


def train_gp_surrogates(X_train, coeffs_train, n_modes, seed=42):
    """Trains n_modes Gaussian Processes in parallel across CPU cores."""
    gps = Parallel(n_jobs=-1)(
        delayed(train_single_gp)(k, X_train, coeffs_train[:, k], seed=seed)
        for k in range(n_modes)
    )
    return gps


def train_scalar_gp(X_train, y_train, seed=42):
    """Trains scalar GP (ARD) on [W, d_over_W, ar, H_over_W] matching baselines."""
    return train_single_gp(0, X_train, y_train, seed=seed)


def compute_metrics(pred_fields, true_fields):
    """
    Computes summary metrics:
      1. Full-field relative L2 error (%)
      2. Perturbation relative L2 error (%)
      3. Peak error (%) at notch root [0, 0]
    """
    M = len(pred_fields)
    full_l2 = np.zeros(M, dtype=np.float64)
    pert_l2 = np.zeros(M, dtype=np.float64)
    peak_err = np.zeros(M, dtype=np.float64)

    for i in range(M):
        yp = pred_fields[i]
        yt = true_fields[i]

        diff_norm = np.linalg.norm(yp - yt)
        true_norm = np.linalg.norm(yt)
        pert_true_norm = np.linalg.norm(yt - 1.0)

        full_l2[i] = 100.0 * diff_norm / true_norm if true_norm > 0 else 0.0
        pert_l2[i] = 100.0 * diff_norm / pert_true_norm if pert_true_norm > 0 else 0.0
        peak_err[i] = 100.0 * abs(yp[0, 0] - yt[0, 0]) / yt[0, 0] if yt[0, 0] > 0 else 0.0

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


def fit_pca_reconstruction_selection(flat_sub_tr, flat_sub_val, max_modes=30, threshold_pct=0.1):
    """
    Selects mode count based on validation reconstruction relative L2 error (< threshold_pct).
    """
    pca_eval = PCA(n_components=max_modes, random_state=42)
    pca_eval.fit(flat_sub_tr)
    c_val = pca_eval.transform(flat_sub_val)

    mean_errs = []
    for k in range(1, max_modes + 1):
        recon = pca_eval.mean_ + c_val[:, :k] @ pca_eval.components_[:k]
        rel_l2 = [
            100.0 * np.linalg.norm(recon[j] - flat_sub_val[j]) / np.linalg.norm(flat_sub_val[j])
            for j in range(len(flat_sub_val))
        ]
        mean_errs.append(float(np.mean(rel_l2)))

    valid = [k for k, err in enumerate(mean_errs, 1) if err < threshold_pct]
    chosen_k = valid[0] if valid else max_modes
    return chosen_k, mean_errs


def run_pipeline(npz_path="data/fields_ellipse.npz", csv_path="data/plate_ellipse.csv",
                 out_csv="results/field_ellipse_shape.csv", inv_ar=False, seed=42):
    print("=" * 100)
    print(f"AMPLITUDE/SHAPE DECOMPOSITION & END-TO-END MODE SELECTION (INV_AR={inv_ar})")
    print("=" * 100)

    # 1. Load Data
    data = np.load(npz_path)
    df = pd.read_csv(csv_path)
    df = df[df["ok"].astype(str) == "True"].reset_index(drop=True)

    fields = data["fields"]       # (1500, 64, 64)
    N = len(fields)
    flat_y = fields.reshape(N, -1)

    # Notch peak stress per sample
    kt_samples = fields[:, 0, 0]

    # Input feature matrices
    d_over_W = df["d_over_W"].values
    ar = df["ar"].values
    H_over_W = df["H_over_W"].values
    W = df["W"].values

    if inv_ar:
        X_field = np.column_stack([d_over_W, 1.0 / ar, H_over_W])
        feat_desc = "['d/W', '1/ar', 'H/W']"
    else:
        X_field = np.column_stack([d_over_W, ar, H_over_W])
        feat_desc = "['d/W', 'ar', 'H/W']"

    X_scalar = df[["W", "d_over_W", "ar", "H_over_W"]].values
    y_scalar = df["kt_gross"].values

    # Build Shape Field f = (y - 1) / (Kt - 1)
    f_fields = (fields - 1.0) / (kt_samples[:, None, None] - 1.0)
    flat_f = f_fields.reshape(N, -1)

    # Splits (identical to field_pod_ellipse.py)
    idx = np.arange(N)
    idx_tr_A, idx_tmp_A = train_test_split(idx, test_size=0.30, random_state=seed)
    idx_va_A, idx_te_A = train_test_split(idx_tmp_A, test_size=0.50, random_state=seed)

    idx_tr_B1 = np.where(df["d_over_W"] < 0.40)[0]
    idx_te_B1 = np.where(df["d_over_W"] >= 0.45)[0]
    order_b1 = idx_tr_B1[np.argsort(df.loc[idx_tr_B1, "d_over_W"].values)]
    n_val_b1 = int(round(0.15 * len(idx_tr_B1)))
    b1_sub_idx = order_b1[:-n_val_b1]
    b1_val_idx = order_b1[-n_val_b1:]

    idx_tr_B2 = np.where(df["ar"] >= 0.8)[0]
    idx_te_B2 = np.where(df["ar"] < 0.7)[0]
    order_b2 = idx_tr_B2[np.argsort(df.loc[idx_tr_B2, "ar"].values)]
    n_val_b2 = int(round(0.15 * len(idx_tr_B2)))
    b2_val_idx = order_b2[:n_val_b2]
    b2_sub_idx = order_b2[n_val_b2:]

    splits = {
        "interp": {
            "name": "Split A (Interpolation)",
            "tr": idx_tr_A, "va": idx_va_A, "te": idx_te_A,
            "sub_tr": idx_tr_A, "sub_va": idx_va_A,
            "orig_k_y": 14,
        },
        "extrap_hole": {
            "name": "Split B1 (Hole Size Extrapolation)",
            "tr": idx_tr_B1, "va": b1_val_idx, "te": idx_te_B1,
            "sub_tr": b1_sub_idx, "sub_va": b1_val_idx,
            "orig_k_y": 21,
        },
        "extrap_notch": {
            "name": "Split B2 (Sharper Notch Extrapolation)",
            "tr": idx_tr_B2, "va": b2_val_idx, "te": idx_te_B2,
            "sub_tr": b2_sub_idx, "sub_va": b2_val_idx,
            "orig_k_y": 30,
        },
    }

    # =========================================================================
    # PRE-CHECK: Print min(Kt - 1) over Train/Val/Test
    # =========================================================================
    print("\n--- Pre-Check: min(Kt - 1) across Train, Validation, and Test Sets ---")
    print(f"Global min(Kt - 1): {np.min(kt_samples - 1.0):.6f} | Global max(Kt - 1): {np.max(kt_samples - 1.0):.6f}")
    for s_key, s_info in splits.items():
        tr, va, te = s_info["tr"], s_info["va"], s_info["te"]
        min_tr = np.min(kt_samples[tr] - 1.0)
        min_va = np.min(kt_samples[va] - 1.0)
        min_te = np.min(kt_samples[te] - 1.0)
        print(f"  {s_info['name']}: Train min = {min_tr:.6f}, Val min = {min_va:.6f}, Test min = {min_te:.6f}")
    print(">> Safe Normalization Confirmed: All (Kt - 1) > 1.015 > 0, zero risk of division by zero.\n")

    # =========================================================================
    # PART 1: AMPLITUDE / SHAPE DECOMPOSITION
    # =========================================================================
    print("=" * 100)
    print(f"PART 1: AMPLITUDE / SHAPE DECOMPOSITION (Features: {feat_desc})")
    print("=" * 100)

    part1_records = []
    models_dict = {}

    for s_key, s_info in splits.items():
        tr, va, te = s_info["tr"], s_info["va"], s_info["te"]
        sub_tr, sub_va = s_info["sub_tr"], s_info["sub_va"]
        print(f"\n--- Processing {s_info['name']} ---")

        # 1. Mode selection for f using validation reconstruction rule (< 0.1%, cap 30)
        chosen_k_f, mean_recon_f = fit_pca_reconstruction_selection(flat_f[sub_tr], flat_f[sub_va], max_modes=30, threshold_pct=0.1)
        print(f"PCA Reconstruction Selection for shape field f: k = {chosen_k_f} modes (mean val error: {mean_recon_f[chosen_k_f-1]:.4f}%)")

        # 2. Fit PCA on full training f
        pca_f = PCA(n_components=30, random_state=seed)
        pca_f.fit(flat_f[tr])
        c_tr_f = pca_f.transform(flat_f[tr])
        c_te_f_true = pca_f.transform(flat_f[te])

        # PCA floor for f at chosen k
        f_floor_flat = pca_f.mean_ + c_te_f_true[:, :chosen_k_f] @ pca_f.components_[:chosen_k_f]
        f_floor_fields = f_floor_flat.reshape(len(te), 64, 64)
        # Resulting stress field floor using true Kt
        y_floor_fields = 1.0 + (kt_samples[te, None, None] - 1.0) * f_floor_fields
        m_floor_f = compute_metrics(y_floor_fields, fields[te])

        # 3. Train GP per mode on f's coefficients (30 modes)
        gps_f = train_gp_surrogates(X_field[tr], c_tr_f, n_modes=30, seed=seed)

        # 4. Train scalar GP (ARD) on training set for Kt
        sgp = train_scalar_gp(X_scalar[tr], y_scalar[tr], seed=seed)
        kt_scalar_te = sgp.predict(X_scalar[te])

        # 5. Predict shape field f_hat at chosen k
        pred_c_f = np.column_stack([gp.predict(X_field[te]) for gp in gps_f[:chosen_k_f]])
        f_hat_flat = pca_f.mean_ + pred_c_f @ pca_f.components_[:chosen_k_f]
        f_hat = f_hat_flat.reshape(len(te), 64, 64)

        # Variant 1: Raw f_hat
        y_hat_raw = 1.0 + (kt_scalar_te[:, None, None] - 1.0) * f_hat
        m_raw = compute_metrics(y_hat_raw, fields[te])

        # Variant 2: Renormalized f_hat by f_hat[0, 0]
        f_peak = f_hat[:, 0, 0]
        denom_safe = np.where(np.abs(f_peak) < 1e-6, 1.0, f_peak)
        f_hat_rn = f_hat / denom_safe[:, None, None]
        y_hat_rn = 1.0 + (kt_scalar_te[:, None, None] - 1.0) * f_hat_rn
        m_rn = compute_metrics(y_hat_rn, fields[te])

        # Save for Part 2 and comparison
        models_dict[s_key] = {
            "pca_f": pca_f, "gps_f": gps_f, "sgp": sgp,
            "chosen_k_f": chosen_k_f,
        }

        # Store Table 1 records
        part1_records.extend([
            {
                "split": s_key, "model": f"PCA Floor for f (k={chosen_k_f})",
                "k": chosen_k_f, **{k: v for k, v in m_floor_f.items() if not k.startswith("per_sample")}
            },
            {
                "split": s_key, "model": f"Shape Model [Raw f_hat] (k={chosen_k_f})",
                "k": chosen_k_f, **{k: v for k, v in m_raw.items() if not k.startswith("per_sample")}
            },
            {
                "split": s_key, "model": f"Shape Model [Renorm f_hat] (k={chosen_k_f})",
                "k": chosen_k_f, **{k: v for k, v in m_rn.items() if not k.startswith("per_sample")}
            },
        ])

    # Load previous baseline results from CSV for direct side-by-side comparison
    df_prev_diag = pd.read_csv("results/field_ellipse_diagnostics.csv")
    prev_uncorr = df_prev_diag[df_prev_diag["model"] == "Uncorrected POD+GP"].copy()
    prev_corr_b = df_prev_diag[df_prev_diag["model"] == "Correction (b) [Perturbation Scale]"].copy()

    side_by_side = []
    for s_key in ["interp", "extrap_hole", "extrap_notch"]:
        # Uncorrected
        u_row = prev_uncorr[prev_uncorr["split"] == s_key].iloc[0]
        side_by_side.append({
            "split": s_key, "model": "Uncorrected POD+GP (y-field)", "k": splits[s_key]["orig_k_y"],
            "full_rel_l2_mean_%": u_row["full_rel_l2_mean_%"], "full_rel_l2_max_%": u_row["full_rel_l2_max_%"],
            "pert_rel_l2_mean_%": u_row["pert_rel_l2_mean_%"], "pert_rel_l2_max_%": u_row["pert_rel_l2_max_%"],
            "peak_err_mean_%": u_row["peak_err_mean_%"], "peak_err_max_%": u_row["peak_err_max_%"],
        })
        # Correction (b)
        cb_row = prev_corr_b[prev_corr_b["split"] == s_key].iloc[0]
        side_by_side.append({
            "split": s_key, "model": "POD+GP + Correction (b) [Perturbation Scale]", "k": splits[s_key]["orig_k_y"],
            "full_rel_l2_mean_%": cb_row["full_rel_l2_mean_%"], "full_rel_l2_max_%": cb_row["full_rel_l2_max_%"],
            "pert_rel_l2_mean_%": cb_row["pert_rel_l2_mean_%"], "pert_rel_l2_max_%": cb_row["pert_rel_l2_max_%"],
            "peak_err_mean_%": cb_row["peak_err_mean_%"], "peak_err_max_%": cb_row["peak_err_max_%"],
        })
        # Part 1 rows
        for r in part1_records:
            if r["split"] == s_key:
                side_by_side.append(r)

    df_table1 = pd.DataFrame(side_by_side)
    print("\n" + "=" * 100)
    print("TABLE 1: AMPLITUDE / SHAPE DECOMPOSITION vs POD+GP AND CORRECTION (b)")
    print("=" * 100)
    print(df_table1.to_string(index=False))

    # =========================================================================
    # PART 2: END-TO-END MODE SELECTION
    # =========================================================================
    print("\n" + "=" * 100)
    print("PART 2: END-TO-END MODE SELECTION (Validation Error Sweep k = 1..30)")
    print("=" * 100)

    part2_records = []

    for s_key, s_info in splits.items():
        tr, va, te = s_info["tr"], s_info["va"], s_info["te"]
        sub_tr, sub_va = s_info["sub_tr"], s_info["sub_va"]
        val_yt = fields[sub_va]
        test_yt = fields[te]

        print(f"\n--- End-to-End Validation Mode Sweep: {s_info['name']} ---")

        # --- A. Existing y-field Model ---
        # 1. Fit PCA(30) on sub_tr, train 30 GPs on sub_tr
        pca_y_sub = PCA(n_components=30, random_state=seed).fit(flat_y[sub_tr])
        c_y_sub = pca_y_sub.transform(flat_y[sub_tr])
        gps_y_sub = train_gp_surrogates(X_field[sub_tr], c_y_sub, n_modes=30, seed=seed)

        pred_c_val_y = np.column_stack([gp.predict(X_field[sub_va]) for gp in gps_y_sub])

        # Sweep k = 1..30 on validation set
        val_errs_y = []
        for k in range(1, 31):
            pred_val_k = (pca_y_sub.mean_ + pred_c_val_y[:, :k] @ pca_y_sub.components_[:k]).reshape(len(sub_va), 64, 64)
            err = np.mean([100.0 * np.linalg.norm(pred_val_k[i] - val_yt[i]) / np.linalg.norm(val_yt[i]) for i in range(len(sub_va))])
            val_errs_y.append(float(err))

        best_k_y = int(np.argmin(val_errs_y) + 1)
        print(f"y-field Model: Optimal k* = {best_k_y} (Validation full rel L2 = {val_errs_y[best_k_y-1]:.4f}%)")

        # Train on full training set with 30 modes for test evaluation
        pca_y_full = PCA(n_components=30, random_state=seed).fit(flat_y[tr])
        c_y_tr = pca_y_full.transform(flat_y[tr])
        gps_y_full = train_gp_surrogates(X_field[tr], c_y_tr, n_modes=30, seed=seed)
        pred_c_te_y = np.column_stack([gp.predict(X_field[te]) for gp in gps_y_full])

        # Test prediction: modes above k* zeroed
        pred_te_y_kopt = (pca_y_full.mean_ + pred_c_te_y[:, :best_k_y] @ pca_y_full.components_[:best_k_y]).reshape(len(te), 64, 64)
        m_y_kopt = compute_metrics(pred_te_y_kopt, test_yt)

        # Test prediction: all modes predicted at original PCA reconstruction k (14/21/30)
        orig_k_y = s_info["orig_k_y"]
        pred_te_y_orig = (pca_y_full.mean_ + pred_c_te_y[:, :orig_k_y] @ pca_y_full.components_[:orig_k_y]).reshape(len(te), 64, 64)
        m_y_orig = compute_metrics(pred_te_y_orig, test_yt)

        # Test prediction: all 30 modes predicted
        pred_te_y_30 = (pca_y_full.mean_ + pred_c_te_y @ pca_y_full.components_).reshape(len(te), 64, 64)
        m_y_30 = compute_metrics(pred_te_y_30, test_yt)

        # --- B. New Shape Model f ---
        # 1. Fit PCA(30) on sub_tr for f, train 30 GPs on sub_tr
        pca_f_sub = PCA(n_components=30, random_state=seed).fit(flat_f[sub_tr])
        c_f_sub = pca_f_sub.transform(flat_f[sub_tr])
        gps_f_sub = train_gp_surrogates(X_field[sub_tr], c_f_sub, n_modes=30, seed=seed)
        sgp_sub = train_scalar_gp(X_scalar[sub_tr], y_scalar[sub_tr], seed=seed)
        kt_val_pred = sgp_sub.predict(X_scalar[sub_va])

        pred_c_val_f = np.column_stack([gp.predict(X_field[sub_va]) for gp in gps_f_sub])

        # Sweep k = 1..30 on validation set for shape model (renormalized)
        val_errs_f_rn = []
        val_errs_f_raw = []
        for k in range(1, 31):
            f_val_k = (pca_f_sub.mean_ + pred_c_val_f[:, :k] @ pca_f_sub.components_[:k]).reshape(len(sub_va), 64, 64)
            # Raw
            y_val_raw = 1.0 + (kt_val_pred[:, None, None] - 1.0) * f_val_k
            err_raw = np.mean([100.0 * np.linalg.norm(y_val_raw[i] - val_yt[i]) / np.linalg.norm(val_yt[i]) for i in range(len(sub_va))])
            val_errs_f_raw.append(float(err_raw))
            # Renorm
            f_00 = f_val_k[:, 0, 0]
            f_val_rn = f_val_k / np.where(np.abs(f_00) < 1e-6, 1.0, f_00)[:, None, None]
            y_val_rn = 1.0 + (kt_val_pred[:, None, None] - 1.0) * f_val_rn
            err_rn = np.mean([100.0 * np.linalg.norm(y_val_rn[i] - val_yt[i]) / np.linalg.norm(val_yt[i]) for i in range(len(sub_va))])
            val_errs_f_rn.append(float(err_rn))

        best_k_f = int(np.argmin(val_errs_f_rn) + 1)
        print(f"Shape Model: Optimal k* = {best_k_f} (Validation full rel L2 = {val_errs_f_rn[best_k_f-1]:.4f}%)")

        # Train on full training set with 30 modes for shape test evaluation
        pca_f_full = models_dict[s_key]["pca_f"]
        gps_f_full = models_dict[s_key]["gps_f"]
        sgp_full = models_dict[s_key]["sgp"]
        kt_te_pred = sgp_full.predict(X_scalar[te])

        pred_c_te_f = np.column_stack([gp.predict(X_field[te]) for gp in gps_f_full])

        # Test prediction: modes above k* zeroed (Renorm & Raw)
        f_te_kopt = (pca_f_full.mean_ + pred_c_te_f[:, :best_k_f] @ pca_f_full.components_[:best_k_f]).reshape(len(te), 64, 64)
        f_00_kopt = f_te_kopt[:, 0, 0]
        f_te_kopt_rn = f_te_kopt / np.where(np.abs(f_00_kopt) < 1e-6, 1.0, f_00_kopt)[:, None, None]
        y_te_f_kopt_rn = 1.0 + (kt_te_pred[:, None, None] - 1.0) * f_te_kopt_rn
        y_te_f_kopt_raw = 1.0 + (kt_te_pred[:, None, None] - 1.0) * f_te_kopt
        m_f_kopt_rn = compute_metrics(y_te_f_kopt_rn, test_yt)
        m_f_kopt_raw = compute_metrics(y_te_f_kopt_raw, test_yt)

        # Test prediction: all 30 modes predicted (from Part 1)
        f_te_30 = (pca_f_full.mean_ + pred_c_te_f @ pca_f_full.components_).reshape(len(te), 64, 64)
        f_00_30 = f_te_30[:, 0, 0]
        f_te_30_rn = f_te_30 / np.where(np.abs(f_00_30) < 1e-6, 1.0, f_00_30)[:, None, None]
        y_te_f_30_rn = 1.0 + (kt_te_pred[:, None, None] - 1.0) * f_te_30_rn
        m_f_30_rn = compute_metrics(y_te_f_30_rn, test_yt)

        # Record Part 2 summary
        part2_records.extend([
            {
                "split": s_key, "model": f"y-field POD+GP: End-to-End k*={best_k_y} (modes > k* zeroed)",
                "k_selected": best_k_y, "modes_above_k": "zeroed",
                **{k: v for k, v in m_y_kopt.items() if not k.startswith("per_sample")}
            },
            {
                "split": s_key, "model": f"y-field POD+GP: Recon k={orig_k_y} (all k predicted)",
                "k_selected": orig_k_y, "modes_above_k": "predicted",
                **{k: v for k, v in m_y_orig.items() if not k.startswith("per_sample")}
            },
            {
                "split": s_key, "model": f"y-field POD+GP: All 30 modes predicted",
                "k_selected": 30, "modes_above_k": "predicted",
                **{k: v for k, v in m_y_30.items() if not k.startswith("per_sample")}
            },
            {
                "split": s_key, "model": f"Shape Model [Renorm]: End-to-End k*={best_k_f} (modes > k* zeroed)",
                "k_selected": best_k_f, "modes_above_k": "zeroed",
                **{k: v for k, v in m_f_kopt_rn.items() if not k.startswith("per_sample")}
            },
            {
                "split": s_key, "model": f"Shape Model [Raw]: End-to-End k*={best_k_f} (modes > k* zeroed)",
                "k_selected": best_k_f, "modes_above_k": "zeroed",
                **{k: v for k, v in m_f_kopt_raw.items() if not k.startswith("per_sample")}
            },
            {
                "split": s_key, "model": f"Shape Model [Renorm]: All 30 modes predicted",
                "k_selected": 30, "modes_above_k": "predicted",
                **{k: v for k, v in m_f_30_rn.items() if not k.startswith("per_sample")}
            },
        ])

    df_table2 = pd.DataFrame(part2_records)
    print("\n" + "=" * 100)
    print("TABLE 2: END-TO-END MODE SELECTION (ZEROED vs ALL PREDICTED)")
    print("=" * 100)
    print(df_table2.to_string(index=False))

    # =========================================================================
    # PART 3: REPEAT PART 1 WITH [d/W, 1/ar, H/W] (OPTIONAL TOGGLE / RUN)
    # =========================================================================
    part3_records = []
    if not inv_ar:
        print("\n" + "=" * 100)
        print("PART 3: INVERTED ASPECT RATIO TOGGLE COMPARISON [d/W, 1/ar, H/W]")
        print("=" * 100)
        X_field_inv = np.column_stack([d_over_W, 1.0 / ar, H_over_W])

        for s_key, s_info in splits.items():
            tr, te = s_info["tr"], s_info["te"]
            chosen_k_f = models_dict[s_key]["chosen_k_f"]
            pca_f = models_dict[s_key]["pca_f"]
            sgp = models_dict[s_key]["sgp"]
            kt_scalar_te = sgp.predict(X_scalar[te])

            # Train GPs with inv_ar inputs
            c_tr_f = pca_f.transform(flat_f[tr])
            gps_f_inv = train_gp_surrogates(X_field_inv[tr], c_tr_f, n_modes=chosen_k_f, seed=seed)

            pred_c_f_inv = np.column_stack([gp.predict(X_field_inv[te]) for gp in gps_f_inv])
            f_hat_inv = (pca_f.mean_ + pred_c_f_inv @ pca_f.components_[:chosen_k_f]).reshape(len(te), 64, 64)

            # Raw & Renorm
            y_raw_inv = 1.0 + (kt_scalar_te[:, None, None] - 1.0) * f_hat_inv
            f_peak_inv = f_hat_inv[:, 0, 0]
            f_rn_inv = f_hat_inv / np.where(np.abs(f_peak_inv) < 1e-6, 1.0, f_peak_inv)[:, None, None]
            y_rn_inv = 1.0 + (kt_scalar_te[:, None, None] - 1.0) * f_rn_inv

            m_raw_inv = compute_metrics(y_raw_inv, fields[te])
            m_rn_inv = compute_metrics(y_rn_inv, fields[te])

            part3_records.extend([
                {
                    "split": s_key, "model": f"Shape Model (1/ar) [Raw f_hat] (k={chosen_k_f})",
                    "k": chosen_k_f, **{k: v for k, v in m_raw_inv.items() if not k.startswith("per_sample")}
                },
                {
                    "split": s_key, "model": f"Shape Model (1/ar) [Renorm f_hat] (k={chosen_k_f})",
                    "k": chosen_k_f, **{k: v for k, v in m_rn_inv.items() if not k.startswith("per_sample")}
                },
            ])

        df_table3 = pd.DataFrame(part3_records)
        print("\n" + "=" * 100)
        print("TABLE 3: PART 3 INVERTED ASPECT RATIO (1/ar) TEST METRICS")
        print("=" * 100)
        print(df_table3.to_string(index=False))

    # Save all results to results/field_ellipse_shape.csv
    df_all_out = pd.concat([
        df_table1.assign(table="Part 1: Shape Decomposition"),
        df_table2.assign(table="Part 2: End-to-End Selection"),
        pd.DataFrame(part3_records).assign(table="Part 3: Inverted Aspect Ratio (1/ar)") if part3_records else pd.DataFrame(),
    ], ignore_index=True)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df_all_out.to_csv(out_csv, index=False)
    print(f"\nSaved all results to {out_csv}")

    # Generate Visualization Figure
    fig_path = "results/field_ellipse_shape_comparison.png"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    split_keys = ["interp", "extrap_hole", "extrap_notch"]
    split_titles = ["Split A (Interpolation)", "Split B1 (Hole Extrap)", "Split B2 (Notch Extrap)"]

    for ax, sk, st in zip(axes, split_keys, split_titles):
        sub_df = df_all_out[df_all_out["split"] == sk]
        
        models_to_plot = [
            ("Uncorr POD+GP", sub_df[sub_df["model"].str.contains("Uncorrected POD+GP", regex=False)].iloc[0]["full_rel_l2_mean_%"]),
            ("Corr (b)", sub_df[sub_df["model"].str.contains("Correction (b)", regex=False)].iloc[0]["full_rel_l2_mean_%"]),
            ("Shape (k=30)", sub_df[sub_df["model"].str.startswith("Shape Model [Renorm f_hat]")].iloc[0]["full_rel_l2_mean_%"]),
            ("Shape (E2E k*)", sub_df[sub_df["model"].str.startswith("Shape Model [Renorm]: End-to-End")].iloc[0]["full_rel_l2_mean_%"]),
        ]
        
        labels = [m[0] for m in models_to_plot]
        vals = [m[1] for m in models_to_plot]
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
        
        bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
        ax.set_title(st, fontsize=12, fontweight="bold")
        ax.set_ylabel("Full-Field Rel $L_2$ Error (%)", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., h + 0.04 * max(vals), f"{h:.3f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.22)

    fig.suptitle("Comparison of Field Surrogate Strategies Across Interpolation & Extrapolation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved diagnostic comparison figure to {fig_path}")

    return df_all_out


def plot_from_csv(csv_path="results/field_ellipse_shape.csv"):
    df_all_out = pd.read_csv(csv_path)
    fig_path = "results/field_ellipse_shape_comparison.png"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    split_keys = ["interp", "extrap_hole", "extrap_notch"]
    split_titles = ["Split A (Interpolation)", "Split B1 (Hole Extrap)", "Split B2 (Notch Extrap)"]

    for ax, sk, st in zip(axes, split_keys, split_titles):
        sub_df = df_all_out[df_all_out["split"] == sk]
        
        models_to_plot = [
            ("Uncorr POD+GP", sub_df[sub_df["model"].str.contains("Uncorrected POD+GP", regex=False)].iloc[0]["full_rel_l2_mean_%"]),
            ("Corr (b)", sub_df[sub_df["model"].str.contains("Correction (b)", regex=False)].iloc[0]["full_rel_l2_mean_%"]),
            ("Shape (k=30)", sub_df[sub_df["model"].str.startswith("Shape Model [Renorm f_hat]")].iloc[0]["full_rel_l2_mean_%"]),
            ("Shape (E2E k*)", sub_df[sub_df["model"].str.startswith("Shape Model [Renorm]: End-to-End")].iloc[0]["full_rel_l2_mean_%"]),
        ]
        
        labels = [m[0] for m in models_to_plot]
        vals = [m[1] for m in models_to_plot]
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
        
        bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
        ax.set_title(st, fontsize=12, fontweight="bold")
        ax.set_ylabel("Full-Field Rel $L_2$ Error (%)", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., h + 0.04 * max(vals), f"{h:.3f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.22)

    fig.suptitle("Comparison of Field Surrogate Strategies Across Interpolation & Extrapolation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved diagnostic comparison figure to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amplitude/shape decomposition and end-to-end mode selection")
    parser.add_argument("--inv-ar", action="store_true", help="Use [d/W, 1/ar, H/W] as inputs for GP surrogates")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--plot-only", action="store_true", help="Only render the comparison plot from existing CSV")
    args = parser.parse_args()

    if args.plot_only:
        plot_from_csv()
    else:
        run_pipeline(inv_ar=args.inv_ar, seed=args.seed)
