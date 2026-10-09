"""
Stress Field Diagnostics for 2D FEA Surrogates (POD+GP vs. CNN-Deconv NN).

Loads:
  - data/fields.npz
  - results/pod_predictions.npz
  - results/nn_predictions.npz

Computes strictly on valid plate pixels:
  1. Relative L2 error of two trivial predictors:
     (a) Uniform field equal to 1.0 everywhere.
     (b) Pixelwise mean training field (averaged over training samples where pixel is valid).
  2. Relative L2 error of the models' perturbation (pred - 1) against (true - 1).
  3. Relative L2 error only in the near-hole region:
     annular zone within 2 * (d_over_W / 2) = d_over_W of the hole center.
  4. Peak error: predicted field peak vs true gridded peak (%), mean and max.

Outputs:
  - results/field_diagnostics.csv
  - results/field_interpolation_vs_d_over_W.png
  - Printed terminal summary table
"""
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_datasets_and_predictions(
    fields_npz_path="data/fields.npz",
    pod_npz_path="results/pod_predictions.npz",
    nn_npz_path="results/nn_predictions.npz",
):
    """
    Loads raw ground truth fields, geometry arrays, and surrogate predictions.
    """
    for p in [fields_npz_path, pod_npz_path, nn_npz_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    data = np.load(fields_npz_path)
    pod = np.load(pod_npz_path)
    nn = np.load(nn_npz_path)

    return data, pod, nn


def build_split_indices(d_over_W, seed=42):
    """
    Reconstructs the identical split indices for Interpolation and Extrapolation.
    """
    N = len(d_over_W)

    # Split A: Interpolation (random 70/15/15)
    idx = np.arange(N)
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.30, random_state=seed)
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, random_state=seed)

    # Split B: Extrapolation (train d/W < 0.40, test d/W >= 0.45)
    tr_ext_all = np.where(d_over_W < 0.40)[0]
    te_ext_idx = np.where(d_over_W >= 0.45)[0]
    sorted_ext = tr_ext_all[np.argsort(d_over_W[tr_ext_all])]
    n_ext_val = int(len(sorted_ext) * 0.15)
    tr_ext_idx = sorted_ext[:-n_ext_val]
    va_ext_idx = sorted_ext[-n_ext_val:]

    return {
        "interp": {"tr": idx_tr, "va": idx_va, "te": idx_te},
        "extrap": {"tr": tr_ext_idx, "va": va_ext_idx, "te": te_ext_idx},
    }


def compute_pixelwise_training_mean(fields, masks, train_idx):
    """
    Computes the pixelwise mean of the training fields over samples where that pixel is valid.
    """
    tr_fields = fields[train_idx]
    tr_masks = masks[train_idx]
    sum_tr = np.sum(tr_fields * tr_masks, axis=0)
    count_tr = np.sum(tr_masks, axis=0)

    mean_field = np.zeros_like(sum_tr, dtype=np.float32)
    valid_pixels = count_tr > 0
    mean_field[valid_pixels] = sum_tr[valid_pixels] / count_tr[valid_pixels]

    return mean_field


def evaluate_trivial_predictors(true_fields, masks, mean_train_field):
    """
    Computes relative L2 error (%) for:
      (a) Uniform field equal to 1.0
      (b) Mean training field
    """
    N = len(true_fields)
    u_err = np.zeros(N, dtype=np.float64)
    m_err = np.zeros(N, dtype=np.float64)

    for i in range(N):
        t = true_fields[i]
        m = masks[i]
        t_valid = t[m]
        norm_t = np.linalg.norm(t_valid)

        if norm_t > 0:
            # (a) Uniform = 1.0
            diff_u = 1.0 - t_valid
            u_err[i] = (np.linalg.norm(diff_u) / norm_t) * 100.0

            # (b) Mean training field
            diff_m = mean_train_field[m] - t_valid
            m_err[i] = (np.linalg.norm(diff_m) / norm_t) * 100.0

    return {
        "uniform_rel_l2_%": u_err,
        "mean_train_rel_l2_%": m_err,
        "uniform_mean": float(np.mean(u_err)),
        "uniform_max": float(np.max(u_err)),
        "mean_train_mean": float(np.mean(m_err)),
        "mean_train_max": float(np.max(m_err)),
    }


def evaluate_model_diagnostics(pred_fields, true_fields, masks, d_over_W, xi, eta):
    """
    Computes all 4 diagnostic metrics for a specific model prediction array:
      - Full-field relative L2 error (%)
      - Perturbation relative L2 error (%) : (pred - 1) vs (true - 1)
      - Near-hole relative L2 error (%)     : within 2 * (d/W / 2) = d/W of hole center
      - Peak error (%)                      : |pred_max - true_max| / true_max
    """
    N = len(pred_fields)
    XI, ETA = np.meshgrid(xi, eta)
    R_sq = XI ** 2 + ETA ** 2

    full_rel_l2 = np.zeros(N, dtype=np.float64)
    pert_rel_l2 = np.zeros(N, dtype=np.float64)
    near_rel_l2 = np.zeros(N, dtype=np.float64)
    peak_errors = np.zeros(N, dtype=np.float64)

    for i in range(N):
        p = pred_fields[i]
        t = true_fields[i]
        m = masks[i]
        dW = float(d_over_W[i])

        p_m = p[m]
        t_m = t[m]
        norm_t = np.linalg.norm(t_m)
        diff = p_m - t_m

        # 0. Full field relative L2 error
        if norm_t > 0:
            full_rel_l2[i] = (np.linalg.norm(diff) / norm_t) * 100.0

        # 2. Perturbation error: (pred - 1) vs (true - 1)
        pert_t = t_m - 1.0
        norm_pert_t = np.linalg.norm(pert_t)
        if norm_pert_t > 0:
            pert_rel_l2[i] = (np.linalg.norm(diff) / norm_pert_t) * 100.0

        # 3. Near-hole region: radius <= 2 * (d_over_W / 2) = d_over_W
        r_outer_sq = dW ** 2
        m_near = m & (R_sq <= r_outer_sq)
        t_near = t[m_near]
        p_near = p[m_near]
        norm_t_near = np.linalg.norm(t_near)
        if norm_t_near > 0:
            near_rel_l2[i] = (np.linalg.norm(p_near - t_near) / norm_t_near) * 100.0

        # 4. Peak stress error
        p_pk = p_m.max() if len(p_m) > 0 else 0.0
        t_pk = t_m.max() if len(t_m) > 0 else 1.0
        peak_errors[i] = (abs(p_pk - t_pk) / t_pk) * 100.0

    return {
        "full_rel_l2_%": full_rel_l2,
        "pert_rel_l2_%": pert_rel_l2,
        "near_rel_l2_%": near_rel_l2,
        "peak_err_%": peak_errors,
        "full_mean": float(np.mean(full_rel_l2)),
        "full_max": float(np.max(full_rel_l2)),
        "pert_mean": float(np.mean(pert_rel_l2)),
        "pert_max": float(np.max(pert_rel_l2)),
        "near_mean": float(np.mean(near_rel_l2)),
        "near_max": float(np.max(near_rel_l2)),
        "peak_mean": float(np.mean(peak_errors)),
        "peak_max": float(np.max(peak_errors)),
    }


def plot_interpolation_error_vs_geometry(
    d_over_W, pod_errors, nn_errors, out_path="results/field_interpolation_vs_d_over_W.png"
):
    """
    Plots relative L2 error (%) versus d_over_W for both POD+GP and CNN-Deconv NN
    on the interpolation test set.
    """
    sort_idx = np.argsort(d_over_W)
    d_sorted = d_over_W[sort_idx]
    pod_sorted = pod_errors[sort_idx]
    nn_sorted = nn_errors[sort_idx]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Scatter points for individual test samples
    ax.scatter(d_over_W, pod_errors, color="#1f77b4", alpha=0.65, s=35,
               edgecolors="none", label="POD+GP samples")
    ax.scatter(d_over_W, nn_errors, color="#ff7f0e", alpha=0.65, s=35,
               edgecolors="none", label="CNN-Deconv NN samples")

    # Polynomial trend curves (degree 2)
    p_poly = np.poly1d(np.polyfit(d_sorted, pod_sorted, deg=2))
    n_poly = np.poly1d(np.polyfit(d_sorted, nn_sorted, deg=2))

    d_dense = np.linspace(d_sorted.min(), d_sorted.max(), 200)
    ax.plot(d_dense, p_poly(d_dense), color="#08519c", linewidth=2.5,
            label="POD+GP trend")
    ax.plot(d_dense, n_poly(d_dense), color="#d94801", linewidth=2.5, linestyle="--",
            label="CNN-Deconv NN trend")

    ax.set_xlabel(r"Hole Diameter Ratio $d/W$", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Relative $L_2$ Field Error (%)", fontsize=12, fontweight="bold")
    ax.set_title(r"Interpolation Accuracy vs. Geometric Parameter $d/W$ (Random 70/15/15 Split)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.95)
    ax.set_ylim(0, max(max(pod_errors), max(nn_errors)) * 1.20)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    alt_path = out_path.replace("_vs_d_over_W.png", ".png")
    if alt_path != out_path:
        fig.savefig(alt_path, dpi=150)
    plt.close(fig)
    print(f"Saved interpolation error plot to {out_path}.")


def run_diagnostics(
    fields_npz_path="data/fields.npz",
    pod_npz_path="results/pod_predictions.npz",
    nn_npz_path="results/nn_predictions.npz",
    out_csv_path="results/field_diagnostics.csv",
    plot_path="results/field_interpolation_vs_d_over_W.png",
):
    """
    Coordinates the full diagnostic evaluation across both splits.
    """
    data, pod, nn = load_datasets_and_predictions(fields_npz_path, pod_npz_path, nn_npz_path)

    fields = data["fields"]
    masks = data["masks"]
    d_over_W = data["d_over_W"]
    xi = data["xi"]
    eta = data["eta"]

    splits = build_split_indices(d_over_W, seed=42)

    rows = []
    interp_errors = {}

    for split_key in ["interp", "extrap"]:
        te_idx = splits[split_key]["te"]
        tr_idx = splits[split_key]["tr"]

        true_te = fields[te_idx]
        mask_te = masks[te_idx]
        dW_te = d_over_W[te_idx]

        # 1. Trivial predictors
        mean_tr_field = compute_pixelwise_training_mean(fields, masks, tr_idx)
        triv_res = evaluate_trivial_predictors(true_te, mask_te, mean_tr_field)

        # Baseline: Uniform = 1.0
        rows.append({
            "split": split_key,
            "model": "Uniform Field (1.0)",
            "mean_rel_l2_%": triv_res["uniform_mean"],
            "max_rel_l2_%": triv_res["uniform_max"],
            "pert_rel_l2_mean_%": 100.0,
            "pert_rel_l2_max_%": 100.0,
            "near_hole_rel_l2_mean_%": float(np.mean([
                np.linalg.norm(1.0 - true_te[i][mask_te[i] & (np.meshgrid(xi, eta)[0]**2 + np.meshgrid(xi, eta)[1]**2 <= dW_te[i]**2)])
                / np.linalg.norm(true_te[i][mask_te[i] & (np.meshgrid(xi, eta)[0]**2 + np.meshgrid(xi, eta)[1]**2 <= dW_te[i]**2)]) * 100.0
                for i in range(len(te_idx))
            ])),
            "near_hole_rel_l2_max_%": float(np.max([
                np.linalg.norm(1.0 - true_te[i][mask_te[i] & (np.meshgrid(xi, eta)[0]**2 + np.meshgrid(xi, eta)[1]**2 <= dW_te[i]**2)])
                / np.linalg.norm(true_te[i][mask_te[i] & (np.meshgrid(xi, eta)[0]**2 + np.meshgrid(xi, eta)[1]**2 <= dW_te[i]**2)]) * 100.0
                for i in range(len(te_idx))
            ])),
            "peak_err_mean_%": float(np.mean([
                abs(1.0 - true_te[i][mask_te[i]].max()) / true_te[i][mask_te[i]].max() * 100.0
                for i in range(len(te_idx))
            ])),
            "peak_err_max_%": float(np.max([
                abs(1.0 - true_te[i][mask_te[i]].max()) / true_te[i][mask_te[i]].max() * 100.0
                for i in range(len(te_idx))
            ])),
        })

        # Baseline: Mean Training Field
        rows.append({
            "split": split_key,
            "model": "Mean Training Field",
            "mean_rel_l2_%": triv_res["mean_train_mean"],
            "max_rel_l2_%": triv_res["mean_train_max"],
            "pert_rel_l2_mean_%": float(np.mean([
                np.linalg.norm(mean_tr_field[mask_te[i]] - true_te[i][mask_te[i]])
                / np.linalg.norm(true_te[i][mask_te[i]] - 1.0) * 100.0
                for i in range(len(te_idx))
            ])),
            "pert_rel_l2_max_%": float(np.max([
                np.linalg.norm(mean_tr_field[mask_te[i]] - true_te[i][mask_te[i]])
                / np.linalg.norm(true_te[i][mask_te[i]] - 1.0) * 100.0
                for i in range(len(te_idx))
            ])),
            "near_hole_rel_l2_mean_%": float(np.mean([
                np.linalg.norm(mean_tr_field[mask_te[i] & (np.meshgrid(xi, eta)[0]**2 + np.meshgrid(xi, eta)[1]**2 <= dW_te[i]**2)]
                               - true_te[i][mask_te[i] & (np.meshgrid(xi, eta)[0]**2 + np.meshgrid(xi, eta)[1]**2 <= dW_te[i]**2)])
                / np.linalg.norm(true_te[i][mask_te[i] & (np.meshgrid(xi, eta)[0]**2 + np.meshgrid(xi, eta)[1]**2 <= dW_te[i]**2)]) * 100.0
                for i in range(len(te_idx))
            ])),
            "near_hole_rel_l2_max_%": float(np.max([
                np.linalg.norm(mean_tr_field[mask_te[i] & (np.meshgrid(xi, eta)[0]**2 + np.meshgrid(xi, eta)[1]**2 <= dW_te[i]**2)]
                               - true_te[i][mask_te[i] & (np.meshgrid(xi, eta)[0]**2 + np.meshgrid(xi, eta)[1]**2 <= dW_te[i]**2)])
                / np.linalg.norm(true_te[i][mask_te[i] & (np.meshgrid(xi, eta)[0]**2 + np.meshgrid(xi, eta)[1]**2 <= dW_te[i]**2)]) * 100.0
                for i in range(len(te_idx))
            ])),
            "peak_err_mean_%": float(np.mean([
                abs(mean_tr_field[mask_te[i]].max() - true_te[i][mask_te[i]].max()) / true_te[i][mask_te[i]].max() * 100.0
                for i in range(len(te_idx))
            ])),
            "peak_err_max_%": float(np.max([
                abs(mean_tr_field[mask_te[i]].max() - true_te[i][mask_te[i]].max()) / true_te[i][mask_te[i]].max() * 100.0
                for i in range(len(te_idx))
            ])),
        })

        # Models: POD+GP and CNN-Deconv NN
        for mod_label, npz_data in [("POD + GP", pod), ("CNN-Deconv NN", nn)]:
            pred_te = npz_data[f"{split_key}_pred_fields"]
            diag = evaluate_model_diagnostics(pred_te, true_te, mask_te, dW_te, xi, eta)

            rows.append({
                "split": split_key,
                "model": mod_label,
                "mean_rel_l2_%": diag["full_mean"],
                "max_rel_l2_%": diag["full_max"],
                "pert_rel_l2_mean_%": diag["pert_mean"],
                "pert_rel_l2_max_%": diag["pert_max"],
                "near_hole_rel_l2_mean_%": diag["near_mean"],
                "near_hole_rel_l2_max_%": diag["near_max"],
                "peak_err_mean_%": diag["peak_mean"],
                "peak_err_max_%": diag["peak_max"],
            })

            if split_key == "interp":
                interp_errors[mod_label] = diag["full_rel_l2_%"]

    # Build DataFrame
    df_diag = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    df_diag.to_csv(out_csv_path, index=False)
    print(f"Saved diagnostics summary table to {out_csv_path}.")

    # Generate interpolation error vs d_over_W plot
    plot_interpolation_error_vs_geometry(
        d_over_W=d_over_W[splits["interp"]["te"]],
        pod_errors=interp_errors["POD + GP"],
        nn_errors=interp_errors["CNN-Deconv NN"],
        out_path=plot_path,
    )

    # Print formatted table
    print("\n" + "=" * 110)
    print("STRESS FIELD SURROGATE DIAGNOSTICS & TRIVIAL PREDICTOR BENCHMARKS")
    print("=" * 110)
    fmt_df = df_diag.copy()
    for col in fmt_df.columns:
        if col not in ["split", "model"]:
            fmt_df[col] = fmt_df[col].map("{:7.3f}%".format)
    print(fmt_df.to_string(index=False))
    print("=" * 110 + "\n")

    return df_diag


def main():
    parser = argparse.ArgumentParser(description="Compute stress field surrogate diagnostics.")
    parser.add_argument("--fields_npz", default="data/fields.npz", help="Path to fields.npz")
    parser.add_argument("--pod_npz", default="results/pod_predictions.npz", help="Path to pod_predictions.npz")
    parser.add_argument("--nn_npz", default="results/nn_predictions.npz", help="Path to nn_predictions.npz")
    parser.add_argument("--out_csv", default="results/field_diagnostics.csv", help="Path to output CSV")
    parser.add_argument("--plot", default="results/field_interpolation_vs_d_over_W.png", help="Path to output plot")
    args = parser.parse_args()

    run_diagnostics(
        fields_npz_path=args.fields_npz,
        pod_npz_path=args.pod_npz,
        nn_npz_path=args.nn_npz,
        out_csv_path=args.out_csv,
        plot_path=args.plot,
    )


if __name__ == "__main__":
    main()
