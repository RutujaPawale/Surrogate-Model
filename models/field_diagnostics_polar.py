"""
Diagnostics and Benchmarks for 2D Stress Field Surrogates on Polar Grid.

Loads:
  - data/fields_polar.npz
  - results/pod_predictions_polar.npz
  - results/nn_predictions_polar.npz
  - (optional for comparison) Cartesian predictions from results/pod_predictions.npz and results/nn_predictions.npz

Computes on polar grid:
  1. Full-field relative L2 error (%) [mean, max]
  2. Perturbation relative L2 error (%) [mean, max] : (pred - 1) vs (true - 1)
  3. Peak error vs true gridded peak (%) [mean, max]
  4. Peak error vs kt_gross from CSV (%) [mean, max]
  For:
    - Uniform Field (1.0) trivial baseline
    - Pixelwise Mean Training Field trivial baseline
    - POD + GP surrogate
    - CNN-Deconv NN surrogate
  Across both splits:
    - Interpolation (random 70/15/15)
    - Extrapolation (train d/W < 0.40, test d/W >= 0.45)

Saves:
  - results/field_diagnostics_polar.csv

Prints:
  - Formatted diagnostics summary table
  - 10 worst test samples by perturbation error on BOTH Cartesian and Polar datasets
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_polar_datasets_and_predictions(
    polar_npz_path="data/fields_polar.npz",
    pod_npz_path="results/pod_predictions_polar.npz",
    nn_npz_path="results/nn_predictions_polar.npz",
):
    """
    Loads ground truth polar fields and surrogate predictions.
    """
    for p in [polar_npz_path, pod_npz_path, nn_npz_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    data = np.load(polar_npz_path)
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


def compute_pixelwise_training_mean(fields, train_idx):
    """
    Computes the pixelwise mean of training fields across all valid pixels (all are valid in polar).
    """
    return np.mean(fields[train_idx], axis=0)


def evaluate_polar_metrics(pred_fields, true_fields, kt_gross_te):
    """
    Computes:
      - full_rel_l2_%
      - pert_rel_l2_%
      - peak_err_grid_%
      - peak_err_kt_%
    """
    N = len(pred_fields)
    full_err = np.zeros(N, dtype=np.float64)
    pert_err = np.zeros(N, dtype=np.float64)
    pk_grid_err = np.zeros(N, dtype=np.float64)
    pk_kt_err = np.zeros(N, dtype=np.float64)

    for i in range(N):
        p = pred_fields[i]
        t = true_fields[i]
        diff = p - t
        norm_t = np.linalg.norm(t)
        pert_t = t - 1.0
        norm_pert_t = np.linalg.norm(pert_t)
        kt = float(kt_gross_te[i])

        # Full field error
        if norm_t > 0:
            full_err[i] = (np.linalg.norm(diff) / norm_t) * 100.0

        # Perturbation error
        if norm_pert_t > 0:
            pert_err[i] = (np.linalg.norm(diff) / norm_pert_t) * 100.0

        # Peak error vs true gridded peak
        p_pk = float(p.max())
        t_pk = float(t.max())
        if t_pk > 0:
            pk_grid_err[i] = (abs(p_pk - t_pk) / t_pk) * 100.0

        # Peak error vs kt_gross from CSV
        if kt > 0:
            pk_kt_err[i] = (abs(p_pk - kt) / kt) * 100.0

    return {
        "full_err": full_err,
        "pert_err": pert_err,
        "pk_grid_err": pk_grid_err,
        "pk_kt_err": pk_kt_err,
        "full_mean": float(np.mean(full_err)),
        "full_max": float(np.max(full_err)),
        "pert_mean": float(np.mean(pert_err)),
        "pert_max": float(np.max(pert_err)),
        "pk_grid_mean": float(np.mean(pk_grid_err)),
        "pk_grid_max": float(np.max(pk_grid_err)),
        "pk_kt_mean": float(np.mean(pk_kt_err)),
        "pk_kt_max": float(np.max(pk_kt_err)),
    }


def get_worst_10_samples(preds, trues, dW, HW, ids, masks=None):
    """
    Extracts the 10 worst samples by perturbation error.
    """
    records = []
    N = len(preds)
    for i in range(N):
        p = preds[i]
        t = trues[i]
        if masks is not None:
            m = masks[i]
            diff = p[m] - t[m]
            pert_t = t[m] - 1.0
            norm_t = np.linalg.norm(t[m])
        else:
            diff = p - t
            pert_t = t - 1.0
            norm_t = np.linalg.norm(t)

        pert_err = (np.linalg.norm(diff) / np.linalg.norm(pert_t)) * 100.0
        full_err = (np.linalg.norm(diff) / norm_t) * 100.0

        records.append({
            "id": int(ids[i]),
            "d_over_W": float(dW[i]),
            "H_over_W": float(HW[i]),
            "pert_err_%": float(pert_err),
            "full_err_%": float(full_err),
        })

    df = pd.DataFrame(records)
    return df.sort_values(by="pert_err_%", ascending=False).head(10).reset_index(drop=True)


def run_polar_diagnostics(
    polar_npz_path="data/fields_polar.npz",
    pod_npz_path="results/pod_predictions_polar.npz",
    nn_npz_path="results/nn_predictions_polar.npz",
    cart_pod_path="results/pod_predictions.npz",
    cart_nn_path="results/nn_predictions.npz",
    out_csv_path="results/field_diagnostics_polar.csv",
):
    """
    Executes polar diagnostics, prints benchmarks, and outputs worst samples comparison.
    """
    data, pod, nn = load_polar_datasets_and_predictions(
        polar_npz_path, pod_npz_path, nn_npz_path
    )

    fields = data["fields"]
    d_over_W = data["d_over_W"]
    kt_gross = data["kt_gross"]

    splits = build_split_indices(d_over_W, seed=42)

    rows = []

    for split_key in ["interp", "extrap"]:
        te_idx = splits[split_key]["te"]
        tr_idx = splits[split_key]["tr"]

        true_te = fields[te_idx]
        kt_te = kt_gross[te_idx]
        N_te = len(te_idx)

        # Baseline 1: Uniform Field (1.0)
        uniform_pred = np.ones_like(true_te)
        u_res = evaluate_polar_metrics(uniform_pred, true_te, kt_te)
        rows.append({
            "split": split_key,
            "model": "Uniform Field (1.0)",
            "full_rel_l2_mean_%": u_res["full_mean"],
            "full_rel_l2_max_%": u_res["full_max"],
            "pert_rel_l2_mean_%": 100.0,
            "pert_rel_l2_max_%": 100.0,
            "peak_err_grid_mean_%": u_res["pk_grid_mean"],
            "peak_err_grid_max_%": u_res["pk_grid_max"],
            "peak_err_kt_mean_%": u_res["pk_kt_mean"],
            "peak_err_kt_max_%": u_res["pk_kt_max"],
        })

        # Baseline 2: Mean Training Field
        mean_tr_field = compute_pixelwise_training_mean(fields, tr_idx)
        mean_pred = np.tile(mean_tr_field[None, :, :], (N_te, 1, 1))
        m_res = evaluate_polar_metrics(mean_pred, true_te, kt_te)
        rows.append({
            "split": split_key,
            "model": "Mean Training Field",
            "full_rel_l2_mean_%": m_res["full_mean"],
            "full_rel_l2_max_%": m_res["full_max"],
            "pert_rel_l2_mean_%": m_res["pert_mean"],
            "pert_rel_l2_max_%": m_res["pert_max"],
            "peak_err_grid_mean_%": m_res["pk_grid_mean"],
            "peak_err_grid_max_%": m_res["pk_grid_max"],
            "peak_err_kt_mean_%": m_res["pk_kt_mean"],
            "peak_err_kt_max_%": m_res["pk_kt_max"],
        })

        # Model 1: POD + GP
        pod_pred = pod[f"{split_key}_pred_fields"]
        p_res = evaluate_polar_metrics(pod_pred, true_te, kt_te)
        rows.append({
            "split": split_key,
            "model": "POD + GP",
            "full_rel_l2_mean_%": p_res["full_mean"],
            "full_rel_l2_max_%": p_res["full_max"],
            "pert_rel_l2_mean_%": p_res["pert_mean"],
            "pert_rel_l2_max_%": p_res["pert_max"],
            "peak_err_grid_mean_%": p_res["pk_grid_mean"],
            "peak_err_grid_max_%": p_res["pk_grid_max"],
            "peak_err_kt_mean_%": p_res["pk_kt_mean"],
            "peak_err_kt_max_%": p_res["pk_kt_max"],
        })

        # Model 2: CNN-Deconv NN
        nn_pred = nn[f"{split_key}_pred_fields"]
        n_res = evaluate_polar_metrics(nn_pred, true_te, kt_te)
        rows.append({
            "split": split_key,
            "model": "CNN-Deconv NN",
            "full_rel_l2_mean_%": n_res["full_mean"],
            "full_rel_l2_max_%": n_res["full_max"],
            "pert_rel_l2_mean_%": n_res["pert_mean"],
            "pert_rel_l2_max_%": n_res["pert_max"],
            "peak_err_grid_mean_%": n_res["pk_grid_mean"],
            "peak_err_grid_max_%": n_res["pk_grid_max"],
            "peak_err_kt_mean_%": n_res["pk_kt_mean"],
            "peak_err_kt_max_%": n_res["pk_kt_max"],
        })

    df_polar_diag = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    df_polar_diag.to_csv(out_csv_path, index=False)
    print(f"Saved polar diagnostics summary table to {out_csv_path}.")

    # Print formatted diagnostics table
    print("\n" + "=" * 125)
    print("POLAR STRESS FIELD SURROGATE DIAGNOSTICS & BENCHMARKS (results/field_diagnostics_polar.csv)")
    print("=" * 125)
    fmt_df = df_polar_diag.copy()
    for col in fmt_df.columns:
        if col not in ["split", "model"]:
            fmt_df[col] = fmt_df[col].map("{:7.3f}%".format)
    print(fmt_df.to_string(index=False))
    print("=" * 125 + "\n")

    # Worst 10 test samples comparison: Cartesian vs. Polar
    if os.path.exists(cart_pod_path) and os.path.exists(cart_nn_path):
        pod_c = np.load(cart_pod_path)
        nn_c = np.load(cart_nn_path)

        for split in ["interp", "extrap"]:
            split_label = "Interpolation (Random 70/15/15)" if split == "interp" else "Extrapolation (d/W >= 0.45)"
            print("\n" + "#" * 90)
            print(f"TOP 10 WORST TEST SAMPLES BY PERTURBATION ERROR: {split_label}")
            print("#" * 90)

            # Cartesian POD+GP
            c_pod_worst = get_worst_10_samples(
                preds=pod_c[f"{split}_pred_fields"],
                trues=pod_c[f"{split}_true_fields"],
                dW=pod_c[f"{split}_d_over_W"],
                HW=pod_c[f"{split}_H_over_W"],
                ids=pod_c[f"{split}_ids"],
                masks=pod_c[f"{split}_masks"],
            )
            # Cartesian NN
            c_nn_worst = get_worst_10_samples(
                preds=nn_c[f"{split}_pred_fields"],
                trues=nn_c[f"{split}_true_fields"],
                dW=nn_c[f"{split}_d_over_W"],
                HW=nn_c[f"{split}_H_over_W"],
                ids=nn_c[f"{split}_ids"],
                masks=nn_c[f"{split}_masks"],
            )
            # Polar POD+GP
            p_pod_worst = get_worst_10_samples(
                preds=pod[f"{split}_pred_fields"],
                trues=pod[f"{split}_true_fields"],
                dW=pod[f"{split}_d_over_W"],
                HW=pod[f"{split}_H_over_W"],
                ids=pod[f"{split}_ids"],
                masks=None,
            )
            # Polar NN
            p_nn_worst = get_worst_10_samples(
                preds=nn[f"{split}_pred_fields"],
                trues=nn[f"{split}_true_fields"],
                dW=nn[f"{split}_d_over_W"],
                HW=nn[f"{split}_H_over_W"],
                ids=nn[f"{split}_ids"],
                masks=None,
            )

            print("\n--- (1) CARTESIAN POD+GP ---")
            print(c_pod_worst.to_string(index=True))
            print("\n--- (2) CARTESIAN CNN-Deconv NN ---")
            print(c_nn_worst.to_string(index=True))
            print("\n--- (3) POLAR POD+GP ---")
            print(p_pod_worst.to_string(index=True))
            print("\n--- (4) POLAR CNN-Deconv NN ---")
            print(p_nn_worst.to_string(index=True))
            print("-" * 90)

    return df_polar_diag


def main():
    parser = argparse.ArgumentParser(description="Compute polar stress field diagnostics.")
    parser.add_argument("--polar_npz", default="data/fields_polar.npz", help="Path to fields_polar.npz")
    parser.add_argument("--pod_npz", default="results/pod_predictions_polar.npz", help="Path to pod_predictions_polar.npz")
    parser.add_argument("--nn_npz", default="results/nn_predictions_polar.npz", help="Path to nn_predictions_polar.npz")
    parser.add_argument("--cart_pod", default="results/pod_predictions.npz", help="Path to cartesian pod_predictions.npz")
    parser.add_argument("--cart_nn", default="results/nn_predictions.npz", help="Path to cartesian nn_predictions.npz")
    parser.add_argument("--out_csv", default="results/field_diagnostics_polar.csv", help="Path to output CSV")
    args = parser.parse_args()

    run_polar_diagnostics(
        polar_npz_path=args.polar_npz,
        pod_npz_path=args.pod_npz,
        nn_npz_path=args.nn_npz,
        cart_pod_path=args.cart_pod,
        cart_nn_path=args.cart_nn,
        out_csv_path=args.out_csv,
    )


if __name__ == "__main__":
    main()
