"""
Comparison of 2D Stress Field Surrogates (POD+GP vs. PyTorch CNN-Deconv)
and Scalar Baseline Models.

Reads:
  - results/pod_results.csv
  - results/nn_results.csv
  - results/baseline_results.csv
  - results/pod_predictions.npz
  - results/nn_predictions.npz
  - data/plate_hole.csv (for FEA reference timing)

Produces:
  - results/field_comparison.csv
  - results/field_comparison_bars.png (Bar chart of mean relative L2 error on both splits)
  - results/field_extrapolation_vs_d_over_W.png (Extrapolation error vs d/W for both models)

Also prints exact latency and speedup comparisons against mean FEA time (t_mesh + t_solve),
explicitly distinguishing between single-sample and batched evaluations.
"""
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_all_results(
    pod_csv_path="results/pod_results.csv",
    nn_csv_path="results/nn_results.csv",
    base_csv_path="results/baseline_results.csv",
    pod_npz_path="results/pod_predictions.npz",
    nn_npz_path="results/nn_predictions.npz",
    fea_csv_path="data/plate_hole.csv"
):
    """
    Loads all result metrics, prediction archives, and FEA execution times.
    """
    for p in [pod_csv_path, nn_csv_path, base_csv_path, pod_npz_path, nn_npz_path, fea_csv_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required result file not found: {p}")

    df_pod = pd.read_csv(pod_csv_path)
    df_nn = pd.read_csv(nn_csv_path)
    df_base = pd.read_csv(base_csv_path)

    npz_pod = np.load(pod_npz_path)
    npz_nn = np.load(nn_npz_path)

    df_fea = pd.read_csv(fea_csv_path)
    df_fea_ok = df_fea[df_fea["ok"] == True]

    t_mesh_mean = float(df_fea_ok["t_mesh"].mean())
    t_solve_mean = float(df_fea_ok["t_solve"].mean())
    t_fea_mean_s = t_mesh_mean + t_solve_mean
    t_fea_mean_us = t_fea_mean_s * 1e6

    fea_timing = {
        "t_mesh_s": t_mesh_mean,
        "t_solve_s": t_solve_mean,
        "t_fea_mean_s": t_fea_mean_s,
        "t_fea_mean_ms": t_fea_mean_s * 1e3,
        "t_fea_mean_us": t_fea_mean_us,
        "n_samples": len(df_fea_ok),
    }

    return df_pod, df_nn, df_base, npz_pod, npz_nn, fea_timing


def compute_per_sample_errors(npz_data, split_prefix="extrap"):
    """
    Computes per-sample relative L2 error (%) over valid plate pixels.
    """
    pred_fields = npz_data[f"{split_prefix}_pred_fields"]
    true_fields = npz_data[f"{split_prefix}_true_fields"]
    masks = npz_data[f"{split_prefix}_masks"]
    d_over_W = npz_data[f"{split_prefix}_d_over_W"]
    H_over_W = npz_data[f"{split_prefix}_H_over_W"]
    ids = npz_data[f"{split_prefix}_ids"]

    N = len(pred_fields)
    rel_l2_errors = np.zeros(N, dtype=np.float64)

    for i in range(N):
        m = masks[i]
        diff = pred_fields[i][m] - true_fields[i][m]
        norm_diff = np.linalg.norm(diff)
        norm_true = np.linalg.norm(true_fields[i][m])
        rel_l2_errors[i] = (norm_diff / norm_true) * 100.0 if norm_true > 0 else 0.0

    return {
        "ids": ids,
        "d_over_W": d_over_W,
        "H_over_W": H_over_W,
        "rel_l2_errors": rel_l2_errors,
    }


def build_comparison_table(df_pod, df_nn, df_base, fea_timing, out_csv_path="results/field_comparison.csv"):
    """
    Constructs a unified comparison DataFrame of field surrogates and scalar baselines
    including exact latency speedup factors vs. mean FEA time.
    """
    t_fea_us = fea_timing["t_fea_mean_us"]

    rows = []

    # 1. Field Models (POD+GP and NN)
    for split in ["interp", "extrap"]:
        # POD row
        p_row = df_pod[df_pod["split"] == split].iloc[0]
        s_single = float(p_row["single_sample_us"])
        s_batch = float(p_row["batched_us_per_sample"])
        rows.append({
            "split": split,
            "category": "Field Surrogate",
            "model": "POD + GP (20 modes)",
            "mean_rel_l2_%": float(p_row["mean_rel_l2_%"]),
            "max_rel_l2_%": float(p_row["max_rel_l2_%"]),
            "mean_abs_pixel_err": float(p_row["mean_abs_pixel_err"]),
            "mean_peak_err_%": float(p_row["mean_peak_err_%"]),
            "max_peak_err_%": float(p_row["max_peak_err_%"]),
            "single_sample_us": s_single,
            "batched_us_per_sample": s_batch,
            "speedup_single_vs_fea": t_fea_us / s_single,
            "speedup_batched_vs_fea": t_fea_us / s_batch,
        })

        # NN row
        n_row = df_nn[df_nn["split"] == split].iloc[0]
        s_single = float(n_row["single_sample_us"])
        s_batch = float(n_row["batched_us_per_sample"])
        rows.append({
            "split": split,
            "category": "Field Surrogate",
            "model": "CNN-Deconv NN",
            "mean_rel_l2_%": float(n_row["mean_rel_l2_%"]),
            "max_rel_l2_%": float(n_row["max_rel_l2_%"]),
            "mean_abs_pixel_err": float(n_row["mean_abs_pixel_err"]),
            "mean_peak_err_%": float(n_row["mean_peak_err_%"]),
            "max_peak_err_%": float(n_row["max_peak_err_%"]),
            "single_sample_us": s_single,
            "batched_us_per_sample": s_batch,
            "speedup_single_vs_fea": t_fea_us / s_single,
            "speedup_batched_vs_fea": t_fea_us / s_batch,
        })

    # 2. Scalar Baseline Models (predicting Kt directly)
    for _, b_row in df_base.iterrows():
        split = b_row["split"]
        m_name = b_row["model"]
        p_lat = float(b_row["predict_us_per_sample"])
        rows.append({
            "split": split,
            "category": "Scalar Baseline (Kt only)",
            "model": f"{m_name}",
            "mean_rel_l2_%": np.nan,
            "max_rel_l2_%": np.nan,
            "mean_abs_pixel_err": float(b_row["MAE"]),
            "mean_peak_err_%": float(b_row["mean_rel_err_%"]),
            "max_peak_err_%": float(b_row["max_rel_err_%"]),
            "single_sample_us": p_lat,
            "batched_us_per_sample": p_lat,
            "speedup_single_vs_fea": t_fea_us / p_lat,
            "speedup_batched_vs_fea": t_fea_us / p_lat,
        })

    comp_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    comp_df.to_csv(out_csv_path, index=False)
    print(f"Saved comprehensive field comparison table to {out_csv_path}.")
    return comp_df


def plot_bar_chart_l2_errors(df_pod, df_nn, out_path="results/field_comparison_bars.png"):
    """
    Figure (1): Grouped bar chart of mean per-sample relative L2 error
    for POD+GP vs NN across both splits.
    """
    splits = ["Interpolation\n(Random 70/15/15)", "Extrapolation\n(d/W >= 0.45)"]
    split_keys = ["interp", "extrap"]

    pod_errors = [
        float(df_pod[df_pod["split"] == k]["mean_rel_l2_%"].iloc[0])
        for k in split_keys
    ]
    nn_errors = [
        float(df_nn[df_nn["split"] == k]["mean_rel_l2_%"].iloc[0])
        for k in split_keys
    ]

    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    bars1 = ax.bar(x - width / 2, pod_errors, width, label="POD + GP (20 modes)",
                   color="#1f77b4", edgecolor="black", linewidth=1.2, alpha=0.9)
    bars2 = ax.bar(x + width / 2, nn_errors, width, label="CNN-Deconv NN",
                   color="#ff7f0e", edgecolor="black", linewidth=1.2, alpha=0.9)

    # Label values on bars
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Mean Relative $L_2$ Error on Field (%)", fontsize=12, fontweight="bold")
    ax.set_title("Stress Field Surrogate Accuracy: POD+GP vs. CNN-Deconv NN",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=11, fontweight="semibold")
    ax.set_ylim(0, max(max(pod_errors), max(nn_errors)) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(fontsize=11, loc="upper left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    # Also save with secondary name if useful
    alt_path = out_path.replace("_bars.png", "_l2.png")
    fig.savefig(alt_path, dpi=150)
    plt.close(fig)
    print(f"Saved Figure (1) bar chart to {out_path} and {alt_path}.")


def plot_extrapolation_error_vs_geometry(
    pod_extrap_dict, nn_extrap_dict, out_path="results/field_extrapolation_vs_d_over_W.png"
):
    """
    Figure (2): Relative L2 error (%) versus d_over_W on the extrapolation test set
    for both POD+GP and NN models.
    """
    d_W = pod_extrap_dict["d_over_W"]
    pod_err = pod_extrap_dict["rel_l2_errors"]
    nn_err = nn_extrap_dict["rel_l2_errors"]

    # Sort by d_over_W for smooth plotting
    sort_idx = np.argsort(d_W)
    d_W_sorted = d_W[sort_idx]
    pod_err_sorted = pod_err[sort_idx]
    nn_err_sorted = nn_err[sort_idx]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Scatter points for individual samples
    ax.scatter(d_W, pod_err, color="#1f77b4", alpha=0.55, edgecolors="none", s=35,
               label="POD+GP samples")
    ax.scatter(d_W, nn_err, color="#ff7f0e", alpha=0.55, edgecolors="none", s=35,
               label="CNN-Deconv NN samples")

    # Fit quadratic trend lines to clearly show error behavior vs d/W
    p_poly = np.poly1d(np.polyfit(d_W_sorted, pod_err_sorted, deg=2))
    n_poly = np.poly1d(np.polyfit(d_W_sorted, nn_err_sorted, deg=2))

    d_dense = np.linspace(d_W_sorted.min(), d_W_sorted.max(), 200)
    ax.plot(d_dense, p_poly(d_dense), color="#08519c", linewidth=2.5,
            label="POD+GP trend")
    ax.plot(d_dense, n_poly(d_dense), color="#d94801", linewidth=2.5, linestyle="--",
            label="CNN-Deconv NN trend")

    # Annotate boundary thresholds
    ax.axvline(0.40, color="green", linestyle=":", linewidth=1.5,
               label="Training cutoff ($d/W = 0.40$)")
    ax.axvline(0.45, color="red", linestyle=":", linewidth=1.5,
               label="Test boundary ($d/W = 0.45$)")

    ax.set_xlabel(r"Hole Diameter Ratio $d/W$", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Relative $L_2$ Field Error (%)", fontsize=12, fontweight="bold")
    ax.set_title(r"Extrapolation Error vs. Geometric Parameter $d/W$ ($d/W \geq 0.45$)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95)

    # Set x limits to show the gap between 0.40 and 0.45
    ax.set_xlim(0.38, d_W.max() + 0.01)
    ax.set_ylim(0, max(pod_err.max(), nn_err.max()) * 1.15)

    # Shaded annotation for the extrapolation test region
    ax.axvspan(0.45, d_W.max() + 0.01, alpha=0.08, color="salmon")
    ax.text(0.46, ax.get_ylim()[1] * 0.92, "Extrapolation Test Region",
            fontsize=10, fontstyle="italic", color="darkred")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    alt_path = out_path.replace("_vs_d_over_W.png", "_error.png")
    fig.savefig(alt_path, dpi=150)
    plt.close(fig)
    print(f"Saved Figure (2) extrapolation error plot to {out_path} and {alt_path}.")


def print_speed_comparison(comp_df, fea_timing):
    """
    Prints the execution speed of each surrogate model compared with
    the mean FEA time per run from data/plate_hole.csv (t_mesh + t_solve),
    explicitly stating whether each timing number is single-sample or batched.
    """
    t_mesh = fea_timing["t_mesh_s"]
    t_solve = fea_timing["t_solve_s"]
    t_fea_s = fea_timing["t_fea_mean_s"]
    t_fea_ms = fea_timing["t_fea_mean_ms"]
    t_fea_us = fea_timing["t_fea_mean_us"]
    n_fea = fea_timing["n_samples"]

    print("\n" + "=" * 80)
    print("SURROGATE MODEL SPEED & SPEEDUP COMPARISON AGAINST FINITE ELEMENT ANALYSIS (FEA)")
    print("=" * 80)
    print(f"Reference FEA Time (Mean of {n_fea} successful simulations in data/plate_hole.csv):")
    print(f"  - Mesh Generation (t_mesh) : {t_mesh * 1e3:8.2f} ms ({t_mesh * 1e6:10,.1f} us)")
    print(f"  - Linear Solve    (t_solve): {t_solve * 1e3:8.2f} ms ({t_solve * 1e6:10,.1f} us)")
    print(f"  - Total FEA Time  (t_total): {t_fea_ms:8.2f} ms ({t_fea_us:10,.1f} us)")
    print("-" * 80)

    print("\n1. 2D Stress Field Surrogates (Full Field Prediction):")
    field_df = comp_df[comp_df["category"] == "Field Surrogate"]
    for _, row in field_df.iterrows():
        split = row["split"]
        model = row["model"]
        s_us = row["single_sample_us"]
        b_us = row["batched_us_per_sample"]
        sp_single = row["speedup_single_vs_fea"]
        sp_batch = row["speedup_batched_vs_fea"]

        print(f"\n  [{split.upper()}] {model}:")
        print(f"    * Single-Sample Latency : {s_us:9,.1f} us ({s_us / 1e3:6.3f} ms) -> {sp_single:8,.1f}x faster than FEA (single-sample)")
        print(f"    * Batched Latency       : {b_us:9,.1f} us ({b_us / 1e3:6.3f} ms) -> {sp_batch:8,.1f}x faster than FEA (batched)")

    print("\n" + "-" * 80)
    print("2. Scalar Baseline Models (Scalar Stress Concentration Factor Kt only):")
    base_df = comp_df[comp_df["category"] == "Scalar Baseline (Kt only)"]
    for _, row in base_df.iterrows():
        split = row["split"]
        model = row["model"]
        lat = row["single_sample_us"]
        sp = row["speedup_single_vs_fea"]
        print(f"  [{split.upper():6s}] {model:14s} : {lat:7.2f} us ({lat / 1e3:7.4f} ms) -> {sp:10,.1f}x faster than FEA (batched/single)")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare 2D stress field surrogates and baselines.")
    parser.add_argument("--pod_csv", default="results/pod_results.csv", help="Path to pod_results.csv")
    parser.add_argument("--nn_csv", default="results/nn_results.csv", help="Path to nn_results.csv")
    parser.add_argument("--base_csv", default="results/baseline_results.csv", help="Path to baseline_results.csv")
    parser.add_argument("--pod_npz", default="results/pod_predictions.npz", help="Path to pod_predictions.npz")
    parser.add_argument("--nn_npz", default="results/nn_predictions.npz", help="Path to nn_predictions.npz")
    parser.add_argument("--fea_csv", default="data/plate_hole.csv", help="Path to plate_hole.csv")
    parser.add_argument("--out_csv", default="results/field_comparison.csv", help="Path to output CSV")
    parser.add_argument("--fig1", default="results/field_comparison_bars.png", help="Path to Figure 1 bar chart")
    parser.add_argument("--fig2", default="results/field_extrapolation_vs_d_over_W.png", help="Path to Figure 2 plot")
    args = parser.parse_args()

    # Load results
    df_pod, df_nn, df_base, npz_pod, npz_nn, fea_timing = load_all_results(
        pod_csv_path=args.pod_csv,
        nn_csv_path=args.nn_csv,
        base_csv_path=args.base_csv,
        pod_npz_path=args.pod_npz,
        nn_npz_path=args.nn_npz,
        fea_csv_path=args.fea_csv,
    )

    # Compute per-sample extrapolation errors
    pod_extrap_dict = compute_per_sample_errors(npz_pod, split_prefix="extrap")
    nn_extrap_dict = compute_per_sample_errors(npz_nn, split_prefix="extrap")

    # Build and save comparison CSV
    comp_df = build_comparison_table(df_pod, df_nn, df_base, fea_timing, out_csv_path=args.out_csv)

    # Generate Figure (1): Bar chart
    plot_bar_chart_l2_errors(df_pod, df_nn, out_path=args.fig1)

    # Generate Figure (2): Extrapolation error vs d/W
    plot_extrapolation_error_vs_geometry(pod_extrap_dict, nn_extrap_dict, out_path=args.fig2)

    # Print speed comparison
    print_speed_comparison(comp_df, fea_timing)


if __name__ == "__main__":
    main()
