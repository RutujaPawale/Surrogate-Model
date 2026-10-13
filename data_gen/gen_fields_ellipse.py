"""
Resamples FEA stress fields for a plate with an elliptical hole onto a body-fitted grid.

Body-fitted coordinate grid:
  x = a * cos(phi) * K**s
  y = b * sin(phi) * K**s
  with K = 2.0, s in [0, 1] (64 values), phi in [0, pi/2] (64 values).

Properties:
  - At s = 0: x = a*cos(phi), y = b*sin(phi) (exact elliptical hole boundary).
  - At s = 0, phi = 0: x = a, y = 0 (notch root, location of peak von Mises stress).
  - At s = 1: x = 2a*cos(phi), y = 2b*sin(phi) (outer boundary ring).
  - Since d/W <= 0.50, 2a <= 0.5W, all grid points strictly lie inside the plate domain [0, W/2] x [0, H].
  - No hole mask is needed; all pixels represent solid material.

Fallback handling:
  Boundary discretization chords may place mathematical boundary points slightly outside
  triangular elements; nearest-node fallback via KD-tree is applied and tracked.

Outputs:
  - data/fields_ellipse.npz (ids, d_over_W, ar, H_over_W, kt_gross, fields (N, 64, 64) float32)
  - results/fields_ellipse_samples.png (4 representative fields in physical coordinates)
"""
import argparse
import os
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Ensure project root is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_gen.gen_data_ellipse import solve_plate


def resample_ellipse_geometry(W, H, a, b, sigma, s, phi, K=2.0, n_hole=64):
    """
    Solves one elliptical plate FEA geometry and samples normalized stress (vm / sigma)
    onto the body-fitted grid defined by:
        x = a * cos(phi) * K**s
        y = b * sin(phi) * K**s

    Parameters:
        W (float): Plate width [mm]
        H (float): Plate height [mm]
        a (float): Semi-axis along x [mm]
        b (float): Semi-axis along y [mm]
        sigma (float): Applied gross tensile stress [MPa]
        s (np.ndarray): 1D array of radial parameter values [0, 1]
        phi (np.ndarray): 1D array of angular parameter values [0, pi/2]
        K (float): Radial expansion factor (default 2.0)
        n_hole (int): Number of elements along the hole arc (default 64)

    Returns:
        field (np.ndarray): (64, 64) float32 sampled normalized stress field
        n_fallback (int): Number of points requiring nearest-node fallback
        val_at_root (float): Sampled normalized stress at (s=0, phi=0)
        peak_vm (float): FEA peak von Mises stress [MPa]
        X (np.ndarray): (64, 64) physical x coordinates [mm]
        Y (np.ndarray): (64, 64) physical y coordinates [mm]
    """
    S, PHI = np.meshgrid(s, phi, indexing="ij")  # shape (ns, nphi)
    X = a * np.cos(PHI) * (K ** S)
    Y = b * np.sin(PHI) * (K ** S)

    # Solve FEA with full nodal stress field returned
    res = solve_plate(W, H, a, b, sigma, n_hole=n_hole, return_field=True)
    mesh = res["mesh"]
    vm_nodal = res["vm_nodal"]
    vm_norm = (vm_nodal / sigma).astype(np.float64)

    # Linear interpolation over planar triangulation
    triang = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    interp = mtri.LinearTriInterpolator(triang, vm_norm)
    vals = interp(X, Y)

    # Handle boundary rounding via KD-tree nearest node fallback
    if np.ma.is_masked(vals):
        fallback_mask = vals.mask
    else:
        fallback_mask = np.isnan(vals)

    n_fallback = int(np.sum(fallback_mask))

    if n_fallback > 0:
        tree = cKDTree(mesh.p.T)
        pts_fallback = np.column_stack([X[fallback_mask], Y[fallback_mask]])
        _, nearest_idx = tree.query(pts_fallback)
        vals_clean = np.array(vals, dtype=np.float64)
        vals_clean[fallback_mask] = vm_norm[nearest_idx]
    else:
        vals_clean = np.array(vals, dtype=np.float64)

    field = vals_clean.astype(np.float32)
    val_at_root = float(field[0, 0])

    return field, n_fallback, val_at_root, res["peak_vm"], X, Y


def plot_example_fields(example_samples, out_path="results/fields_ellipse_samples.png"):
    """
    Plots 4 representative fields (flat ellipse, circle, tall ellipse, large hole)
    in physical x-y coordinates.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))

    for ax, ex in zip(axes, example_samples):
        X = ex["X"]
        Y = ex["Y"]
        field = ex["field"]
        a = ex["a"]
        b = ex["b"]
        label = ex["label"]
        ar = ex["ar"]
        d_over_W = ex["d_over_W"]
        kt_gross = ex["kt_gross"]

        # Body-fitted stress field colormesh
        c = ax.pcolormesh(X, Y, field, shading="gouraud", cmap="inferno")

        # Boundary curves: inner hole arc and outer ring s=1
        t = np.linspace(0.0, np.pi / 2.0, 150)
        ax.plot(a * np.cos(t), b * np.sin(t), "w-", lw=2.0, label="Hole boundary ($s=0$)")
        ax.plot(2.0 * a * np.cos(t), 2.0 * b * np.sin(t), "w--", lw=1.2, label="Outer boundary ($s=1$)")

        # Notch root marker at (a, 0)
        ax.plot(a, 0, "ro", markersize=6, label=f"Notch root ($K_t={kt_gross:.2f}$)")

        ax.set_aspect("equal")
        ax.set_xlabel("Physical $x$ [mm]", fontsize=10)
        ax.set_ylabel("Physical $y$ [mm]", fontsize=10)
        ax.set_title(
            f"{label}\n$ar={ar:.2f},\\ d/W={d_over_W:.2f},\\ K_t={kt_gross:.2f}$",
            fontsize=11,
            pad=8,
        )
        ax.grid(True, linestyle=":", alpha=0.4)

        cbar = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\sigma_{\mathrm{vm}} / \sigma$", fontsize=10)

    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.8)
    fig.suptitle("Representative Body-Fitted Elliptical Stress Fields in Physical Coordinates", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Also duplicate to results/field_samples_ellipse.png
    alt_path = "results/field_samples_ellipse.png"
    shutil.copyfile(out_path, alt_path)
    print(f"Saved representative sample plots to {out_path} and {alt_path}")


def generate_ellipse_fields(
    csv_path="data/plate_ellipse.csv",
    out_path="data/fields_ellipse.npz",
    fig_path="results/fields_ellipse_samples.png",
    n_hole=64,
    ns=64,
    nphi=64,
    K=2.0,
    limit=None,
):
    """
    Iterates through all usable geometries in data/plate_ellipse.csv, solves each,
    resamples the stress field onto the body-fitted grid, prints diagnostic checks,
    and saves data/fields_ellipse.npz and example plots.
    """
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    n_total = len(df)
    df_clean = df[df["ok"].astype(str) == "True"].copy().reset_index(drop=True)
    print(f"Loaded {len(df_clean)}/{n_total} usable runs from CSV.")

    if limit is not None and limit > 0:
        df_clean = df_clean.iloc[:limit].copy()
        print(f"Limiting execution to first {len(df_clean)} samples...")

    s = np.linspace(0.0, 1.0, ns, dtype=np.float64)
    phi = np.linspace(0.0, np.pi / 2.0, nphi, dtype=np.float64)

    # Identify 4 target representative geometries for the physical coordinates plot
    idx_flat = int((df_clean["ar"] - 0.50).abs().idxmin())
    idx_circle = int((df_clean["ar"] - 1.00).abs().idxmin())
    idx_tall = int((df_clean["ar"] - 2.00).abs().idxmin())
    idx_large = int((df_clean["d_over_W"] - 0.50).abs().idxmin())

    target_example_indices = {
        idx_flat: "Flat Ellipse (ar ~ 0.5)",
        idx_circle: "Circle (ar ~ 1.0)",
        idx_tall: "Tall Ellipse (ar ~ 2.0)",
        idx_large: "Large Hole (d/W ~ 0.50)",
    }

    example_samples = []

    ids_list = []
    d_over_W_list = []
    ar_list = []
    H_over_W_list = []
    kt_gross_list = []
    fields_list = []

    diff_root_list = []
    ring_mean_pert_list = []
    ring_max_pert_list = []

    total_fallback_points = 0
    failed_runs = []
    t_start = time.perf_counter()

    print("=" * 85)
    print("RESAMPLING BODY-FITTED ELLIPTICAL STRESS FIELDS")
    print(f"Grid: ns = {ns}, nphi = {nphi}, K = {K}")
    print(f"Mesh resolution: n_hole = {n_hole}")
    print(f"Output archive:  {out_path}")
    print("=" * 85)

    for idx, row in df_clean.iterrows():
        sample_id = int(row["id"])
        W = float(row["W"])
        H = float(row["H"])
        a = float(row["a"])
        b = float(row["b"])
        sigma = float(row["sigma"])
        d_over_W = float(row["d_over_W"])
        ar = float(row["ar"])
        H_over_W = float(row["H_over_W"])
        kt_gross = float(row["kt_gross"])

        try:
            field, n_fb, val_root, peak_vm, X, Y = resample_ellipse_geometry(
                W=W, H=H, a=a, b=b, sigma=sigma, s=s, phi=phi, K=K, n_hole=n_hole
            )

            # Check (1): relative difference at notch root (s=0, phi=0) vs kt_gross
            diff_root_pct = abs(val_root - kt_gross) / kt_gross * 100.0

            # Check (2): |field - 1| on outer ring s = 1
            outer_ring_vals = field[-1, :]
            outer_pert = np.abs(outer_ring_vals - 1.0)
            mean_ring_pert = float(np.mean(outer_pert))
            max_ring_pert = float(np.max(outer_pert))

            ids_list.append(sample_id)
            d_over_W_list.append(d_over_W)
            ar_list.append(ar)
            H_over_W_list.append(H_over_W)
            kt_gross_list.append(kt_gross)
            fields_list.append(field)

            diff_root_list.append(diff_root_pct)
            ring_mean_pert_list.append(mean_ring_pert)
            ring_max_pert_list.append(max_ring_pert)
            total_fallback_points += n_fb

            # Store representative sample for plotting if matched
            if idx in target_example_indices:
                example_samples.append({
                    "label": target_example_indices[idx],
                    "X": X,
                    "Y": Y,
                    "field": field,
                    "a": a,
                    "b": b,
                    "ar": ar,
                    "d_over_W": d_over_W,
                    "kt_gross": kt_gross,
                })

        except Exception as e:
            failed_runs.append((sample_id, str(e)))
            print(f"[Run {idx+1}/{len(df_clean)}] FAILED for ID={sample_id}: {e}")

        # Progress reporting every 50 runs
        if (idx + 1) % 50 == 0 or (idx + 1) == len(df_clean):
            elapsed = time.perf_counter() - t_start
            rate = (idx + 1) / elapsed
            rem = (len(df_clean) - (idx + 1)) / rate if rate > 0 else 0.0
            curr_diff = diff_root_list[-1] if diff_root_list else 0.0
            curr_ring = ring_mean_pert_list[-1] if ring_mean_pert_list else 0.0
            print(
                f"[{idx+1:4d}/{len(df_clean):4d}] "
                f"ID={sample_id:4d} | Root diff: {curr_diff:.4f}% | "
                f"Outer ring |f-1|: {curr_ring:.4f} | Fallback pts: {total_fallback_points} | "
                f"Rate: {rate:.2f} runs/s | ETA: {rem:.1f}s"
            )

    # Save 4 representative fields plot
    if len(example_samples) == 4:
        plot_example_fields(example_samples, out_path=fig_path)
    else:
        print(f"Warning: captured {len(example_samples)} example fields instead of 4.")

    diff_arr = np.array(diff_root_list)
    ring_mean_arr = np.array(ring_mean_pert_list)
    ring_max_arr = np.array(ring_max_pert_list)

    print("\n" + "=" * 85)
    print("BODY-FITTED ELLIPTICAL STRESS FIELD GENERATION COMPLETE")
    print("=" * 85)
    print(f"Total processed samples:             {len(fields_list)} / {len(df_clean)}")

    print("\n--- CHECK (1): Relative difference at notch root (s=0, phi=0) vs kt_gross ---")
    print(f"  Mean relative difference:          {diff_arr.mean():.6f}%")
    print(f"  Max relative difference:           {diff_arr.max():.6f}%")
    print(f"  Samples with diff < 0.01%:         {np.sum(diff_arr < 0.01)} / {len(diff_arr)} "
          f"({100.0 * np.sum(diff_arr < 0.01) / len(diff_arr):.2f}%)")

    print("\n--- CHECK (2): Outer ring (s=1) perturbation |field - 1| ---")
    print(f"  Mean perturbation on outer ring:   {ring_mean_arr.mean():.6f}")
    print(f"  Max perturbation on outer ring:    {ring_max_arr.max():.6f}")

    print("\n--- CHECK (3): Fallback points (points outside mesh boundary) ---")
    print(f"  Total fallback points:             {total_fallback_points}")
    print(f"  Mean fallback points per sample:   {total_fallback_points / len(fields_list):.4f} / {ns * nphi} "
          f"({100.0 * total_fallback_points / (len(fields_list) * ns * nphi):.4f}%)")

    if failed_runs:
        print(f"\nWARNING: {len(failed_runs)} runs failed:")
        for fid, msg in failed_runs:
            print(f"  ID {fid}: {msg}")

    # Save to compressed NPZ archive
    fields_arr = np.stack(fields_list, axis=0)  # shape (N, 64, 64)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        ids=np.array(ids_list, dtype=np.int32),
        d_over_W=np.array(d_over_W_list, dtype=np.float32),
        ar=np.array(ar_list, dtype=np.float32),
        H_over_W=np.array(H_over_W_list, dtype=np.float32),
        kt_gross=np.array(kt_gross_list, dtype=np.float32),
        fields=fields_arr,
        s=s.astype(np.float32),
        phi=phi.astype(np.float32),
    )

    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nSuccessfully saved fields archive to {out_path} ({file_size_mb:.2f} MB).")
    print(f"Saved arrays: ids={len(ids_list)}, fields={fields_arr.shape} ({fields_arr.dtype})")
    print("=" * 85 + "\n")

    return fields_arr


def main():
    parser = argparse.ArgumentParser(
        description="Generate 2D body-fitted stress fields for plate with elliptical hole."
    )
    parser.add_argument("--csv", default="data/plate_ellipse.csv", help="Input CSV path")
    parser.add_argument("--out", default="data/fields_ellipse.npz", help="Output .npz archive path")
    parser.add_argument("--fig", default="results/fields_ellipse_samples.png", help="Output sample plot path")
    parser.add_argument("--n_hole", type=int, default=64, help="Elements along hole arc")
    parser.add_argument("--ns", type=int, default=64, help="Radial resolution s [0, 1]")
    parser.add_argument("--nphi", type=int, default=64, help="Angular resolution phi [0, pi/2]")
    parser.add_argument("--K", type=float, default=2.0, help="Radial expansion factor K")
    parser.add_argument("--limit", type=int, default=None, help="Process limited samples (for testing)")
    args = parser.parse_args()

    generate_ellipse_fields(
        csv_path=args.csv,
        out_path=args.out,
        fig_path=args.fig,
        n_hole=args.n_hole,
        ns=args.ns,
        nphi=args.nphi,
        K=args.K,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
