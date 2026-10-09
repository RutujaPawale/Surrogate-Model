"""
Step 1c: Resample FEA stress fields onto a polar coordinate grid around the hole.

Reads data/plate_hole.csv, re-runs each geometry with n_hole=32 and return_field=True,
and samples the normalized von Mises stress field (vm / sigma) on a polar grid:
    rho = r + s * (W/2 - r), with s in [0, 1] (64 values)
    theta in [0, pi/2] (64 values)
    x = rho * cos(theta), y = rho * sin(theta)

Because the grid covers the physical plate quadrant from the hole boundary (rho=r)
to the right edge (rho=W/2), all points lie strictly on solid plate material.
No hole mask is needed.

Fallback:
    Boundary points falling slightly outside the triangular elements due to discretization
    chords are mapped to the nearest mesh node. Fallback count is tracked and reported.

Validation:
    At s=0, theta=0 (the root of the notch at (r, 0)), the stress value is compared with
    kt_gross (FEA peak_vm / sigma). The relative difference should be virtually zero.

Output:
    data/fields_polar.npz containing:
        - ids        : (N,) geometry sample identifiers
        - d_over_W   : (N,) hole diameter ratio
        - H_over_W   : (N,) plate aspect ratio
        - kt_gross   : (N,) stress concentration factor
        - fields     : (N, 64, 64) float32 normalized stress fields
        - s          : (64,) normalized radial coordinate [0, 1]
        - theta      : (64,) polar angle coordinate [0, pi/2]
"""
import argparse
import os
import sys
import time

import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Ensure project root is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_gen.gen_data import solve_plate


def resample_polar_geometry(W, H, r, sigma, s, theta, n_hole=32):
    """
    Solves one FEA geometry and resamples normalized stress onto a (64, 64) polar grid.

    Parameters:
        W (float): Plate width [mm]
        H (float): Plate height [mm]
        r (float): Hole radius [mm]
        sigma (float): Applied gross tensile stress [MPa]
        s (np.ndarray): 1D array of normalized radial coordinates [0, 1], length 64
        theta (np.ndarray): 1D array of polar angle coordinates [0, pi/2], length 64
        n_hole (int): Element resolution along the hole arc

    Returns:
        field (np.ndarray): (64, 64) float32 normalized stress field
        n_fallback (int): Number of points requiring nearest-node fallback
        val_at_0_0 (float): Sampled normalized stress at (s=0, theta=0)
        peak_vm (float): FEA peak von Mises stress
    """
    S, THETA = np.meshgrid(s, theta, indexing="ij")  # shape (64, 64)
    rho = r + S * (W / 2.0 - r)
    X = rho * np.cos(THETA)
    Y = rho * np.sin(THETA)

    # Run FEA
    res = solve_plate(W, H, r, sigma, n_hole=n_hole, return_field=True)
    mesh = res["mesh"]
    vm_nodal = res["vm_nodal"]
    vm_norm = (vm_nodal / sigma).astype(np.float64)

    # Triangulation interpolation
    triang = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    interp = mtri.LinearTriInterpolator(triang, vm_norm)
    vals = interp(X, Y)

    # Identify any boundary points needing fallback
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
    val_at_0_0 = float(field[0, 0])

    return field, n_fallback, val_at_0_0, res["peak_vm"]


def generate_polar_fields(
    csv_path="data/plate_hole.csv",
    out_path="data/fields_polar.npz",
    n_hole=32,
    ns=64,
    ntheta=64,
    limit=None,
):
    """
    Processes all successful geometries from the CSV dataset and saves fields_polar.npz.
    """
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    n_total = len(df)
    df_clean = df[df["ok"].astype(str) == "True"].copy().reset_index(drop=True)
    print(f"Loaded {len(df_clean)}/{n_total} usable runs from CSV.")

    if limit is not None and limit > 0:
        df_clean = df_clean.iloc[:limit].copy()
        print(f"Processing subset of {len(df_clean)} runs for testing...")

    s = np.linspace(0.0, 1.0, ns, dtype=np.float64)
    theta = np.linspace(0.0, np.pi / 2.0, ntheta, dtype=np.float64)

    ids_list = []
    d_over_W_list = []
    H_over_W_list = []
    kt_gross_list = []
    fields_list = []
    diff_at_root_list = []

    total_fallback_points = 0
    failed_runs = []
    t_start = time.perf_counter()

    for idx, row in df_clean.iterrows():
        sample_id = int(row["id"])
        W = float(row["W"])
        H = float(row["H"])
        r = float(row["r"])
        sigma = float(row["sigma"])
        d_over_W = float(row["d_over_W"])
        H_over_W = float(row["H_over_W"])
        kt_gross = float(row["kt_gross"])

        try:
            field, n_fb, val_0_0, peak_vm = resample_polar_geometry(
                W=W, H=H, r=r, sigma=sigma, s=s, theta=theta, n_hole=n_hole
            )

            # Compare value at s=0, theta=0 with kt_gross
            rel_diff_pct = abs(val_0_0 - kt_gross) / kt_gross * 100.0

            ids_list.append(sample_id)
            d_over_W_list.append(d_over_W)
            H_over_W_list.append(H_over_W)
            kt_gross_list.append(kt_gross)
            fields_list.append(field)
            diff_at_root_list.append(rel_diff_pct)
            total_fallback_points += n_fb

        except Exception as e:
            failed_runs.append((sample_id, str(e)))
            print(f"[Run {idx+1}/{len(df_clean)}] FAILED for ID={sample_id}: {e}")

        # Progress reporting every 50 runs
        if (idx + 1) % 50 == 0 or (idx + 1) == len(df_clean):
            elapsed = time.perf_counter() - t_start
            rate = (idx + 1) / elapsed
            rem = (len(df_clean) - (idx + 1)) / rate if rate > 0 else 0
            curr_diff = diff_at_root_list[-1] if diff_at_root_list else 0.0
            print(
                f"[{idx+1:4d}/{len(df_clean):4d}] "
                f"ID={sample_id:4d} | Root diff: {curr_diff:.4f}% | "
                f"Fallback pts: {total_fallback_points} | "
                f"Rate: {rate:.2f} runs/s | ETA: {rem:.1f}s"
            )

    print("\n" + "=" * 70)
    print("POLAR FIELD GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total processed: {len(fields_list)} / {len(df_clean)}")
    print(f"Total points requiring nearest-node fallback: {total_fallback_points}")
    print(
        f"Points requiring fallback per sample: {total_fallback_points / len(fields_list):.2f} "
        f"out of {ns * ntheta} grid points ({(total_fallback_points / (len(fields_list) * ns * ntheta)) * 100:.3f}%)"
    )

    diff_arr = np.array(diff_at_root_list)
    print("\nComparison at s=0, theta=0 vs kt_gross (FEA peak):")
    print(f"  Mean relative difference: {diff_arr.mean():.6f}%")
    print(f"  Max relative difference:  {diff_arr.max():.6f}%")
    print(f"  Median relative diff:     {np.median(diff_arr):.6f}%")
    print(f"  Samples with diff < 0.01%: {np.sum(diff_arr < 0.01)} / {len(diff_arr)}")

    if failed_runs:
        print(f"\nWARNING: {len(failed_runs)} runs failed during generation:")
        for fid, msg in failed_runs:
            print(f"  ID {fid}: {msg}")

    # Save to NPZ
    fields_arr = np.stack(fields_list, axis=0)  # shape (N, 64, 64)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        ids=np.array(ids_list, dtype=np.int32),
        d_over_W=np.array(d_over_W_list, dtype=np.float32),
        H_over_W=np.array(H_over_W_list, dtype=np.float32),
        kt_gross=np.array(kt_gross_list, dtype=np.float32),
        fields=fields_arr,
        s=s.astype(np.float32),
        theta=theta.astype(np.float32),
        masks=np.ones_like(fields_arr, dtype=bool),  # all pixels are solid plate
    )

    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nSuccessfully saved polar fields to {out_path} ({file_size_mb:.2f} MB).")
    print(f"Array shape: fields={fields_arr.shape}, dtype={fields_arr.dtype}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Generate 2D polar stress fields from FEA.")
    parser.add_argument("--csv", default="data/plate_hole.csv", help="Path to plate_hole.csv")
    parser.add_argument("--out", default="data/fields_polar.npz", help="Output path for fields_polar.npz")
    parser.add_argument("--n_hole", type=int, default=32, help="Elements along hole arc")
    parser.add_argument("--ns", type=int, default=64, help="Grid points in radial direction s [0, 1]")
    parser.add_argument("--ntheta", type=int, default=64, help="Grid points in angular direction theta [0, pi/2]")
    parser.add_argument("--limit", type=int, default=None, help="Process limited samples (for test)")
    args = parser.parse_args()

    generate_polar_fields(
        csv_path=args.csv,
        out_path=args.out,
        n_hole=args.n_hole,
        ns=args.ns,
        ntheta=args.ntheta,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
