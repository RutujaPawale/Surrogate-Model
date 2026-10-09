"""
Step 1b: Resample FEA stress fields onto a standardized 2D grid.

Reads data/plate_hole.csv, re-runs each geometry with n_hole=32 and return_field=True,
and interpolates the normalized von Mises stress field (vm / sigma) onto a fixed
grid in normalized coordinates:
    xi  = x / W in [0, 0.5]  (nx = 64 points)
    eta = y / W in [0, 1.0]  (ny = 128 points)

A boolean mask is generated per sample:
    True  = valid plate material (xi^2 + eta^2 >= (d_over_W / 2)^2)
    False = circular hole interior (field set to 0.0, not evaluated via FEA)

Boundary rounding fallback:
    Valid points falling slightly outside the triangular mesh due to polygonal
    boundary discretization are mapped to the nearest mesh node using a KD-tree.
    The fallback count is strictly tracked and reported.

Output:
    data/fields.npz containing:
        - ids        : (N,) geometry sample identifiers
        - d_over_W   : (N,) hole diameter ratio
        - H_over_W   : (N,) plate aspect ratio
        - kt_gross   : (N,) stress concentration factor
        - fields     : (N, ny, nx) float32 normalized stress fields
        - masks      : (N, ny, nx) bool plate masks
        - xi         : (nx,) normalized x coordinates
        - eta        : (ny,) normalized y coordinates
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


def resample_geometry(W, H, r, sigma, d_over_W, xi, eta, n_hole=32):
    """
    Solves one FEA geometry and resamples normalized stress onto the fixed (xi, eta) grid.

    Parameters:
        W (float): Plate width [mm]
        H (float): Plate height [mm]
        r (float): Hole radius [mm]
        sigma (float): Applied gross tensile stress [MPa]
        d_over_W (float): Hole diameter to plate width ratio
        xi (np.ndarray): 1D array of normalized x coordinates [0, 0.5]
        eta (np.ndarray): 1D array of normalized y coordinates [0, 1.0]
        n_hole (int): Element resolution along the hole arc

    Returns:
        field (np.ndarray): (ny, nx) float32 normalized stress field (0 inside hole)
        mask (np.ndarray): (ny, nx) bool mask (True = plate, False = hole)
        n_fallback (int): Number of valid pixels requiring nearest-node fallback
        peak_vm (float): Peak von Mises stress from FEA
    """
    nx = len(xi)
    ny = len(eta)
    XI, ETA = np.meshgrid(xi, eta)  # shapes (ny, nx)

    # 1. Mask valid plate material (exclude circular hole)
    r_norm_sq = (d_over_W / 2.0) ** 2
    mask = (XI ** 2 + ETA ** 2) >= r_norm_sq

    # 2. Run FEA with field output
    res = solve_plate(W, H, r, sigma, n_hole=n_hole, return_field=True)
    mesh = res["mesh"]
    vm_nodal = res["vm_nodal"]
    vm_norm = (vm_nodal / sigma).astype(np.float64)

    # 3. Interpolate onto valid plate coordinates
    X_valid = XI[mask] * W
    Y_valid = ETA[mask] * W

    triang = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    interp = mtri.LinearTriInterpolator(triang, vm_norm)
    vals = interp(X_valid, Y_valid)

    # Identify valid plate pixels falling outside mesh boundary due to discretization chords
    if np.ma.is_masked(vals):
        fallback_needed = vals.mask
    else:
        fallback_needed = np.isnan(vals)

    n_fallback = int(np.sum(fallback_needed))

    # 4. Nearest mesh node fallback
    if n_fallback > 0:
        tree = cKDTree(mesh.p.T)
        pts_fallback = np.column_stack([X_valid[fallback_needed], Y_valid[fallback_needed]])
        _, nearest_idx = tree.query(pts_fallback)
        vals_clean = np.array(vals, dtype=np.float64)
        vals_clean[fallback_needed] = vm_norm[nearest_idx]
    else:
        vals_clean = np.array(vals, dtype=np.float64)

    # 5. Populate fixed 2D grid
    field = np.zeros((ny, nx), dtype=np.float32)
    field[mask] = vals_clean.astype(np.float32)

    return field, mask, n_fallback, res["peak_vm"]


def generate_fields(csv_path="data/plate_hole.csv", out_path="data/fields.npz",
                    n_hole=32, nx=64, ny=128, limit=None):
    """
    Processes all successful geometries from the CSV dataset and saves fields.npz.
    """
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    n_total = len(df)
    df_clean = df[df["ok"].astype(str) == "True"].copy().reset_index(drop=True)
    print(f"Loaded {len(df_clean)}/{n_total} usable runs from CSV.")

    if limit is not None and limit > 0:
        df_clean = df_clean.iloc[:limit].copy()
        print(f"Processing subset of {len(df_clean)} runs for testing...")

    xi = np.linspace(0.0, 0.5, nx)
    eta = np.linspace(0.0, 1.0, ny)

    ids_list = []
    d_over_W_list = []
    H_over_W_list = []
    kt_gross_list = []
    fields_list = []
    masks_list = []

    total_fallback_pixels = 0
    failed_runs = []
    t_start = time.perf_counter()

    n_runs = len(df_clean)
    print(f"Starting field generation on {n_runs} geometries (nx={nx}, ny={ny}, n_hole={n_hole})...")

    for i, row in df_clean.iterrows():
        sample_id = int(row["id"]) if "id" in row else i
        W = float(row["W"])
        H = float(row["H"])
        r = float(row["r"])
        sigma = float(row["sigma"])
        d_over_W = float(row["d_over_W"])
        H_over_W = float(row["H_over_W"])
        kt_gross = float(row["kt_gross"])

        try:
            field, mask, n_fb, _ = resample_geometry(
                W=W, H=H, r=r, sigma=sigma, d_over_W=d_over_W,
                xi=xi, eta=eta, n_hole=n_hole
            )
            total_fallback_pixels += n_fb

            ids_list.append(sample_id)
            d_over_W_list.append(d_over_W)
            H_over_W_list.append(H_over_W)
            kt_gross_list.append(kt_gross)
            fields_list.append(field)
            masks_list.append(mask)

        except Exception as e:
            failed_runs.append({"id": sample_id, "error": repr(e)})
            print(f"Run {sample_id} (index {i}) failed: {e!r}")

        # Progress reporting every 50 runs
        if (i + 1) % 50 == 0 or (i + 1) == n_runs:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed
            remaining = (n_runs - (i + 1)) / rate if rate > 0 else 0
            print(f"[{i + 1}/{n_runs}] runs processed ({rate:.1f} runs/s, ~{remaining:.0f}s remaining). "
                  f"Cumulative fallback pixels: {total_fallback_pixels}")

    n_success = len(fields_list)
    print(f"\nProcessing complete: {n_success}/{n_runs} geometries succeeded.")
    print(f"Total valid pixels requiring boundary fallback: {total_fallback_pixels} "
          f"({total_fallback_pixels / (n_success * nx * ny) * 100:.3f}% of all pixels)")

    if failed_runs:
        print(f"Warning: {len(failed_runs)} runs failed. Error summary:")
        for fr in failed_runs:
            print(f"  ID {fr['id']}: {fr['error']}")

    # Save to compressed .npz archive
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        ids=np.array(ids_list, dtype=np.int32),
        d_over_W=np.array(d_over_W_list, dtype=np.float32),
        H_over_W=np.array(H_over_W_list, dtype=np.float32),
        kt_gross=np.array(kt_gross_list, dtype=np.float32),
        fields=np.stack(fields_list, axis=0),
        masks=np.stack(masks_list, axis=0),
        xi=xi.astype(np.float32),
        eta=eta.astype(np.float32),
    )

    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved {out_path} ({file_size_mb:.2f} MB)")
    print(f"Archive contents:")
    print(f"  fields   : shape {np.stack(fields_list, axis=0).shape}, dtype float32")
    print(f"  masks    : shape {np.stack(masks_list, axis=0).shape}, dtype bool")
    print(f"  ids      : shape ({len(ids_list)},)")
    print(f"  d_over_W : shape ({len(d_over_W_list)},)")
    print(f"  H_over_W : shape ({len(H_over_W_list)},)")
    print(f"  kt_gross : shape ({len(kt_gross_list)},)")
    print(f"  xi       : shape ({len(xi)},), range [{xi[0]:.2f}, {xi[-1]:.2f}]")
    print(f"  eta      : shape ({len(eta)},), range [{eta[0]:.2f}, {eta[-1]:.2f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2D normalized stress fields dataset.")
    parser.add_argument("--csv", default="data/plate_hole.csv", help="Input CSV path")
    parser.add_argument("--out", default="data/fields.npz", help="Output .npz path")
    parser.add_argument("--n_hole", type=int, default=32, help="Elements along hole arc")
    parser.add_argument("--nx", type=int, default=64, help="Grid points in xi")
    parser.add_argument("--ny", type=int, default=128, help="Grid points in eta")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of runs")
    args = parser.parse_args()

    generate_fields(
        csv_path=args.csv,
        out_path=args.out,
        n_hole=args.n_hole,
        nx=args.nx,
        ny=args.ny,
        limit=args.limit
    )
