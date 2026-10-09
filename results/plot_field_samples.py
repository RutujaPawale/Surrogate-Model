"""
Visualizes 4 example normalized stress fields from data/fields.npz.
Displays the masked hole region as blank and includes a colorbar.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_field_samples(npz_path="data/fields.npz", out_path="results/field_samples.png"):
    """
    Loads fields.npz and saves a 4-panel figure of representative stress fields.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{npz_path} not found. Run data_gen/gen_fields.py first.")

    data = np.load(npz_path)
    fields = data["fields"]       # (N, ny, nx)
    masks = data["masks"]         # (N, ny, nx)
    d_over_W = data["d_over_W"]   # (N,)
    H_over_W = data["H_over_W"]   # (N,)
    kt_gross = data["kt_gross"]   # (N,)
    xi = data["xi"]               # (nx,)
    eta = data["eta"]             # (ny,)

    n_samples = len(fields)
    print(f"Loaded {n_samples} fields from {npz_path}.")

    # Select 4 representative samples spanning the range of hole sizes (d/W)
    sort_idx = np.argsort(d_over_W)
    chosen_indices = [
        sort_idx[int(n_samples * 0.05)],  # Small hole (~5th percentile)
        sort_idx[int(n_samples * 0.35)],  # Medium-low hole (~35th percentile)
        sort_idx[int(n_samples * 0.65)],  # Medium-high hole (~65th percentile)
        sort_idx[int(n_samples * 0.95)],  # Large hole (~95th percentile)
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), sharex=True, sharey=True, constrained_layout=True)

    # Global min and max across chosen samples for a unified colorbar
    vmin = 0.5
    vmax = max(kt_gross[idx] for idx in chosen_indices) * 1.02

    im = None
    for ax, idx in zip(axes, chosen_indices):
        f = fields[idx].copy()
        m = masks[idx]

        # Masked region (hole) set to NaN so it renders blank
        f_masked = np.where(m, f, np.nan)

        im = ax.imshow(
            f_masked,
            origin="lower",
            extent=[xi[0], xi[-1], eta[0], eta[-1]],
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            interpolation="nearest"
        )

        d_val = d_over_W[idx]
        hw_val = H_over_W[idx]
        kt_val = kt_gross[idx]

        ax.set_title(
            f"d/W = {d_val:.2f}\nH/W = {hw_val:.2f}\n$K_t$ = {kt_val:.2f}",
            fontsize=10,
            fontweight="bold"
        )
        ax.set_xlabel(r"$\xi = x/W$")
        ax.grid(False)

    axes[0].set_ylabel(r"$\eta = y/W$")

    # Unified colorbar
    cbar = fig.colorbar(im, ax=axes, location="right", fraction=0.02, pad=0.02)
    cbar.set_label(r"Normalized Stress $\sigma_{vm} / \sigma$", fontsize=11)

    fig.suptitle(
        r"FEA Normalized Stress Fields ($\xi \in [0, 0.5], \eta \in [0, 1.0]$) with Masked Hole",
        fontsize=13,
        fontweight="bold"
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved field samples plot to {out_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sample 2D stress fields from fields.npz.")
    parser.add_argument("--npz", default="data/fields.npz", help="Path to fields.npz")
    parser.add_argument("--out", default="results/field_samples.png", help="Output PNG path")
    args = parser.parse_args()

    plot_field_samples(npz_path=args.npz, out_path=args.out)
