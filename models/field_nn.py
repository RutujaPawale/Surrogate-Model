"""
Deep Learning (PyTorch CNN/Deconv) Surrogate Model for 2D Stress Field Prediction.

Inputs: d_over_W, H_over_W (dimensionless geometry) standardized with training statistics.
Target: 2D normalized stress field (sigma_vm / sigma) on fixed (128, 64) grid.

Architecture:
  - Input: (B, 2)
  - FC layers: 2 -> 128 -> 256 -> (256 * 4 * 8) with ReLU
  - Reshape to feature map: (B, 256, 4, 8)
  - 4 Transposed-convolution stages with ReLU:
      Stage 1: 256 -> 128, (4, 8) -> (8, 16)
      Stage 2: 128 -> 64,  (8, 16) -> (16, 32)
      Stage 3: 64  -> 32,  (16, 32) -> (32, 64)
      Stage 4: 32  -> 1,   (32, 64) -> (64, 128)
  - Bilinear resize to match target grid: (B, 1, 128, 64)

Loss:
  - Masked MSE on valid solid plate pixels only (hole pixels multiplied by 0).

Training:
  - Adam optimizer, initial lr=1e-3 with CosineAnnealingLR decay.
  - Up to 1500 epochs, batch size 32.
  - Early stopping on validation loss (patience 100), seed 42.
  - Logs progress every 100 epochs.

Splits:
  (A) Interpolation: random 70/15/15 by geometry (seed 42).
  (B) Extrapolation: train on d_over_W < 0.40, test on d_over_W >= 0.45.
      Validation set: last 15% of the training geometries (highest d_over_W).

Outputs:
  - results/nn_predictions.npz
  - results/nn_results.csv
  - results/nn_best_worst.png
"""
import argparse
import copy
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# ---- Model Architecture ------------------------------------------------------
class StressFieldNN(nn.Module):
    """
    Fully-connected expansion followed by transposed convolution decoder
    to predict 2D stress fields from geometric parameters.
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256 * 4 * 8),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        h = self.fc(x)
        h = h.view(-1, 256, 4, 8)
        out = self.deconv(h)  # Shape: (B, 1, 64, 128)
        if out.shape[-2:] != (128, 64):
            out = F.interpolate(out, size=(128, 64), mode="bilinear", align_corners=False)
        return out


# ---- Masked Loss -------------------------------------------------------------
def masked_mse_loss(pred, target, mask):
    """
    Mean Squared Error calculated strictly over valid plate pixels.
    Hole interior pixels have mask = 0 and contribute zero loss.
    """
    diff = (pred - target) * mask
    valid_pixels = mask.sum().clamp(min=1.0)
    return (diff ** 2).sum() / valid_pixels


# ---- Training Loop -----------------------------------------------------------
def train_model(model, train_loader, X_val, y_val, m_val, max_epochs=1500,
                patience=100, lr=1e-3, print_interval=100):
    """
    Trains the network using Adam with cosine annealing and early stopping.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    patience_counter = 0

    t_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_train_loss = 0.0
        n_samples = 0

        for bx, by, bm in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = masked_mse_loss(pred, by, bm)
            loss.backward()
            optimizer.step()

            bs = bx.size(0)
            total_train_loss += loss.item() * bs
            n_samples += bs

        scheduler.step()
        train_loss = total_train_loss / n_samples

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val)
            val_loss = masked_mse_loss(pred_val, y_val, m_val).item()

        # Early stopping check
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % print_interval == 0 or patience_counter >= patience or epoch == max_epochs:
            elapsed = time.perf_counter() - t_start
            print(f"Epoch {epoch:4d}/{max_epochs} | Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} (Best: {best_val_loss:.6f} @ Ep {best_epoch}) | "
                  f"Time: {elapsed:.1f}s")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (no validation improvement for {patience} epochs).")
            break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"Restored best model weights from epoch {best_epoch} (Val Loss: {best_val_loss:.6f}).")

    return model, best_epoch


# ---- Inference & Metric Calculations -----------------------------------------
def predict_fields_nn(model, X_tensor, d_over_W, xi, eta):
    """
    Generates predicted fields from input tensor and applies the analytic hole mask.
    """
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor)  # (M, 1, 128, 64)
    pred_fields = preds.squeeze(1).cpu().numpy().astype(np.float32)

    M, ny, nx = pred_fields.shape
    XI, ETA = np.meshgrid(xi, eta)
    XI_sq_ETA_sq = XI ** 2 + ETA ** 2

    masks = np.zeros((M, ny, nx), dtype=bool)
    for i in range(M):
        r_sq = (float(d_over_W[i]) / 2.0) ** 2
        m = XI_sq_ETA_sq >= r_sq
        masks[i] = m
        pred_fields[i][~m] = 0.0

    return pred_fields, masks


def compute_field_metrics(pred_fields, true_fields, masks):
    """
    Computes relative L2 error, MAE, and peak error strictly on valid plate pixels.
    """
    M = len(pred_fields)
    rel_l2_errors = np.zeros(M, dtype=np.float64)
    mae_errors = np.zeros(M, dtype=np.float64)
    peak_errors = np.zeros(M, dtype=np.float64)

    for i in range(M):
        m = masks[i]
        yt_valid = true_fields[i][m]
        yp_valid = pred_fields[i][m]

        # Relative L2 error (%)
        l2_diff = np.linalg.norm(yp_valid - yt_valid)
        l2_true = np.linalg.norm(yt_valid)
        rel_l2_errors[i] = 100.0 * l2_diff / l2_true if l2_true > 0 else 0.0

        # Mean absolute pixel error
        mae_errors[i] = np.mean(np.abs(yp_valid - yt_valid))

        # Peak error (%)
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


def time_inference_nn(model, X_tensor, d_over_W, xi, eta, n_repeats=100):
    """Measures single-sample and batched inference latency in microseconds."""
    model.eval()
    sample_x = X_tensor[:1]
    sample_d = d_over_W[:1]

    # Single sample latency
    with torch.no_grad():
        # Warmup
        model(sample_x)
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            predict_fields_nn(model, sample_x, sample_d, xi, eta)
        single_us = 1e6 * (time.perf_counter() - t0) / n_repeats

        # Batched latency
        model(X_tensor)
        t1 = time.perf_counter()
        predict_fields_nn(model, X_tensor, d_over_W, xi, eta)
        batched_us = 1e6 * (time.perf_counter() - t1) / len(X_tensor)

    return single_us, batched_us


def plot_best_and_worst_samples(true_fields, pred_fields, masks, d_over_W, H_over_W,
                                rel_l2_list, ids, xi, eta, out_path="results/nn_best_worst.png"):
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
        axes[row, 1].set_title(f"CNN-Deconv Predicted Field\nPeak = {np.nanmax(yp):.2f}", fontsize=10)
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

    fig.suptitle("PyTorch CNN-Deconv 2D Stress Field Reconstruction: Best vs. Worst Test Samples", fontsize=13, fontweight="bold")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved diagnostic best/worst figure to {out_path}.")


# ---- Main Pipeline -----------------------------------------------------------
def run_nn_pipeline(npz_path="data/fields.npz", max_epochs=1500, patience=100,
                    batch_size=32, lr=1e-3, seed=42):
    """
    Executes the neural network surrogate training and evaluation across both splits.
    """
    # Deterministic seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(6)

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

    # Model architecture verification
    dummy_model = StressFieldNN()
    n_params = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
    print(f"Model architecture initialized: {n_params:,} trainable parameters.")

    # Target fields with channel dimension: (N, 1, 128, 64)
    y_all = fields[:, None, :, :].astype(np.float32)
    m_all = masks[:, None, :, :].astype(np.float32)
    X_raw = np.column_stack([d_over_W, H_over_W]).astype(np.float32)

    # =========================================================================
    # (A) INTERPOLATION SPLIT (70/15/15, seed 42)
    # =========================================================================
    idx = np.arange(N)
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.30, random_state=seed)
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, random_state=seed)

    print(f"\n=======================================================")
    print(f"SPLIT (A): Interpolation (train={len(idx_tr)}, val={len(idx_va)}, test={len(idx_te)})")
    print(f"=======================================================")

    # Standardize inputs using training statistics
    mean_A = X_raw[idx_tr].mean(axis=0)
    std_A = X_raw[idx_tr].std(axis=0)
    std_A[std_A == 0] = 1.0

    X_tr_A = (X_raw[idx_tr] - mean_A) / std_A
    X_va_A = (X_raw[idx_va] - mean_A) / std_A
    X_te_A = (X_raw[idx_te] - mean_A) / std_A

    t_X_tr_A = torch.from_numpy(X_tr_A)
    t_y_tr_A = torch.from_numpy(y_all[idx_tr])
    t_m_tr_A = torch.from_numpy(m_all[idx_tr])

    t_X_va_A = torch.from_numpy(X_va_A)
    t_y_va_A = torch.from_numpy(y_all[idx_va])
    t_m_va_A = torch.from_numpy(m_all[idx_va])

    t_X_te_A = torch.from_numpy(X_te_A)

    train_loader_A = DataLoader(
        TensorDataset(t_X_tr_A, t_y_tr_A, t_m_tr_A),
        batch_size=batch_size,
        shuffle=True
    )

    model_A = StressFieldNN()
    model_A, best_ep_A = train_model(
        model=model_A,
        train_loader=train_loader_A,
        X_val=t_X_va_A,
        y_val=t_y_va_A,
        m_val=t_m_va_A,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        print_interval=100
    )

    pred_fields_te_A, pred_masks_te_A = predict_fields_nn(
        model_A, t_X_te_A, d_over_W[idx_te], xi, eta
    )
    metrics_A = compute_field_metrics(pred_fields_te_A, fields[idx_te], masks[idx_te])
    single_us_A, batch_us_A = time_inference_nn(
        model_A, t_X_te_A, d_over_W[idx_te], xi, eta
    )

    print("\n--- Interpolation Test Set Results ---")
    print(f"Mean Relative L2 Error: {metrics_A['mean_rel_l2_%']:.3f}%")
    print(f"Max Relative L2 Error:  {metrics_A['max_rel_l2_%']:.3f}%")
    print(f"Mean Abs Pixel Error:   {metrics_A['mean_abs_pixel_err']:.4f}")
    print(f"Mean Peak Error:        {metrics_A['mean_peak_err_%']:.3f}%")
    print(f"Max Peak Error:         {metrics_A['max_peak_err_%']:.3f}%")
    print(f"Single-sample latency:  {single_us_A:,.1f} us ({single_us_A / 1000:.3f} ms)")
    print(f"Batched latency:        {batch_us_A:,.1f} us/sample ({batch_us_A / 1000:.4f} ms/sample)")

    # Save diagnostic best/worst figure
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
        out_path="results/nn_best_worst.png"
    )

    # =========================================================================
    # (B) EXTRAPOLATION SPLIT (train d/W < 0.40, test d/W >= 0.45)
    # Validation: last 15% of training samples sorted by d/W (highest d/W)
    # =========================================================================
    tr_ext_all = np.where(d_over_W < 0.40)[0]
    te_ext_idx = np.where(d_over_W >= 0.45)[0]

    # Sort training geometries by d/W and hold out the highest 15% as validation
    sorted_ext_order = tr_ext_all[np.argsort(d_over_W[tr_ext_all])]
    n_ext_val = int(len(sorted_ext_order) * 0.15)
    tr_ext_idx = sorted_ext_order[:-n_ext_val]
    va_ext_idx = sorted_ext_order[-n_ext_val:]

    print(f"\n=======================================================")
    print(f"SPLIT (B): Extrapolation (train={len(tr_ext_idx)}, val={len(va_ext_idx)} [highest 15% d/W < 0.40], "
          f"test={len(te_ext_idx)} [d/W >= 0.45])")
    print(f"=======================================================")

    mean_B = X_raw[tr_ext_idx].mean(axis=0)
    std_B = X_raw[tr_ext_idx].std(axis=0)
    std_B[std_B == 0] = 1.0

    X_tr_B = (X_raw[tr_ext_idx] - mean_B) / std_B
    X_va_B = (X_raw[va_ext_idx] - mean_B) / std_B
    X_te_B = (X_raw[te_ext_idx] - mean_B) / std_B

    t_X_tr_B = torch.from_numpy(X_tr_B)
    t_y_tr_B = torch.from_numpy(y_all[tr_ext_idx])
    t_m_tr_B = torch.from_numpy(m_all[tr_ext_idx])

    t_X_va_B = torch.from_numpy(X_va_B)
    t_y_va_B = torch.from_numpy(y_all[va_ext_idx])
    t_m_va_B = torch.from_numpy(m_all[va_ext_idx])

    t_X_te_B = torch.from_numpy(X_te_B)

    train_loader_B = DataLoader(
        TensorDataset(t_X_tr_B, t_y_tr_B, t_m_tr_B),
        batch_size=batch_size,
        shuffle=True
    )

    model_B = StressFieldNN()
    model_B, best_ep_B = train_model(
        model=model_B,
        train_loader=train_loader_B,
        X_val=t_X_va_B,
        y_val=t_y_va_B,
        m_val=t_m_va_B,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        print_interval=100
    )

    pred_fields_te_B, pred_masks_te_B = predict_fields_nn(
        model_B, t_X_te_B, d_over_W[te_ext_idx], xi, eta
    )
    metrics_B = compute_field_metrics(pred_fields_te_B, fields[te_ext_idx], masks[te_ext_idx])
    single_us_B, batch_us_B = time_inference_nn(
        model_B, t_X_te_B, d_over_W[te_ext_idx], xi, eta
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
    out_preds_path = "results/nn_predictions.npz"
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
        extrap_ids=ids[te_ext_idx],
        extrap_pred_fields=pred_fields_te_B,
        extrap_true_fields=fields[te_ext_idx],
        extrap_masks=masks[te_ext_idx],
        extrap_d_over_W=d_over_W[te_ext_idx],
        extrap_H_over_W=H_over_W[te_ext_idx],
        extrap_kt_gross=kt_gross[te_ext_idx],
        xi=xi,
        eta=eta
    )
    print(f"\nSaved predictions archive to {out_preds_path} ({os.path.getsize(out_preds_path)/(1024*1024):.2f} MB).")

    # 2. Results CSV
    out_csv_path = "results/nn_results.csv"
    res_df = pd.DataFrame([
        {
            "split": "interp",
            "trainable_params": n_params,
            "mean_rel_l2_%": metrics_A["mean_rel_l2_%"],
            "max_rel_l2_%": metrics_A["max_rel_l2_%"],
            "mean_abs_pixel_err": metrics_A["mean_abs_pixel_err"],
            "mean_peak_err_%": metrics_A["mean_peak_err_%"],
            "max_peak_err_%": metrics_A["max_peak_err_%"],
            "single_sample_us": single_us_A,
            "batched_us_per_sample": batch_us_A,
            "epochs_trained": best_ep_A,
        },
        {
            "split": "extrap",
            "trainable_params": n_params,
            "mean_rel_l2_%": metrics_B["mean_rel_l2_%"],
            "max_rel_l2_%": metrics_B["max_rel_l2_%"],
            "mean_abs_pixel_err": metrics_B["mean_abs_pixel_err"],
            "mean_peak_err_%": metrics_B["mean_peak_err_%"],
            "max_peak_err_%": metrics_B["max_peak_err_%"],
            "single_sample_us": single_us_B,
            "batched_us_per_sample": batch_us_B,
            "epochs_trained": best_ep_B,
        }
    ])
    res_df.to_csv(out_csv_path, index=False)
    print(f"Saved metrics table to {out_csv_path}.")
    print("\nSummary Results Table:")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CNN-Deconv 2D stress field surrogate.")
    parser.add_argument("--npz", default="data/fields.npz", help="Path to fields.npz")
    parser.add_argument("--epochs", type=int, default=1500, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_nn_pipeline(
        npz_path=args.npz,
        max_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )
