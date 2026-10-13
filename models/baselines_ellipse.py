"""
Baseline scalar surrogates for the plate with an elliptical hole dataset (data/plate_ellipse.csv).

Target : kt_gross = peak von Mises stress / applied stress
Inputs : W, d_over_W, ar, H_over_W

Evaluations (all with seed 42):
  A)  Interpolation: random 70/15/15 split by row.
  B1) Extrapolation in hole size: train d_over_W < 0.40, test d_over_W >= 0.45.
  B2) Extrapolation to sharper notches: train ar >= 0.8, test ar < 0.7.

Models:
  1. Mean predictor: trivial baseline predicting the training set mean.
  2. Linear: standardized linear regression.
  3. Poly3+Ridge: degree-3 polynomial features + Ridge on all 4 standardized inputs.
  4. Poly3 (1/ar): degree-3 polynomial features + Ridge on [d_over_W, 1/ar, H_over_W] only (W dropped, ar inverted).
  5. GradBoost: Gradient Boosting Regressor (300 trees, depth 3, lr 0.05).
  6. GP (ARD): Gaussian Process with anisotropic RBF kernel (separate length scale per input).
  7. MLP: Multilayer Perceptron (2 hidden layers of 64 tanh units, L-BFGS solver).

Outputs:
  - results/baseline_ellipse_results.csv
  - results/baseline_ellipse_pred_vs_true.png
  - results/baseline_ellipse_extrapolation.png
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
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler

FEATS = ["W", "d_over_W", "ar", "H_over_W"]
TARGET = "kt_gross"


def transform_inv_ar(X):
    """
    Transforms feature matrix from [W, d_over_W, ar, H_over_W] to [d_over_W, 1/ar, H_over_W].
    Drops plate width W and inverts aspect ratio (b/a -> a/b) to align with Inglis theory.
    Supports both pandas DataFrames and NumPy 2D arrays.
    """
    if isinstance(X, pd.DataFrame):
        d_over_W = X["d_over_W"].values[:, None]
        inv_ar = (1.0 / X["ar"].values)[:, None]
        H_over_W = X["H_over_W"].values[:, None]
    else:
        d_over_W = X[:, 1:2]
        inv_ar = 1.0 / X[:, 2:3]
        H_over_W = X[:, 3:4]
    return np.hstack([d_over_W, inv_ar, H_over_W])


def make_models(seed=42, n_feats=4):
    """
    Instantiates the 7 surrogate models with identical hyperparameters to baselines.py.
    """
    return {
        "Mean predictor": DummyRegressor(strategy="mean"),
        "Linear": make_pipeline(StandardScaler(), LinearRegression()),
        "Poly3+Ridge": make_pipeline(StandardScaler(), PolynomialFeatures(3), Ridge(alpha=1e-3)),
        "Poly3 (1/ar)": make_pipeline(
            FunctionTransformer(transform_inv_ar),
            StandardScaler(),
            PolynomialFeatures(3),
            Ridge(alpha=1e-3),
        ),
        "GradBoost": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed
        ),
        "GP (ARD)": make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=C(1.0) * RBF([1.0] * n_feats) + WhiteKernel(1e-5, (1e-10, 1e-1)),
                normalize_y=True,
                n_restarts_optimizer=2,
                random_state=seed,
            ),
        ),
        "MLP": TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(64, 64),
                    activation="tanh",
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=seed,
                ),
            ),
            transformer=StandardScaler(),
        ),
    }


def metrics(y_true, y_pred):
    """
    Computes standard regression metrics and relative percentage errors.
    """
    rel = 100.0 * np.abs(y_pred - y_true) / y_true
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "mean_rel_err_%": float(rel.mean()),
        "max_rel_err_%": float(rel.max()),
        "R2": float(r2_score(y_true, y_pred)),
    }


def fit_eval(model, Xtr, ytr, Xte, yte):
    """
    Fits the model on training data, evaluates predictions on test data,
    and returns predictions with fit time (s) and batched prediction latency (us/sample).
    """
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred = model.predict(Xte)
    pred_us = 1e6 * (time.perf_counter() - t1) / len(Xte)  # microseconds per sample (batched)

    return pred, {"fit_s": fit_s, "predict_us_per_sample": pred_us}


def run_baselines(csv_path="data/plate_ellipse.csv", seed=42):
    """
    Main evaluation pipeline across interpolation and both extrapolation splits.
    """
    df = pd.read_csv(csv_path)
    n_all = len(df)
    df = df[df["ok"].astype(str) == "True"].reset_index(drop=True)
    print(f"{len(df)}/{n_all} rows usable from {csv_path}.")
    print(f"Inputs: {FEATS}. Target: {TARGET}")

    fea_time = (df["t_mesh"] + df["t_solve"]).mean()
    print(f"Mean FEA time per run (mesh + solve): {fea_time:.3f} s")
    os.makedirs("results", exist_ok=True)

    # =========================================================================
    # A) INTERPOLATION SPLIT (70/15/15 by row, seed 42)
    # =========================================================================
    X, y = df[FEATS].values, df[TARGET].values
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, random_state=seed)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, random_state=seed)

    print("\n" + "=" * 80)
    print("SPLIT A: Interpolation (Random 70/15/15 Split)")
    print(f"Train rows: {len(Xtr)} | Val rows: {len(Xva)} | Test rows: {len(Xte)}")
    print("=" * 80)

    rows = []
    preds_A = {}
    models_A = make_models(seed=seed, n_feats=len(FEATS))

    for name, m in models_A.items():
        print(f"  Training {name:<15} on interpolation split...", end="", flush=True)
        pte, timing = fit_eval(m, Xtr, ytr, Xte, yte)
        pva = m.predict(Xva)
        preds_A[name] = pte
        val = metrics(yva, pva)
        te = metrics(yte, pte)
        print(f" done in {timing['fit_s']:.2f}s | Test Mean Rel Err: {te['mean_rel_err_%']:.3f}%")
        rows.append({
            "split": "interp",
            "model": name,
            "val_mean_rel_err_%": val["mean_rel_err_%"],
            **te,
            **timing,
        })
    res_A = pd.DataFrame(rows)

    # GP Learned Length Scales for Interpolation
    gp_interp = models_A["GP (ARD)"].named_steps["gaussianprocessregressor"]
    ls_interp = gp_interp.kernel_.k1.k2.length_scale
    print("\nGP (ARD) learned length scales on Interpolation (inputs standardized):")
    for f, l in zip(FEATS, np.atleast_1d(ls_interp)):
        print(f"   {f:10s}: {l:.4g}")

    # =========================================================================
    # B1) EXTRAPOLATION IN HOLE SIZE: train d/W < 0.40, test d/W >= 0.45
    # =========================================================================
    tr_B1 = df[df["d_over_W"] < 0.40]
    te_B1 = df[df["d_over_W"] >= 0.45]
    print("\n" + "=" * 80)
    print("SPLIT B1: Extrapolation in Hole Size")
    print(f"Train rows (d/W < 0.40): {len(tr_B1)} | Test rows (d/W >= 0.45): {len(te_B1)}")
    print("=" * 80)

    rows_B1 = []
    preds_B1 = {}
    models_B1 = make_models(seed=seed, n_feats=len(FEATS))

    for name, m in models_B1.items():
        print(f"  Training {name:<15} on B1 (hole size)...", end="", flush=True)
        p, timing = fit_eval(
            m,
            tr_B1[FEATS].values,
            tr_B1[TARGET].values,
            te_B1[FEATS].values,
            te_B1[TARGET].values,
        )
        preds_B1[name] = p
        te = metrics(te_B1[TARGET].values, p)
        print(f" done in {timing['fit_s']:.2f}s | Test Mean Rel Err: {te['mean_rel_err_%']:.3f}%")
        rows_B1.append({
            "split": "extrap_hole",
            "model": name,
            "val_mean_rel_err_%": np.nan,
            **te,
            **timing,
        })
    res_B1 = pd.DataFrame(rows_B1)

    # GP Learned Length Scales for B1
    gp_B1 = models_B1["GP (ARD)"].named_steps["gaussianprocessregressor"]
    ls_B1 = gp_B1.kernel_.k1.k2.length_scale
    print("\nGP (ARD) learned length scales on B1 (hole size, inputs standardized):")
    for f, l in zip(FEATS, np.atleast_1d(ls_B1)):
        print(f"   {f:10s}: {l:.4g}")

    # =========================================================================
    # B2) EXTRAPOLATION TO SHARPER NOTCHES: train ar >= 0.8, test ar < 0.7
    # =========================================================================
    tr_B2 = df[df["ar"] >= 0.8]
    te_B2 = df[df["ar"] < 0.7]
    print("\n" + "=" * 80)
    print("SPLIT B2: Extrapolation to Sharper Notches")
    print(f"Train rows (ar >= 0.8): {len(tr_B2)} | Test rows (ar < 0.7): {len(te_B2)}")
    print("=" * 80)

    rows_B2 = []
    preds_B2 = {}
    models_B2 = make_models(seed=seed, n_feats=len(FEATS))

    for name, m in models_B2.items():
        print(f"  Training {name:<15} on B2 (sharper notches)...", end="", flush=True)
        p, timing = fit_eval(
            m,
            tr_B2[FEATS].values,
            tr_B2[TARGET].values,
            te_B2[FEATS].values,
            te_B2[TARGET].values,
        )
        preds_B2[name] = p
        te = metrics(te_B2[TARGET].values, p)
        print(f" done in {timing['fit_s']:.2f}s | Test Mean Rel Err: {te['mean_rel_err_%']:.3f}%")
        rows_B2.append({
            "split": "extrap_notch",
            "model": name,
            "val_mean_rel_err_%": np.nan,
            **te,
            **timing,
        })
    res_B2 = pd.DataFrame(rows_B2)

    # GP Learned Length Scales for B2
    gp_B2 = models_B2["GP (ARD)"].named_steps["gaussianprocessregressor"]
    ls_B2 = gp_B2.kernel_.k1.k2.length_scale
    print("\nGP (ARD) learned length scales on B2 (sharper notches, inputs standardized):")
    for f, l in zip(FEATS, np.atleast_1d(ls_B2)):
        print(f"   {f:10s}: {l:.4g}")

    # =========================================================================
    # COMBINE RESULTS AND SAVE CSV
    # =========================================================================
    res_all = pd.concat([res_A, res_B1, res_B2], ignore_index=True)
    out_csv = "results/baseline_ellipse_results.csv"
    res_all.to_csv(out_csv, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    print("\n" + "=" * 105)
    print("=== RESULTS SUMMARY: PLATE WITH ELLIPTICAL HOLE ===")
    print("=" * 105)
    print(res_all.drop(columns=["val_mean_rel_err_%"]).to_string(index=False))
    print("-" * 105)

    # Best models per split
    best_A = res_A.sort_values("mean_rel_err_%").iloc[0]
    speedup_A = fea_time / (best_A["predict_us_per_sample"] * 1e-6)
    print(f"\nBest Interpolation Model:      {best_A['model']} (mean rel err: {best_A['mean_rel_err_%']:.3f}%) | "
          f"Speedup vs FEA: {speedup_A:,.0f}x")

    best_B1 = res_B1.sort_values("mean_rel_err_%").iloc[0]
    print(f"Best B1 (Hole Size) Extrap:   {best_B1['model']} (mean rel err: {best_B1['mean_rel_err_%']:.3f}%)")

    best_B2 = res_B2.sort_values("mean_rel_err_%").iloc[0]
    print(f"Best B2 (Sharper Notch) Extrap:{best_B2['model']} (mean rel err: {best_B2['mean_rel_err_%']:.3f}%)")

    # =========================================================================
    # PLOTS
    # =========================================================================
    # 1. Predicted vs True figure for Interpolation (7 subplots)
    fig, axes = plt.subplots(1, len(preds_A), figsize=(3.5 * len(preds_A), 3.5), sharex=True, sharey=True)
    lo, hi = yte.min() * 0.95, yte.max() * 1.05
    for ax, (name, p) in zip(np.atleast_1d(axes), preds_A.items()):
        ax.scatter(yte, p, s=12, alpha=0.6, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("FEA Kt (true)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(True, linestyle=":", alpha=0.5)
    np.atleast_1d(axes)[0].set_ylabel("Model Kt (predicted)")
    fig.tight_layout()
    fig_pred_path = "results/baseline_ellipse_pred_vs_true.png"
    fig.savefig(fig_pred_path, dpi=150)
    plt.close(fig)
    shutil.copyfile(fig_pred_path, "results/baselines_ellipse_pred_vs_true.png")

    # 2. Extrapolation Error figure: 2 subplots (B1: error vs d_over_W, B2: error vs ar)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # B1 panel: relative error vs d_over_W
    order_B1 = np.argsort(te_B1["d_over_W"].values)
    x_B1 = te_B1["d_over_W"].values[order_B1]
    for name, p in preds_B1.items():
        rel = 100.0 * (p - te_B1[TARGET].values) / te_B1[TARGET].values
        ax1.plot(x_B1, rel[order_B1], "o-", markersize=3.5, alpha=0.65, lw=1.2, label=name)
    ax1.axhline(0, color="k", lw=0.8, linestyle="--")
    ax1.set_xlabel("d/W (unseen test range >= 0.45)")
    ax1.set_ylabel("Relative prediction error (%)")
    ax1.set_title("B1: Hole size extrapolation (d/W >= 0.45)")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(fontsize=8)

    # B2 panel: relative error vs ar
    order_B2 = np.argsort(te_B2["ar"].values)
    x_B2 = te_B2["ar"].values[order_B2]
    for name, p in preds_B2.items():
        rel = 100.0 * (p - te_B2[TARGET].values) / te_B2[TARGET].values
        ax2.plot(x_B2, rel[order_B2], "o-", markersize=3.5, alpha=0.65, lw=1.2, label=name)
    ax2.axhline(0, color="k", lw=0.8, linestyle="--")
    ax2.set_xlabel("ar = b/a (unseen sharper notches < 0.7)")
    ax2.set_ylabel("Relative prediction error (%)")
    ax2.set_title("B2: Notch sharpness extrapolation (ar < 0.7)")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig_extrap_path = "results/baseline_ellipse_extrapolation.png"
    fig.savefig(fig_extrap_path, dpi=150)
    plt.close(fig)
    shutil.copyfile(fig_extrap_path, "results/baselines_ellipse_extrapolation.png")

    print(f"\nSaved:")
    print(f"  - {out_csv}")
    print(f"  - {fig_pred_path}")
    print(f"  - {fig_extrap_path}")
    print("=" * 105 + "\n")

    return res_all


def main():
    parser = argparse.ArgumentParser(
        description="Baseline surrogate models for plate with an elliptical hole."
    )
    parser.add_argument("--csv", default="data/plate_ellipse.csv", help="Path to elliptical plate CSV dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting and model training")
    args = parser.parse_args()

    run_baselines(csv_path=args.csv, seed=args.seed)


if __name__ == "__main__":
    main()
