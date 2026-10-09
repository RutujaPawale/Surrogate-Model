"""
Step 2: baseline scalar surrogates for the plate-with-a-hole dataset.

Target : kt_gross = peak von Mises stress / applied stress  (geometry-only quantity,
         because the problem is linear; peak stress = kt_gross * sigma)
Inputs : W, d_over_W, H_over_W   (W should turn out to be irrelevant; that is a physics check)

Two evaluations:
  A) Interpolation : random 70/15/15 split (one row = one geometry, so no leakage)
  B) Extrapolation : train on d_over_W < 0.40, test on d_over_W >= 0.45 (outside training range)

Run from the surrogate_fea folder:
    .\\venv\\Scripts\\python.exe models\\baselines.py
    .\\venv\\Scripts\\python.exe models\\baselines.py --csv data\\plate_hole.csv --seed 42
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FEATS = ["W", "d_over_W", "H_over_W"]
TARGET = "kt_gross"


def make_models(seed, n_feats):
    return {
        "Linear": make_pipeline(StandardScaler(), LinearRegression()),
        "Poly3+Ridge": make_pipeline(StandardScaler(), PolynomialFeatures(3), Ridge(alpha=1e-3)),
        "GradBoost": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed),
        "GP (ARD)": make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=C(1.0) * RBF([1.0] * n_feats) + WhiteKernel(1e-5, (1e-10, 1e-1)),
                normalize_y=True, n_restarts_optimizer=2, random_state=seed)),
        "MLP": TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", solver="lbfgs",
                             max_iter=3000, random_state=seed)),
            transformer=StandardScaler()),
    }


def metrics(y_true, y_pred):
    rel = 100 * np.abs(y_pred - y_true) / y_true
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "mean_rel_err_%": rel.mean(),
        "max_rel_err_%": rel.max(),
        "R2": r2_score(y_true, y_pred),
    }


def fit_eval(model, Xtr, ytr, Xte, yte):
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    pred = model.predict(Xte)
    pred_us = 1e6 * (time.perf_counter() - t1) / len(Xte)   # microseconds per sample (batched)
    return pred, {"fit_s": fit_s, "predict_us_per_sample": pred_us}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/plate_hole.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_W", action="store_true", help="drop W from the inputs")
    a = ap.parse_args()

    feats = [f for f in FEATS if not (a.no_W and f == "W")]
    df = pd.read_csv(a.csv)
    n_all = len(df)
    df = df[df["ok"].astype(str) == "True"].reset_index(drop=True)
    print(f"{len(df)}/{n_all} rows usable. Inputs: {feats}. Target: {TARGET}")

    fea_time = (df["t_mesh"] + df["t_solve"]).mean()
    print(f"Mean FEA time per run (mesh + solve): {fea_time:.3f} s")
    os.makedirs("results", exist_ok=True)

    # ---------- A) interpolation: random split by geometry ----------
    X, y = df[feats].values, df[TARGET].values
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, random_state=a.seed)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, random_state=a.seed)
    print(f"\nA) Interpolation split: train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

    rows, preds_A = [], {}
    models = make_models(a.seed, len(feats))
    for name, m in models.items():
        pte, timing = fit_eval(m, Xtr, ytr, Xte, yte)
        pva = m.predict(Xva)
        preds_A[name] = pte
        val = metrics(yva, pva)
        te = metrics(yte, pte)
        rows.append({"split": "interp", "model": name,
                     "val_mean_rel_err_%": val["mean_rel_err_%"], **te, **timing})
    res_A = pd.DataFrame(rows)

    # ---------- B) extrapolation: unseen range of d_over_W ----------
    tr = df[df["d_over_W"] < 0.40]
    te_ = df[df["d_over_W"] >= 0.45]
    print(f"B) Extrapolation: train d/W<0.40 ({len(tr)} rows), test d/W>=0.45 ({len(te_)} rows)")
    rows, preds_B = [], {}
    models_B = make_models(a.seed, len(feats))
    for name, m in models_B.items():
        p, timing = fit_eval(m, tr[feats].values, tr[TARGET].values,
                             te_[feats].values, te_[TARGET].values)
        preds_B[name] = p
        rows.append({"split": "extrap", "model": name, "val_mean_rel_err_%": np.nan,
                     **metrics(te_[TARGET].values, p), **timing})
    res_B = pd.DataFrame(rows)

    res = pd.concat([res_A, res_B], ignore_index=True)
    res.to_csv("results/baseline_results.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    print("\n=== Results (interp = random split test set, extrap = unseen d/W range) ===")
    print(res.drop(columns=["val_mean_rel_err_%"]).to_string(index=False))

    # ---------- speedup ----------
    best = res_A.sort_values("mean_rel_err_%").iloc[0]
    speedup = fea_time / (best["predict_us_per_sample"] * 1e-6)
    print(f"\nBest interpolation model: {best['model']} "
          f"(mean rel err {best['mean_rel_err_%']:.3f}%). "
          f"Speedup vs FEA (batched inference): about {speedup:,.0f}x")

    # ---------- is W irrelevant? (GP length scales: large = input barely matters) ----------
    gp = models["GP (ARD)"].named_steps["gaussianprocessregressor"]
    ls = gp.kernel_.k1.k2.length_scale
    print("\nGP learned length scales (large = input barely matters):")
    for f, l in zip(feats, np.atleast_1d(ls)):
        print(f"   {f:10s} {l:.3g}")

    # ---------- plots ----------
    fig, axes = plt.subplots(1, len(preds_A), figsize=(4 * len(preds_A), 4), sharex=True, sharey=True)
    lo, hi = yte.min(), yte.max()
    for ax, (name, p) in zip(np.atleast_1d(axes), preds_A.items()):
        ax.scatter(yte, p, s=8)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(name)
        ax.set_xlabel("FEA Kt (true)")
    np.atleast_1d(axes)[0].set_ylabel("Model Kt (predicted)")
    fig.tight_layout()
    fig.savefig("results/baselines_pred_vs_true.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, p in preds_B.items():
        rel = 100 * (p - te_[TARGET].values) / te_[TARGET].values
        order = np.argsort(te_["d_over_W"].values)
        ax.plot(te_["d_over_W"].values[order], rel[order], label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("d/W (outside the training range)")
    ax.set_ylabel("Prediction error (%)")
    ax.set_title("Extrapolation error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/baselines_extrapolation.png", dpi=150)
    print("\nSaved: results/baseline_results.csv, baselines_pred_vs_true.png, baselines_extrapolation.png")


if __name__ == "__main__":
    main()
