"""
Step 1: FEA data generation for the plate-with-a-hole surrogate project.

Problem
-------
Rectangular plate (width W, height H) with a central circular hole (radius r),
pulled by a uniform tensile stress `sigma` on the top/bottom edges.
Linear elastic, plane stress. We model ONE QUARTER of the plate using symmetry:
    - left edge   (x = 0): u_x = 0
    - bottom edge (y = 0): u_y = 0
    - top edge    (y = H/2): traction sigma in +y
    - right edge and hole edge: free

Output of interest: peak von Mises stress (it occurs at the hole edge, (r, 0)).

Install (run on your own machine):
    pip install scikit-fem gmsh numpy scipy pandas

Usage:
    python gen_data.py --verify            # check against theory + mesh convergence
    python gen_data.py --n 500 --out data/plate_hole.csv

NOTE: written without being able to run it in the authoring environment.
Run --verify first and fix anything that breaks before generating data.
"""
import argparse
import os
import time

import gmsh
import numpy as np
import pandas as pd
from scipy.stats import qmc
from skfem import (Basis, ElementTriP1, ElementTriP2, ElementVector, FacetBasis,
                   LinearForm, MeshTri, asm, condense, solve)
from skfem.models.elasticity import lame_parameters, linear_elasticity

# ---- material (steel-like, units: mm, N, MPa) --------------------------------
E = 210e3      # Young's modulus [MPa]
NU = 0.30      # Poisson's ratio

# ---- parameter ranges for the dataset ----------------------------------------
RANGES = {
    "W": (60.0, 200.0),         # plate width [mm]
    "d_over_W": (0.05, 0.50),   # hole diameter / width (keep < 0.5 so Peterson's fit holds)
    "H_over_W": (2.0, 3.0),     # aspect ratio; >= 2 keeps end effects away from the hole
    "sigma": (50.0, 200.0),     # applied gross stress [MPa]
}


# ---- mesh --------------------------------------------------------------------
def make_mesh(W, H, r, n_hole=12, far_div=10):
    """Quarter plate with quarter hole, refined near the hole. Returns skfem MeshTri."""
    h_hole = (np.pi * r / 2) / n_hole   # element size along the hole edge
    h_far = W / far_div

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("plate")
        occ = gmsh.model.occ
        rect = occ.addRectangle(0, 0, 0, W / 2, H / 2)
        disk = occ.addDisk(0, 0, 0, r, r)
        occ.cut([(2, rect)], [(2, disk)])
        occ.synchronize()

        # find the hole arc: the curve whose bounding box sits inside [0, r] x [0, r]
        hole_curves = []
        for _, tag in gmsh.model.getEntities(1):
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, tag)
            if xmax <= r * 1.001 and ymax <= r * 1.001:
                hole_curves.append(tag)
        assert hole_curves, "hole arc not found"

        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", hole_curves)
        gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 200)
        f_thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thr, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", h_hole)
        gmsh.model.mesh.field.setNumber(f_thr, "SizeMax", h_far)
        gmsh.model.mesh.field.setNumber(f_thr, "DistMin", 0.2 * r)
        gmsh.model.mesh.field.setNumber(f_thr, "DistMax", 1.5 * W / 2)
        gmsh.model.mesh.field.setAsBackgroundMesh(f_thr)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        gmsh.model.mesh.generate(2)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords).reshape(-1, 3)[:, :2]
        node_tags = np.asarray(node_tags)
        _, _, elem_nodes = gmsh.model.mesh.getElements(2)
        tri_tags = np.asarray(elem_nodes[0]).reshape(-1, 3)
    finally:
        gmsh.finalize()

    # map gmsh node tags -> 0..N-1 and drop nodes not used by any triangle
    lookup = {int(t): i for i, t in enumerate(node_tags)}
    tri = np.vectorize(lookup.get)(tri_tags)
    used = np.unique(tri)
    remap = -np.ones(len(coords), dtype=int)
    remap[used] = np.arange(len(used))
    mesh = MeshTri(coords[used].T.copy(), remap[tri].T.copy())

    mesh = mesh.with_boundaries({
        "left": lambda x: np.isclose(x[0], 0.0),
        "bottom": lambda x: np.isclose(x[1], 0.0),
        "top": lambda x: np.isclose(x[1], H / 2),
    })
    return mesh


# ---- solver ------------------------------------------------------------------
def solve_plate(W, H, r, sigma, n_hole=12, return_field=False):
    """Run one FEA. Returns dict with peak von Mises stress and timings."""
    t0 = time.perf_counter()
    mesh = make_mesh(W, H, r, n_hole=n_hole)
    t_mesh = time.perf_counter() - t0

    t1 = time.perf_counter()
    basis = Basis(mesh, ElementVector(ElementTriP2()))

    lam, mu = lame_parameters(E, NU)
    lam = 2 * lam * mu / (lam + 2 * mu)          # plane stress correction
    K = asm(linear_elasticity(lam, mu), basis)

    fb = FacetBasis(mesh, basis.elem, facets=mesh.boundaries["top"])

    @LinearForm
    def traction(v, w):
        return sigma * v.value[1]

    f = asm(traction, fb)

    def comp_dofs(facets, comp):
        d = basis.get_dofs(facets)
        return np.concatenate([d.nodal[comp], d.facet[comp]])

    D = np.concatenate([
        comp_dofs(mesh.boundaries["left"], "u^1"),    # u_x = 0 on x = 0
        comp_dofs(mesh.boundaries["bottom"], "u^2"),  # u_y = 0 on y = 0
    ])
    u = solve(*condense(K, f, D=D))

    # stresses at quadrature points (plane stress, Hooke's law)
    g = basis.interpolate(u).grad                    # shape (2, 2, nel, nq): du_i/dx_j
    exx, eyy = g[0, 0], g[1, 1]
    exy = 0.5 * (g[0, 1] + g[1, 0])
    c = E / (1 - NU ** 2)
    sxx = c * (exx + NU * eyy)
    syy = c * (eyy + NU * exx)
    sxy = E / (1 + NU) * exy
    vm = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3 * sxy ** 2)

    # smooth to nodes (L2 projection onto P1) so the peak is not underestimated
    p1 = Basis(mesh, ElementTriP1(), quadrature=basis.quadrature)
    vm_nodal = p1.project(vm)
    t_solve = time.perf_counter() - t1

    res = {
        "peak_vm": float(vm_nodal.max()),
        "n_nodes": int(mesh.p.shape[1]),
        "n_elems": int(mesh.t.shape[1]),
        "t_mesh": t_mesh,
        "t_solve": t_solve,
    }
    if return_field:
        res["mesh"] = mesh
        res["W"] = W
        res["r"] = r
        res["sigma"] = sigma
        res["vm_nodal"] = vm_nodal
    return res


# ---- theory (for validation) -------------------------------------------------
def peak_stress_theory(sigma, d_over_W):
    """Peterson's fit for a finite-width plate with a central hole, tension.
    Kt_net = 3.00 - 3.13 (d/W) + 3.66 (d/W)^2 - 1.53 (d/W)^3, based on NET section stress.
    Peak stress = Kt_net * sigma_net, with sigma_net = sigma * W / (W - d).
    """
    x = d_over_W
    kt_net = 3.00 - 3.13 * x + 3.66 * x ** 2 - 1.53 * x ** 3
    return kt_net * sigma / (1 - x)


# ---- modes -------------------------------------------------------------------
def verify():
    print("1) Theory check (expect error of a few % or less; mesh-dependent)")
    for x in (0.05, 0.2, 0.4):
        W, H, sigma = 100.0, 250.0, 100.0
        r = x * W / 2
        res = solve_plate(W, H, r, sigma, n_hole=16)
        th = peak_stress_theory(sigma, x)
        err = 100 * (res["peak_vm"] - th) / th
        print(f"   d/W={x:.2f}  FEA={res['peak_vm']:.1f}  theory={th:.1f}  err={err:+.2f}%  "
              f"({res['n_elems']} elems, {res['t_mesh'] + res['t_solve']:.2f}s)")

    print("2) Mesh convergence (peak stress should flatten out as n_hole grows)")
    W, H, sigma, x = 100.0, 250.0, 100.0, 0.2
    r = x * W / 2
    for n in (4, 8, 16, 32, 64):
        res = solve_plate(W, H, r, sigma, n_hole=n)
        print(f"   n_hole={n:3d}  peak={res['peak_vm']:.2f}  elems={res['n_elems']}")
    print("Pick n_hole where the change drops below ~1% and use it for the dataset.")


def generate(n, out, n_hole, seed):
    names = list(RANGES)
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    lo = [RANGES[k][0] for k in names]
    hi = [RANGES[k][1] for k in names]
    X = qmc.scale(sampler.random(n), lo, hi)

    rows = []
    for i, vals in enumerate(X):
        p = dict(zip(names, vals))
        W = p["W"]
        H = p["H_over_W"] * W
        r = p["d_over_W"] * W / 2
        row = {"id": i, **p, "H": H, "r": r}
        try:
            res = solve_plate(W, H, r, p["sigma"], n_hole=n_hole)
            th = peak_stress_theory(p["sigma"], p["d_over_W"])
            row.update(res)
            row["peak_theory"] = th
            row["err_vs_theory_pct"] = 100 * (res["peak_vm"] - th) / th
            row["kt_gross"] = res["peak_vm"] / p["sigma"]
            row["ok"] = True
        except Exception as e:  # log failures instead of silently dropping them
            row["ok"] = False
            row["error"] = repr(e)
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{n} done")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"saved {out}: {int(df['ok'].sum())}/{n} runs succeeded")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="data/plate_hole.csv")
    ap.add_argument("--n_hole", type=int, default=16, help="elements along the hole arc")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.verify:
        verify()
    else:
        generate(a.n, a.out, a.n_hole, a.seed)
