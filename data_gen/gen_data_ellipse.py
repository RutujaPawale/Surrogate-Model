"""
FEA data generation and validation for plate with a central elliptical hole.

Problem
-------
Rectangular plate (width W, height H) with a central elliptical hole:
    - semi-axis a along x (perpendicular to load direction)
    - semi-axis b along y (parallel to load direction)
Parameters:
    - d_over_W = 2a / W
    - ar = b / a (aspect ratio of the hole)

Pulled by uniform tensile stress `sigma` on top/bottom edges in the y-direction.
Linear elastic, plane stress (E = 210 GPa, NU = 0.30).
Modeled as ONE QUARTER of the plate using symmetry:
    - left edge   (x = 0): u_x = 0
    - bottom edge (y = 0): u_y = 0
    - top edge    (y = H/2): traction sigma in +y
    - right edge and ellipse edge: traction-free

Mesh:
    - Built in gmsh with OpenCASCADE (occ.addEllipse + plane surface)
    - Cut from rectangular quarter-plate [0, W/2] x [0, H/2]
    - Hole curve identified by bounding box within [0, a] x [0, b]
    - Local refinement using Distance and Threshold fields
    - Optional tip refinement (--tip_refine) using notch radius of curvature b^2/a
    - Quadratic (P2) triangular elements

Post-processing:
    - Quadrature-point von Mises stress projected via L2 onto continuous P1 nodal field
"""
import argparse
import os
import sys
import time

import gmsh
import numpy as np
import pandas as pd
from scipy.stats import qmc
from skfem import (Basis, ElementTriP1, ElementTriP2, ElementVector, FacetBasis,
                   LinearForm, MeshTri, asm, condense, solve)
from skfem.models.elasticity import lame_parameters, linear_elasticity

# Material (steel-like, units: mm, N, MPa)
E = 210e3      # Young's modulus [MPa]
NU = 0.30      # Poisson's ratio

# Parameter ranges for dataset generation
RANGES = {
    "W": (60.0, 200.0),         # plate width [mm]
    "d_over_W": (0.05, 0.50),   # 2a / W
    "ar": (0.5, 2.0),           # b / a
    "H_over_W": (2.5, 4.0),     # aspect ratio H / W
    "sigma": (50.0, 200.0),     # applied gross stress [MPa]
}


def ellipse_arc_length(a, b):
    """
    Ramanujan's approximation for the perimeter of a quarter-ellipse arc.
    Exact for a circle (a = b): returns pi * a / 2.
    """
    return (np.pi / 4.0) * (3.0 * (a + b) - np.sqrt((3.0 * a + b) * (a + 3.0 * b)))


def make_mesh(W, H, a, b, n_hole=16, far_div=10, tip_refine=False, n_tip=8):
    """
    Quarter plate with quarter elliptical hole, refined near the hole boundary.
    If tip_refine is True, limits element size near the notch root (a, 0)
    using the radius of curvature rho_tip = b^2 / a.
    Returns an skfem MeshTri object with boundary facets: 'left', 'bottom', 'top'.
    """
    h_hole = ellipse_arc_length(a, b) / n_hole   # target element size along ellipse boundary
    tip_radius = (b ** 2) / a

    if tip_refine:
        tip_size = tip_radius * (np.pi / 2.0) / n_tip
        h_hole = min(h_hole, tip_size)
        r_ref = min(a, b, tip_radius)
    else:
        r_ref = min(a, b)

    h_far = W / far_div                          # coarse element size far away

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("plate_ellipse")
        occ = gmsh.model.occ

        # Base quarter plate rectangle [0, W/2] x [0, H/2]
        rect = occ.addRectangle(0, 0, 0, W / 2.0, H / 2.0)

        # Build full ellipse surface centered at origin
        # OpenCASCADE requires major radius >= minor radius
        if a >= b:
            # Major radius a along x-axis, minor radius b along y-axis
            e = occ.addEllipse(0, 0, 0, a, b, zAxis=[0, 0, 1], xAxis=[1, 0, 0])
        else:
            # Major radius b along y-axis, minor radius a along x-axis
            e = occ.addEllipse(0, 0, 0, b, a, zAxis=[0, 0, 1], xAxis=[0, 1, 0])

        w = occ.addWire([e])
        s = occ.addPlaneSurface([w])

        # Cut elliptical void from rectangle
        occ.cut([(2, rect)], [(2, s)])
        occ.synchronize()

        # Find hole curve: the unique curve whose bounding box sits within [0, a] x [0, b]
        hole_curves = []
        for _, tag in gmsh.model.getEntities(1):
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, tag)
            if xmax <= a * 1.001 and ymax <= b * 1.001:
                hole_curves.append(tag)
        assert hole_curves, f"Hole arc not found for a={a}, b={b}"

        # Refinement field near the hole using Distance and Threshold
        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", hole_curves)
        gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 200)

        f_thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thr, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", h_hole)
        gmsh.model.mesh.field.setNumber(f_thr, "SizeMax", h_far)
        gmsh.model.mesh.field.setNumber(f_thr, "DistMin", 0.2 * r_ref)
        gmsh.model.mesh.field.setNumber(f_thr, "DistMax", 1.5 * W / 2.0)
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

    # Map gmsh node tags -> 0..N-1 and filter unused nodes
    lookup = {int(t): i for i, t in enumerate(node_tags)}
    tri = np.vectorize(lookup.get)(tri_tags)
    used = np.unique(tri)
    remap = -np.ones(len(coords), dtype=int)
    remap[used] = np.arange(len(used))
    mesh = MeshTri(coords[used].T.copy(), remap[tri].T.copy())

    mesh = mesh.with_boundaries({
        "left": lambda x: np.isclose(x[0], 0.0),
        "bottom": lambda x: np.isclose(x[1], 0.0),
        "top": lambda x: np.isclose(x[1], H / 2.0),
    })
    return mesh


def solve_plate(W, H, a, b, sigma, n_hole=16, return_field=False, tip_refine=False, n_tip=8):
    """
    Solves linear elasticity on a quarter plate with an elliptical hole.
    Returns dict with peak von Mises stress, element/node counts, and runtimes.
    """
    t0 = time.perf_counter()
    mesh = make_mesh(W, H, a, b, n_hole=n_hole, tip_refine=tip_refine, n_tip=n_tip)
    t_mesh = time.perf_counter() - t0

    t1 = time.perf_counter()
    basis = Basis(mesh, ElementVector(ElementTriP2()))

    # Plane stress correction for 2D isotropic elasticity
    lam, mu = lame_parameters(E, NU)
    lam = 2.0 * lam * mu / (lam + 2.0 * mu)
    K = asm(linear_elasticity(lam, mu), basis)

    fb = FacetBasis(mesh, basis.elem, facets=mesh.boundaries["top"])

    @LinearForm
    def traction(v, w):
        return sigma * v.value[1]

    f = asm(traction, fb)

    def comp_dofs(facets, comp):
        d = basis.get_dofs(facets)
        return np.concatenate([d.nodal[comp], d.facet[comp]])

    # Symmetry Dirichlet boundary conditions
    D = np.concatenate([
        comp_dofs(mesh.boundaries["left"], "u^1"),    # u_x = 0 on x = 0
        comp_dofs(mesh.boundaries["bottom"], "u^2"),  # u_y = 0 on y = 0
    ])
    u = solve(*condense(K, f, D=D))

    # Quadrature-point stresses (plane stress, Hooke's law)
    g = basis.interpolate(u).grad
    exx, eyy = g[0, 0], g[1, 1]
    exy = 0.5 * (g[0, 1] + g[1, 0])
    c = E / (1.0 - NU ** 2)
    sxx = c * (exx + NU * eyy)
    syy = c * (eyy + NU * exx)
    sxy = E / (1.0 + NU) * exy
    vm = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2)

    # Smooth quadrature stresses to nodes (L2 projection onto P1)
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
        res["H"] = H
        res["a"] = a
        res["b"] = b
        res["sigma"] = sigma
        res["vm_nodal"] = vm_nodal
    return res


def inglis_stress(sigma, a, b):
    """
    Theoretical maximum stress for an elliptical hole in an infinite plate under tension:
        sigma_max = sigma * (1 + 2 * a / b)
    where:
        a is semi-axis perpendicular to load direction (along x)
        b is semi-axis parallel to load direction (along y)
    """
    return sigma * (1.0 + 2.0 * a / b)


def verify():
    """
    Verification mode:
        Plate: W = 200, H = 500, sigma = 100, d_over_W = 0.05 (a = 5 mm).
        ar in [0.5, 1.0, 1.5, 2.0] (where ar = b/a).
        n_hole in [16, 32, 64].
    Compares against Inglis theory sigma * (1 + 2*a/b).
    At ar = 1.0, also compares directly against circular-hole FEA from gen_data.py.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    from data_gen.gen_data import solve_plate as solve_plate_circle

    W = 200.0
    H = 500.0
    sigma = 100.0
    d_over_W = 0.05
    a = d_over_W * W / 2.0  # a = 5.0 mm

    print("=" * 86)
    print("ELLIPTICAL HOLE FEA VERIFICATION AGAINST INGLIS THEORY & CIRCULAR BENCHMARK")
    print(f"Plate Dimensions: W = {W:.1f} mm, H = {H:.1f} mm, sigma = {sigma:.1f} MPa")
    print(f"Hole Parameter:   d/W = 2a/W = {d_over_W:.3f} -> Semi-axis a = {a:.2f} mm")
    print("=" * 86)

    for ar in [0.5, 1.0, 1.5, 2.0]:
        b = ar * a
        th = inglis_stress(sigma, a, b)
        print(f"\nAspect Ratio ar = b/a = {ar:.2f} (b = {b:.2f} mm) | Inglis Value = {th:6.2f} MPa")
        print("-" * 86)
        print(f"  {'n_hole':>6} | {'FEA peak [MPa]':>14} | {'Inglis [MPa]':>12} | {'Diff [%]':>9} | {'Elems':>7} | {'Time [s]':>8}")
        print("-" * 86)

        for n_hole in [16, 32, 64]:
            res = solve_plate(W, H, a, b, sigma, n_hole=n_hole)
            diff_pct = 100.0 * (res["peak_vm"] - th) / th
            t_total = res["t_mesh"] + res["t_solve"]
            print(f"  {n_hole:6d} | {res['peak_vm']:14.2f} | {th:12.2f} | {diff_pct:+8.2f}% | {res['n_elems']:7d} | {t_total:7.2f}s")

            # At ar = 1.0, compare against circular-hole implementation
            if ar == 1.0:
                res_circ = solve_plate_circle(W, H, r=a, sigma=sigma, n_hole=n_hole)
                circ_diff_pct = 100.0 * (res["peak_vm"] - res_circ["peak_vm"]) / res_circ["peak_vm"]
                print(f"         --> vs Circle FEA (r={a:.1f}): Peak = {res_circ['peak_vm']:.2f} MPa | Diff = {circ_diff_pct:+.4f}% | Elems = {res_circ['n_elems']}")

    print("\n" + "=" * 86)
    print("Verification completed successfully.")
    print("=" * 86)


def verify2():
    """
    Advanced verification mode (--verify2):
    (a) Small aspect ratios ar in [0.35, 0.5, 0.75] with d/W = 0.05 across fine meshes
        n_hole in [32, 64, 128, 256].
        Evaluated BOTH without tip refinement and with tip refinement (--tip_refine).
        Reports peak von Mises, Inglis value, % diff, % change from previous n_hole,
        element count, and run time.
    (b) Finite-width circular holes (ar = 1.0) with d/W in [0.1, 0.2, 0.3, 0.4]
        at n_hole in [32, 64].
        Compared against Peterson's analytical formula (peak_stress_theory) from gen_data.py.
        Reports % diff vs Peterson and % change between n_hole=32 and n_hole=64.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    from data_gen.gen_data import peak_stress_theory

    W = 200.0
    H = 500.0
    sigma = 100.0

    print("=" * 96)
    print("VERIFY2: EXTENDED CONVERGENCE & COMPARATIVE ANALYSIS")
    print("=" * 96)

    # -------------------------------------------------------------------------
    # (1a) Convergence on small aspect ratios ar in [0.35, 0.5, 0.75]
    # -------------------------------------------------------------------------
    d_over_W_a = 0.05
    a_a = d_over_W_a * W / 2.0  # 5.0 mm
    n_holes_a = [32, 64, 128, 256]

    print("\n" + "#" * 96)
    print("(1a) CONVERGENCE ON SMALL ASPECT RATIOS: ar in [0.35, 0.5, 0.75], d/W=0.05 (a=5 mm)")
    print("#" * 96)

    for use_tip in [False, True]:
        mode_label = "WITH Tip Refinement (--tip_refine)" if use_tip else "WITHOUT Tip Refinement (Standard)"
        print(f"\n>>> MODE: {mode_label}")
        print("=" * 96)

        for ar in [0.35, 0.5, 0.75]:
            b = ar * a_a
            th = inglis_stress(sigma, a_a, b)
            tip_rad = (b ** 2) / a_a

            print(f"\nAspect Ratio ar = {ar:.2f} (a = {a_a:.2f} mm, b = {b:.2f} mm, tip radius = {tip_rad:.4f} mm)")
            print(f"Inglis Theoretical Peak = {th:.2f} MPa")
            print("-" * 96)
            print(f"  {'n_hole':>6} | {'FEA peak [MPa]':>14} | {'Inglis [MPa]':>12} | {'Diff [%]':>9} | {'% Change':>10} | {'Elems':>7} | {'Time [s]':>8}")
            print("-" * 96)

            prev_peak = None
            for n_hole in n_holes_a:
                res = solve_plate(W, H, a_a, b, sigma, n_hole=n_hole, tip_refine=use_tip)
                diff_pct = 100.0 * (res["peak_vm"] - th) / th

                if prev_peak is not None:
                    pct_change = 100.0 * (res["peak_vm"] - prev_peak) / prev_peak
                    change_str = f"{pct_change:+9.2f}%"
                else:
                    change_str = "       ---"

                t_total = res["t_mesh"] + res["t_solve"]
                print(f"  {n_hole:6d} | {res['peak_vm']:14.2f} | {th:12.2f} | {diff_pct:+8.2f}% | {change_str} | {res['n_elems']:7d} | {t_total:7.2f}s")
                prev_peak = res["peak_vm"]

    # -------------------------------------------------------------------------
    # (1b) Finite-width circular holes (ar=1.0) compared with Peterson's formula
    # -------------------------------------------------------------------------
    print("\n" + "#" * 96)
    print("(1b) CIRCULAR HOLE FINITE-WIDTH CHECK: ar=1.0, d/W in [0.1, 0.2, 0.3, 0.4] vs Peterson")
    print("#" * 96)
    print(f"  {'d/W':>5} | {'Peterson':>9} | {'Peak n=32':>10} | {'Diff32':>8} | {'Peak n=64':>10} | {'Diff64':>8} | {'% Change':>9} | {'Elems32':>7} | {'Elems64':>7}")
    print("-" * 96)

    for dw in [0.1, 0.2, 0.3, 0.4]:
        a_b = dw * W / 2.0
        b_b = 1.0 * a_b
        th_peterson = peak_stress_theory(sigma, dw)

        res32 = solve_plate(W, H, a_b, b_b, sigma, n_hole=32)
        res64 = solve_plate(W, H, a_b, b_b, sigma, n_hole=64)

        diff32 = 100.0 * (res32["peak_vm"] - th_peterson) / th_peterson
        diff64 = 100.0 * (res64["peak_vm"] - th_peterson) / th_peterson
        pct_change = 100.0 * (res64["peak_vm"] - res32["peak_vm"]) / res32["peak_vm"]

        print(
            f"  {dw:5.2f} | {th_peterson:8.2f}M | {res32['peak_vm']:9.2f}M | {diff32:+7.2f}% | "
            f"{res64['peak_vm']:9.2f}M | {diff64:+7.2f}% | {pct_change:+8.2f}% | "
            f"{res32['n_elems']:7d} | {res64['n_elems']:7d}"
        )

    print("\n" + "=" * 96)
    print("Verify2 completed successfully.")
    print("=" * 96)


def selfcheck(n_samples=20, n_hole=32, seed=42, tip_refine=False):
    """
    Self-check mode (--selfcheck N):
    Samples N geometries by Latin Hypercube Sampling:
        - d_over_W in [0.05, 0.50]
        - ar in [0.5, 2.0]
        - H_over_W in [2.0, 3.0]
        - W in [60.0, 200.0]
    Solves each sample at n_hole and 2*n_hole (default 32 and 64),
    with tip refinement enabled if specified.
    Prints:
        - % change in peak stress per sample
        - Mean and max % change
        - The 5 worst samples (largest relative change) with their parameters.
    """
    param_names = ["W", "d_over_W", "ar", "H_over_W"]
    ranges_check = {
        "W": (60.0, 200.0),
        "d_over_W": (0.05, 0.50),
        "ar": (0.5, 2.0),
        "H_over_W": (2.0, 3.0),
    }

    sigma = 100.0  # Constant load (Kt is load-independent)
    n1 = n_hole
    n2 = 2 * n_hole

    print("=" * 105)
    print(f"SELFCHECK MODE: {n_samples} LHS GEOMETRIES EVALUATED AT n_hole={n1} vs {n2} (seed={seed})")
    print(f"Tip Refinement: {'ENABLED (--tip_refine)' if tip_refine else 'DISABLED'}")
    print("=" * 105)

    sampler = qmc.LatinHypercube(d=len(param_names), seed=seed)
    lo = [ranges_check[k][0] for k in param_names]
    hi = [ranges_check[k][1] for k in param_names]
    X = qmc.scale(sampler.random(n_samples), lo, hi)

    records = []
    print(f"\n  {'ID':>3} | {'W [mm]':>7} | {'d/W':>6} | {'ar':>5} | {'H/W':>5} | {'a [mm]':>6} | {'b [mm]':>6} | {'Peak n1':>8} | {'Peak n2':>8} | {'% Change':>9} | {'Elems1':>6} | {'Elems2':>6}")
    print("-" * 105)

    for i, vals in enumerate(X):
        p = dict(zip(param_names, vals))
        W = float(p["W"])
        d_over_W = float(p["d_over_W"])
        ar = float(p["ar"])
        H_over_W = float(p["H_over_W"])

        H = H_over_W * W
        a = (d_over_W * W) / 2.0
        b = ar * a

        res1 = solve_plate(W, H, a, b, sigma, n_hole=n1, tip_refine=tip_refine)
        res2 = solve_plate(W, H, a, b, sigma, n_hole=n2, tip_refine=tip_refine)

        p1 = res1["peak_vm"]
        p2 = res2["peak_vm"]
        pct_change = 100.0 * abs(p2 - p1) / p1
        signed_change = 100.0 * (p2 - p1) / p1

        record = {
            "id": i,
            "W": W,
            "d_over_W": d_over_W,
            "ar": ar,
            "H_over_W": H_over_W,
            "a": a,
            "b": b,
            "peak1": p1,
            "peak2": p2,
            "pct_change": pct_change,
            "signed_change": signed_change,
            "elems1": res1["n_elems"],
            "elems2": res2["n_elems"],
        }
        records.append(record)

        print(
            f"  {i:3d} | {W:7.2f} | {d_over_W:6.3f} | {ar:5.2f} | {H_over_W:5.2f} | "
            f"{a:6.2f} | {b:6.2f} | {p1:8.2f} | {p2:8.2f} | {signed_change:+8.3f}% | "
            f"{res1['n_elems']:6d} | {res2['n_elems']:6d}"
        )

    df = pd.DataFrame(records)
    mean_change = df["pct_change"].mean()
    max_change = df["pct_change"].max()

    print("\n" + "=" * 105)
    print("SELFCHECK CONVERGENCE SUMMARY")
    print("=" * 105)
    print(f"Total Geometries Evaluated: {n_samples}")
    print(f"Mesh Levels:                 n_hole = {n1} vs {n2}")
    print(f"Mean Peak Stress % Change:   {mean_change:.4f}%")
    print(f"Max Peak Stress % Change:    {max_change:.4f}%")
    print("-" * 105)

    # 5 worst samples (largest relative discrepancy)
    df_worst = df.sort_values(by="pct_change", ascending=False).head(5).reset_index(drop=True)

    print("\nTOP 5 WORST (LEAST CONVERGED) SAMPLES:")
    print("-" * 105)
    print(f"  {'Rank':>4} | {'ID':>3} | {'W [mm]':>7} | {'d/W':>6} | {'ar':>5} | {'H/W':>5} | {'a [mm]':>6} | {'b [mm]':>6} | {'Peak n1':>8} | {'Peak n2':>8} | {'% Change':>9}")
    print("-" * 105)
    for rank, row in df_worst.iterrows():
        print(
            f"  {rank + 1:4d} | {int(row['id']):3d} | {row['W']:7.2f} | {row['d_over_W']:6.3f} | "
            f"{row['ar']:5.2f} | {row['H_over_W']:5.2f} | {row['a']:6.2f} | {row['b']:6.2f} | "
            f"{row['peak1']:8.2f} | {row['peak2']:8.2f} | {row['pct_change']:8.3f}%"
        )
    print("=" * 105 + "\n")


def hcheck():
    """
    Height sensitivity check mode (--hcheck):
    Evaluates sensitivity of peak stress to plate aspect ratio H/W.
    Parameters:
        W = 100.0 mm, sigma = 100.0 MPa, n_hole = 64, tip_refine = False.
        Geometry pairs (d_over_W, ar) in:
            [(0.5, 2.0), (0.5, 0.5), (0.3, 2.0), (0.05, 2.0)]
        Aspect ratios H_over_W in:
            [2.0, 2.5, 3.0, 4.0, 5.0]
    Solves each configuration and prints:
        - peak von Mises stress
        - % difference from the H_over_W = 5.0 result for the same (d_over_W, ar)
        - element count
        - run time
    """
    W = 100.0
    sigma = 100.0
    n_hole = 64
    tip_refine = False

    geom_pairs = [
        (0.50, 2.0),
        (0.50, 0.5),
        (0.30, 2.0),
        (0.05, 2.0),
    ]
    H_over_W_list = [2.0, 2.5, 3.0, 4.0, 5.0]

    print("=" * 105)
    print("HEIGHT SENSITIVITY CHECK (--hcheck): IMPACT OF H/W ON PEAK STRESS")
    print(f"Setup: W = {W:.1f} mm, sigma = {sigma:.1f} MPa, n_hole = {n_hole}, tip_refine = {tip_refine}")
    print("Baseline reference: H/W = 5.0 for each (d/W, ar) geometry")
    print("=" * 105)

    for d_over_W, ar in geom_pairs:
        a = (d_over_W * W) / 2.0
        b = ar * a
        th_inglis = inglis_stress(sigma, a, b)

        print(f"\n>>> Geometry: d/W = {d_over_W:.2f}, ar = {ar:.2f} (a = {a:.2f} mm, b = {b:.2f} mm) | Inglis = {th_inglis:.2f} MPa")
        print("-" * 105)
        print(f"  {'H/W':>5} | {'H [mm]':>7} | {'FEA peak [MPa]':>14} | {'Diff vs H=5.0':>14} | {'Diff vs Inglis':>15} | {'Elems':>7} | {'Time [s]':>8}")
        print("-" * 105)

        # Solve across all H/W values
        results_by_H = {}
        for HW in H_over_W_list:
            H = HW * W
            res = solve_plate(W, H, a, b, sigma, n_hole=n_hole, tip_refine=tip_refine)
            results_by_H[HW] = res

        # Baseline peak at H/W = 5.0
        peak_base = results_by_H[5.0]["peak_vm"]

        for HW in H_over_W_list:
            res = results_by_H[HW]
            pk = res["peak_vm"]
            diff_from_base = 100.0 * (pk - peak_base) / peak_base
            diff_inglis = 100.0 * (pk - th_inglis) / th_inglis
            t_total = res["t_mesh"] + res["t_solve"]

            if HW == 5.0:
                diff_base_str = "    baseline"
            else:
                diff_base_str = f"{diff_from_base:+13.4f}%"

            print(
                f"  {HW:5.2f} | {HW*W:7.1f} | {pk:14.2f} | {diff_base_str} | "
                f"{diff_inglis:+14.2f}% | {res['n_elems']:7d} | {t_total:7.2f}s"
            )

    print("\n" + "=" * 105)
    print("Hcheck completed successfully.")
    print("=" * 105 + "\n")


def generate(n=1500, out="data/plate_ellipse.csv", n_hole=64, seed=42, tip_refine=False):
    """
    Generates a tabular dataset of elliptical-hole simulations using Latin Hypercube Sampling:
        W: 60.0 to 200.0 mm
        d_over_W: 0.05 to 0.50
        ar: 0.5 to 2.0
        H_over_W: 2.5 to 4.0
        sigma: 50.0 to 200.0 MPa
    Uses n_hole=64 and no tip refinement.
    For a random 5% subset of runs (seed 42), re-solves at n_hole=128 to evaluate mesh convergence (mesh_check_pct).
    Saves to data/plate_ellipse.csv with columns:
        id, W, d_over_W, ar, H_over_W, sigma, H, a, b,
        peak_vm, kt_gross, n_nodes, n_elems, t_mesh, t_solve,
        mesh_check_pct, ok, error
    """
    names = list(RANGES)
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    lo = [RANGES[k][0] for k in names]
    hi = [RANGES[k][1] for k in names]
    X = qmc.scale(sampler.random(n), lo, hi)

    # 5% random subset of runs (seed 42) for n_hole=128 mesh check
    rng = np.random.default_rng(seed=seed)
    n_check = int(round(n * 0.05))
    check_set = set(rng.choice(n, size=n_check, replace=False))

    print("=" * 80)
    print(f"GENERATING ELLIPTICAL PLATE FEA DATASET: {n} SAMPLES")
    print(f"Ranges: W in [60, 200], d/W in [0.05, 0.50], ar in [0.5, 2.0],")
    print(f"        H/W in [2.5, 4.0], sigma in [50, 200]")
    print(f"Mesh: n_hole = {n_hole}, tip_refine = {tip_refine}")
    print(f"Convergence checks: 5% random subset ({n_check} runs) re-solved at n_hole=128")
    print(f"Output target: {out}")
    print("=" * 80)

    rows = []
    t_start = time.perf_counter()

    for i, vals in enumerate(X):
        p = dict(zip(names, vals))
        W = float(p["W"])
        d_over_W = float(p["d_over_W"])
        ar = float(p["ar"])
        H_over_W = float(p["H_over_W"])
        sigma = float(p["sigma"])

        H = H_over_W * W
        a = (d_over_W * W) / 2.0
        b = ar * a

        row = {
            "id": i,
            "W": W,
            "d_over_W": d_over_W,
            "ar": ar,
            "H_over_W": H_over_W,
            "sigma": sigma,
            "H": H,
            "a": a,
            "b": b,
            "peak_vm": np.nan,
            "kt_gross": np.nan,
            "n_nodes": np.nan,
            "n_elems": np.nan,
            "t_mesh": np.nan,
            "t_solve": np.nan,
            "mesh_check_pct": np.nan,
            "ok": False,
            "error": None,
        }

        try:
            res = solve_plate(W, H, a, b, sigma, n_hole=n_hole, tip_refine=tip_refine)
            row["peak_vm"] = res["peak_vm"]
            row["kt_gross"] = res["peak_vm"] / sigma
            row["n_nodes"] = res["n_nodes"]
            row["n_elems"] = res["n_elems"]
            row["t_mesh"] = res["t_mesh"]
            row["t_solve"] = res["t_solve"]
            row["ok"] = True

            # 5% subset re-solved at n_hole=128
            if i in check_set:
                res128 = solve_plate(W, H, a, b, sigma, n_hole=128, tip_refine=tip_refine)
                pct_change = 100.0 * (res128["peak_vm"] - res["peak_vm"]) / res["peak_vm"]
                row["mesh_check_pct"] = pct_change

        except Exception as e:
            row["ok"] = False
            row["error"] = repr(e)

        rows.append(row)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed
            rem = (n - (i + 1)) / rate if rate > 0 else 0.0
            print(f"[{i + 1:4d}/{n:4d}] runs done | Rate: {rate:.2f} runs/s | Elapsed: {elapsed:.1f}s | ETA: {rem:.1f}s")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    ok_count = int(df["ok"].sum())
    valid_checks = df["mesh_check_pct"].dropna()

    print("\n" + "=" * 80)
    print("ELLIPTICAL PLATE DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"Saved:                {out} ({len(df)} rows)")
    print(f"Successful runs (ok): {ok_count} / {n}")
    if len(valid_checks) > 0:
        print(f"Mesh checks (n=128):  {len(valid_checks)} runs (5% subset)")
        print(f"mesh_check_pct mean:  {valid_checks.mean():.4f}%")
        print(f"mesh_check_pct max:   {valid_checks.max():.4f}%")
    print("=" * 80 + "\n")

    return df


def main():
    parser = argparse.ArgumentParser(description="FEA data generation and validation for plate with an elliptical hole.")
    parser.add_argument("--verify", action="store_true", help="Run standard verification against Inglis theory and circular benchmark")
    parser.add_argument("--verify2", action="store_true", help="Run extended verification (fine meshes, small ar, tip refinement comparison, Peterson check)")
    parser.add_argument("--hcheck", action="store_true", help="Check sensitivity of peak stress to plate aspect ratio H/W")
    parser.add_argument("--tip_refine", action="store_true", help="Enable notch tip curvature refinement (rho_tip = b^2/a)")
    parser.add_argument("--selfcheck", type=int, default=None, metavar="N", help="Run self-check on N LHS geometries at n_hole and 2*n_hole")
    parser.add_argument("--n", type=int, default=1500, help="Number of Latin hypercube samples to generate")
    parser.add_argument("--out", default="data/plate_ellipse.csv", help="Output CSV path")
    parser.add_argument("--n_hole", type=int, default=64, help="Elements along the hole arc")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.hcheck:
        hcheck()
    elif args.verify2:
        verify2()
    elif args.selfcheck is not None:
        selfcheck(n_samples=args.selfcheck, n_hole=args.n_hole, seed=args.seed, tip_refine=args.tip_refine)
    elif args.verify:
        verify()
    else:
        generate(n=args.n, out=args.out, n_hole=args.n_hole, seed=args.seed, tip_refine=args.tip_refine)


if __name__ == "__main__":
    main()
