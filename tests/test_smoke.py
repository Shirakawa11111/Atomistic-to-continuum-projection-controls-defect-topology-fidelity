"""Smoke + unit tests for the Route A benchmark machinery (fast: small grids)."""
import numpy as np
from scipy.spatial import cKDTree

from route_a.cases import make_T0, make_T2, get_case, CASES
from route_a.nondim import Nondimensionalizer, setup_grid
from route_a.kernels import get_kernel, KERNELS
from route_a.postprocess import get_postprocess
from route_a.reference_density import get_reference
from route_a.sc_ch_model import SCCHModel
from route_a import metrics as M
from route_a.pipeline import (
    run_protocol, Protocol, reduced_protocol_matrix,
    discrimination_verdict, add_relative_observables,
)


def _grid_atoms(case, nx=96):
    nd = Nondimensionalizer((case.box[0], case.box[1]), (case.box[2], case.box[3]))
    a = nd.nondimensionalize_coords(case.atoms)
    g = setup_grid(nd.Lx_nd, nd.Ly_nd, nx)
    return nd, a, g


def test_model_relaxes_and_is_healthy():
    case = make_T0(8, 8)
    _, a, g = _grid_atoms(case, 96)
    field = get_postprocess("P4").apply(get_kernel("K3").map(a, g))
    m = SCCHModel.reference(field, dx=g.dx, x_bounds=g.x_bounds, y_bounds=g.y_bounds)
    F0 = m.free_energy()
    res = m.relax(max_steps=400, chunk=200)
    assert m.check_density_health()
    assert res["F_final"] <= F0 + 1e-9  # free energy must not increase


def test_all_kernels_conserve_atom_count():
    case = make_T0(8, 8)
    _, a, g = _grid_atoms(case, 96)
    for name in KERNELS:
        rho = get_kernel(name).map(a, g)
        assert abs(rho.sum() * g.dx**2 - case.n_atoms) / case.n_atoms < 0.02


def test_reference_only_affects_P2():
    case = make_T0(8, 8)
    _, a, g = _grid_atoms(case, 96)
    rho = get_kernel("K3").map(a, g)
    rho0 = get_reference("R2").compute(get_kernel("K3"), g, perfect_atoms_nd=a)
    p3, p2 = get_postprocess("P3"), get_postprocess("P2")
    assert np.allclose(p3.apply(rho, rho0=rho0), p3.apply(rho, rho0=rho0 * 0.5 + 1e-6))
    assert not np.allclose(p2.apply(rho, rho0=rho0), p2.apply(rho, rho0=rho0 * 0.5 + 1e-6))


def test_topology_pristine_is_all_hexagons():
    case = make_T0(16, 16)
    _, a, g = _grid_atoms(case, 512)
    field = get_postprocess("P4").apply(get_kernel("K3").map(a, g))
    nn = float(np.median(cKDTree(a[:, :2]).query(a[:, :2], k=2)[0][:, 1]))
    pk = M.reconstruct_peaks(field, g, min_sep_nd=0.7 * nn)
    h = M.ring_histogram(pk)
    assert abs(len(pk) - case.n_atoms) <= 5          # reconstruction fidelity
    assert h.get(6, 0) > 300 and h.get(5, 0) == 0 and h.get(7, 0) == 0


def test_all_cases_generate_and_have_atoms():
    for k in CASES:
        c = get_case(k)
        assert c.n_atoms > 100
        assert c.atoms.shape[1] == 3


def test_pipeline_run_protocol():
    case = make_T2(10, 10)
    row = run_protocol(case, Protocol("K3", "R2", "P4"), nx_long_edge=96, relax_max_steps=300)
    assert np.isfinite(row["F_final"])
    assert row["n_atoms"] == case.n_atoms
    assert "tau_57" in row


def test_protocol_matrix_size_and_baseline():
    protos = reduced_protocol_matrix()
    assert len(protos) == 18
    assert Protocol("K3", "R2", "P4") in protos


def test_relative_observable_and_verdict():
    rows = [
        {"case": "T0_pristine", "protocol": "K3/R2/P4", "F_final": 1.0, "tau_57": 1.0,
         "l2_change": 0.1, "ring_l1_init": 0.05, "roughness_init": 0.50},
        {"case": "T1_vacancy", "protocol": "K3/R2/P4", "F_final": 1.2, "tau_57": 0.8,
         "l2_change": 0.2, "ring_l1_init": 0.15, "roughness_init": 0.60},
    ]
    add_relative_observables(rows)
    assert abs(rows[1]["Ef_rel"] - 0.2) < 1e-9
    v = discrimination_verdict(rows)
    assert v["outcome"].startswith(("A", "B"))
