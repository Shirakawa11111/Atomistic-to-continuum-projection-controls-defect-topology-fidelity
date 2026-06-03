"""Frozen reference configuration of the SC-CH continuum testbed (the *fixed model*).

Route A holds this model fixed and varies ONLY the MD→ψ mapping protocol.
Values are the synchronized manuscript-scale defaults from Paper 1
(``GrapheneSimulation.XPFC_PARAMS`` / ``EVOLUTION_PARAMS``), audited in
``revision_support/01_xpfc_parameter_audit`` of the Paper 1 repository.

Positioning (locked decision): **SC-CH mesoscale**. With ``a0`` non-dimensional
and hardcoded to 1.0 while a 1024-atom sheet maps to ``a0_nd ≈ 0.047``, the XPFC
structural kernels resonate at a mesoscopic wavelength (~O(1)) far from the
atomic spacing — the known order-of-magnitude scale offset. The relaxation is
therefore effectively a *constrained Cahn–Hilliard* flow whose attractor is the
protocol-mapped field ``n0`` (morphological constraint, weight ``lambda``). We
benchmark protocol-induced *relative variance*, not absolute energetics.
"""
from __future__ import annotations

import numpy as np

# --- non-dimensional lattice constant used by Paper 1 (hardcoded a0_nd = 1.0) ---
A0_ND: float = 1.0

# --- frozen XPFC / SC-CH model parameters (do NOT vary in Route A) ---
REFERENCE_XPFC_PARAMS: dict = dict(
    a0=A0_ND,
    R=4.0 * np.pi / np.sqrt(3.0) * A0_ND,   # 7.255197456937  (first reciprocal-lattice magnitude)
    X=2.5,
    m=3,
    eta=0.25,
    chi=0.5,
    r0_a0_ratio=1.22604,
    morphological_constraint_weight=0.01,   # lambda — dominant practical sensitivity axis, FIXED here
    vacuum_threshold_for_mask=0.05,
)

# --- frozen evolution parameters ---
REFERENCE_EVOLUTION_PARAMS: dict = dict(
    dt=1.0e-7,
    mobility=1.0,
    scheme="semi-implicit",
)
REFERENCE_STEPS: int = 20000               # baseline fixed-step budget
RELAX_TOL: float = 1.0e-5                  # |ΔF|/|F| convergence threshold (design-doc relaxation criterion)

# --- postprocess (P3/P4) field targets — Paper 1 baseline ---
FIELD_MEAN_TARGET: float = 0.18
FIELD_PEAK_TARGET: float = 0.98
VACUUM_NOISE_CUTOFF: float = 0.01          # zero-out field below this after rescale

# --- grid ---
DEFAULT_NX_LONG_EDGE: int = 512

# --- Voronoi (K3) deposition params — Paper 1 baseline ---
VORONOI_PARAMS: dict = dict(
    sigma_normal=0.005,
    sigma_defect=0.002,
    w_normal=1.0,
    w_defect=2.5,
    defect_radius=0.012,
)

# --- physical constants ---
A0_GRAPHENE_ANG: float = 2.46              # graphene lattice constant (Å)
BOND_GRAPHENE_ANG: float = 1.42            # C–C bond length (Å)
