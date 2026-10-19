"""
Bulk 8x8 k.p Hamiltonian of one node (form and parameters: see
physics.kp.parameters). Basis order: S, X, Y, Z (spin up), S, X, Y, Z
(spin down). k in nm^-1, energies in eV.
"""
from __future__ import annotations

import numpy as np

from physics.kp.parameters import HB


def _spin_orbit_6x6() -> np.ndarray:
    """L.sigma on (X, Y, Z) x (up, down), basis X^,Y^,Z^,Xv,Yv,Zv.
    (L_k)_ij = -i eps_kij; eigenvalues +1 (x4, j = 3/2) and -2 (x2,
    j = 1/2), so (Delta_so / 3) L.sigma splits the j = 1/2 doublet
    Delta_so below the j = 3/2 quartet."""
    eps = np.zeros((3, 3, 3))
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        eps[a, b, c], eps[a, c, b] = 1.0, -1.0
    L = [-1j * eps[k] for k in range(3)]
    sig = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]),
           np.array([[1, 0], [0, -1]], complex)]
    return sum(np.kron(sig[k], L[k]) for k in range(3))


_LS = _spin_orbit_6x6()
_VB = np.array([1, 2, 3, 5, 6, 7])       # X, Y, Z of both spins in the 8x8 basis


def bulk_hamiltonian(p: dict, kx: float, ky: float, kz: float) -> np.ndarray:
    """8x8 Hermitian H(k) for the node parameters p (KPParams.node(j))."""
    kt2 = kx * kx + ky * ky
    k2 = kt2 + kz * kz
    h = np.zeros((4, 4), complex)
    h[0, 0] = p['Ec'] + HB * k2 + p['A2p'] * kt2 + p['A1p'] * kz * kz
    h[0, 1] = 1j * p['P2'] * kx
    h[0, 2] = 1j * p['P2'] * ky
    h[0, 3] = 1j * p['P1'] * kz
    h[1, 1] = p['XY0'] + HB * k2 + p['L1p'] * kx * kx + p['M1'] * ky * ky + p['M2'] * kz * kz
    h[2, 2] = p['XY0'] + HB * k2 + p['M1'] * kx * kx + p['L1p'] * ky * ky + p['M2'] * kz * kz
    h[3, 3] = p['Z0'] + HB * k2 + p['M3'] * kt2 + p['L2p'] * kz * kz
    h[1, 2] = p['N1p'] * kx * ky
    n3 = 1j * p['N3']
    h[1, 3] = p['N2p'] * kx * kz - n3 * kx
    h[2, 3] = p['N2p'] * ky * kz - n3 * ky
    h = np.triu(h) + np.triu(h, 1).conj().T
    H = np.zeros((8, 8), complex)
    H[:4, :4] = h
    H[4:, 4:] = h
    H[np.ix_(_VB, _VB)] += (p['Dso'] / 3.0) * _LS
    return H
