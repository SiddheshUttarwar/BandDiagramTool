"""
Quantum-confined Stark effect in a GaN/AlN quantum well.

    python examples/02_quantum_well_stark_effect.py

For GaN wells of increasing width between AlN barriers (coherent to AlN),
prints the built-in field, the e1-h1 transition energy and the squared
electron-hole overlap. The transition falls below the GaN gap and the overlap
collapses as the well widens.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from devices.device import AlGaNDevice  # noqa: E402
from devices.layer import AbruptLayer, Contact, QuantumRegionMarker  # noqa: E402


def well(width_nm):
    layers = [AbruptLayer(x_Al=1.0, thickness_nm=60),
              QuantumRegionMarker('start'),
              AbruptLayer(x_Al=1.0, thickness_nm=8, dx_nm=0.05),
              AbruptLayer(x_Al=0.0, thickness_nm=width_nm, dx_nm=0.02),
              AbruptLayer(x_Al=1.0, thickness_nm=8, dx_nm=0.05),
              QuantumRegionMarker('end'),
              AbruptLayer(x_Al=1.0, thickness_nm=30)]
    # insulating AlN on both sides: Fermi level at mid-gap
    contacts = [Contact('bottom', 'schottky', 'Ni', barrier_eV=3.0),
                Contact('top', 'schottky', 'Ni', barrier_eV=3.0)]
    r = AlGaNDevice(layers, contacts, dx_nm=0.2).solve(V_applied=0.0, quantum=True)
    x = np.asarray(r.x_nm)
    inside = (x > 68.3) & (x < 68 + width_nm - 0.3)
    field = float(np.mean(np.abs(np.asarray(r.E_field)[inside]))) / 1e8
    return field, r.qcse_transition_eV, r.qcse_overlap


if __name__ == '__main__':
    print('width [nm]   field [MV/cm]   E(e1-h1) [eV]   overlap^2')
    for w in (1.0, 1.5, 2.0, 2.6, 3.5):
        f, e, o = well(w)
        print(f'{w:8.1f} {f:14.1f} {e:15.3f} {o:13.2e}')
