"""
AlGaN/GaN HEMT: polarization-induced two-dimensional electron gas.

    python examples/01_hemt_2deg.py

Solves a 25 nm Al0.3Ga0.7N barrier on GaN at zero bias, classically and with
the Schrodinger equation, and compares the two polarization parameter sets
and the two crystal polarities.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from devices.analysis import sheet_density  # noqa: E402
from devices.device import AlGaNDevice  # noqa: E402
from devices.layer import AbruptLayer, Contact, QuantumRegionMarker  # noqa: E402

BUFFER, BARRIER = 200.0, 25.0   # nm


def hemt(quantum=False, **device_kwargs):
    layers = [AbruptLayer(x_Al=0.0, thickness_nm=BUFFER - 20, n_doping=1e16),
              QuantumRegionMarker('start'),
              AbruptLayer(x_Al=0.0, thickness_nm=20, dx_nm=0.1),
              AbruptLayer(x_Al=0.3, thickness_nm=BARRIER, dx_nm=0.1),
              QuantumRegionMarker('end')]
    contacts = [Contact('bottom', 'ohmic', 'Ti'),
                # free surface: Fermi level pinned 0.84 + 1.3 x eV below Ec (Ambacher 2000)
                Contact('top', 'schottky', 'Ni', barrier_eV=0.84 + 1.3 * 0.3)]
    device = AlGaNDevice(layers, contacts, dx_nm=0.5, **device_kwargs)
    return device.solve(V_applied=0.0, quantum=quantum)


if __name__ == '__main__':
    r = hemt()
    print(f'classical           n_s = {sheet_density(r, 150, 225):.3e} cm^-2')

    rq = hemt(quantum=True)
    print(f'Schrodinger-Poisson n_s = {sheet_density(rq, 150, 225):.3e} cm^-2')
    print('  lowest subbands relative to the Fermi level [meV]:',
          ', '.join(f'{1e3 * e:+.0f}' for e in rq.E_e[:3]))

    rd = hemt(polarization_model='dreyer2016')
    print(f'Dreyer 2016 constants n_s = {sheet_density(rd, 150, 225):.3e} cm^-2')

    rn = hemt(polarity='N')
    print(f'N-polar: electrons under the barrier {sheet_density(rn, 150, 200):.1e}, '
          f'holes under the barrier {sheet_density(rn, 150, 200, "p"):.3e} cm^-2')
