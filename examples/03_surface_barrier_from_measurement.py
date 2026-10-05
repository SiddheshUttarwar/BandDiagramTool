"""
Surface barrier from a measured 2DEG density (inverse problem).

    python examples/03_surface_barrier_from_measurement.py

Under a barrier a few nanometres thick the 2DEG density depends so strongly
on the surface barrier that a forward prediction is not meaningful. This
script finds the barrier that reproduces published Hall densities of
AlN/GaN heterojunctions (Cao & Jena, APL 90, 182112 (2007); Cao et al., APL
92, 152112 (2008)).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from devices.analysis import extract_surface_barrier  # noqa: E402
from devices.layer import AbruptLayer  # noqa: E402

SAMPLES = [(2.0, 5.0e12), (2.3, 1.45e13), (4.0, 3.1e13)]   # AlN thickness [nm], n_s [cm^-2]

if __name__ == '__main__':
    print('AlN [nm]   measured n_s [cm^-2]   surface barrier [eV]')
    for t_aln, n_s in SAMPLES:
        layers = [AbruptLayer(x_Al=0.0, thickness_nm=200, n_doping=1e16),
                  AbruptLayer(x_Al=1.0, thickness_nm=t_aln, dx_nm=0.05)]
        phi_b = extract_surface_barrier(layers, n_s, window_nm=(150, 200 + t_aln), bounds_eV=(1.0, 5.8))
        print(f'{t_aln:7.1f} {n_s:20.2e} {phi_b:20.2f}')
