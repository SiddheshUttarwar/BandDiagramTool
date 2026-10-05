"""
Reciprocal space map of thick Al0.6Ga0.4N on an AlN substrate.

    python examples/05_reciprocal_space_map.py

The structure of Rathkanthiwar et al., Appl. Phys. Lett. 120, 202105 (2022):
0.5 um homoepitaxial AlN on a (0001) AlN substrate, then 1.8 um
Al0.6Ga0.4N. The paper's (10.5) map shows the AlGaN peak vertically aligned
in q_x with AlN (pseudomorphic), and wafer curvature gives a compressive
stress of 3.3-3.8 GPa.

The script simulates the (10-15) map for the layer grown coherently and,
for contrast, fully relaxed, prints where the AlGaN peak falls in each
case, and compares the biaxial stress of the coherent layer with the
measured one. The map needs only the strain profile, so the device grid is
used directly; no Poisson solve is run.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from devices.device import AlGaNDevice  # noqa: E402
from devices.layer import AbruptLayer, Contact  # noqa: E402
from physics.rsm import simulate_rsm  # noqa: E402

X_AL, T_ALN, T_ALGAN = 0.60, 500.0, 1800.0     # composition, thicknesses in nm
# C11, C12 [GPa] (Vurgaftman & Meyer 2003); C13, C33 as in physics.materials.algan
C11 = {'GaN': 390.0, 'AlN': 396.0}
C12 = {'GaN': 145.0, 'AlN': 137.0}
C13 = {'GaN': 106.0, 'AlN': 108.0}
C33 = {'GaN': 398.0, 'AlN': 373.0}


def grid(relaxed: bool):
    layers = [AbruptLayer(x_Al=1.0, thickness_nm=T_ALN),
              AbruptLayer(x_Al=X_AL, thickness_nm=T_ALGAN, relaxed=relaxed)]
    contacts = [Contact('bottom', 'ohmic', 'Ti'), Contact('top', 'ohmic', 'Ni')]
    return AlGaNDevice(layers, contacts, dx_nm=1.0).build_grid()


def algan_peak(rsm):
    """(q_x, q_z) of the strongest point above the AlN peak's rod-and-height."""
    name = f'Al{X_AL:.2f}Ga{1 - X_AL:.2f}N'
    return next((qx, qz) for n, qx, qz in rsm.strained_points if n == name)


if __name__ == '__main__':
    g = grid(relaxed=False)
    top = np.asarray(g.x_Al) < 0.99
    exx = float(np.mean(np.asarray(g.eps_xx)[top]))
    ezz = float(np.mean(np.asarray(g.eps_zz)[top]))

    def vegard(c):
        return X_AL * c['AlN'] + (1 - X_AL) * c['GaN']
    modulus = vegard(C11) + vegard(C12) - 2 * vegard(C13) ** 2 / vegard(C33)
    print(f'Coherent Al{X_AL:.1f}Ga{1 - X_AL:.1f}N on AlN:  eps_xx = {100 * exx:+.3f} %,  eps_zz = {100 * ezz:+.3f} %')
    print(f'  biaxial stress  = {modulus * exx:+.2f} GPa   (biaxial modulus {modulus:.0f} GPa)')
    print('  measured (wafer curvature, Table I of the paper): -3.8, -3.6, -3.3 GPa')

    coherent = simulate_rsm(g, (1, 0, 5))
    relaxed = simulate_rsm(grid(relaxed=True), (1, 0, 5))
    q_aln = coherent.qx_substrate
    for label, rsm in (('coherent', coherent), ('fully relaxed', relaxed)):
        qx, qz = algan_peak(rsm)
        print(f'  (10-15) AlGaN peak, {label:<14}: q_x = {qx:.4f}, q_z = {qz:.4f} 1/A'
              f'   (q_x - q_x(AlN) = {qx - q_aln:+.4f})')

    # The paper's Fig. 2 is in reciprocal lattice units, q_rlu = lambda / (2 d) x 10^4
    # (Cu K-alpha1). Digitized from the figure: (00.4) AlN 6185, AlGaN 6055;
    # (10.5) AlN 7737, AlGaN 7573.
    rlu = 1.540598 / (4 * np.pi) * 1e4
    print("  q_z in the units of the paper's Fig. 2 (measured: AlN / AlGaN 6185 / 6055 and 7737 / 7573):")
    for hkl, tag in (((0, 0, 4), '(00.4)'), ((1, 0, 5), '(10.5)')):
        coh, rel = simulate_rsm(g, hkl), simulate_rsm(grid(relaxed=True), hkl)
        aln = next(qz for n, _qx, qz in coh.strained_points if n == 'AlN')
        print(f'    {tag}: AlN {aln * rlu:6.0f}   AlGaN coherent {algan_peak(coh)[1] * rlu:6.0f}'
              f'   AlGaN relaxed {algan_peak(rel)[1] * rlu:6.0f}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from visualization.plotter import plot_rsm
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, rsm, title in ((axes[0], coherent, 'coherent to AlN (as reported)'),
                               (axes[1], relaxed, 'fully relaxed (for contrast)')):
            plot_rsm(rsm, ax=ax)
            ax.set_title(f'(10-15), 1.8 um Al0.6Ga0.4N on AlN: {title}', fontsize=10)
            ax.set_xlim(q_aln - 0.03, q_aln + 0.012)
            ax.set_ylim(6.14, 6.36)
        fig.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rsm_algan_on_aln.png')
        fig.savefig(out, dpi=140)
        print('figure written to', out)
    except ImportError:
        pass
