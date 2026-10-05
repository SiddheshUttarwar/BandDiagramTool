"""
Point-defect reduction by illumination during growth (command line only).

    python examples/06_defects_under_illumination.py
    python examples/06_defects_under_illumination.py my_device.json --wavelength 365 --power 2
    python examples/06_defects_under_illumination.py --help

For every doped layer of a structure, taken as the free growth surface at
the growth temperature, prints the quasi-Fermi level splitting the light
produces and the factor by which the charged state of each compensating
defect is reduced (physics.dqfl; read its docstring for the model and its
limits). Without a project file it runs a Mg-doped GaN layer on Si-doped
GaN, the case of Klump et al., J. Appl. Phys. 127, 045702 (2020).

This calculation is not in the GUI: its result depends strongly on the
carrier lifetime and on a capture-ratio parameter that is not known, and it
agrees with the published measurements only to within a factor of two to
eight.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from devices.layer import AbruptLayer  # noqa: E402
from physics.dqfl import simulate_illumination  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('project', nargs='?', help='project file saved from the GUI (.json); default: Mg:GaN on Si:GaN')
    ap.add_argument('--temperature', type=float, default=1040.0, help='growth temperature [C] (default 1040)')
    ap.add_argument('--wavelength', type=float, default=300.0, help='wavelength of the light [nm] (default 300)')
    ap.add_argument('--power', type=float, default=1.0, help='power density at the wafer [W/cm^2] (default 1)')
    ap.add_argument('--lifetime', type=float, default=0.3, help='minority-carrier lifetime [ns] (default 0.3)')
    ap.add_argument('--absorption', type=float, default=3e5, help='absorption above the gap [1/cm] (default 3e5)')
    ap.add_argument('--surface-velocity', type=float, default=0.0,
                    help='surface recombination velocity [cm/s] (default 0)')
    ap.add_argument('--capture-ratio', type=float, default=1.0,
                    help='capture cross-section for the attracted carrier / the repelled one (default 1)')
    ap.add_argument('--no-diffusion', action='store_true', help='keep the carriers where they are generated')
    args = ap.parse_args()

    if args.project:
        from gui.project_io import load_project
        layers = load_project(args.project).layers
    else:
        layers = [AbruptLayer(x_Al=0.0, thickness_nm=1300, n_doping=2e18),
                  AbruptLayer(x_Al=0.0, thickness_nm=700, p_doping=2e19)]

    r = simulate_illumination(layers, T_growth_C=args.temperature, power_W_cm2=args.power,
                              wavelength_nm=args.wavelength, lifetime_ns=args.lifetime,
                              absorption_cm=args.absorption, capture_asymmetry=args.capture_ratio,
                              surface_velocity_cm_s=args.surface_velocity, diffusion=not args.no_diffusion)

    print(f'{r.wavelength_nm:g} nm ({r.photon_eV:.3f} eV), {r.power_W_cm2:g} W/cm^2, {r.T_K - 273.15:.0f} C, '
          f'lifetime {r.lifetime_s * 1e9:g} ns, capture ratio {r.capture_asymmetry:g}, '
          f'surface velocity {r.surface_velocity_cm_s:g} cm/s')
    print('Each layer as the free, neutral film at the growth surface (no contacts).\n')
    print(' #  layer             type     Eg[eV]  n0[cm^-3]  p0[cm^-3]  L[nm]  excess     Ep[eV]')
    for la in r.layers:
        print(f'{la.index:>2}  {la.name:<17} {la.doping:<8} {la.Eg_eV:5.2f}  {la.n0:9.2e}  {la.p0:9.2e}'
              f'  {la.diffusion_length_nm:5.0f}  {la.excess:9.2e}  {la.Ep_eV:6.3f}')
        if la.absorption_cm == 0.0:
            print('      not absorbed: the gap is wider than the photon energy')
            continue
        for d in la.defects:
            print(f'      {d.compensator.name:<24} level Ev+{d.level_eV:.2f} eV   dE = {d.dE_eV:+.3f} eV'
                  f'   charged state x {d.factor:.2e}')


if __name__ == '__main__':
    main()
