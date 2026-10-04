# The mathematics behind BandDiagramTool

BandDiagramTool computes the band diagram, charge distribution, quantum states and current of a
wurtzite III-nitride layer stack grown along the c-axis. These notes state every equation the
solver uses, where each number comes from, and how well the result agrees with experiment.
They are written for a device or epitaxy engineer who knows semiconductor physics but has not
read the code.

## What the tool solves

One spatial dimension, z, along the growth direction, from the substrate (bottom, z = 0) to the
surface (top). The unknowns at every grid point are

- the electrostatic potential $\phi(z)$,
- the electron and hole quasi-Fermi levels $E_{Fn}(z)$, $E_{Fp}(z)$,
- optionally, the confined electron and hole states $\{E_k, \psi_k(z)\}$ of a quantum region.

Three coupled equations determine them:

| Equation | Unknown | Document |
|---|---|---|
| Poisson, with polarization charge | $\phi$ | [05](05_poisson_equilibrium.md) |
| Schrödinger, effective mass | $E_k$, $\psi_k$ | [06](06_schrodinger_quantum.md) |
| Electron and hole continuity (drift-diffusion) | $E_{Fn}$, $E_{Fp}$ | [07](07_drift_diffusion_bias.md) |

At zero bias the quasi-Fermi levels are flat and equal, and only Poisson (plus Schrödinger) is solved.

## Reading order

1. [Device model, grid and energy reference](01_device_model_and_grid.md)
2. [Material model: Al(x)In(y)Ga(1-x-y)N](02_material_model.md)
3. [Strain, polarization and strained band edges](03_strain_and_polarization.md)
4. [Carrier statistics and dopant ionization](04_carrier_statistics.md)
5. [Poisson equation and the equilibrium solve](05_poisson_equilibrium.md)
6. [Schrödinger equation and quantum charge](06_schrodinger_quantum.md)
7. [Drift-diffusion and the biased solve](07_drift_diffusion_bias.md)
8. [Optical outputs: transition energy, overlap, gain](08_optical_outputs.md)
9. [Validation against nextnano++ and 102 papers](09_validation.md)
10. [Assumptions and limits](10_assumptions_and_limits.md)

## Notation and units

| Symbol | Meaning | Unit in these notes |
|---|---|---|
| $z$ | position along growth direction, 0 at the substrate side | nm |
| $x$, $y$ | Al and In mole fractions of Al$_x$In$_y$Ga$_{1-x-y}$N | - |
| $\phi$ | electrostatic potential | V |
| $E_c$, $E_v$ | conduction and topmost valence band edge | eV |
| $E_{Fn}$, $E_{Fp}$ | electron and hole quasi-Fermi levels | eV |
| $n$, $p$ | electron and hole density | cm$^{-3}$ |
| $N_D$, $N_A$ | donor (Si) and acceptor (Mg) concentration | cm$^{-3}$ |
| $P$ | polarization, $P_{sp} + P_{pz}$ | C/m$^2$ |
| $\varepsilon_{xx}$, $\varepsilon_{zz}$ | in-plane and out-of-plane strain | - |
| $\varepsilon_r$ | relative permittivity along c | - |
| $q$ | elementary charge (positive) | C |
| $k_BT$ | thermal energy | eV |

Sign conventions used throughout:

- **Growth is metal-polar (Ga-face, [0001]).** Spontaneous polarization is negative.
- **Energies are electron energies.** A positive potential lowers the bands: $E_c = E_{c0} - \phi$.
- **The Fermi level is the zero of energy at equilibrium.**
- **Forward bias is positive** and is applied to the top contact.

## Source files

| Topic | File |
|---|---|
| Layers, contacts | `devices/layer.py` |
| Grid and material profiles | `devices/grid_builder.py`, `physics/grid_utils.py` |
| Material parameters | `physics/materials/algan.py`, `physics/materials/nitrides.py`, `physics/materials/metals.py` |
| Strain and polarization | `physics/polarization.py` |
| Fermi-Dirac statistics | `physics/fermi_dirac.py` |
| Poisson | `physics/poisson.py` |
| Schrödinger | `physics/schrodinger.py` |
| Equilibrium and outer loops | `physics/self_consistent.py` |
| Biased drift-diffusion Newton solver | `physics/dd_newton.py`, `physics/drift_diffusion.py` |
| Optical diagnostics | `physics/optical.py`, `physics/gain.py` |
| Mott density (display only) | `physics/mott.py` |
