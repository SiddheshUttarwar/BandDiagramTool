# BandDiagramTool model reference

BandDiagramTool is a one-dimensional Schrödinger-Poisson-current solver for wurtzite
Al-In-Ga-N layer stacks grown along the c-axis. This reference states the equations it solves,
the parameters that enter them, and the algorithms that couple them.

It is organised by physical model, in the same order the program evaluates them. Equations are
numbered by chapter, so (5.4) is the fourth equation of chapter 5.

## Chapters

| # | Chapter | Contents |
|---|---|---|
| 1 | [Structure and grid](01_structure_and_grid.md) | layers, contacts, coordinates, finite-volume grid |
| 2 | [Materials](02_materials.md) | interpolation schemes, band gaps, band offsets, masses, parameter database |
| 3 | [Strain](03_strain.md) | strain tensor, pseudomorphic solution, band-edge shifts |
| 4 | [Polarization](04_polarization.md) | pyroelectric and piezoelectric polarization, interface and volume charge |
| 5 | [Electrostatics](05_electrostatics.md) | Poisson equation, classical charge densities, doping, surface states, boundary conditions |
| 6 | [Quantum mechanics](06_quantum.md) | envelope-function Schrödinger equation, quantum charge densities |
| 7 | [Currents](07_currents.md) | drift-diffusion, mobility, generation-recombination, discretised flux |
| 8 | [Optics](08_optics.md) | transition energies, overlap, gain spectrum |
| 9 | [Solution algorithms](09_solution_algorithms.md) | the four solve modes and how each one iterates |
| 10 | [Validation](10_validation.md) | comparison with nextnano++ and with 102 published experiments |
| 11 | [Limits and comparison with nextnano++](11_limits_and_comparison.md) | what is not modelled; feature-by-feature comparison |

## Program flow

```
 layer stack + contacts
        |
        v
 1. grid                                     chapter 1
 2. material parameters at every node        chapter 2
 3. strain  ->  band-edge shifts             chapter 3
 4. polarization  ->  fixed charge           chapter 4
        |
        v
 5. self-consistent loop                     chapter 9
      Poisson            phi                 chapter 5
      Schrodinger        E_k, psi_k          chapter 6   (quantum modes)
      current equation   E_Fn, E_Fp          chapter 7   (biased modes)
        |
        v
 6. post-processing: sheet densities, fields,
    transition energies, overlap, gain       chapter 8
```

Steps 1-4 are evaluated once. They depend only on the structure and the temperature. Step 5 is
the only iterative part.

## Unknowns and the equations that determine them

| Unknown | Symbol | Determined by |
|---|---|---|
| electrostatic potential | $`\phi(z)`$ | Poisson equation (5.1) |
| electron and hole quasi-Fermi levels | $`E_{Fn}(z)`$, $`E_{Fp}(z)`$ | current equations (7.3) |
| subband energies and envelope functions | $`E_k`$, $`\psi_k(z)`$ | Schrödinger equation (6.1) |

The carrier densities are not independent unknowns. They are functions of the others,

```math
n = n(z;\ \phi,\ E_{Fn}), \qquad p = p(z;\ \phi,\ E_{Fp}),
```

evaluated classically (5.4)-(5.5) or quantum mechanically (6.7)-(6.8). Keeping this dependence in
view makes each algorithm in chapter 9 easy to read: every step holds some arguments fixed and
solves one equation for the remaining one.

## Solve modes

| Mode | Call | Equations solved |
|---|---|---|
| Poisson | `solve(V_applied=0, quantum=False)` | (5.1) |
| Schrödinger-Poisson | `solve(V_applied=0, quantum=True)` | (5.1), (6.1) |
| Current-Poisson | `solve(V_applied=V, quantum=False)` | (5.1), (7.3) |
| Schrödinger-current-Poisson | `solve(V_applied=V, quantum=True)` | (5.1), (6.1), (7.3) |
| Flat quasi-Fermi levels under bias | `solve(V_applied=V, flat_qfl=True)` | (5.1), optionally (6.1) |

## Conventions

| Symbol | Meaning | Unit |
|---|---|---|
| $`z`$ | position along the growth axis, 0 at the substrate side | nm |
| $`x`$, $`y`$ | Al and In mole fractions in Al<sub>x</sub>In<sub>y</sub>Ga<sub>1−x−y</sub>N | - |
| $`\phi`$ | electrostatic potential | V |
| $`E_c`$, $`E_v`$ | conduction band edge and topmost valence band edge | eV |
| $`n`$, $`p`$, $`N_D`$, $`N_A`$ | carrier and dopant densities | cm<sup>−3</sup> |
| $`P`$ | polarization | C/m<sup>2</sup> |
| $`\varepsilon_{ij}`$ | strain tensor | - |
| $`\varepsilon_r`$ | static relative permittivity along c | - |
| $`q`$ | elementary charge, positive | C |

- **Polarity.** Metal-polar (Ga-face) growth: the $`+z`$ axis is the crystal [0001] direction.
- **Energy.** Electron energies. A positive potential lowers the bands, $`E_c = E_{c,0} - q\phi`$.
  With energies in eV and potential in V the factor $`q`$ is numerically 1 and is omitted below.
- **Reference.** At equilibrium the Fermi level is the zero of energy.
- **Bias.** Positive $`V`$ is forward bias, applied to the top contact.
- **Internal units.** SI, except densities in cm<sup>−3</sup> and energies in eV.

## Source files

| Model | File |
|---|---|
| Structure | `devices/layer.py`, `devices/device.py` |
| Grid and node profiles | `devices/grid_builder.py`, `physics/grid_utils.py` |
| Materials | `physics/materials/algan.py`, `nitrides.py`, `metals.py` |
| Strain and polarization | `physics/polarization.py` |
| Carrier statistics | `physics/fermi_dirac.py` |
| Poisson | `physics/poisson.py` |
| Schrödinger | `physics/schrodinger.py` |
| Current | `physics/dd_newton.py`, `physics/drift_diffusion.py` |
| Algorithms | `physics/self_consistent.py` |
| Optics | `physics/optical.py`, `physics/gain.py` |
