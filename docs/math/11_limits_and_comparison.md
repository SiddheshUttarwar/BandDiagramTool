# 11. Limits, and comparison with nextnano++

## 11.1 Summary of assumptions

| Area | Assumption | Chapter |
|---|---|---|
| Geometry | one dimension along the growth axis; laterally uniform | 1 |
| Crystal | wurtzite Al-In-Ga-N, metal-polar c-plane | 2, 4 |
| Temperature | uniform; only $k_BT$ and $P_{sp}$ depend on it | 2, 4 |
| Strain | coherent to the bottom layer; biaxial; linear elasticity | 3 |
| Polarization | linear piezoelectricity; fixed bound charge | 4 |
| Statistics | Fermi-Dirac; one donor and one acceptor level; all dopants active | 5 |
| Surfaces | fixed barrier, or user-defined trap states | 5 |
| Quantum | single-band envelope functions, three decoupled valence bands, Hartree | 6 |
| Transport | drift-diffusion, constant mobility, steady state | 7 |
| Optics | band-to-band transitions; band-edge gain | 8 |

## 11.2 What is not modelled

**Structure and materials**

- two- and three-dimensional geometry: gates, mesas, current spreading, nanowires, dots
- N-polar, semi-polar and non-polar orientations; zincblende
- alloys beyond Al-In-Ga-N (ScAlN, BAlN), dielectrics, metals as layers
- alloy disorder, interface roughness, indium clustering
- relaxation, critical thickness, cracking, dislocations

**Electrostatics**

- temperature-dependent band gaps and masses
- compensation, native defects, DX centres, impurity bands
- buffer traps (Fe, C); surface donors unless entered explicitly
- band-gap narrowing at high doping

**Quantum mechanics**

- multi-band k·p, valence-band mixing, nonparabolicity
- excitons and many-body effects
- tunnelling, resonant states, superlattice minibands

**Transport**

- thermionic emission and tunnelling at heterobarriers
- field-, doping- and temperature-dependent mobility; velocity saturation
- impact ionization, breakdown, optical generation
- self-heating; transient and small-signal response

**Optics**

- optical modes, confinement factor, threshold, extraction
- spontaneous emission spectra and efficiency
- in-plane dispersion in the gain; polarization selection rules

## 11.3 Comparison with nextnano++

nextnano++ is a commercial Schrödinger-Poisson-current solver. The numerical agreement of the two
codes on identical inputs is documented in section 10.1. The table compares the models, based on
the nextnano++ model reference.

| Model | nextnano++ | BandDiagramTool |
|---|---|---|
| Dimensions | 1D, 2D, 3D | 1D |
| Materials | group IV, III-V, II-VI; zincblende and wurtzite; database editable | wurtzite Al-In-Ga-N; parameters in source |
| Crystal orientation | arbitrary, by Miller indices | c-plane, metal-polar |
| Alloy interpolation | linear, quadratic and cubic schemes; alloys of up to eight components | linear with pairwise bowing; three components |
| Band-offset definition | average valence-band energy $E_{v,av}$ per material | top valence-band energy per material, (2.4) |
| Band gap vs temperature | Varshni parameters in the database | none |
| Strain | analytic pseudomorphic solution, or numerical minimisation of elastic energy | analytic pseudomorphic solution (3.5); per-layer override |
| Polarization | pyroelectric and piezoelectric, any orientation | pyroelectric and piezoelectric, c-axis component (4.5) |
| Poisson equation | nonlinear, classical or quantum densities | same, (5.1) |
| Classical densities | Fermi-Dirac, summed over all conduction valleys and valence bands | Fermi-Dirac, $\Gamma$ valley and three valence bands, (5.4)-(5.7) |
| Doping | several species per region, incomplete ionization, $g_D = 2$, $g_A = 4$ | one donor and one acceptor, same ionization formula (5.8) |
| Surface | Schottky barrier, fixed surface charge, surface states | Schottky barrier, surface states (5.11), fixed dipole |
| Quantum model | single-band and multi-band k·p (6- and 8-band) | single-band, (6.1) |
| Quantum density | analytic for single-band; k-space integration for k·p | analytic, (6.6)-(6.8) |
| Current model | drift-diffusion in quasi-Fermi-level form | same, (7.2) |
| Mobility | several models (constant, doping-dependent, high-field) | constant, linear in composition |
| Recombination | SRH with doping-dependent lifetimes, radiative, Auger with separate $C_n$, $C_p$, optical generation | SRH, radiative, Auger with fixed coefficients (7.5)-(7.7) |
| Coupling of current and Poisson | decoupled iteration: Poisson at fixed quasi-Fermi levels, then current equation | fully coupled Newton on $(\phi, E_{Fn}, E_{Fp})$, (9.4) |
| Optical spectra | semiclassical spectra; k·p absorption and gain; excitons | transition energies and overlap; band-edge gain (8.4) |

**Where the two give the same answer.** Band diagrams, sheet densities and single-band subband
energies of c-plane nitride layer stacks at 300 K, when the same parameters are used.

**Where nextnano++ should be preferred.** Anything needing valence-band mixing (polarization of
emission, accurate hole subbands, gain magnitude), temperature-dependent gaps, crystal
orientations other than c-plane, lateral geometry, or materials outside Al-In-Ga-N.

**What this tool adds.** Composition-dependent Si and Mg ionization energies for the nitrides
built in; polarization charge that is exactly conservative on non-uniform grids; a fully coupled
drift-diffusion Newton solver with a current-conservation diagnostic; and validation tables
against measured nitride devices (chapter 10).

## 11.4 Related modules

`physics/defects/` contains a point-defect formation-energy calculator, a phenomenological
DX-centre model, a surface-pinning model and a step-flow growth profile. They evaluate defect
thermodynamics on a converged band diagram. They do not contribute charge to the solution
described in this reference.
