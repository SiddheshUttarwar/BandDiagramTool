# 10. Validation

Two questions are answered separately: whether the equations are solved correctly, and whether
the equations and parameters describe real devices.

## 10.1 Numerical verification against nextnano++

Twelve structures (eight AlGaN, four containing In) were solved at zero bias in both codes, using
the nextnano++ free edition (100-point grid, no strain module).

**Same parameters.** With nextnano's material parameters injected into this tool, so that only the
numerics differ:

| Quantity | Difference |
|---|---|
| band edges | about 10 meV (12-23 meV for the In-containing structures) |
| 2DEG and 2DHG sheet densities | 0-3% |
| ground subband of an Al<sub>0.3</sub>Ga<sub>0.7</sub>N/GaN well | $`-122.3`$ vs $`-121.3`$ meV |
| 2DEG density of the same structure, quantum | $`1.229\times10^{13}`$ vs $`1.234\times10^{13}`$ cm<sup>−2</sup> |

**Default parameters.** With each code using its own database, the mean agreement score over the
eight AlGaN devices is 7.9/10. The remaining differences are parameter choices, not numerics.

## 10.2 Comparison with published experiments

102 published structures were rebuilt from the layer data in each paper and solved with default
settings at 300 K and zero bias.

**Rules fixed before simulating**

| Unknown | Rule |
|---|---|
| surface barrier | the paper's value if stated; otherwise $`0.84 + 1.3x`$ eV (Ambacher 2000) |
| GaN-capped surface | 0.84 eV |
| insulating AlN or Al-rich buffer | Fermi level at mid-gap |
| unintentionally doped GaN buffer | $`N_D = 10^{16}`$ cm<sup>−3</sup> |
| measurement below 77 K | simulated at 77 K |

No parameter was adjusted for any individual paper.

**Scoring**

| Score | Energy error | Density, field or charge error |
|---|---|---|
| 10 | ≤ 30 meV | ≤ 10% |
| 9 | ≤ 60 meV | ≤ 20% |
| 8 | ≤ 100 meV | ≤ 30% |
| 7 / 6.5 | ≤ 150 meV | ≤ 50% |
| 5 | ≤ 250 meV | ≤ factor 2 |
| 3 | ≤ 400 meV | ≤ factor 3 |
| 1 | larger | larger |

**Result: 7.2/10 over 102 papers and 153 measured values (1969-2026).**

| Quantity | Model chapter | Papers | Score | Typical deviation |
|---|---|---|---|---|
| dopant ionization energies | 5.3 | 9 | 9.3 | below 50 meV |
| polarization interface charge | 4.3 | 2 | 9.7 | below 20% |
| 2DHG sheet density | 4, 5 | 4 | 8.8 | below 20% |
| band gaps and bowing | 2.2 | 14 | 8.1 | below 50 meV at 300 K |
| Schottky barrier heights | 5.6 | 2 | 8.0 | below 0.1 eV, except Pt |
| intersubband energies | 6 | 2 | 7.5 | 60-120 meV high |
| 2DEG sheet density | 4, 5 | 39 | 7.3 | median +19% |
| quantum-well fields | 4, 5 | 10 | 6.7 | InGaN within 30%; GaN/AlGaN 1.7x high |
| polarization-doped layers | 4.4 | 3 | 6.1 | 20-60% |
| band offsets | 2.3 | 9 | 4.8 | 0.1-0.45 eV |
| quantum-well emission energy | 8 | 7 | 4.6 | see 10.3 |

By reliability of the source: 7.7 where the paper states structure and value explicitly (69
papers), 6.2 where a dimension had to be assumed (33 papers). By date: 7.6 from 2000 onward, 5.3
before.

2DEG sheet density in detail (56 values):

| Group | Values | Median tool / measured | Range |
|---|---|---|---|
| all | 56 | 1.19 | 59% within 30%, 88% within a factor 2 |
| InAlN and InAlGaN barriers | 12 | 1.06 | 0.82-1.50 |
| AlN/GaN/AlN quantum wells | 10 | 1.16 | 0.70-1.55 |

## 10.3 Known deviations, by model

| Model | Deviation | Probable cause |
|---|---|---|
| 5.6 surface boundary | 2DEG overestimated 2-5x for barriers below 3 nm (AlN at 2-2.5 nm, Al<sub>0.72</sub>Ga<sub>0.28</sub>N at 2.5 nm) | fixed surface barrier; conduction/valence split of the AlN deformation potentials |
| 4, 5 | 2DEG about 19% high overall | ideal surface, no buffer traps, piezoelectric constants at the high end of the literature |
| 2.2 band gap | mid-composition AlGaN gap about 0.1 eV high | bowing 0.7 eV; measurements favour about 1.0 eV |
| 2.2 band gap | low-temperature optical data off by 60-110 meV | no $`E_g(T)`$ |
| 2.3 offsets | InN-containing offsets off by 0.1-0.45 eV | the measurements themselves span 0.5 eV |
| 6 quantum | intersubband energies 60-120 meV high | parabolic conduction band |
| 6, 8 | emission of 1-3 monolayer GaN/AlN wells 0.45-0.65 eV low | envelope-function approximation not valid |
| 8 optics | zero-bias $`E_{11}`$ of Al-rich wells 0.5 eV below electroluminescence | field screening under injection; $`O_{11} \lt 1\%`$ |
| 4 polarization | GaN/AlGaN quantum-well fields 1.7x the optically extracted value | long-standing gap between polarization theory and QW optics |
| 5.6 contacts | Pt barrier 0.5 eV high | Schottky-Mott rule, no pinning |

## 10.4 Guidance

| Use | Applies to |
|---|---|
| directly | sheet densities for barriers of 4 nm or more; hole gases; polarization charge; InGaN well fields; critical barrier thickness; carrier concentrations from Si and Mg doping |
| with a correction | emission energies (subtract exciton binding; add 60-110 meV for low temperature); AlGaN gaps near $`x = 0.5`$ (about 0.1 eV lower); intersubband energies (60-120 meV lower) |
| not at all | absolute 2DEG density below 3 nm barrier thickness; monolayer wells; zero-bias transition energy as a stand-in for electroluminescence of wide Al-rich wells; Pt barriers from the default metal model |

## 10.5 Limits of the validation

- About half of the 78 papers added in the last round come from two groups that publish open
  copies. Their growth method and surface preparation are over-represented.
- Nominally identical structures in different papers differ by 30-70% in measured sheet density.
  Agreement better than that cannot be established from the literature.
- Nothing under bias has been compared with measurement: current-voltage curves, efficiency and
  gain are unvalidated.
- N-polar structures and tunnel junctions are not covered.

## 10.6 Reproducing

| Benchmark | Command |
|---|---|
| nextnano++ comparison | `python benchmarks/run_all.py` (needs nextnano++ and nextnanopy) |
| literature comparison | `python benchmarks/literature/lit_bench2.py -j 6`, then `gen_report100.py` |

The `benchmarks/` folder holds the scripts, the per-paper tables and the source of every measured
value. It is kept outside version control.
