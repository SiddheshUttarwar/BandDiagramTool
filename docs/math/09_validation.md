# 9. Validation

Two independent checks have been run. The first asks whether the equations are solved correctly;
the second asks whether the equations and parameters describe real devices.

## 9.1 Numerics: comparison with nextnano++

Eight AlGaN devices and four In-containing devices were solved in both codes at zero bias
(nextnano++ free edition, 100-point grid). With nextnano's own material parameters injected into
this tool, so that only the numerics differ:

- band edges agree to about 10 meV (12-23 meV for the In-containing devices);
- sheet densities agree within 0-3%;
- the first subband energy of an Al$_{0.3}$Ga$_{0.7}$N/GaN well agrees within 1 meV
  ($-122.3$ vs $-121.3$ meV), and the 2DEG density within 0.4% ($1.229\times10^{13}$ vs $1.234\times10^{13}$ cm$^{-2}$).

The solver numerics are therefore not the limiting factor. With each code using its own default
parameters the agreement score is 7.9/10, and what remains is parameter choice.

## 9.2 Physics: comparison with 102 published experiments

Each published structure was rebuilt and solved with default settings at 300 K and zero bias, and
the predicted quantity compared with the measured one. Rules for unknowns (surface barrier,
background doping, buffer boundary) were fixed before any simulation, and nothing was tuned per paper.

**Overall: 7.2/10 across 102 papers and 153 measured values, 1969-2026.**

| Quantity | Papers | Score /10 | Typical error |
|---|---|---|---|
| Dopant ionization energies | 9 | 9.3 | under 50 meV in most cases |
| Polarization interface charge | 2 | 9.7 | under 20% |
| 2D hole gas density | 4 | 8.8 | under 20% |
| Band gaps and bowing | 14 | 8.1 | under 50 meV at 300 K; 50-150 meV at low T and mid-alloy |
| Schottky barrier heights | 2 | 8.0 | under 0.1 eV except Pt |
| Intersubband energies | 2 | 7.5 | 60-120 meV high |
| 2D electron gas density | 39 | 7.3 | median +19%; 88% within a factor of 2 |
| Quantum-well fields | 10 | 6.7 | InGaN within 30%; GaN/AlGaN about 1.7x high |
| Polarization-doped layers | 3 | 6.1 | 20-60% |
| Band offsets | 9 | 4.8 | 0.1-0.45 eV, largest for InN |
| Quantum-well emission energy | 7 | 4.6 | 60-130 meV for ordinary wells; fails for monolayer wells and for LEDs under injection |

Scoring: an energy within 30 meV scores 10, within 100 meV 8, within 250 meV 5; a density, field or
charge within 10% scores 10, within 30% 8, within a factor of two 5.

Papers whose structure and measured value are both stated explicitly average 7.7; papers where a
dimension had to be assumed average 6.2. Papers from 2000 onward average 7.6; earlier ones 5.3,
mostly because early material was defect-dominated.

## 9.3 What the validation says, for a user

**Use with confidence**

- 2DEG density in AlGaN/GaN, InAlN/GaN and AlN/GaN/AlN structures with barriers of 4 nm or more.
  Expect the prediction to be 10-20% above a Hall measurement.
- 2D hole gas density at GaN/AlN.
- Polarization charge, built-in fields in InGaN wells, the critical barrier thickness of an AlGaN/GaN HEMT.
- Hole and electron concentrations from Mg and Si doping across composition.

**Use with a correction in mind**

- Emission wavelength of ordinary quantum wells: subtract exciton binding, and add 60-110 meV when
  comparing with low-temperature data.
- AlGaN band gap near $x = 0.5$: the tool is about 0.1 eV high.
- Intersubband energies: 60-120 meV high.
- GaN/AlGaN quantum-well fields: the tool gives the theoretical polarization field, which is
  consistently above what optical experiments extract.

**Do not rely on**

- Absolute 2DEG density for barriers thinner than about 3 nm. The tool overestimates by 2-5 times.
- Emission from GaN wells 1-3 monolayers thick.
- Zero-bias transition energy as a predictor of electroluminescence from wide, Al-rich polar wells.
- InN-containing band offsets to better than 0.3 eV. The measurements themselves disagree by that much.
- Pt Schottky barriers from the default metal model; give the measured barrier explicitly.

## 9.4 Limits of the validation itself

- About half of the added papers come from two research groups that publish open-access copies,
  so their growth and processing conditions are over-represented.
- Nominally identical structures from different papers differ by 30-70% in measured sheet density.
  Agreement better than that cannot be demonstrated from the literature.
- Nothing under bias has been validated against measurement: current-voltage curves, efficiency
  and gain are untested.
- N-polar devices and tunnel junctions are not covered.

The full tables, with every paper, structure, measured value, predicted value and source, are
produced by `benchmarks/literature/lit_bench2.py` and `gen_report100.py`. That folder is kept
outside version control.
