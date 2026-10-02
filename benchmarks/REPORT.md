# BandDiagramTool vs nextnano++ at V = 0 (2026-10-01)

## UPDATE: after the material/model fixes (same day)
All six recommended changes are implemented. Re-run with `python benchmarks/run_all.py`.

| Device | Before (old defaults) | After (new defaults) | nextnano params (ceiling) |
|---|---|---|---|
| GaN p-n | 9 | 9.5 | 10 |
| Al0.3GaN/GaN HEMT | 1 | **9** (2DEG 1.23e13 vs NN 1.24e13) | 10 |
| AlN/GaN HEMT | 1 | **7.5** (5.12e13 vs 5.28e13) | 9 |
| AlGaN/AlN/GaN HEMT | 3 | **7.5** (1.43e13 vs 1.45e13) | 7.5* |
| Polarization-doped graded p-n | 1 | **9** (+0–1 %) | 10 |
| DUV LED 3xMQW | 3.5 | **7.5** | 7.5* |
| UVC LD | 1 | **4** | 5.5* |
| AlGaN/GaN HBT | 4.5 | **9** | 10 |
| **Average** | **3.0** | **7.9** | 8.7 |

\* These scores are limited by nextnano's 100-point grid, not by our solver.

Quantum (Schrödinger–Poisson) Al0.3GaN HEMT with the new defaults: n_s = 1.229e13 (nextnano 1.234e13), E1 = −122.3 meV (nextnano −121.3), E2 = 4.8 meV (nextnano 5.0).

What changed:
1. Band alignment: valence-band offset 0.80 eV, type-I. χ is now derived from it (χ_AlN ≈ 2.26 eV).
2. Spontaneous polarization is on by default (device, grid builder, GUI).
3. Psp = −0.090x − 0.034(1−x) + 0.021x(1−x) (Ambacher 2002). There is now one shared function, used by both `algan` and `polarization`.
4. Permittivity along c: GaN 10.1, AlN 8.57. Piezo constants: Bernardini/Ambacher set. AlN C13/C33 = 108/373 GPa.
5. Contacts: `Contact.barrier_eV` adds an explicit Schottky barrier (GUI spin box, "auto" by default). With the corrected χ, the default Schottky–Mott estimate gives Ni ≈ 1.0 eV on GaN and 1.4 eV on Al0.3GaN.
6. Dopants: Si Ed(x) and Mg Ea(x) are composition-dependent and used consistently in the contacts, initial guess and every solve path.
7. Valence band:
   * `Ev` is now the TOPMOST valence band.
   * HH/LH/SO are depths below it (`dEv_hh/lh/so`).
   * Nv uses anisotropic per-band DOS masses.
   * `Ev_hh` is plotted and exported to CSV.

Regressions checked:
* 62/62 tests pass. The LED golden values and the Mott threshold were re-recorded, with justification in the test docstrings.
* UV-LED example projects converge at 0/3 V (all three) and at 6 V (ungraded).
* The graded projects fail at 6 V **both before and after** these changes. This is a pre-existing high-bias issue.

---
# Original (pre-fix) report


Run: `python benchmarks/run_all.py` (needs nextnanopy + nextnano++ free edition).
Plots and metrics are in `benchmarks/results/`.

## Method
* nextnano++ 2.4.27 **free edition** has three limits: a 100-point grid, no strain module, and no database edits.
  * The grid is clustered at the interfaces (`nngrid.py`).
  * Piezoelectric charge is computed from nextnano's own database (Vurgaftman 2003) and injected as fixed charge per cell. nextnano still computes the spontaneous (pyro) part itself.
  * Band edges are unstrained in both codes.
* Dopant model is matched to our solver (Ed = 20 meV, g = 2; Ea = 170 meV, g = 4). Schottky barriers are shared: e·φB = 0.84 + 1.3x eV (Ambacher 2000).
* Each device runs four ways:
  * **NN**: nextnano.
  * **def**: our tool with GUI defaults.
  * **psp**: our tool with Psp on.
  * **nnp**: our tool with nextnano's parameters injected. This isolates solver numerics.

## Scores (/10)
Score = 10 − band penalty − sheet penalty. The band penalty uses the 95th-percentile |ΔEc|, |ΔEv|; the sheet penalty uses the 2DEG/2DHG error. Both are measured against NN.

| Device | def | psp | nnp (numerics) |
|---|---|---|---|
| GaN p-n | 9 | 9 | 10 |
| Al0.3GaN/GaN HEMT | 1 | 2.5 | 10 |
| AlN/GaN HEMT | 1 | 3 | 9 |
| AlGaN/AlN/GaN HEMT | 3 | 4 | 7.5* |
| Polarization-doped graded p-n | 1 | 4 | 10 |
| DUV LED 3xMQW | 3.5 | 3.5 | 7.5* |
| UVC LD, graded p-clad | 1 | 1 | 6.5* |
| AlGaN/GaN HBT | 4.5 | 4.5 | 10 |

\* These scores are capped by nextnano's own 100-point grid. For the UVC LD, moving nextnano's points changed its hole sheet by 19%; ours did not move.

Quantum check, Al0.3GaN HEMT (nnp vs NN): E1 = −121.6 vs −121.3 meV, E2 = 4.8 vs 5.0 meV, n_s = 1.233e13 vs 1.234e13 cm⁻².

## Bug found and fixed
`physics/polarization.compute_pol_charge` used the non-uniform 3-point derivative, which does not conserve charge.
* Wherever the grid spacing changed at a polarization step, the sheet charge was scaled by h1/h2.
* The tool's own auto-refinement of thin (<6-point) layers triggers this, so default users were affected too.
* On the interlayer HEMT, the 2DEG varied from 9.2e12 to 3.6e13 depending on mesh.
* After the fix it is 1.42–1.46e13 for every mesh (nextnano: 1.45e13).
* Regression test: `tests/physics/test_polarization.py::test_pol_charge_conserved_on_nonuniform_grid`.

## Material/model issues (NOT changed; need a decision)
1. **Band offsets are wrong (type-II).** χ is interpolated linearly from 4.1 to 0.6 eV.
   * That gives GaN/AlN ΔEc = 3.5 eV, with the AlN valence band 0.86 eV *above* GaN.
   * Literature and nextnano: ΔEc ≈ 1.9 eV, ΔEv ≈ 0.7–0.8 eV, type-I.
   * This is the dominant error in every heterostructure. It also removes the hole-blocking ΔEv that the HBT emitter exists to provide.
2. **Psp is off by default.** The AlGaN HEMT 2DEG comes out at 5.1e12 instead of 1.24e13 (Ambacher measured ≈1.0–1.3e13).
3. **Psp bowing sign is flipped.** The code uses −0.021x(1−x); Ambacher 2002 (and nextnano's computed pyro charge) use **+**0.021x(1−x). That is about 2.8e12 cm⁻² at x = 0.3.
4. **Permittivity and piezo constants differ.**
   * GaN ε = 8.9 vs 10.1 (static, along c).
   * GaN e33 = 0.65 vs 1.27 (Vurgaftman) / 0.73 (Bernardini).
5. **Schottky barrier uses Schottky–Mott with the same χ.** Ni gives 2.05 eV on Al0.3GaN and 4.5 eV on AlN; typical values are about 1.2–1.3 eV and ~2 eV.
6. **Dopant energies are composition-independent** (Si 20 meV, Mg 170 meV), even in Al-rich layers. Real values reach ~0.25 eV (Si) and ~0.5 eV (Mg) in AlN.
7. **Only Al(x)Ga(1-x)N is supported.** InGaN LEDs and lasers cannot be benchmarked.
