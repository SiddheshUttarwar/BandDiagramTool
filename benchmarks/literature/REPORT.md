# BandDiagramTool vs 20 published AlN/GaN/AlGaN experiments (1969–2024)

## UPDATE 2026-10-01: strain-dependent band edges added
- **Model:** deformation potentials (Vurgaftman & Meyer 2003, the same values nextnano uses) in the strained Chuang–Chang Hamiltonian. They shift Ec and the HH/LH/CH valence bands at every grid point using that point's actual strain: pseudomorphic, relaxed, or the layer's `custom_strain_xx`.
- **Switch:** `AlGaNDevice(include_strain_band_shift=True)` is the default. The nextnano benchmark passes False, because the free nextnano edition can't strain its band edges.
- **Bug fixed on the way:** the unstrained Γ7 valence block was missing its (Δ1−Δ2)/2 centre. GaN A–B / A–C splittings are now 5 / 43 meV (were 24 / 62), and AlN's crystal-field band sits 163 meV above HH (was 251).
- **Custom strain verified:**
  - A custom strain equal to the pseudomorphic value reproduces the default (interface charge identical, 2DEG within 0.15%).
  - `relaxed=True` removes both the band shift and the piezo charge.
  - The relaxed Al0.3GaN/GaN interface carries 7.734e12 cm⁻², matching nextnano's own spontaneous-polarization charge (7.733e12).
- **Effect on the benchmark:**
  - **GaN/AlN quantum-well PL** (Adelmann; GaN compressed by 2.4%): wide-well E1H1 is 2.37 eV vs 2.30 eV measured. That's +65 meV raw, or about +15 meV after the typical exciton binding; it was −97 meV without the strain shift. The 0.7 nm (~3 monolayer) well overshoots by 129 meV, the limit of the effective-mass model.
  - **AlGaN/GaN 2DEGs** move by only ~1%.
  - **The 2 nm AlN barrier is NOT fixed.** In Vurgaftman's split, tensile AlN's gap shrinkage is mostly a valence-band rise and its Ec rises 0.10 eV. Cao & Jena's fit assumed the opposite split.
- **Rathkanthiwar 2022 added (paper #21, 8.1/10):**
  - Al₀.₆Ga₀.₄N stays fully pseudomorphic on AlN up to 3.5 µm. That matches the tool's assumption, though the tool has no relaxation or critical-thickness model of its own.
  - The strained c-lattice is predicted within 0.1%.
  - Stress, from the tool's strain plus Vurgaftman C11/C12, is 22–41% above the curvature-derived values.
  - On the band side, the tool's strain shift raises Al₀.₆GaN's gap by 162 meV (4.80 → 4.97 eV) and gives −2.8×10¹³ cm⁻² of polarization charge at the AlN/AlGaN interface.
- **New overall: 6.8 / 10.** The two PL points were added with this change and scored with the same rubric.

To reproduce, run `python benchmarks/literature/lit_bench.py`. Results are in `lit_results.json`; the plot is `lit_parity.png`.

## Method
- Each structure is built exactly as published. All runs use the tool's **default** physics: classical, 300 K, V = 0.
- The **predicted** observable is compared with the paper's **measured** value.
- Rules fixed before simulating:
  - **Surface barrier:** the paper's own value if it states one; else Ambacher (0.84 + 1.3x eV).
  - **Insulating AlN:** mid-gap Fermi level (3.0 eV barrier).
  - **UID GaN background:** 1e16 cm⁻³.
- **Score:**
  - Energies: ≤30 meV → 10, ≤60 → 9, ≤100 → 8, ≤150 → 7, ≤250 → 5, ≤400 → 3, else 1.
  - Densities, fields and charges (symmetric ratio error): ≤10% → 10, ≤20% → 9, ≤30% → 8, ≤50% → 6.5, ≤×2 → 5, ≤×3 → 3, else 1.
  - Critical thickness: ≤0.5 nm → 10, ≤1 nm → 8, ≤2 nm → 6, else 3.
- **Confidence** says how completely the paper specifies the structure and value:
  - **High:** text or table.
  - **Medium:** figure reading or one assumed dimension.
  - **Low:** major assumption.

## Results
| Year | Paper | Observable | Measured | Tool | Score | Conf. |
|---|---|---|---|---|---|---|
| 1969 | Maruska & Tietjen, APL 15, 327 | GaN Eg (RT) | 3.39 eV | 3.39 eV | 10 | High |
| 1973 | Yim et al., JAP 44, 292 | AlN Eg (RT) | 6.2 eV | 6.03 eV | 5 | High |
| 1974 | Monemar, PRB 10, 676 | GaN Eg (1.6 K) | 3.503 eV | 3.39 eV | 7 | High |
| 1992 | Khan et al., APL 60, 3027 | first GaN/Al0.13GaN 2DEG | 1e11 cm⁻² | 4.1e12 | 1 | Low |
| 1997 | Brunner et al., JAP 82, 5090 | AlGaN bowing (as Eg at x=0.5) | b = 1.3 eV | b = 0.7 (+150 meV) | 5 | High |
| 1997 | Li, Jiang, Khan et al., JVST B 15, 1117 | n-Al0.1GaN/GaN 2DEG (dark) | 0.85e12 | 2.7e12 | 1 | Medium |
| 1999 | Ambacher et al., JAP 85, 3222 | 2DEG at x=0.15 / 0.31 | 6e12 / 2e13 | 5.0e12 / 1.32e13 | 7 | Medium |
| 1999 | Grandjean et al., JAP 86, 3714 | GaN/Al0.27GaN QW field | ~1 MV/cm | 2.17 MV/cm | 3 | Medium |
| 2000 | Ibbetson et al., APL 77, 250 | critical barrier thickness (Al0.34) | 3.5 nm | 3.5 nm | 10 | High |
| 2000 | Kaufmann et al., PRB 62, 10867 | GaN:Mg holes at [Mg]=2e19 | 6e17 | 5.5e17 | 10 | High |
| 2002 | Jena et al., APL 81, 4395 | graded 0-10/20/30% 3DES; Al0.2 2DEG | 1.7/7.8/8.9e12; 7.8e12 | 2.3/6.1/10.4e12; 6.5e12 | 7.9 | High |
| 2003 | Heikman et al., JAP 93, 10114 | Al0.32/GaN polarization charge | 1.6–1.7e13 | 1.60e13 | 10 | High |
| 2003 | Adelmann et al., cond-mat/0304124 | GaN/AlN QW field | 9.2 MV/cm | 9.8 MV/cm | 10 | Medium |
| 2006 | Taniyasu et al., Nature 441, 325 | AlN LED emission | 5.90 eV (210 nm) | 6.03 eV | 7 | High |
| 2007 | Cao & Jena, APL 90, 182112 | AlN/GaN 2DEG at 2/4/6 nm AlN | 0.5/3.5/5.0e13 | 2.3/4.1/4.8e13 | 6.7 | Medium |
| 2017 | Zhu et al., arXiv:1704.03001 | graded 0–20% / 600 nm, n | ~1e17 cm⁻³ | 1.6e17 | 5 | Medium |
| 2019 | Chaudhuri et al., Science 365, 1454 | undoped GaN/AlN 2DHG | 4e13 | 4.3e13 | 10 | High |
| 2022 | Rathkanthiwar et al., APL 120, 202105 | pseudomorphic Al0.6GaN on AlN: c-lattice; stress (0.95/1.8/3.5 µm) | 5.086 Å; −3.8/−3.6/−3.3 GPa | 5.091 Å; −4.6 GPa | 8.1 | High |
| 2023 | Mukhopadhyay et al., arXiv:2304.05593 | 5 AlGaN/AlN/GaN HEMTs | 1.12–1.63e13 | 1.69–1.82e13 | 7.5 | High |
| 2023 | Knight et al., JAP 134, 185701 | n_s at x = 0.07–0.42 | 2.3e12–1.45e13 | 1.4e12–1.96e13 | 6.5 | Medium |
| 2024 | Chen et al., APL 124, 152111 | 4 AlN/GaN/AlN QW-HEMTs | 1.99–3.68e13 | 2.28–4.54e13 | 8.1 | High |

**Overall: 6.9 / 10.**
- High-confidence papers: 8.1 (n = 12); medium/low: 5.0 (n = 8).
- Post-2000: 8.2; pre-2000: 4.9.

## What the misses say about the tool
1. **No temperature-dependent band gap.** Eg is fixed at 300 K values (Monemar 1974: −113 meV at 1.6 K). This affects any low-temperature calculation. Fix: Varshni α, β.
2. **AlGaN bowing of 0.7 eV is low** versus 1.0–1.3 eV measured (Brunner 1997). Mid-composition gaps come out about 0.1–0.15 eV too large. It's a literature-dependent choice (Vurgaftman recommends 0.8).
3. **No exciton binding** in emission predictions. AlN emits 120 meV below the tool's Eg.
4. ~~No strain-dependent band edges~~ **Fixed (see update above).** Ultrathin 2 nm AlN still gets too much 2DEG. The remaining cause is the CB/VB split of the deformation potentials and/or the surface barrier.
5. **Systematic ~+20% 2DEG overestimate on modern HEMTs** (2023–2024). Likely causes:
   - ideal surface barrier and no surface or buffer traps;
   - classical rather than quantum charge;
   - Bernardini piezo constants on the high side.
6. **Ideal material only.** There is no compensation, trap or relaxation model, so early (1990s) defect-dominated samples and partially relaxed barriers (Knight x = 0.42) are overestimated.
7. **QW fields match polarization theory, not AlGaN/GaN QW experiments** (Grandjean 1999). This ~2× gap is a long-standing open issue in the literature itself. For GaN/AlN QWs the tool matches experiment (Adelmann, +7%).
8. **Scope:** Al(x)Ga(1-x)N only. InGaN LEDs and lasers, N-polar devices and ScAlN can't be tested. Zhang 2019 (271.8 nm laser) was dropped because its well composition isn't public.

## Excluded (no verifiable structure or number found)
- Smorchkova 1999 and Shen 2001: numbers not accessible.
- Leroux 1998: barrier composition not found.
- Nakajima 2010 and Hickman 2019: layer stack not published.
- Pampili 2026: sheet resistance only, no n_s.
- Simon 2008: no hole density measured.
