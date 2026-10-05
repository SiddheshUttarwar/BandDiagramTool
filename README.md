# BandDiagramTool

A one-dimensional Schrödinger-Poisson-current solver for wurtzite III-nitride heterostructures
(GaN, AlN, InN and their alloys), written in Python, with a desktop GUI.

Given a layer stack grown along the c-axis it computes band edges, electric field, polarization
charge, electron and hole densities, confined states, quantum-well transition energies and, under
bias, the drift-diffusion current.

**Documentation: <https://siddheshuttarwar.github.io/BandDiagramTool/>**
Every equation the solver uses, the parameter tables and their sources, the solution algorithms,
and the validation results.

## What it is, and what it is not

It is a small, readable, nitride-specific tool whose physics is documented equation by equation
and whose accuracy has been measured against published experiments.

It is not a replacement for a general device simulator. It is one-dimensional, c-plane only,
single-band (no k·p), and its transport model is plain drift-diffusion without tunnelling. The
[limits chapter](https://siddheshuttarwar.github.io/BandDiagramTool/11_limits_and_comparison.html)
lists what is left out and compares the models with nextnano++ feature by feature.

## Validation

| Check | Result |
|---|---|
| Numerics vs nextnano++ (12 structures, same parameters) | sheet densities within 3%; band edges within about 10 meV where the comparison grid resolves the structure |
| 102 published experiments, 153 measured values, default settings, nothing tuned per paper | 7.2 / 10 overall |
| 2DEG sheet density (56 values) | median 20% above measurement; 59% within 30%, 86% within a factor of two |
| Dopant levels, polarization charge, hole gases | within 20% or 50 meV |
| Weakest areas | band offsets involving InN; emission energy of monolayer wells and of LEDs under injection; forward prediction for barriers thinner than about 4 nm |

The [validation chapter](https://siddheshuttarwar.github.io/BandDiagramTool/10_validation.html)
gives the method, the failures and their causes, and the
[appendix](https://siddheshuttarwar.github.io/BandDiagramTool/12_benchmark_tables.html) lists every
paper with its measured and computed value. The benchmark was assembled with machine assistance and
has not been reviewed by an independent expert; corrections are welcome.

## Install

Python 3.10 or newer.

```
git clone https://github.com/SiddheshUttarwar/BandDiagramTool.git
cd BandDiagramTool
pip install -r requirements.txt
```

## Use

**GUI**

```
python run_gui.py
```

Build a stack from layer cards, set the contacts, and solve one bias point or a voltage sweep.

**Python**

```python
from devices.device import AlGaNDevice
from devices.layer import AbruptLayer, Contact
from devices.analysis import sheet_density

layers = [AbruptLayer(x_Al=0.0, thickness_nm=200, n_doping=1e16),   # GaN buffer
          AbruptLayer(x_Al=0.3, thickness_nm=25)]                   # Al0.3Ga0.7N barrier
contacts = [Contact('bottom', 'ohmic', 'Ti'),
            Contact('top', 'schottky', 'Ni', barrier_eV=1.23)]      # surface pinning

result = AlGaNDevice(layers, contacts).solve(V_applied=0.0, quantum=False)
print(sheet_density(result, 150, 225))    # 2DEG density, about 1.2e13 cm^-2
```

Layers are listed from the substrate side up. `AbruptLayer` and `GradedLayer` take Al and In
fractions (`x_Al`, `x_In`), thickness and doping. Useful device options:

| Option | Meaning |
|---|---|
| `T` | temperature in K (band gaps follow Varshni) |
| `polarity='N'` | N-polar growth; default is metal-polar |
| `polarization_model='dreyer2016'` | Dreyer et al. 2016 constants; default is Ambacher 2002 |
| `solve(quantum=True)` | include the Schrödinger equation |
| `solve(V_applied=V)` | drift-diffusion solve under bias |
| `solve(V_applied=V, flat_qfl=True)` | band diagram under bias without solving for current |

## Examples

| Script | Shows |
|---|---|
| `examples/01_hemt_2deg.py` | AlGaN/GaN 2DEG: classical and quantum, both polarization sets, N-polar |
| `examples/02_quantum_well_stark_effect.py` | field, transition energy and overlap of GaN/AlN wells vs width |
| `examples/03_surface_barrier_from_measurement.py` | surface barrier extracted from published thin-barrier Hall data |
| `examples/04_pn_diode_under_bias.py` | forward current of a GaN p-n diode with the current-conservation check |

## Tests

```
python -m pytest            # all tests, about two minutes
python -m pytest -m "not slow"
```

## Repository layout

| Path | Contents |
|---|---|
| `devices/` | layers, contacts, grid construction, analysis helpers |
| `physics/` | materials, strain and polarization, Poisson, Schrödinger, drift-diffusion, optics |
| `gui/`, `visualization/` | PyQt6 interface, plots, CSV export |
| `examples/`, `tests/` | runnable examples and the test suite |
| `docs/` | the published documentation site (`docs/build/` regenerates it) |

## Citing sources

Material parameters and models are taken from the literature; each chapter of the documentation
names its sources. The main ones are Vurgaftman and Meyer (2003) for band parameters and
deformation potentials, Ambacher et al. (2002) and Bernardini et al. (1997) for polarization,
and Dreyer et al. (2016) for the alternative polarization set.

## Licence

No licence has been chosen yet. Until one is added, the code is published for reading and
evaluation only.
