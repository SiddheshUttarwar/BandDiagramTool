# 10. Assumptions and limits

Everything the model assumes, in one place. Each item names the chapter with the detail.

## Geometry

| Assumption | Consequence |
|---|---|
| One dimension, along the growth axis ([1](01_device_model_and_grid.md)) | No current spreading, gate-edge fields, mesa or sidewall effects |
| Metal-polar (Ga-face) c-plane only ([3](03_strain_and_polarization.md)) | N-polar, semi-polar and non-polar orientations are not available |
| Abrupt, laterally uniform interfaces | No interface roughness, alloy fluctuation or indium clustering |

## Materials

| Assumption | Consequence |
|---|---|
| Wurtzite Al-In-Ga-N only | No Sc- or B-containing alloys, no dielectrics, no substrate materials |
| Parameters at 300 K; only spontaneous polarization and the thermal energy change with temperature ([2](02_material_model.md)) | Band gaps, masses and dopant levels are 300 K values at any simulation temperature |
| AlGaN gap bowing 0.7 eV | Mid-composition gaps are about 0.1 eV above most measurements |
| Linear valence-band offset, InN from one XPS study | In-containing offsets uncertain by about 0.3 eV |
| Parabolic, decoupled bands | No nonparabolicity, no valence-band mixing |

## Strain

| Assumption | Consequence |
|---|---|
| Coherent to the bottom layer at any thickness unless overridden ([3](03_strain_and_polarization.md)) | No critical thickness, no partial relaxation, no cracking |
| Biaxial strain, linear elasticity and piezoelectricity | No shear, no second-order piezoelectric terms |

## Charges

| Assumption | Consequence |
|---|---|
| Every dopant atom is active, single level per dopant ([4](04_carrier_statistics.md)) | No compensation, self-compensation, DX centres or impurity bands in the electrostatics |
| Surface described by a fixed barrier, or by trap states the user adds | The default does not model surface donors, passivation or dielectric charge |
| No buffer traps (Fe, C) | Buffer depletion of the 2DEG is not captured |

## Quantum mechanics

| Assumption | Consequence |
|---|---|
| Single-band envelope functions ([6](06_schrodinger_quantum.md)) | Not valid for wells of 1-3 monolayers |
| Hartree potential only | No exchange-correlation, no excitons |
| Quantum charge only inside the quantum region | States outside it are treated classically |

## Transport

| Assumption | Consequence |
|---|---|
| Drift-diffusion with field-independent mobility ([7](07_drift_diffusion_bias.md)) | No velocity saturation, hot carriers, tunnelling or thermionic emission |
| Generic recombination constants | Current magnitude and efficiency are indicative only |
| Isothermal | No self-heating |
| Steady state | No transients, capacitance-voltage or frequency response |

## Optics

| Assumption | Consequence |
|---|---|
| Band-to-band transition energies ([8](08_optical_outputs.md)) | No exciton binding, Stokes shift or localization |
| Band-edge gain model | Peak position and sign are meaningful; magnitude is order-of-magnitude |
| No optical mode calculation | No confinement factor, threshold or extraction efficiency |

## Related modules not covered here

The repository also contains a point-defect formation-energy calculator, a phenomenological
DX-centre model, a surface-pinning model and a step-flow growth (Burton-Cabrera-Frank) profile
under `physics/defects/`. They read a converged band diagram and evaluate defect thermodynamics on
top of it. They do not feed charge back into the device solution described in these notes.
