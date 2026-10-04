# 3. Strain, polarization and strained band edges

Polarization is what makes nitride devices different from other III-V devices: sheet charges of
order $10^{13}$ cm$^{-2}$ appear at every heterointerface without any doping. This chapter gives
the chain from lattice mismatch to the charge that enters Poisson's equation.

## 3.1 Strain

A layer grown coherently on a substrate with in-plane lattice constant $a_{sub}$ takes that
lattice constant. Its in-plane strain is

$$
\varepsilon_{xx} = \varepsilon_{yy} = \frac{a_{sub} - a_0(x,y)}{a_0(x,y)},
$$

positive for tension (AlGaN on GaN), negative for compression (InGaN on GaN, GaN on AlN).
The crystal is free to relax along the growth direction, which gives

$$
\varepsilon_{zz} = -2\,\frac{C_{13}}{C_{33}}\,\varepsilon_{xx}.
$$

A layer flagged **relaxed** has $\varepsilon_{xx} = 0$. A layer with a **custom strain** uses the
given $\varepsilon_{xx}$, and $\varepsilon_{zz}$ still follows from the relation above.

## 3.2 Piezoelectric polarization

$$
P_{pz} = e_{33}\,\varepsilon_{zz} + 2\,e_{31}\,\varepsilon_{xx}
       = 2\,\varepsilon_{xx}\left(e_{31} - e_{33}\frac{C_{13}}{C_{33}}\right).
$$

The bracket is negative for all nitrides, so tensile layers have $P_{pz} < 0$ (adding to the
spontaneous polarization) and compressive layers have $P_{pz} > 0$ (opposing it).

## 3.3 Spontaneous polarization

Ambacher et al. 2002, with bowing for each cation pair:

$$
P_{sp}(x,y) = -0.090\,x - 0.042\,y - 0.034\,z_{Ga} + 0.021\,x z_{Ga} + 0.037\,y z_{Ga} + 0.070\,x y \quad [\text{C/m}^2].
$$

Temperature enters through the pyroelectric coefficient:

$$
P_{sp}(T) = P_{sp}(300\text{ K}) + (T - 300)\left[x\,p_{AlN} + (1 - x)\,p_{GaN}\right],
$$

with $p_{GaN} = 6.0\times10^{-5}$ and $p_{AlN} = 4.5\times10^{-5}$ C/(m²K). Spontaneous polarization
can be switched off for the whole device.

## 3.4 Interface sheet charge

The total polarization is $P = P_{sp} + P_{pz}$. At an abrupt interface between a lower layer 1
and an upper layer 2 the bound sheet charge is

$$
\sigma = P_1 - P_2.
$$

For a tensile AlGaN barrier on GaN, $P_2$ is more negative than $P_1$, so $\sigma > 0$: a positive
fixed charge that attracts a two-dimensional electron gas. For GaN on AlN the sign reverses and a
hole gas forms.

Worked example, Al$_{0.3}$Ga$_{0.7}$N on relaxed GaN:
$\varepsilon_{xx} = (3.189 - 3.1659)/3.1659 = 0.0073$; the barrier has
$P_{sp} = -0.0464$, $P_{pz} = -0.0114$ C/m²; GaN has $P = -0.034$ C/m².
$\sigma = 0.0238$ C/m² $= 1.49\times10^{13}$ cm$^{-2}$.

## 3.5 Polarization charge

In general the bound charge density is the divergence of the polarization,

$$
\rho_{pol}(z) = -\frac{dP}{dz}.
$$

An abrupt interface gives a sheet charge; a **graded layer** gives a uniform volume charge

$$
\rho_{pol} = -\frac{P_{end} - P_{start}}{d},
$$

which is the basis of polarization doping. Grading from GaN up to Al$_{0.3}$Ga$_{0.7}$N over
100 nm on a metal-polar surface gives a fixed positive charge of about $1.5\times10^{18}$
cm$^{-3}$, neutralized by a slab of mobile electrons with no donors present. Grading downward
gives a hole slab.

Discretely, the charge at node $i$ is

$$
\rho_i\, w_i = -\tfrac12 \left(P_{i+1} - P_{i-1}\right),
$$

with $w_i$ the node's control-volume width. Summed over the nodes around an interface this
telescopes to exactly $P_1 - P_2$, independent of how the grid spacing changes there. An ordinary
non-uniform central difference does not have this property and makes the 2DEG density depend on
the mesh.

## 3.6 Strain shift of the band edges

Strain also moves the band edges (enabled by default). The conduction band shifts by

$$
\Delta E_c = a_{cz}\,\varepsilon_{zz} + 2\,a_{ct}\,\varepsilon_{xx},
\qquad a_{cz} = a_1 + D_1,\quad a_{ct} = a_2 + D_2,
$$

and the top valence band by the change in the largest eigenvalue of the strained valence
Hamiltonian of [section 2.4](02_material_model.md#24-valence-band-structure):

$$
\Delta E_v = \max\{E_{HH}, E_\pm\}_{strained} - \max\{E_{HH}, E_\pm\}_{unstrained}.
$$

Both shifts are added to the flat-band edges at every node, using that node's actual strain. The
gap changes by $\Delta E_c - \Delta E_v$. Examples: GaN compressed to the AlN lattice constant
(2.4%) has its gap raised by 0.18 eV; Al$_{0.6}$Ga$_{0.4}$N on AlN goes from 4.80 to 4.97 eV.

## 3.7 What this model does not include

- **N-polar growth.** All signs assume metal polarity.
- **Relaxation.** Coherent strain is assumed at any thickness unless overridden per layer.
- **Second-order piezoelectricity** and strain dependence of the piezoelectric constants.
- **Screening of the polarization charge by surface or interface traps**, unless the user adds a
  surface-charge layer explicitly.
