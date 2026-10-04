# 2. Material model: Al(x)In(y)Ga(1-x-y)N

One quaternary model covers GaN, AlN, InN and every alloy between them. AlGaN, InGaN and InAlGaN
in the user interface are views of the same function, so band offsets between any two layers are
always mutually consistent. All values are for the wurtzite phase at 300 K.

Write $x$ = Al fraction, $y$ = In fraction, $z_{Ga} = 1 - x - y$.

## 2.1 Interpolation rule

Every parameter $Q$ is interpolated linearly between the three binaries (Vegard's law),

$$
Q(x, y) = x\,Q_{AlN} + y\,Q_{InN} + z_{Ga}\,Q_{GaN},
$$

except the band gap and the spontaneous polarization, which have bowing terms.

## 2.2 Band gap

$$
E_g(x,y) = x E_g^{AlN} + y E_g^{InN} + z_{Ga} E_g^{GaN}
 - b_{AlGa}\, x z_{Ga} - b_{InGa}\, y z_{Ga} - b_{AlIn}(u)\, x y
$$

| Quantity | Value | Source |
|---|---|---|
| $E_g$ GaN / AlN / InN | 3.39 / 6.026 / 0.69 eV | NSM (Ioffe); Wu, Davydov for InN |
| $b_{AlGa}$ | 0.7 eV | Vurgaftman & Meyer 2003 |
| $b_{InGa}$ | 1.4 eV | Vurgaftman & Meyer 2003 |
| $b_{AlIn}(u)$ | $6.43 / (1 + 1.21 u^2)$ eV, $u = y/(x+y)$ | Sakalauskas 2010 |

The gap does not depend on temperature in the tool, and it is the band-to-band gap: no exciton
binding energy is subtracted.

## 2.3 Band alignment

The line-up is defined through the **valence band**. The top of the valence band relative to GaN is

$$
\Delta E_v(x,y) = -0.80\,x + 0.58\,y + 2.6\,x y \quad [\text{eV}],
$$

that is, AlN lies 0.80 eV below GaN (Vurgaftman & Meyer), InN lies 0.58 eV above GaN
(King et al. 2008, XPS), and the Al-In bowing term reproduces the measured 0.2 eV valence offset
of lattice-matched Al$_{0.83}$In$_{0.17}$N on GaN (Akazawa 2010). The electron affinity then follows,
anchored at $\chi_{GaN} = 4.1$ eV:

$$
E_v^{abs} = -(\chi_{GaN} + E_g^{GaN}) + \Delta E_v, \qquad \chi(x,y) = -\left(E_v^{abs} + E_g(x,y)\right).
$$

The conduction offset between any two materials is $\Delta E_c = \chi_a - \chi_b$. For GaN/AlN this
gives $\Delta E_v = 0.80$ eV and $\Delta E_c = 1.84$ eV, type I.

## 2.4 Valence-band structure

At the zone centre the three valence bands (heavy hole HH, and two bands of mixed
light-hole/crystal-field character) come from the Chuang-Chang wurtzite Hamiltonian with
$\Delta_1 = \Delta_{cr}$ and $\Delta_2 = \Delta_3 = \Delta_{so}/3$:

$$
E_{HH} = \Delta_1 + \Delta_2 + \lambda + \theta,
$$

$$
E_{\pm} = \text{eigenvalues of }
\begin{pmatrix} \Delta_1 - \Delta_2 + \lambda + \theta & \sqrt2\,\Delta_3 \\ \sqrt2\,\Delta_3 & \lambda \end{pmatrix},
$$

with the strain terms

$$
\lambda = D_1 \varepsilon_{zz} + 2 D_2 \varepsilon_{xx}, \qquad \theta = D_3 \varepsilon_{zz} + 2 D_4 \varepsilon_{xx}.
$$

| | GaN | AlN | InN |
|---|---|---|---|
| $\Delta_{cr}$ (eV) | 0.040 | -0.169 | 0.040 |
| $\Delta_{so}$ (eV) | 0.008 | 0.019 | 0.005 |
| $D_1 \dots D_4$ (eV) | -3.7, 4.5, 8.2, -4.1 | -17.1, 7.9, 8.8, -3.9 | -3.7, 4.5, 8.2, -4.1 |

Because $\Delta_{cr}$ is negative in AlN, the crystal-field band is on top in Al-rich AlGaN and the
heavy-hole band is on top in GaN. The tool always references $E_v$ and $E_g$ to whichever band is
on top, and reports the HH, LH and CH edges separately. Unstrained, the model gives GaN A-B and A-C
splittings of 5 and 43 meV, and puts the AlN crystal-field band 163 meV above HH (measured
A-B exciton splitting: 218 meV).

The three bands are treated as independent parabolic bands (no k·p mixing away from the zone centre).

## 2.5 Effective masses and density of states

| | GaN | AlN | InN |
|---|---|---|---|
| $m_e$ | 0.20 | 0.40 | 0.07 |
| $m_{hh}$ along c | 1.4 | 3.53 | 1.63 |
| $m_{lh}$ along c | 0.3 | 3.53 | 0.27 |
| $m_{ch}$ along c | 0.6 | 0.25 | 0.65 |

Conduction-band effective density of states:

$$
N_c = 2 \left( \frac{2\pi m_e k_B T}{h^2} \right)^{3/2}.
$$

The valence-band density of states sums the three bands, each weighted by how far it lies below
the top band ($\delta_b \ge 0$):

$$
N_v = 2 \left( \frac{2\pi m_0 k_B T}{h^2} \right)^{3/2}
\sum_{b \in \{HH, LH, CH\}} \left(m_b^{dos}\right)^{3/2} e^{-\delta_b / k_B T},
\qquad m_b^{dos} = \left(m_{\parallel} m_{\perp}^2\right)^{1/3}.
$$

The density-of-states masses use the anisotropic in-plane and c-axis masses of each band; the
confinement masses in the table above are the c-axis masses used in the Schrödinger equation.

## 2.6 Dielectric, lattice, elastic and piezoelectric constants

| | GaN | AlN | InN |
|---|---|---|---|
| $\varepsilon_r$ along c | 10.1 | 8.57 | 14.4 |
| $a_0$ (Å) | 3.189 | 3.112 | 3.545 |
| $c_0$ (Å) | 5.186 | 4.982 | 5.703 |
| $C_{13}$ / $C_{33}$ (GPa) | 106 / 398 | 108 / 373 | 92 / 224 |
| $e_{31}$ / $e_{33}$ (C/m²) | -0.49 / 0.73 | -0.60 / 1.46 | -0.57 / 0.97 |
| $P_{sp}$ (C/m²) | -0.034 | -0.090 | -0.042 |
| Conduction-band deformation potentials $a_1$, $a_2$ (eV) | -4.9, -11.3 | -3.4, -11.8 | -3.5, -3.5 |

Piezoelectric constants are Bernardini et al. 1997; spontaneous polarization is Ambacher et al. 2002;
deformation potentials are Vurgaftman & Meyer 2003.

## 2.7 Dopant ionization energies

**Silicon donor**, depth below $E_c$, piecewise linear in Al fraction:

| $x$ | 0 | 0.1 | 0.4 | 0.6 | 1.0 |
|---|---|---|---|---|---|
| $E_D$ (meV) | 18 | 18 | 50 | 85 | 85 |

Indium content does not change $E_D$.

**Magnesium acceptor**, height above the top valence band:

$$
E_A(x, y) = \left(0.17 + 0.34\,x\right) \frac{E_g(x, y)}{E_g(x, 0)} \quad [\text{eV}],
$$

170 meV in GaN rising to 510 meV in AlN (Nam et al. 2003). The gap-ratio factor for In-containing
alloys is a heuristic; it gives 107 meV at In$_{0.35}$Ga$_{0.65}$N (measured 43) and 35 meV in InN (measured 60).

## 2.8 Mobility and recombination (biased solves only)

Low-field mobility is linear in Al fraction between GaN (electrons 300, holes 10 cm²/Vs) and AlN
(25 and 2 cm²/Vs). Recombination coefficients are fixed: Shockley-Read-Hall lifetimes
$\tau_n = \tau_p = 1$ ns, radiative coefficient $B = 10^{-11}$ cm³/s, Auger coefficient
$C = 10^{-30}$ cm⁶/s. These set the magnitude of the current, not the band diagram, and are generic
values rather than fits to any device.

## 2.9 Metal work functions

Used only for the Schottky-Mott barrier when the user gives no barrier height:
Ni 5.10, Au 5.10, Pt 5.65, Pd 5.12, Ti 4.33, Al 4.28, Ag 4.26, Cr 4.50, W 4.55, Mo 4.60 eV.
