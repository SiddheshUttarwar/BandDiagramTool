# 2. Materials

One material model covers the whole wurtzite Al$_x$In$_y$Ga$_{1-x-y}$N system. The classes
`AlGaN`, `InGaN` and `InAlGaN` are views of the same function, `get_nitride_params(x_Al, x_In)`,
so any two layers are always described by mutually consistent parameters.

Component materials are the three binaries GaN, AlN and InN. Write $z_{Ga} = 1 - x - y$.

## 2.1 Interpolation schemes

**Linear.** Used for every parameter unless stated otherwise:

$$
Q(x, y) = x\,Q_{AlN} + y\,Q_{InN} + z_{Ga}\,Q_{GaN}. \tag{2.1}
$$

**Quadratic, pairwise bowing.** Used for the band gap and the spontaneous polarization. Each pair
of components contributes its own bowing term:

$$
Q(x, y) = Q_{lin}(x,y) - b_{AlGa}\,x\,z_{Ga} - b_{InGa}\,y\,z_{Ga} - b_{AlIn}\,x\,y . \tag{2.2}
$$

For a ternary this reduces to the familiar $Q = x Q_A + (1-x) Q_B - b\,x(1-x)$.

**Composition-dependent bowing.** For the Al-In pair of the band gap, $b_{AlIn}$ depends on the
In share of that pair, $u = y/(x+y)$:

$$
b_{AlIn}(u) = \frac{6.43}{1 + 1.21\,u^2}\ \text{eV}. \tag{2.3}
$$

**Tabulated.** The Si donor energy is piecewise linear in $x$ (section 5.3).

## 2.2 Band gap

The gap is the energy between the conduction band minimum at $\Gamma$ and the **topmost** valence
band, for the unstrained crystal at 300 K, from (2.2):

| Parameter | Value | Source |
|---|---|---|
| $E_g$ GaN | 3.39 eV | NSM / Ioffe |
| $E_g$ AlN | 6.026 eV | NSM / Ioffe |
| $E_g$ InN | 0.69 eV | Wu 2002, Davydov 2002 |
| $b_{AlGa}$ | 0.7 eV | Vurgaftman & Meyer 2003 |
| $b_{InGa}$ | 1.4 eV | Vurgaftman & Meyer 2003 |
| $b_{AlIn}$ | (2.3) | Sakalauskas 2010 |

> [!IMPORTANT]
> The gap has no temperature dependence. Running at 77 K changes the thermal energy and the
> spontaneous polarization, but $E_g$ keeps its 300 K value. Strain shifts are added separately
> (section 3.4), so in a strained layer $E_c - E_v \neq E_g$.

## 2.3 Band offsets

Band alignment is defined through the valence band. Each material is assigned the energy of its
top valence band relative to GaN:

$$
\Delta E_v(x,y) = x\,\Delta E_v^{AlN} + y\,\Delta E_v^{InN} + b_v\,x\,y , \tag{2.4}
$$

| Parameter | Value | Source |
|---|---|---|
| $\Delta E_v^{AlN}$ | $-0.80$ eV | Vurgaftman & Meyer 2003 |
| $\Delta E_v^{InN}$ | $+0.58$ eV | King 2008 (XPS) |
| $b_v$ (Al-In pair) | 2.6 eV | fitted to $\Delta E_v = 0.2$ eV at Al$_{0.83}$In$_{0.17}$N/GaN (Akazawa 2010) |

Absolute band edges follow by anchoring GaN at an electron affinity $\chi_{GaN} = 4.1$ eV below
the vacuum level:

$$
E_v^{abs}(x,y) = -\left(\chi_{GaN} + E_g^{GaN}\right) + \Delta E_v(x,y), \tag{2.5}
$$

$$
E_c^{abs}(x,y) = E_v^{abs}(x,y) + E_g(x,y), \qquad \chi(x,y) = -E_c^{abs}(x,y). \tag{2.6}
$$

The conduction-band offset between two materials is the difference of their $\chi$; the
valence-band offset is the difference of their $\Delta E_v$. Their sum is the gap difference by
construction.

| Heterojunction | $\Delta E_c$ | $\Delta E_v$ | Type |
|---|---|---|---|
| GaN / AlN | 1.84 eV | 0.80 eV | I |
| InN / GaN | 2.12 eV | 0.58 eV | I |
| Al$_{0.83}$In$_{0.17}$N / GaN | 0.65 eV | 0.20 eV | I |

> [!NOTE]
> Offsets are among the least certain nitride parameters. Published InN/GaN valence offsets range
> from 0.5 to 1.05 eV. If a device depends on an offset, check the value against the literature
> for that material pair.

## 2.4 Valence bands at the zone centre

Wurtzite has three valence bands at $\Gamma$. With $\Delta_1 = \Delta_{cr}$ (crystal field) and
$\Delta_2 = \Delta_3 = \Delta_{so}/3$ (spin-orbit), their energies are

$$
E_{HH} = \Delta_1 + \Delta_2 + \lambda_\varepsilon + \theta_\varepsilon , \tag{2.7}
$$

$$
E_{\pm} = \operatorname{eig}
\begin{pmatrix}
\Delta_1 - \Delta_2 + \lambda_\varepsilon + \theta_\varepsilon & \sqrt2\,\Delta_3 \\
\sqrt2\,\Delta_3 & \lambda_\varepsilon
\end{pmatrix}, \tag{2.8}
$$

where $\lambda_\varepsilon$ and $\theta_\varepsilon$ are the strain terms of (3.9), zero for an
unstrained crystal. Of the two eigenstates of (2.8), the one of mainly in-plane ($X \pm iY$)
character is labelled LH and the one of mainly $Z$ character is labelled CH.

| | GaN | AlN | InN |
|---|---|---|---|
| $\Delta_{cr}$ (eV) | 0.040 | $-0.169$ | 0.040 |
| $\Delta_{so}$ (eV) | 0.008 | 0.019 | 0.005 |

Because $\Delta_{cr} < 0$ in AlN, the band order inverts with composition: HH is on top in GaN and
CH is on top in Al-rich AlGaN. The tool always takes $E_v$ as the highest of the three and stores
the depth of each band below it,

$$
\delta_b = \max(E_{HH}, E_{LH}, E_{CH}) - E_b \ \ge 0, \qquad b \in \{HH, LH, CH\}. \tag{2.9}
$$

The three bands are then treated as independent parabolic bands with edges $E_v - \delta_b$.

## 2.5 Effective masses

| | GaN | AlN | InN |
|---|---|---|---|
| $m_e$ (isotropic) | 0.20 | 0.40 | 0.07 |
| HH: $m_\parallel$, $m_\perp$ | 1.1, 1.6 | 3.53, 10.42 | 1.63, 1.63 |
| LH: $m_\parallel$, $m_\perp$ | 1.1, 0.15 | 3.53, 0.24 | 0.27, 0.27 |
| CH: $m_\parallel$, $m_\perp$ | 0.15, 1.1 | 0.25, 3.81 | 0.65, 0.65 |
| quantisation mass $m_{hh}$, $m_{lh}$, $m_{ch}$ | 1.4, 0.3, 0.6 | 3.53, 3.53, 0.25 | 1.63, 0.27, 0.65 |

$\parallel$ is along c, $\perp$ is in the plane. Two masses are derived from these:

- the **density-of-states mass** of a band, $m^{dos} = (m_\parallel\,m_\perp^2)^{1/3}$, used for bulk carrier statistics (5.6)-(5.7);
- the **quantisation mass** along c, used in the Schrödinger equation (6.1).

## 2.6 Dielectric, lattice, elastic and piezoelectric parameters

| | GaN | AlN | InN | Source |
|---|---|---|---|---|
| $\varepsilon_r$ along c | 10.1 | 8.57 | 14.4 | Tsai 1999, Fonoberov 2003, NSM |
| $a_0$ (Å) | 3.189 | 3.112 | 3.545 | NSM |
| $c_0$ (Å) | 5.186 | 4.982 | 5.703 | NSM |
| $C_{13}$ (GPa) | 106 | 108 | 92 | NSM, V&M 2003 |
| $C_{33}$ (GPa) | 398 | 373 | 224 | NSM, V&M 2003 |
| $e_{31}$ (C/m²) | $-0.49$ | $-0.60$ | $-0.57$ | Bernardini 1997 |
| $e_{33}$ (C/m²) | 0.73 | 1.46 | 0.97 | Bernardini 1997 |
| $P_{sp}$ (C/m²) | $-0.034$ | $-0.090$ | $-0.042$ | Ambacher 2002 |
| pyroelectric coefficient (C/m²K) | $6.0\times10^{-5}$ | $4.5\times10^{-5}$ | $6.0\times10^{-5}$ | PRB 93, 081205 (2016) |

## 2.7 Deformation potentials

| (eV) | GaN | AlN | InN |
|---|---|---|---|
| $a_1$, $a_2$ (conduction band) | $-4.9$, $-11.3$ | $-3.4$, $-11.8$ | $-3.5$, $-3.5$ |
| $D_1$ | $-3.7$ | $-17.1$ | $-3.7$ |
| $D_2$ | 4.5 | 7.9 | 4.5 |
| $D_3$ | 8.2 | 8.8 | 8.2 |
| $D_4$ | $-4.1$ | $-3.9$ | $-4.1$ |

All from Vurgaftman & Meyer 2003. Their use is in section 3.4.

## 2.8 Transport and recombination parameters

| Parameter | Value |
|---|---|
| electron mobility | 300 cm²/Vs (GaN) to 25 cm²/Vs (AlN), linear in $x$ |
| hole mobility | 10 cm²/Vs (GaN) to 2 cm²/Vs (AlN), linear in $x$ |
| SRH lifetimes $\tau_n$, $\tau_p$ | 1 ns |
| radiative coefficient $B$ | $10^{-11}$ cm³/s |
| Auger coefficient $C$ | $10^{-30}$ cm⁶/s |

These are generic values, not fitted to any device. They are the same for all In fractions.

## 2.9 Metals

Work functions used for the ideal Schottky barrier (5.17) when no barrier is given:

| Ni | Au | Pt | Pd | Ti | Al | Ag | Cr | W | Mo |
|---|---|---|---|---|---|---|---|---|---|
| 5.10 | 5.10 | 5.65 | 5.12 | 4.33 | 4.28 | 4.26 | 4.50 | 4.55 | 4.60 |
