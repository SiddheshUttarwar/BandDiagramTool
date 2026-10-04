# 5. Electrostatics

## 5.1 Poisson equation

The electrostatic potential $`\phi(z)`$ and the total charge density $`\rho`$ are related by

```math
\frac{d}{dz}\left[\varepsilon_0\,\varepsilon_r(z)\,\frac{d\phi}{dz}\right] = -\rho(z, \phi), \tag{5.1}
```

with $`\varepsilon_r`$ the static dielectric constant along c. The charge density collects every
contribution:

```math
\rho(z,\phi) = q\left[\,p(z,\phi) - n(z,\phi) + N_D^+(z,\phi) - N_A^-(z,\phi)\,\right] + \rho_{pol}(z) + \rho_{it}(z,\phi) + \rho_{dip}(z). \tag{5.2}
```

| Term | Meaning | Defined in |
|---|---|---|
| $`n`$, $`p`$ | mobile electrons and holes, classical or quantum | 5.2, chapter 6 |
| $`N_D^+`$, $`N_A^-`$ | ionized donors and acceptors | 5.3 |
| $`\rho_{pol}`$ | polarization charge | (4.10) |
| $`\rho_{it}`$ | interface trap states | 5.4 |
| $`\rho_{dip}`$ | fixed interface dipoles | 4.6 |

The band edges follow the potential rigidly:

```math
E_c(z) = E_{c,0}(z) - \phi(z), \qquad E_v(z) = E_{v,0}(z) - \phi(z), \tag{5.3}
```

with $`E_{c,0}`$, $`E_{v,0}`$ the strained flat-band edges (3.11). Because the mobile and dopant
charges depend on $`E_c`$ and $`E_v`$, (5.1) is a nonlinear equation for $`\phi`$.

## 5.2 Classical charge densities

When the Schrödinger equation is not solved, or outside the quantum region, carriers are computed
in the Thomas-Fermi approximation from the local band edges and quasi-Fermi levels:

```math
n(z) = N_c(T)\ \mathcal{F}_{1/2}\!\left(\frac{E_{Fn}(z) - E_c(z)}{k_B T}\right), \tag{5.4}
```

```math
p(z) = N_v(T)\ \mathcal{F}_{1/2}\!\left(\frac{E_v(z) - E_{Fp}(z)}{k_B T}\right). \tag{5.5}
```

$`\mathcal{F}_{1/2}`$ is the Fermi-Dirac integral of order 1/2, normalised so that
$`\mathcal{F}_{1/2}(\eta) \to e^\eta`$ for $`\eta \ll 0`$:

```math
\mathcal{F}_{1/2}(\eta) = \frac{2}{\sqrt\pi}\int_0^\infty \frac{\sqrt t}{1 + e^{\,t-\eta}}\,dt .
```

The effective densities of states are

```math
N_c = 2\left(\frac{2\pi\,m_e\,k_B T}{h^2}\right)^{3/2}, \tag{5.6}
```

```math
N_v = 2\left(\frac{2\pi\,m_0\,k_B T}{h^2}\right)^{3/2} \sum_{b}\left(m_b^{dos}\right)^{3/2} \exp\!\left(-\frac{\delta_b}{k_B T}\right), \qquad b \in \{HH, LH, CH\}. \tag{5.7}
```

(5.7) refers all three valence bands to the top one: each contributes its own density of states,
weighted by its depth $`\delta_b`$ below the top (2.9). This is exact for non-degenerate holes and
follows the band reordering in Al-rich AlGaN automatically.

> [!NOTE]
> Fermi-Dirac statistics are used everywhere; the Boltzmann approximation is never made. A
> $`10^{13}`$ cm<sup>−2</sup> electron gas confined to 2 nm has a density of $`5\times10^{19}`$ cm<sup>−3</sup>,
> well into degeneracy.

$`\mathcal{F}_{1/2}`$ is tabulated once by quadrature and interpolated. Its inverse, used to
recover a quasi-Fermi level from a density, is the Joyce-Dixon form:
$`\eta \approx \ln r + r/\sqrt8`$ for small $`r = n/N_c`$, $`\eta \approx (3\sqrt\pi\,r/4)^{2/3}`$ for
large $`r`$.

## 5.3 Doping

Each layer carries one donor species (Si) and one acceptor species (Mg). Their ionized densities are

```math
N_D^+ = \frac{N_D}{1 + g_D\,\exp\!\left(\dfrac{E_{Fn} - E_D}{k_B T}\right)}, \qquad
N_A^- = \frac{N_A}{1 + g_A\,\exp\!\left(\dfrac{E_A - E_{Fp}}{k_B T}\right)}, \tag{5.8}
```

with degeneracy factors $`g_D = 2`$, $`g_A = 4`$. The impurity levels move with the band edges:

```math
E_D(z) = E_{c,0}(z) - \phi(z) - E_D^{ion}, \qquad E_A(z) = E_{v,0}(z) - \phi(z) + E_A^{ion}. \tag{5.9}
```

**Ionization energies.**

Si donor, piecewise linear in Al fraction, independent of In fraction:

| $`x`$ | 0 | 0.1 | 0.4 | 0.6 | 1.0 |
|---|---|---|---|---|---|
| $`E_D^{ion}`$ (meV) | 18 | 18 | 50 | 85 | 85 |

Mg acceptor:

```math
E_A^{ion}(x,y) = \left(0.17 + 0.34\,x\right)\,\frac{E_g(x,y)}{E_g(x,0)}\quad[\text{eV}]. \tag{5.10}
```

170 meV in GaN, 510 meV in AlN (Nam 2003). The gap ratio for In-containing alloys is a heuristic.

| Example at 300 K | Result |
|---|---|
| GaN, $`[\text{Mg}] = 2\times10^{19}`$ cm<sup>−3</sup> | $`p = 5.5\times10^{17}`$ cm<sup>−3</sup> (3% ionized) |

> [!IMPORTANT]
> Each dopant has a single level whose depth does not change with concentration. There is no
> impurity band, no compensation by native defects and no DX behaviour. The Mott density
> $`N_{Mott} = 1/(4.2\,a_B^3)`$, with $`a_B = a_H\,\varepsilon_r\,m_0/m^*`$, is drawn on the band
> diagram as a warning where doping exceeds it, but does not change the result.

## 5.4 Interface trap states

A `SurfaceCharge` places trap levels at one plane. Each level has an areal density $`N_{it}`$ and an
energy measured from the local band edge: depth below $`E_c`$ for a donor-like level, height above
$`E_v`$ for an acceptor-like level. They ionize exactly as bulk dopants do, (5.8)-(5.9):

| Type | Charge when ionized | Neutral when | Degeneracy |
|---|---|---|---|
| donor-like | $`+q`$ | occupied by an electron | 2 |
| acceptor-like | $`-q`$ | empty | 4 |

The areal density is assigned to the node at that plane as the volume density $`N_{it}/w_i`$, so

```math
\rho_{it,i} = \frac{q}{w_i}\left(N_{it,D}^+ - N_{it,A}^-\right). \tag{5.11}
```

This is the mechanism for modelling surface donors as the origin of a 2DEG, instead of
prescribing the surface barrier.

## 5.5 Discrete Poisson equation

Applying (1.3) to (5.1):

```math
\frac{1}{w_i}\left[\varepsilon_{i+1/2}\,\frac{\phi_{i+1} - \phi_i}{h_{i+1/2}} - \varepsilon_{i-1/2}\,\frac{\phi_i - \phi_{i-1}}{h_{i-1/2}}\right] = -\frac{\rho_i(\phi_i)}{\varepsilon_0}. \tag{5.12}
```

In matrix form, with $`A`$ the symmetric positive-definite tridiagonal matrix of the left side
(sign reversed):

```math
A\,\boldsymbol\phi = \frac{\boldsymbol\rho(\boldsymbol\phi)}{\varepsilon_0}. \tag{5.13}
```

The normal displacement $`\varepsilon_0\varepsilon_r\,d\phi/dz`$ is continuous across material
interfaces by construction. A sheet charge appears as a jump in it.

**Linearisation.** Raising $`\phi`$ at a node lowers the bands there, adding electrons and removing
holes. The derivative of the charge is therefore negative, and Newton's method for (5.13) reads

```math
\left(A + D\right)\boldsymbol\phi^{new} = D\,\boldsymbol\phi + \frac{\boldsymbol\rho(\boldsymbol\phi)}{\varepsilon_0},
\qquad
D_{ii} = -\frac{1}{\varepsilon_0}\frac{\partial\rho_i}{\partial\phi_i} \ \ge 0 . \tag{5.14}
```

For non-degenerate carriers and fully ionized dopants $`D_{ii} = q\,(n_i + p_i)/(\varepsilon_0 k_B T)`$,
which is $`1/L_D^2`$ with $`L_D`$ the local Debye length. The general expression uses
$`\mathcal{F}'_{1/2}`$ and the derivatives of (5.8). Since $`D \ge 0`$, the matrix $`A + D`$ is positive
definite for any carrier density.

How (5.14) is iterated to convergence is described in section 9.2.

## 5.6 Boundary conditions

Dirichlet conditions are imposed on $`\phi`$ at both ends. The bottom contact is the reference,
$`\phi(0) = 0`$.

**Ohmic contact.** The semiconductor at the contact is charge-neutral and in equilibrium with the
metal. The band position there is the solution $`E_c^{*}`$ of

```math
N_D^+ - N_A^- + p - n = 0 \quad\text{at } E_{Fn} = E_{Fp} = E_F, \tag{5.15}
```

evaluated with (5.4)-(5.5) and (5.8). At the bottom this fixes the constant in (3.11); at the top
it fixes $`\phi_{top}^{eq} = E_{c,0}(z_{top}) - E_c^{*}`$.

**Schottky contact.** The conduction band edge is held a fixed energy above the metal Fermi level:

```math
E_c - E_F = \Phi_B \quad\text{at the contact.} \tag{5.16}
```

$`\Phi_B`$ is the `barrier_eV` given by the user. If none is given, the ideal (Schottky-Mott) value
is used:

```math
\Phi_B = W_m - \chi . \tag{5.17}
```

> [!NOTE]
> (5.17) ignores Fermi-level pinning. It gives 1.0 eV for Ni on GaN, close to measurement, but
> 1.55 eV for Pt against a measured 1.1 eV. For a free surface, or whenever a measured barrier is
> available, enter `barrier_eV` explicitly.

**Applied bias.** With a voltage $`V`$ on the top contact:

```math
\phi(z_{top}) = \phi_{top}^{eq} + V, \qquad
E_{Fn} = E_{Fp} = 0 \ \text{(bottom)}, \qquad E_{Fn} = E_{Fp} = -V \ \text{(top)}. \tag{5.18}
```

Both carrier types are in equilibrium with the metal at each contact.

## 5.7 Output quantities

| Quantity | Definition |
|---|---|
| electric field | $`F = -d\phi/dz`$, by (1.4) |
| quasi-field of a graded layer | $`F_{quasi} = d\chi/dz`$ |
| sheet density in a window $`[z_1, z_2]`$ | $`n_s = \int_{z_1}^{z_2} n\,dz`$ |
| interface polarization charges | (4.8) at every composition step |

**Check by hand.** For an undoped barrier of thickness $`d`$ and permittivity $`\varepsilon`$ on GaN
with surface barrier $`\Phi_B`$, a constant field in the barrier and charge neutrality give

```math
q\,n_s \approx \sigma_{pol} - \frac{\varepsilon_0\varepsilon}{q\,d}\left(\Phi_B + E_F - \Delta E_c\right), \tag{5.19}
```

with $`E_F`$ the Fermi level above the GaN band edge at the interface. Setting $`n_s = 0`$ gives the
critical barrier thickness. The full solution reproduces the measured 3.5 nm for
Al<sub>0.34</sub>Ga<sub>0.66</sub>N with a 1.65 eV surface level (Ibbetson 2000).
