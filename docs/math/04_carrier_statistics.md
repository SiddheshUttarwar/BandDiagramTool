# 4. Carrier statistics and dopant ionization

## 4.1 Free carriers

Nitride devices routinely reach degenerate densities (a $10^{13}$ cm$^{-2}$ electron gas confined
to 2 nm is $5\times10^{19}$ cm$^{-3}$), so Boltzmann statistics are not used anywhere. Carrier
densities are

$$
n = N_c\,\mathcal{F}_{1/2}\!\left(\frac{E_{Fn} - E_c}{k_B T}\right), \qquad
p = N_v\,\mathcal{F}_{1/2}\!\left(\frac{E_v - E_{Fp}}{k_B T}\right),
$$

with the normalized Fermi-Dirac integral

$$
\mathcal{F}_{1/2}(\eta) = \frac{2}{\sqrt\pi} \int_0^\infty \frac{\sqrt{t}}{1 + e^{\,t - \eta}}\,dt,
\qquad \mathcal{F}_{1/2}(\eta) \to e^{\eta} \text{ for } \eta \ll 0.
$$

$\mathcal{F}_{1/2}$ is evaluated from a table computed once by numerical quadrature. The biased
solver uses a smooth cubic spline of $\ln \mathcal{F}_{1/2}$ so that its derivative, needed in the
Jacobian, is exactly consistent with the function.

Inverting for the Fermi level from a density uses the Joyce-Dixon form
$\eta \approx \ln r + r/\sqrt8$ for small $r = n/N_c$ and
$\eta \approx (3\sqrt\pi\, r/4)^{2/3}$ for large $r$, blended in between.

## 4.2 Incomplete dopant ionization

Mg in GaN is 170 meV deep and Si in AlN is 85 meV or more, so only a fraction of the dopants are
ionized at room temperature. The ionized densities are

$$
N_D^+ = \frac{N_D}{1 + 2\exp\!\left(\dfrac{E_{Fn} - (E_c - E_D)}{k_B T}\right)}, \qquad
N_A^- = \frac{N_A}{1 + 4\exp\!\left(\dfrac{(E_v + E_A) - E_{Fp}}{k_B T}\right)},
$$

with degeneracy factors 2 (donor) and 4 (acceptor) and the composition-dependent $E_D$, $E_A$ of
[section 2.7](02_material_model.md#27-dopant-ionization-energies).

Worked example: GaN with $[\text{Mg}] = 2\times10^{19}$ cm$^{-3}$ gives $p \approx 5.5\times10^{17}$
cm$^{-3}$ at 300 K, about 3% activation (Kaufmann et al. measured $6\times10^{17}$).

The dopant level is a single discrete level. Its depth does not shrink with doping density, and
there is no impurity band. The Mott density

$$
N_{Mott} = \frac{1}{4.2\,a_B^3}, \qquad a_B = a_H\,\frac{\varepsilon_r}{m^*/m_0},
$$

is computed and shown on the band diagram so the user can see where a layer's doping exceeds it,
but it does not feed back into the carrier density.

There is no compensation model: every dopant atom entered is electrically active, and no native
defects or DX behaviour are included in the electrostatics.

## 4.3 Surface and interface states

A surface-charge layer places one or more trap levels at a single node, each with an areal density
$N_s$ (cm$^{-2}$) and an energy relative to the local band edge. They ionize with the same
statistics as bulk dopants:

- donor-like: neutral when filled, $+q$ when empty, occupation factor with degeneracy 2;
- acceptor-like: neutral when empty, $-q$ when filled, degeneracy 4.

The areal density is converted to an equivalent volume density over that node's control volume,
so it enters Poisson's equation like any other charge and responds self-consistently to the
Fermi level. This is the mechanism for modelling surface donors as the source of a 2DEG instead of
fixing the surface barrier.

## 4.4 Fixed interface dipole

An interface-dipole layer adds two fixed sheets $\pm\sigma$ separated by $d$. It produces a
potential step $\Delta\phi = \sigma d / (\varepsilon_0 \varepsilon_r)$ and carries no net charge.
It is a structural charge and does not respond to the Fermi level.

## 4.5 Total charge density

The right-hand side of Poisson's equation is

$$
\rho = q\left(p - n + N_D^+ - N_A^-\right) + \rho_{pol} + \rho_{surface} + \rho_{dipole}.
$$

Inside a quantum region $n$ and $p$ are replaced by the subband densities of
[chapter 6](06_schrodinger_quantum.md).
