# 5. Poisson equation and the equilibrium solve

## 5.1 The equation

$$
\frac{d}{dz}\left(\varepsilon_0\,\varepsilon_r(z)\,\frac{d\phi}{dz}\right) = -\rho(z),
$$

with $\rho$ the total charge density of [section 4.5](04_carrier_statistics.md#45-total-charge-density).
Because $n$, $p$, $N_D^+$ and $N_A^-$ all depend on $\phi$ through $E_c = E_{c0} - \phi$, this is a
nonlinear equation for $\phi$.

At zero bias the Fermi level is flat, $E_{Fn} = E_{Fp} = 0$, and Poisson (with Schrödinger, if
enabled) is the whole problem.

## 5.2 Discretization

Integrating over the control volume of node $i$ gives the flux balance

$$
\frac{1}{w_i}\left[
\varepsilon_{i+1/2}\,\frac{\phi_{i+1} - \phi_i}{h_{i+1/2}}
- \varepsilon_{i-1/2}\,\frac{\phi_i - \phi_{i-1}}{h_{i-1/2}}
\right] = -\frac{\rho_i}{\varepsilon_0},
$$

with $\varepsilon_{i\pm1/2}$ the permittivity on the edge. This is a symmetric tridiagonal system.
The first and last rows are replaced by the Dirichlet conditions of
[section 1.4](01_device_model_and_grid.md#14-boundary-conditions).
The displacement field $\varepsilon_r\,d\phi/dz$ is continuous across heterointerfaces by
construction; the sheet charges of chapter 3 appear as jumps in it.

## 5.3 Newton linearization

Write the discrete equation as $A\phi = -\rho(\phi)/\varepsilon_0$, with $A$ the (negative
semi-definite) operator on the left of 5.2, or equivalently $A_{pos}\,\phi = \rho(\phi)/\varepsilon_0$
with $A_{pos} = -A$. Raising $\phi$ at a node lowers the bands there, which adds electrons and
removes holes:

$$
\frac{\partial \rho}{\partial \phi} = -\frac{q}{k_B T}\left(n' + p'\right) - q\left(\frac{\partial N_A^-}{\partial\phi} - \frac{\partial N_D^+}{\partial\phi}\right) \le 0,
$$

where $n' = N_c\,\mathcal{F}'_{1/2}$ reduces to $n$ in the non-degenerate limit. One Newton step
solves

$$
\left(A_{pos} + D\right)\phi^{new} = D\,\phi + \frac{\rho(\phi)}{\varepsilon_0},
\qquad D_{ii} = \frac{q\,(n_i + p_i)}{\varepsilon_0\,k_B T} + (\text{dopant terms}).
$$

$D$ is non-negative, so the matrix is positive definite for any carrier densities, including the
$10^{20}$ cm$^{-3}$ of a polarization-induced gas. Physically $D$ is the local screening: the
update at a node is limited to about a Debye length's worth of response.

## 5.4 Starting guess

The potential is initialized from a linear Poisson solve with fixed charges only (ionized dopants
and polarization, no free carriers). This already has the right qualitative band bending and
saves most of the Newton iterations a flat start would need.

## 5.5 Iteration and convergence

Each outer iteration:

1. Band edges from the current $\phi$.
2. If quantum mode is on, solve Schrödinger in the quantum region ([chapter 6](06_schrodinger_quantum.md)).
3. Carrier densities: subband densities inside the quantum region, Fermi-Dirac outside.
4. One Newton-Poisson step, giving $\phi^{new}$.
5. Accept or reject the step (below).
6. Stop when $\max_i |\phi^{new}_i - \phi_i| <$ tolerance (default $10^{-6}$ V).

The Newton step of 5.3 is not applied at full length. Step 4 is taken with **pseudo-transient
continuation**: a pseudo-time term is added to the Jacobian,

$$
\left(A_{pos} + D + rac{1}{\Delta t}\,Iight)\phi^{new} = D\,\phi + rac{\phi}{\Delta t} + rac{ho(\phi)}{arepsilon_0}.
$$

Small $\Delta t$ gives a short, heavily damped step; $\Delta t 	o \infty$ recovers the plain Newton
step. After each trial the true nonlinear residual $\lVert A_{pos}\phi - ho(\phi)/arepsilon_0 Vert$
is evaluated:

- if it did not increase, the step is accepted and $\Delta t$ grows in proportion to the improvement (at most 3x per step);
- if it increased, $\Delta t$ is multiplied by 0.3 and the step is retried from the same $\phi$.

Damping the Jacobian this way is more robust than computing a full Newton step and then mixing a
fraction of it into the old potential. Mixing still trusts the direction of an oversized step, and
on stacks with many closely spaced high-polarization interfaces (short-period superlattices) it
was found to settle into a limit cycle that never converges.

## 5.6 What the equilibrium solution gives

- Band edges $E_c(z)$, $E_v(z)$ and the three valence-band edges.
- Electric field $F = -d\phi/dz$, and the quasi-field in graded layers.
- $n(z)$, $p(z)$, and sheet densities by integration over any window:
  $n_s = \int n\,dz$.
- Polarization, strain and interface sheet charges.
- In quantum mode, subband energies and wavefunctions.

## 5.7 A check that can be done by hand

For an undoped AlGaN barrier of thickness $d$ on GaN with surface barrier $\Phi_B$, charge
neutrality and a constant field in the barrier give the familiar result

$$
n_s \approx \frac{\sigma}{q} - \frac{\varepsilon_0\varepsilon_r}{q^2 d}\left(\Phi_B + E_F - \Delta E_c\right),
$$

where $E_F$ is the Fermi level above the GaN conduction band edge at the interface. The full
solution adds the dependence of $E_F$ on $n_s$ and the field in the buffer. The tool reproduces the
critical thickness this formula implies (the $d$ at which $n_s \to 0$): 3.5 nm for
Al$_{0.34}$Ga$_{0.66}$N with a 1.65 eV surface donor level, as measured by Ibbetson et al.
