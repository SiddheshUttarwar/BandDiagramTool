# 7. Drift-diffusion and the biased solve

With a voltage applied, the electron and hole populations are no longer described by one Fermi
level. The tool solves the steady-state drift-diffusion system for $\phi$, $E_{Fn}$ and $E_{Fp}$
together.

## 7.1 Equations

$$
\frac{d}{dz}\left(\varepsilon_0\varepsilon_r\frac{d\phi}{dz}\right) = -\rho,
\qquad
\frac{1}{q}\frac{dJ_n}{dz} = R,
\qquad
\frac{1}{q}\frac{dJ_p}{dz} = -R.
$$

Written with quasi-Fermi levels, the currents are simply

$$
J_n = \mu_n\,n\,\frac{dE_{Fn}}{dz}, \qquad J_p = \mu_p\,p\,\frac{dE_{Fp}}{dz},
$$

which contain drift and diffusion together and remain correct for Fermi-Dirac statistics and for
the quantum-corrected densities of [section 6.6](06_schrodinger_quantum.md#66-quantum-correction-under-bias).
A flat quasi-Fermi level means zero current, exactly.

Net recombination is the sum of three mechanisms:

$$
R = \underbrace{\frac{np - n_i^2}{\tau_p (n + n_i) + \tau_n (p + n_i)}}_{\text{Shockley-Read-Hall}}
  + \underbrace{B\,(np - n_i^2)}_{\text{radiative}}
  + \underbrace{C\,(n + p)(np - n_i^2)}_{\text{Auger}} .
$$

## 7.2 Flux discretization

A plain finite difference of $n\,dE_{Fn}/dz$ is unstable when the potential changes by many $k_BT$
between nodes, which is the normal situation at a nitride heterointerface. The tool uses a
Scharfetter-Gummel flux generalized to arbitrary statistics. For an edge from node $a$ to node $c$
of length $h$, define

$$
L = \ln\frac{n_c}{n_a}, \qquad b = \frac{E_{Fn,c} - E_{Fn,a}}{k_B T}, \qquad d = L - b,
$$

and the Bernoulli function $B(u) = u/(e^u - 1)$. The particle flux is

$$
G = \frac{D_n}{h}\,B(d)\,n_c\left(1 - e^{-b}\right) = \frac{D_n}{h}\,B(-d)\,n_a\left(e^{b} - 1\right),
\qquad J_n = q\,G,
$$

with $D_n = \mu_n k_BT/q$. Three properties make this the right choice:

1. **Boltzmann limit.** $d$ becomes the normalized potential step and $G$ is the standard
   Scharfetter-Gummel expression.
2. **Zero current at equilibrium.** $b = 0$ gives $G = 0$ for any density profile, degenerate or
   quantum-corrected. Formulations that are only approximately zero leave spurious currents that
   swamp the real current in wide-gap devices.
3. **Small steps.** $G \to \mu_n n\,\Delta E_{Fn}/h$.

Whichever of the two equivalent forms keeps its exponentials bounded is used on each edge.
Holes use the mirror-image expressions with $\ln p$ and $-E_{Fp}$.

Densities are carried as logarithms throughout. In a UV LED the minority density at a contact can
be $e^{-85}$ of the majority density; working with $\ln n$ keeps such rows well conditioned.

## 7.3 Newton solve

The three equations at the $N$ nodes form $3N$ nonlinear equations in $3N$ unknowns:

$$
\begin{aligned}
&\varepsilon_0 (A_{pos}\phi)_i - \rho_i = 0, \\
&\frac{G_{i+1/2} - G_{i-1/2}}{w_i} - R_i = 0, \\
&\frac{H_{i+1/2} - H_{i-1/2}}{w_i} + R_i = 0,
\end{aligned}
$$

where $H$ is the hole flux. Each equation at node $i$ involves only nodes $i-1$, $i$, $i+1$, so the
Jacobian is block-tridiagonal with 3×3 blocks. It is assembled analytically (derivatives of
$\mathcal{F}_{1/2}$, of the Bernoulli function, of the ionization fractions and of $R$) and solved
by sparse direct factorization.

- **Update limiting.** Each Newton update is capped per step so that one bad linearization cannot
  throw the iterate into a different solution branch.
- **Convergence test.** The full, undamped Newton update must be below a tolerance in volts in
  every unknown. Testing the update instead of a normalized residual avoids declaring convergence
  on a residual that is small only because of how it was scaled. Current conservation is reported
  separately as a diagnostic: $(\max J - \min J)/\max|J|$ over the device.

## 7.4 Bias continuation

The solver does not jump to the target voltage. It starts from the converged equilibrium
solution and ramps the bias in steps, using each converged state as the starting guess for the
next. A step that fails is retried at half the increment. If the adaptive ramp stalls, which
happens where the current-voltage curve is S-shaped, the solver retries from the start with
fixed coarser steps (0.02 to 0.8 V) that cross the difficult stretch in one stride.

Contacts under bias: $\phi_{top} = \phi_{top}^{eq} + V$; $E_{Fn} = E_{Fp} = 0$ at the bottom
contact and $E_{Fn} = E_{Fp} = -V$ at the top contact (both carriers in equilibrium with the metal
at each contact).

## 7.5 Series resistance

Each layer may carry a series resistance; they add to a total $R_s$ (Ω·cm²). The junction voltage
$V_{int}$ is found from

$$
V_{int} + |J(V_{int})|\,R_s = V_{applied},
$$

by fixed-point iteration, with bisection as a fallback. Each trial is an independent solve at
$V_{int}$ with $R_s = 0$. The result reports both $V_{applied}$ and $V_{int}$.

## 7.6 Flat quasi-Fermi-level mode

For band diagrams and wavefunctions under bias when the current itself is not needed, the tool
offers a mode with no transport equations: $E_{Fn} = 0$ and $E_{Fp} = -V$ at every interior node,
and only Poisson (and Schrödinger) is solved. This is the approximation several commercial
Schrödinger-Poisson tools use. It gives no current and ignores series resistance.

## 7.7 What the transport model leaves out

- **Tunnelling and thermionic emission** across barriers. Transport over a heterobarrier is
  drift-diffusion only, so current through thin AlN barriers and tunnel junctions is not modelled.
- **Hot carriers and velocity saturation.** Mobility is field-independent.
- **Device-specific recombination.** Lifetimes and coefficients are generic constants.
- **Self-heating.** Temperature is uniform and fixed.
- **Blocking contacts.** A Schottky contact to p-type material or an ohmic contact on undoped
  AlN can leave the problem without a physical steady-state solution; the solver reports
  failure in those cases.
