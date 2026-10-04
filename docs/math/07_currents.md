# 7. Currents

Under applied bias the electron and hole populations each have their own quasi-Fermi level. The
tool determines them from the drift-diffusion model.

## 7.1 Continuity equations

With generation rate $G$ and recombination rate $R$ of electron-hole pairs, and particle flux
densities $j_n$, $j_p$ (units of area$^{-1}$ time$^{-1}$, positive along $+z$):

$$
\frac{\partial n}{\partial t} + \frac{\partial j_n}{\partial z} = G - R, \qquad
\frac{\partial p}{\partial t} + \frac{\partial j_p}{\partial z} = G - R . \tag{7.1}
$$

## 7.2 Drift-diffusion model

The flux of each carrier is driven by the gradient of its quasi-Fermi level:

$$
j_n = -\frac{\mu_n}{q}\,n\,\frac{dE_{Fn}}{dz}, \qquad j_p = +\frac{\mu_p}{q}\,p\,\frac{dE_{Fp}}{dz} . \tag{7.2}
$$

This single term contains both drift and diffusion and holds for Fermi-Dirac statistics and for
the quantum-corrected densities (6.11). The charge current densities are $J_n = -q\,j_n$,
$J_p = +q\,j_p$, and the total current $J = J_n + J_p$ is independent of $z$ in steady state.

The tool solves the stationary problem, $\partial_t n = \partial_t p = 0$, with $G = 0$:

$$
\frac{d}{dz}\left[\mu_n\,n\,\frac{dE_{Fn}}{dz}\right] = q\,R, \qquad
\frac{d}{dz}\left[\mu_p\,p\,\frac{dE_{Fp}}{dz}\right] = -q\,R . \tag{7.3}
$$

These are the **current equations**. Given $\phi$, they relate the quasi-Fermi levels to the
carrier densities $n(z;\phi,E_{Fn})$ and $p(z;\phi,E_{Fp})$. Boundary values are (5.18).

## 7.3 Mobility

A constant low-field mobility is used, interpolated linearly in Al fraction (section 2.8):

$$
\mu(x) = (1 - x)\,\mu^{GaN} + x\,\mu^{AlN} . \tag{7.4}
$$

There is no dependence on doping, temperature, field or In fraction. The diffusion coefficient
follows from the Einstein relation, $D = \mu\,k_B T/q$.

## 7.4 Recombination

Three mechanisms are summed, $R = R_{SRH} + R_{rad} + R_{Auger}$:

$$
R_{SRH} = \frac{n\,p - n_i^2}{\tau_p\,(n + n_i) + \tau_n\,(p + n_i)}, \tag{7.5}
$$

$$
R_{rad} = B\left(n\,p - n_i^2\right), \tag{7.6}
$$

$$
R_{Auger} = C\,(n + p)\left(n\,p - n_i^2\right). \tag{7.7}
$$

$n_i = \sqrt{N_c N_v}\,\exp(-E_g/2k_BT)$ is the local intrinsic density. The SRH form assumes a
mid-gap trap; the Auger form uses one coefficient for both carriers. Parameter values are in
section 2.8.

## 7.5 Discretised flux

A direct difference of (7.2) is unstable when the potential changes by many $k_BT$ between
neighbouring nodes, which is the rule at nitride heterointerfaces. The tool uses the
Scharfetter-Gummel scheme in a form valid for arbitrary statistics. Consider the edge from node
$a$ to node $c$ of length $h$, and define

$$
L = \ln\frac{n_c}{n_a}, \qquad b = \frac{E_{Fn,c} - E_{Fn,a}}{k_B T}, \qquad d = L - b . \tag{7.8}
$$

With the Bernoulli function $\mathcal{B}(u) = u/(e^u - 1)$, the flux on the edge is

$$
G_{ac} = \frac{D_n}{h}\,\mathcal{B}(d)\ n_c\left(1 - e^{-b}\right)
       = \frac{D_n}{h}\,\mathcal{B}(-d)\ n_a\left(e^{b} - 1\right). \tag{7.9}
$$

The two expressions are identical; on each edge the one whose exponentials stay bounded is
evaluated. $G_{ac}$ is the discrete form of $\mu_n n\,(dE_{Fn}/dz)/q$, so $J_n = q\,G$.

Properties of (7.9):

| Limit | Result |
|---|---|
| Boltzmann statistics | $d$ equals the normalised potential step, and (7.9) is the classical Scharfetter-Gummel flux |
| flat quasi-Fermi level, $b = 0$ | $G = 0$ exactly, for any density profile |
| small steps | $G \to \mu_n\,n\,\Delta E_{Fn}/(q\,h)$ |

Holes use the same expressions with $\ln p$ in place of $\ln n$ and $-E_{Fp}$ in place of $E_{Fn}$;
call the hole flux $H$.

> [!NOTE]
> The second property matters in wide-gap devices. A scheme that is only approximately zero at
> equilibrium leaves a spurious current that can exceed the real one by many orders of magnitude
> when minority densities are as low as $10^{-30}$ cm$^{-3}$.

## 7.6 Discrete current equations

With (1.3):

$$
\frac{G_{i+1/2} - G_{i-1/2}}{w_i} - R_i = 0, \qquad
\frac{H_{i+1/2} - H_{i-1/2}}{w_i} + R_i = 0 . \tag{7.10}
$$

Together with the discrete Poisson equation (5.12) these are $3N$ equations for the $3N$ unknowns
$\{\phi_i, E_{Fn,i}, E_{Fp,i}\}$. Densities are carried as logarithms throughout, so that rows
involving densities of $e^{-85}$ relative to the majority carrier remain well conditioned.

The solution method is described in section 9.4.

## 7.7 Current output and conservation check

Edge currents are evaluated from (7.9) after convergence. The tool reports

$$
\Delta_J = \frac{\max_z J - \min_z J}{\max_z |J|}, \tag{7.11}
$$

which is zero for an exact steady state. It is meaningful only when $|J|$ is above the
double-precision resolution of the quasi-Fermi gradients, about $10^{-6}$ A/cm².

## 7.8 Series resistance

A lumped series resistance $R_s$ (Ω·cm², the sum of the layers' `series_resistance`) reduces the
junction voltage below the applied one:

$$
V_{int} + \left|J(V_{int})\right| R_s = V . \tag{7.12}
$$

(7.12) is solved for $V_{int}$ by damped fixed-point iteration, with bisection as fallback. Each
trial is a complete solve at $V_{int}$ with $R_s = 0$. Both $V$ and $V_{int}$ are reported.

## 7.9 Scope

| Not included | Consequence |
|---|---|
| thermionic emission and tunnelling at heterobarriers | current through thin barriers and tunnel junctions is not modelled |
| high-field mobility, velocity saturation | no hot-carrier effects |
| doping- and temperature-dependent mobility and lifetimes | current magnitude is indicative only |
| generation (optical, impact ionization) | no photocurrent, no breakdown |
| heat flow | isothermal |
| time dependence | no transients, no small-signal analysis |
