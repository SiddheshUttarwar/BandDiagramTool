# 9. Solution algorithms

Chapters 5-7 define three equations. This chapter describes how they are combined in each solve
mode. It helps to keep the functional dependencies in view:

```math
n = n(z;\ \phi,\ E_{Fn}), \qquad p = p(z;\ \phi,\ E_{Fp}), \qquad \rho = \rho(z;\ \phi,\ E_{Fn},\ E_{Fp}).
```

Each algorithm below fixes some of the arguments and solves one equation for the others.

## 9.1 Overview

| Mode | Unknowns | Fixed | Algorithm |
|---|---|---|---|
| Poisson | $`\phi`$ | $`E_{Fn} = E_{Fp} = 0`$ | 9.2 |
| Schrödinger-Poisson | $`\phi`$, $`\{E_k,\psi_k\}`$ | $`E_{Fn} = E_{Fp} = 0`$ | 9.3 |
| Current-Poisson | $`\phi`$, $`E_{Fn}`$, $`E_{Fp}`$ | - | 9.4 |
| Schrödinger-current-Poisson | $`\phi`$, $`E_{Fn}`$, $`E_{Fp}`$, $`\{E_k,\psi_k\}`$ | - | 9.5 |
| Flat quasi-Fermi levels | $`\phi`$ (and states) | $`E_{Fn} = 0`$, $`E_{Fp} = -V`$ | 9.6 |

## 9.2 Poisson (equilibrium, classical)

At zero bias the Fermi level is constant and the densities are the classical (5.4)-(5.5), so the
only equation is the nonlinear Poisson equation (5.13),
$`\mathcal{R}(\boldsymbol\phi) \equiv A\boldsymbol\phi - \boldsymbol\rho(\boldsymbol\phi)/\varepsilon_0 = 0`$.

**Initial guess.** A linear solve of (5.13) with the fixed charges only (polarization and fully
ionized dopants, no mobile carriers).

**Iteration.** Newton's method (5.14), stabilised by pseudo-transient continuation: a term
$`1/\Delta t`$ is added to the diagonal,

```math
\left(A + D + \frac{1}{\Delta t}\,I\right)\boldsymbol\phi^{new} = D\,\boldsymbol\phi + \frac{\boldsymbol\phi}{\Delta t} + \frac{\boldsymbol\rho(\boldsymbol\phi)}{\varepsilon_0}. \tag{9.1}
```

Small $`\Delta t`$ gives a short, strongly damped step. $`\Delta t \to \infty`$ recovers the full
Newton step. The pseudo-time step is controlled by the true residual:

| After a trial step | Action |
|---|---|
| $`\lVert\mathcal{R}(\boldsymbol\phi^{new})\rVert \le \lVert\mathcal{R}(\boldsymbol\phi)\rVert`$ | accept; $`\Delta t \leftarrow \Delta t\cdot\min\left(\lVert\mathcal{R}\rVert/\lVert\mathcal{R}^{new}\rVert,\ 3\right)`$ |
| otherwise | reject; $`\Delta t \leftarrow 0.3\,\Delta t`$; retry from the same $`\boldsymbol\phi`$ |

**Convergence.**

```math
\max_i\left|\phi_i^{new} - \phi_i\right| < \texttt{tol} \qquad (\text{default } 10^{-6}\ \text{V}). \tag{9.2}
```

> [!NOTE]
> Damping the Jacobian is more robust than computing a full Newton step and mixing a fraction of
> it into the previous potential. Mixing keeps the direction of an oversized step. On
> short-period superlattices with large polarization steps at every interface it was observed to
> enter a limit cycle and never converge; (9.1) does not.

## 9.3 Schrödinger-Poisson (equilibrium, quantum)

The same loop as 9.2, with the Schrödinger equation solved at the start of every iteration:

1. Band edges from the current $`\phi`$, by (5.3).
2. Solve (6.1) in the quantum region for electrons and for the three hole bands, keeping states up to the cutoff (6.9).
3. Densities: quantum (6.7)-(6.8) inside the region, classical (5.4)-(5.5) outside.
4. One step of (9.1), with step control as in 9.2.
5. Test (9.2); otherwise return to 1.

The matrix $`D`$ in (9.1) is still built from the classical derivative $`\partial\rho/\partial\phi`$.
It is used only as an approximate Jacobian: a small change of potential changes the quantum charge
by nearly the same amount as it would change a classical charge of the same density, so the
iteration converges to the solution of the coupled Schrödinger and Poisson equations even though
the linearisation is classical. At convergence $`n`$, $`p`$ and $`\phi`$ satisfy (5.1) and (6.1)
simultaneously.

## 9.4 Current-Poisson (biased, classical)

Under bias all three unknowns are solved **simultaneously**. The discrete system is (5.12) and
(7.10) at every node:

```math
\mathbf{F}(\mathbf{u}) = 0, \qquad \mathbf{u} = \left(\phi_0, E_{Fn,0}, E_{Fp,0},\ \dots,\ \phi_{N-1}, E_{Fn,N-1}, E_{Fp,N-1}\right). \tag{9.3}
```

**Newton step.**

```math
\mathbf{J}(\mathbf{u})\,\delta\mathbf{u} = -\mathbf{F}(\mathbf{u}), \qquad \mathbf{u} \leftarrow \mathbf{u} + \operatorname{clip}(\delta\mathbf{u}). \tag{9.4}
```

Each equation at node $`i`$ involves nodes $`i-1`$, $`i`$, $`i+1`$ only, so $`\mathbf{J}`$ is block
tridiagonal with $`3\times3`$ blocks. It is assembled analytically, from the derivatives of
$`\mathcal{F}_{1/2}`$, of the Bernoulli function in (7.9), of the ionization fractions (5.8) and of
the recombination rates (7.5)-(7.7), and factorised by sparse direct LU.

| Element | Choice |
|---|---|
| step limit | each component of the update is clipped to $`\pm0.5`$ V per step |
| convergence | full Newton update $`\max\vert \delta\mathbf{u}\vert \lt 10^{-7}`$ V in every unknown |
| row scaling | applied for the linear solve only; it does not enter the convergence test |

Convergence is judged on the update, not on a scaled residual. A residual can be made small by
its normalisation without the current being conserved; the update cannot.

**Bias continuation.** The target voltage is not approached in one step.

1. Solve the equilibrium problem (9.2 or 9.3). This is the state at $`V = 0`$.
2. Increase the bias by an increment $`\Delta V`$, apply (5.18), and solve (9.4) starting from the previous converged state.
3. If Newton fails, halve $`\Delta V`$ and retry. If it succeeds, continue toward the target.
4. If the adaptive ramp stalls, restart from step 1 with fixed increments of 0.02, 0.05, 0.1, 0.2, 0.4 or 0.8 V in turn.

Step 4 handles current-voltage characteristics with a steep or folded section, which a ramp
cannot pass in small increments from one side but can step across from well behind it.

> [!NOTE]
> Many device simulators, nextnano++ among them, decouple this problem: Poisson and the densities
> are iterated to convergence at fixed quasi-Fermi levels, then the current equations update the
> quasi-Fermi levels, and the cycle repeats. The fully coupled Newton method (9.4) solves for all
> three unknowns at once. It needs the analytic Jacobian, but converges quadratically and
> conserves current to solver precision.

## 9.5 Schrödinger-current-Poisson (biased, quantum)

The current equations are written for classical-form densities. Quantum effects enter through the
correction factors $`\gamma_n`$, $`\gamma_p`$ of (6.10), held fixed during each current-Poisson solve:

1. Solve the classical current-Poisson problem (9.4) at the target bias. Set $`\gamma = 1`$.
2. With the resulting $`\phi`$, $`E_{Fn}`$, $`E_{Fp}`$: solve (6.1), evaluate (6.7)-(6.8), and form new factors $`\gamma^{new}`$ by (6.10).
3. Under-relax in the logarithm: $`\ln\gamma \leftarrow (1-\beta)\ln\gamma + \beta\ln\gamma^{new}`$, with $`\beta`$ starting at 0.5 and reduced (not below 0.02) if the mismatch grows.
4. Solve (9.4) again with densities (6.11).
5. Repeat from 2 until the potential satisfies (9.2) and

```math
\left|\ln\frac{\gamma^{new}}{\gamma}\right| < 10^{-2}\quad\text{wherever the density exceeds } 10^{10}\ \text{cm}^{-3}. \tag{9.5}
```

(9.5) states that the densities used in the current equation and the densities from the
Schrödinger equation agree to 1%.

## 9.6 Flat quasi-Fermi levels

With `flat_qfl=True` the current equations are not solved. The quasi-Fermi levels are prescribed,

```math
E_{Fn}(z) = 0, \qquad E_{Fp}(z) = -V \qquad\text{at every interior node,} \tag{9.6}
```

and only Poisson (and Schrödinger) is solved, by 9.2 or 9.3 with the potential boundary condition
(5.18). The result is a band diagram with $`E_{Fn} - E_{Fp} = V`$ throughout: appropriate for band
profiles, wavefunctions and transition energies under bias, not for currents. $`J = 0`$ is returned
and series resistance is ignored.

## 9.7 Voltage sweeps

`sweep_voltage(V_start, V_stop, n_steps)` solves each bias point by 9.4 or 9.5. Each point is
solved independently: no state is carried over from the previous voltage, so a poor solution at
one point cannot propagate to the next. Points that fail are returned with `converged=False` and
listed in `last_sweep_failed_voltages`.

## 9.8 When a solve fails

Every result carries a `converged` flag. A result with `converged=False` is the last iterate, not
a solution, and should not be used.

| Symptom | Known cause |
|---|---|
| bias ramp halts at a voltage | a blocking contact: a Schottky contact to p-type material, or an ohmic contact on undoped AlN, has no steady state at that bias |
| $`\Delta_J`$ of (7.11) is of order 1 at low bias | the current is below the numerical resolution of the quasi-Fermi gradients; the diagnostic is not meaningful there |
