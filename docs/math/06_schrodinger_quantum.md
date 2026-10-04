# 6. Schrödinger equation and quantum charge

Classical statistics put the peak electron density exactly at the interface and ignore
quantization. In quantum mode the tool solves for the confined states and uses them for the
charge inside a designated quantum region.

## 6.1 The equation

Single-band effective-mass Schrödinger equation along z, in BenDaniel-Duke form so that the
probability current is continuous where the mass changes:

$$
-\frac{\hbar^2}{2}\,\frac{d}{dz}\left(\frac{1}{m^*(z)}\,\frac{d\psi}{dz}\right) + V(z)\,\psi = E\,\psi.
$$

- Electrons: $V = E_c(z)$, $m^* = m_e(z)$.
- Holes: three independent problems, one per valence band $b \in \{HH, LH, CH\}$, with
  $V = -E_{v,b}(z)$ and the c-axis mass $m_b(z)$. The sign flip turns the top of the valence band
  into the bottom of a well; the physical subband energy is $E_{v,k} = -E_k$.

Boundary conditions are hard walls at the ends of the quantum region.

## 6.2 Discretization

In finite-volume form, with edge masses $m_{i+1/2} = \tfrac12(m_i + m_{i+1})$:

$$
-\frac{\hbar^2}{2 w_i}\left[\frac{\psi_{i+1} - \psi_i}{m_{i+1/2}\,h_{i+1/2}} - \frac{\psi_i - \psi_{i-1}}{m_{i-1/2}\,h_{i-1/2}}\right] + V_i\,\psi_i = E\,\psi_i.
$$

On a non-uniform grid this is a generalized eigenproblem $A\psi = E\,W\psi$ with $W = \text{diag}(w_i)$.
The substitution $\varphi = W^{1/2}\psi$ turns it into an ordinary symmetric tridiagonal
eigenproblem, solved directly for the lowest states. The eigenvectors come out normalized as

$$
\sum_i |\psi_{k,i}|^2\, w_i = 1.
$$

## 6.3 The quantum region

Schrödinger is solved only between the user's quantum-region markers, or, if none are given, over
the undoped span of the device (wells, barriers, electron-blocking layer). Solving over the whole
device would treat a 200 nm doped buffer as a very wide well whose closely spaced levels swamp the
states of interest. Outside the region, carriers are classical.

## 6.4 Charge from the subbands

Each subband is a two-dimensional band with constant density of states $m^*/\pi\hbar^2$. Its sheet
density is

$$
N_k = \frac{m_e\,k_B T}{\pi\hbar^2}\,\ln\!\left[1 + \exp\!\left(\frac{E_{Fn} - E_k}{k_B T}\right)\right],
$$

and the electron density is

$$
n(z) = \sum_k N_k\,|\psi_k(z)|^2 .
$$

Holes are the mirror image, summed over the three valence bands:

$$
p(z) = \sum_{b}\sum_k \frac{m_b\,k_B T}{\pi\hbar^2}\,\ln\!\left[1 + \exp\!\left(\frac{E_{v,b,k} - E_{Fp}}{k_B T}\right)\right]|\psi_{b,k}(z)|^2 .
$$

A single average in-plane mass over the region is used in the prefactor, so that the sheet
density of a subband is one number and not a function of position.

**How many subbands.** The number of states is not fixed. States are added until the highest one
lies at least $10\,k_BT$ above the Fermi level, so that everything left out carries negligible
charge. A fixed small count fails on multi-quantum-well stacks, where the lowest states of nine
wells use up the budget before any excited state is reached.

## 6.5 Self-consistency

Schrödinger and Poisson are iterated together inside the loop of
[section 5.5](05_poisson_equilibrium.md#55-iteration-and-convergence): the potential fixes the
states, the states fix the charge, the charge fixes the potential. The Newton step still uses the
classical screening term $D$ as its Jacobian. This is the usual predictor-corrector idea: the
quantum charge responds to a small change in potential almost as the classical charge does, so
the classical derivative is a good preconditioner even though the charge itself is quantum.

## 6.6 Quantum correction under bias

The biased solver ([chapter 7](07_drift_diffusion_bias.md)) works with quasi-Fermi levels and
classical-form densities. Quantum effects enter through a correction factor defined at every node,

$$
\gamma_n(z) = \frac{n_{quantum}(z)}{n_{classical}(z)}, \qquad \gamma_p(z) = \frac{p_{quantum}(z)}{p_{classical}(z)},
$$

equal to 1 outside the quantum region. The drift-diffusion equations are then solved with

$$
n = \gamma_n\,N_c\,\mathcal{F}_{1/2}\!\left(\frac{E_{Fn} - E_c}{k_B T}\right),
$$

which is equivalent to adding a quantum potential $k_BT\ln\gamma_n$ to the conduction band edge.
$\gamma$ is frozen during one drift-diffusion solve, then recomputed from a new Schrödinger solve,
with under-relaxation, until the quantum and modelled densities agree to about 1% wherever the
density is significant. $\gamma$ is clipped to $[10^{-8}, 10^{8}]$ so that nodes with no charge do
not produce meaningless potentials.

The first quantum bias solve starts from the converged classical solution at the same bias.

## 6.7 What the quantum model leaves out

- **Nonparabolicity.** Subbands are parabolic. In deep GaN/AlN wells this overestimates the second
  electron level by roughly 60-120 meV.
- **Valence-band mixing.** HH, LH and CH are decoupled.
- **Exchange-correlation and many-body shifts.** Hartree only.
- **Tunnelling current.** Wavefunctions penetrate barriers, but transport is still drift-diffusion.
- **Monolayer-scale wells.** The envelope-function picture is not valid for wells of 1-3 monolayers.
