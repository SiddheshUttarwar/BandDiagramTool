# 6. Quantum mechanics

The classical densities (5.4)-(5.5) depend only on the local band edge. They put the peak of a
2DEG exactly at the interface, give zero density inside a barrier, and know nothing of quantised
levels. In the quantum modes the tool solves for the confined states of a designated region and
uses them for its charge.

## 6.1 Envelope-function Schrödinger equation

Each band is treated in the single-band effective-mass (envelope-function) approximation. With the
in-plane motion separated as plane waves, the envelope $\psi(z)$ along the growth axis satisfies

$$
-\frac{\hbar^2}{2}\,\frac{d}{dz}\left[\frac{1}{m^*(z)}\,\frac{d\psi}{dz}\right] + V(z)\,\psi(z) = E\,\psi(z). \tag{6.1}
$$

The operator ordering (BenDaniel-Duke) keeps the Hamiltonian Hermitian for a position-dependent
mass and makes $\psi$ and $\frac{1}{m^*}\frac{d\psi}{dz}$ continuous at interfaces.

| Carrier | Potential $V(z)$ | Mass | Bands |
|---|---|---|---|
| electron | $E_c(z)$ | $m_e(z)$ | 1 |
| hole | $-\left[E_v(z) - \delta_b(z)\right]$ | quantisation mass of band $b$ | 3: HH, LH, CH |

For holes the sign of the potential is reversed, so that the top of the valence band becomes the
bottom of a well and the same eigen-solver returns the states nearest the band edge first. An
eigenvalue $E^{(h)}$ of the reversed problem corresponds to a hole subband at

$$
E_{v,b,k} = -E^{(h)}_{b,k} \tag{6.2}
$$

on the common energy scale. The three valence bands are solved as three independent problems.

Boundary conditions: $\psi = 0$ at both ends of the quantum region.

## 6.2 Discrete form

Applying (1.3) to (6.1) with edge masses $m_{i+1/2} = \tfrac12(m_i + m_{i+1})$:

$$
-\frac{\hbar^2}{2\,w_i}\left[\frac{\psi_{i+1} - \psi_i}{m_{i+1/2}\,h_{i+1/2}} - \frac{\psi_i - \psi_{i-1}}{m_{i-1/2}\,h_{i-1/2}}\right] + V_i\,\psi_i = E\,\psi_i . \tag{6.3}
$$

Multiplying by $w_i$ gives a generalised eigenproblem $H\boldsymbol\psi = E\,W\boldsymbol\psi$ with
$H$ symmetric tridiagonal and $W = \operatorname{diag}(w_i)$. The substitution
$\boldsymbol\varphi = W^{1/2}\boldsymbol\psi$ turns it into the standard symmetric tridiagonal
problem

$$
\left(W^{-1/2} H\,W^{-1/2}\right)\boldsymbol\varphi = E\,\boldsymbol\varphi , \tag{6.4}
$$

which is solved directly for the lowest states only. The eigenvectors are normalised as

$$
\sum_i \left|\psi_{k,i}\right|^2 w_i = 1 . \tag{6.5}
$$

## 6.3 Quantum region

(6.1) is solved on a sub-interval of the device, the quantum region.

| How it is set | Region used |
|---|---|
| `QuantumRegionMarker('start')` and `('end')` in the layer list | from 2 nm before the start marker to 2 nm after the end marker |
| no markers | the contiguous undoped span of the device (wells, barriers, blocking layers) |

Outside the region carriers are classical. The restriction is necessary: solved over a whole
device, a thick doped contact layer acts as a very wide well whose dense ladder of states would be
returned before any state of the actual quantum well.

## 6.4 Quantum charge densities

In a structure quantised along one axis, each eigenstate $k$ is the bottom of a two-dimensional
subband. For a parabolic in-plane dispersion the sum over in-plane wave vectors can be done
analytically. The general result for a system quantised in $(3-d')$ directions and free in $d'$ is a
Fermi-Dirac integral of order $(d'-2)/2$; for the present case $d' = 2$ it is of order zero,

$$
\mathcal{F}_0(\eta) = \ln\left(1 + e^{\eta}\right),
$$

and the sheet density of electron subband $k$ is

$$
N_k = \frac{m_e\,k_B T}{\pi\hbar^2}\ \ln\!\left[1 + \exp\!\left(\frac{E_{Fn} - E_k}{k_B T}\right)\right]. \tag{6.6}
$$

The prefactor $m_e/\pi\hbar^2$ is the two-dimensional density of states including spin. The
electron density is

$$
n(z) = \sum_k N_k\ \left|\psi_k(z)\right|^2 . \tag{6.7}
$$

For holes, summing over the three valence bands:

$$
p(z) = \sum_{b}\sum_k \frac{m_b\,k_B T}{\pi\hbar^2}\ \ln\!\left[1 + \exp\!\left(\frac{E_{v,b,k} - E_{Fp}}{k_B T}\right)\right]\left|\psi_{b,k}(z)\right|^2 . \tag{6.8}
$$

The mass in the prefactor is averaged over the quantum region, so that a subband has one sheet
density and not a position-dependent one.

As in the classical case, these densities depend on the potential, here through the Hamiltonian:
$n = n(z;\ \phi,\ E_{Fn})$. They replace (5.4)-(5.5) in the charge density (5.2) inside the
quantum region.

## 6.5 Number of subbands

The sums in (6.7)-(6.8) are truncated by energy, not by count. States are added, doubling the
number requested, until the highest one lies at least

$$
E_k - E_{Fn} \ \ge\ 10\,k_B T \tag{6.9}
$$

above the Fermi level (and correspondingly for holes), at which point the neglected occupation is
below $e^{-10}$. `n_states_e` and `n_states_h` are starting values only.

> [!NOTE]
> A fixed count fails on multi-quantum-well structures. With nine wells and six requested states,
> all six are ground states of different wells and no well has its second level included.

## 6.6 Quantum correction for the current equation

The current equation (chapter 7) is written for densities of the classical form (5.4). To use it
with quantum densities, the tool defines at every node the ratio

$$
\gamma_n(z) = \frac{n^{qm}(z)}{n^{cl}(z)}, \qquad \gamma_p(z) = \frac{p^{qm}(z)}{p^{cl}(z)}, \tag{6.10}
$$

with both densities evaluated at the same $\phi$ and quasi-Fermi levels, and $\gamma = 1$ outside
the quantum region. The current equation is then solved with

$$
n = \gamma_n\,N_c\ \mathcal{F}_{1/2}\!\left(\frac{E_{Fn} - E_c}{k_B T}\right), \qquad
p = \gamma_p\,N_v\ \mathcal{F}_{1/2}\!\left(\frac{E_v - E_{Fp}}{k_B T}\right). \tag{6.11}
$$

In the non-degenerate limit this is the same as adding a quantum potential
$-k_B T\ln\gamma_n$ to the conduction band edge. $\gamma$ is limited to
$[10^{-8}, 10^{8}]$, which only affects nodes with no significant charge.

## 6.7 Scope

| Not included | Consequence |
|---|---|
| multi-band k·p coupling | no valence-band mixing; in-plane dispersion is parabolic |
| conduction-band nonparabolicity | upper electron levels in deep GaN/AlN wells are 60-120 meV too high |
| exchange-correlation | Hartree potential only |
| excitons | transition energies are band-to-band |
| tunnelling current | envelope functions penetrate barriers, but transport remains drift-diffusion |
| atomistic effects | not valid for wells of 1-3 monolayers |
