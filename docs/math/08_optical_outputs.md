# 8. Optical outputs: transition energy, overlap, gain

These quantities are computed from a converged quantum solution. They are post-processing: none
of them feeds back into the electrostatics.

## 8.1 The quantum-confined Stark effect

The polarization field tilts a quantum well. Electrons collect at one side, holes at the other.
Two things follow, and the tool reports both:

- the transition energy falls below the flat-well value (red shift);
- the electron and hole wavefunctions separate, so the optical matrix element falls.

A rough estimate for a well of width $L_w$ in field $F$ is
$E_{e1h1} \approx E_g + E_{e1} + E_{h1} - qFL_w$. The tool does not use this estimate; it takes the
energies and wavefunctions from the Schrödinger solution in the actual tilted potential.

## 8.2 Transition energy

For electron subband $i$ and hole subband $j$:

$$
E_{ij} = E_{e,i} - E_{v,j},
$$

with $E_{v,j} = -E_{h,j}$ the hole subband energy on the same absolute scale as the band edges
([section 6.1](06_schrodinger_quantum.md#61-the-equation)). The headline number is the ground
transition $E_{11}$ of the well, using the topmost of the HH, LH and CH ground states.

**Which state belongs to the well.** In a full device the lowest-energy electron state is often
not in the quantum well at all, but in a polarization-induced notch elsewhere. The tool therefore
selects the lowest state whose probability lies mostly inside the well layer,

$$
\sum_{i \in \text{well}} |\psi_i|^2 w_i \ge \tfrac12 .
$$

If no globally solved state qualifies (typical for multi-well p-i-n stacks), it solves a local
problem on the well plus a few nanometres of barrier on each side and reports that, flagged as a
local solve.

## 8.3 Wavefunction overlap

$$
\left|\langle \psi_{e,i} | \psi_{h,j} \rangle\right|^2 = \left| \sum_k \psi_{e,i}(z_k)\,\psi_{h,j}(z_k)\,w_k \right|^2 ,
$$

between 0 and 1. It is proportional to the oscillator strength of the transition and so to the
radiative rate. In polar wells wider than about 3 nm it falls by orders of magnitude; in that
case a higher pair of subbands can carry more oscillator strength than the ground pair, and the
tool reports the highest-overlap pair among the lowest few as well.

## 8.4 Gain spectrum

A band-edge quantum-well gain model, summed over all electron subbands $i$, hole subbands $j$
and hole bands $b$:

$$
g(\hbar\omega) = \sum_{i,j,b}
\frac{\pi q^2 \hbar}{n_r\,\varepsilon_0\,c\,m_0^2\,E_{ij}}\;
|M_T|^2\;
\left|\langle\psi_{e,i}|\psi_{h,j}\rangle\right|^2\;
\frac{m_{r,ij}}{\pi\hbar^2 L_z}\;
\left[f_c(E_{e,i}) - f_v(E_{v,j})\right]\;
\mathcal{L}(\hbar\omega - E_{ij}).
$$

| Factor | Meaning | Value used |
|---|---|---|
| $\lvert M_T\rvert^2 = \tfrac{m_0}{6}E_p$ | bulk momentum matrix element | Kane energy $E_p$ 14 eV (GaN) to 18 eV (AlN), linear in $x$ |
| $m_{r} = (1/m_e + 1/m_h)^{-1}$ | reduced mass | from the confinement masses |
| $m_r/\pi\hbar^2 L_z$ | two-dimensional joint density of states per unit volume, both spins | $L_z$ = well width |
| $f_c$, $f_v$ | Fermi occupations with $E_{Fn}$, $E_{Fp}$ | from the biased solution |
| $\mathcal{L}$ | Lorentzian line shape | 10 meV half-width |
| $n_r$ | refractive index | 2.5 |

$g > 0$ (gain) requires $f_c > f_v$, which is the Bernard-Duraffourg condition
$E_{Fn} - E_{Fp} > E_{ij}$.

**How far to trust it.** The occupations are evaluated at each subband edge and the in-plane
dispersion is not integrated, so the model gives the right peak position, the right sign change
from absorption to gain with injection, and a reasonable line shape. The absolute value in
cm$^{-1}$ is an order-of-magnitude figure. It has been checked against bulk GaN absorption for the
matrix-element convention, not against measured quantum-well gain.

## 8.5 Comparing with measurements

The reported transition energy is a single-particle band-to-band energy at 300 K. A measured
emission peak differs from it by effects the tool does not include:

| Effect | Size | Direction |
|---|---|---|
| Exciton binding | 20-30 meV (GaN), up to 80 meV (AlN), more in narrow wells | measured lower |
| Temperature (gap is fixed at 300 K) | +60 to +110 meV at 10 K | measured higher at low T |
| Screening of the field under injection | up to several hundred meV in Al-rich wells | measured higher |
| Alloy and well-width fluctuation (localization) | 20-100 meV | measured lower |
| Stokes shift between absorption and emission | 0-100 meV | emission lower |

For a wide polar well at zero bias, the ground transition can have an overlap below 1%. A real LED
does not emit from that state: by the time it emits, injected carriers have screened part of the
field. Comparing the zero-bias $E_{11}$ with electroluminescence is then misleading; a biased
solution is the appropriate comparison.
