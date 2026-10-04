# 8. Optics

The optical quantities are evaluated from a converged quantum solution. They are post-processing
and do not feed back into the electrostatics.

## 8.1 Interband transition energies

For electron subband $i$ and hole subband $j$ of valence band $b$:

$$
E_{ij}^{(b)} = E_{e,i} - E_{v,b,j}, \tag{8.1}
$$

with the hole energy on the common scale by (6.2). These are single-particle, band-to-band
energies at the simulation temperature's thermal energy but the 300 K gap.

## 8.2 Envelope overlap

The interband matrix element between two states factorises into a bulk part and the overlap of the
envelope functions,

$$
O_{ij} = \left|\langle\psi_{e,i}|\psi_{h,j}\rangle\right|^2 = \left|\sum_k \psi_{e,i}(z_k)\,\psi_{h,j}(z_k)\,w_k\right|^2 , \qquad 0 \le O_{ij} \le 1. \tag{8.2}
$$

$O_{ij}$ scales the oscillator strength and hence the radiative rate of the transition.

## 8.3 Quantum-confined Stark effect

The polarization field $F$ across a well of width $L_w$ separates electrons and holes to opposite
sides. Two consequences follow, and they are the two numbers the tool reports for a well:

| Quantity | Effect of the field |
|---|---|
| ground transition energy $E_{11}$ | lowered, roughly by $q F L_w$ for wide wells |
| overlap $O_{11}$ | reduced, by orders of magnitude for $L_w \gtrsim 3$ nm |

No approximate formula is used. Both come from the solution of (6.1) in the actual tilted,
self-consistent potential.

**Selecting the well's states.** The lowest eigenstate of a device is frequently not in the
quantum well but in a polarization-induced notch elsewhere. For the well's own transition the
tool takes the lowest state $k$ with

$$
\sum_{i\,\in\,\text{well}} \left|\psi_{k,i}\right|^2 w_i \ \ge\ \tfrac12 . \tag{8.3}
$$

If no state of the global solution satisfies (8.3), which is typical for multi-well p-i-n
structures, (6.1) is solved again on the well plus 4 nm of barrier on each side, and that result
is reported and flagged (`qcse_local_solve`).

**Dominant transition.** When $O_{11}$ is very small, a higher pair can carry more oscillator
strength. The tool also reports the pair with the largest $O_{ij}$ among the lowest few subbands.

| Output field | Content |
|---|---|
| `qcse_transition_eV` | $E_{11}$ of the well, using the topmost hole band |
| `qcse_overlap` | $O_{11}$ |
| `qcse_pair` | subband indices used |
| `qcse_in_well`, `qcse_local_solve` | how the states were selected |

## 8.4 Gain spectrum

The material gain of a quantum well is computed from Fermi's golden rule in the band-edge
approximation: parabolic subbands, a wave-vector-independent momentum matrix element, and
occupation factors evaluated at each subband edge.

$$
g(\hbar\omega) = \sum_{i,j,b}\
\frac{\pi q^2\hbar}{n_r\,\varepsilon_0\,c\,m_0^2\,E_{ij}}\
\left|M_T\right|^2\ O_{ij}\
\rho_{2D,ij}\
\left[f_c(E_{e,i}) - f_v(E_{v,b,j})\right]\
\mathcal{L}\!\left(\hbar\omega - E_{ij}\right). \tag{8.4}
$$

The factors are:

$$
\left|M_T\right|^2 = \frac{m_0}{6}\,E_P, \tag{8.5}
$$

the bulk momentum matrix element, with $E_P$ the Kane energy;

$$
\rho_{2D,ij} = \frac{m_{r,ij}}{\pi\hbar^2 L_z}, \qquad \frac{1}{m_{r,ij}} = \frac{1}{m_{e}} + \frac{1}{m_{h,b}}, \tag{8.6}
$$

the reduced two-dimensional density of states per unit volume, both spins included;

$$
f_c(E) = \frac{1}{1 + e^{(E - E_{Fn})/k_BT}}, \qquad f_v(E) = \frac{1}{1 + e^{(E - E_{Fp})/k_BT}}, \tag{8.7}
$$

the electron occupations of the conduction and valence states; and the Lorentzian line shape

$$
\mathcal{L}(\Delta) = \frac{1}{\pi}\,\frac{\Gamma}{\Delta^2 + \Gamma^2}. \tag{8.8}
$$

| Parameter | Value |
|---|---|
| Kane energy $E_P$ | 14 eV (GaN) to 18 eV (AlN), linear in $x$ |
| broadening $\Gamma$ | 10 meV |
| refractive index $n_r$ | 2.5 |
| $L_z$ | well width |

Gain ($g > 0$) requires $f_c > f_v$, that is $E_{Fn} - E_{Fp} > E_{ij}$ (Bernard-Duraffourg).
Below that separation (8.4) is negative and is the absorption coefficient.

> [!WARNING]
> (8.4) gives a reliable peak position, the correct change of sign with injection and a reasonable
> line shape. Its magnitude in cm$^{-1}$ is an order-of-magnitude estimate: the in-plane
> dispersion is not integrated, valence-band mixing is absent, and the matrix-element convention
> has been checked only against bulk GaN absorption.

## 8.5 Relation to measured spectra

The reported $E_{11}$ is not a prediction of a luminescence peak. The difference is made up of
effects outside the model:

| Effect | Typical size | Shifts measurement |
|---|---|---|
| exciton binding | 20-30 meV in GaN, up to 80 meV in AlN, larger in narrow wells | down |
| temperature (gap fixed at its 300 K value) | 60-110 meV between 300 K and 10 K | up at low $T$ |
| screening of the field by injected carriers | up to several 100 meV in Al-rich wells | up |
| localisation in alloy and width fluctuations | 20-100 meV | down |
| Stokes shift | 0-100 meV | down, emission only |

For a wide polar well at zero bias $O_{11}$ can be below 1%. A device does not emit from that
state; by the time it emits, the injected carriers have partly screened the field. The comparison
with electroluminescence should then be made with a biased solution.
