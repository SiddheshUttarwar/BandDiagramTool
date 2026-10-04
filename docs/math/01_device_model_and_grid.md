# 1. Device model, grid and energy reference

## 1.1 The device

A device is an ordered list of layers, bottom (substrate side) to top (surface), and two contacts.

| Layer type | What it specifies |
|---|---|
| Abrupt layer | Al fraction $x$, In fraction $y$, thickness, donor and acceptor doping, optional strain override |
| Graded layer | start and end composition, thickness, doping, grading profile |
| Surface charge | zero-thickness sheet of donor-like or acceptor-like traps at an interface |
| Interface dipole | two fixed sheets $\pm\sigma$ a small distance apart |
| Quantum-region marker | where the Schrödinger region starts and ends |

Each contact is either **ohmic** or **Schottky** (with a metal, or an explicit barrier height).

Everything is uniform in the plane of the wafer. Lateral effects (current spreading, gate edges,
mesa sidewalls) are outside the model.

## 1.2 The grid

The stack is discretized into $N$ nodes $z_0 < z_1 < \dots < z_{N-1}$. Each layer may carry its
own spacing, so thin wells and barriers can be resolved at 0.02-0.05 nm while thick buffers use
0.2-1 nm. Node $i$ has

- left and right edge lengths $h_{i-1/2} = z_i - z_{i-1}$ and $h_{i+1/2} = z_{i+1} - z_i$,
- a control-volume width $w_i = \tfrac12 (h_{i-1/2} + h_{i+1/2})$; at the two ends $w$ is the single adjacent edge.

All differential operators are written in **finite-volume** form: the flux through each edge is
computed once and shared by the two nodes it joins, and each node's balance is divided by its own
$w_i$. Two consequences matter in practice:

1. **Charge is conserved exactly on any grid.** A polarization step of $\Delta P$ produces
   exactly $\Delta P$ of sheet charge whether or not the spacing changes at that interface
   (see [03](03_strain_and_polarization.md#35-polarization-charge)).
2. **On a uniform grid every stencil reduces to the textbook central-difference formula.**

Where a plain derivative of a profile is needed (electric field, quasi-field), the three-point
non-uniform formula is used:

$$
f'(z_i) \approx \frac{h_-^2 f_{i+1} - h_+^2 f_{i-1} - (h_-^2 - h_+^2) f_i}{h_- h_+ (h_- + h_+)},
\qquad h_- = h_{i-1/2},\; h_+ = h_{i+1/2}.
$$

## 1.3 Energy reference and band line-up

The equilibrium Fermi level is the zero of energy. Flat-band (zero-field) band edges are built
from the electron affinity $\chi$ and the gap $E_g$ by Anderson's rule, anchored at the bottom node:

$$
E_{c0}(z) = E_{c0}(0) + \chi(0) - \chi(z), \qquad E_{v0}(z) = E_{c0}(z) - E_g(z).
$$

$\chi$ is not an independent input. It is derived from a valence-band offset so that the
heterojunction line-up matches measured offsets (see [02](02_material_model.md#23-band-alignment)).
$E_v$ always means the **topmost** valence band at that composition and strain.

The actual band edges follow the electrostatic potential:

$$
E_c(z) = E_{c0}(z) - \phi(z), \qquad E_v(z) = E_{v0}(z) - \phi(z).
$$

The electric field reported by the tool is $F = -d\phi/dz$. In graded layers there is in addition
a **quasi-field** from the changing affinity, $F_{quasi} = d\chi/dz$ (with $\chi$ in eV, giving V/m), which acts on
electrons but is not a solution of Poisson's equation.

## 1.4 Boundary conditions

The bottom contact is the potential reference, $\phi(0) = 0$.

**Ohmic contact.** The semiconductor at the contact is assumed charge-neutral. The band position
relative to the Fermi level is found by solving the local neutrality condition

$$
N_D^+ - N_A^- + p - n = 0
$$

for $E_c$ at that node, with Fermi-Dirac carriers and incompletely ionized dopants
([04](04_carrier_statistics.md)). An ohmic contact on an undoped wide-gap layer therefore pins the
Fermi level near mid-gap; it does not create carriers that are not there.

**Schottky contact (or a free surface with a pinned Fermi level).**

$$
E_c - E_F = \Phi_B \quad \text{at the contact,}
$$

where $\Phi_B$ is the barrier height given by the user, or the Schottky-Mott estimate
$\Phi_B = W_m - \chi$ from the metal work function $W_m$ when none is given. A bare surface is
modelled the same way, with $\Phi_B$ the surface pinning position.

**Under bias** $V$ the top-contact potential becomes $\phi_{top} = \phi_{top}^{eq} + V$. Both
quasi-Fermi levels are held at 0 at the bottom contact and at $-V$ at the top contact, so each
contact is in local equilibrium with its metal. With series resistance $R_s$ the junction sees $V_{int}$ with
$V_{int} + |J(V_{int})| R_s = V$ ([07](07_drift_diffusion_bias.md#75-series-resistance)).

## 1.5 Strain reference

The bottom layer is taken as the relaxed substrate. Every layer above is strained to its in-plane
lattice constant unless the layer is flagged relaxed or given an explicit in-plane strain.
There is no critical-thickness or relaxation model: a layer stays coherent however thick it is
unless the user says otherwise.
