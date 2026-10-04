# 1. Structure and grid

## 1.1 Simulation domain

The device is a stack of layers along one axis $`z`$, from the substrate side ($`z = 0`$, "bottom")
to the surface ("top"). All quantities are uniform in the plane of the wafer.

A structure is an ordered list of regions and exactly two contacts.

| Region | Class | Defines |
|---|---|---|
| Uniform layer | `AbruptLayer` | composition $`(x, y)`$, thickness, $`N_D`$, $`N_A`$ |
| Graded layer | `GradedLayer` | start and end composition, thickness, doping, profile |
| Interface trap states | `SurfaceCharge` | one or more donor- or acceptor-like levels at a single plane |
| Interface dipole | `InterfaceDipole` | two fixed sheets $`\pm\sigma`$ a distance $`d`$ apart |
| Quantum region marker | `QuantumRegionMarker` | start or end of the region where (6.1) is solved |

Grading profiles: `linear`, `parabolic`, `stepped` (a staircase of `n_steps` uniform sublayers)
and `abrupt`.

Per-layer options:

| Option | Effect |
|---|---|
| `relaxed` | the layer carries no strain (chapter 3) |
| `custom_strain_xx` | in-plane strain set by the user instead of computed |
| `dx_nm` | grid spacing inside this layer |
| `series_resistance` | contribution to the external series resistance, Ω·cm² |

## 1.2 Grid

The domain is covered by $`N`$ nodes $`z_0 \lt z_1 \lt \dots \lt z_{N-1}`$ with edges

```math
h_{i+1/2} = z_{i+1} - z_i . \tag{1.1}
```

The spacing is uniform inside a layer and may differ from layer to layer. Typical values are
0.02-0.05 nm in quantum wells and thin barriers and 0.2-1 nm in buffers.

Each node owns a control volume of width

```math
w_i = \tfrac12\left(h_{i-1/2} + h_{i+1/2}\right), \qquad w_0 = h_{1/2}, \quad w_{N-1} = h_{N-3/2}. \tag{1.2}
```

## 1.3 Discretisation rule

Every differential equation in this reference has the form "divergence of a flux equals a source".
It is discretised by integrating over the control volume of node $`i`$:

```math
\frac{\mathcal{J}_{i+1/2} - \mathcal{J}_{i-1/2}}{w_i} = S_i , \tag{1.3}
```

where the flux $`\mathcal{J}_{i+1/2}`$ on an edge is evaluated once and used by both nodes it joins.

| Equation | Flux $`\mathcal{J}`$ | Source $`S`$ |
|---|---|---|
| Poisson (5.1) | $`\varepsilon_0\varepsilon_r\,d\phi/dz`$ | $`-\rho`$ |
| Polarization charge (4.9) | $`-P`$ | $`\rho_{pol}`$ |
| Schrödinger (6.1) | $`-(\hbar^2/2m^*)\,d\psi/dz`$ | $`(E - V)\psi`$ |
| Current (7.3) | $`\mu n\,dE_{Fn}/dz`$ | $`R - G`$ |

> [!NOTE]
> Because each edge flux is shared, whatever leaves one control volume enters the next. Total
> charge, probability and current are conserved to rounding error on any grid, including where the
> spacing changes abruptly. On a uniform grid (1.3) is identical to the usual three-point
> central-difference stencil.

Where the derivative of a nodal profile is needed for output (the electric field, the
quasi-field), the second-order non-uniform formula is used:

```math
f'(z_i) = \frac{h_-^2 f_{i+1} - h_+^2 f_{i-1} - (h_-^2 - h_+^2) f_i}{h_- h_+ (h_- + h_+)},
\qquad h_\pm = h_{i\pm1/2}. \tag{1.4}
```

## 1.4 Node profiles

After the grid is built, every material quantity of chapters 2-4 is evaluated at every node from
that node's composition, strain and doping. The solver never refers back to "layers"; it sees only
node arrays. A graded layer is therefore not a special case: it is simply a region in which the
composition array varies.

Edge values needed by the fluxes are averages of the two adjacent nodes, for example
$`\varepsilon_{i+1/2} = \tfrac12(\varepsilon_i + \varepsilon_{i+1})`$ and
$`m_{i+1/2} = \tfrac12(m_i + m_{i+1})`$.

## 1.5 Contacts

| Type | Boundary condition | Section |
|---|---|---|
| `ohmic` | local charge neutrality fixes the band position at the contact | 5.6 |
| `schottky` | $`E_c - E_F = \Phi_B`$ at the contact | 5.6 |

A free surface is entered as a Schottky contact whose barrier is the surface Fermi-level pinning
position. The bottom contact is the potential reference.

## 1.6 Substrate

The first (bottom) layer defines the in-plane lattice constant to which all other layers are
strained (chapter 3). It is the only role the substrate plays; there is no separate substrate
object.
