# 3. Strain

Strain enters the model in two places: it shifts the band edges (section 3.4) and it produces the
piezoelectric polarization (chapter 4).

## 3.1 Strain and stress

In linear elasticity the deformation of a crystal is described by the symmetric strain tensor

$$
\varepsilon_{ij} = \frac12\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right), \tag{3.1}
$$

with $\mathbf{u}$ the displacement field. Stress and strain are related by Hooke's law,
$\sigma_{ij} = C_{ijkl}\,\varepsilon_{kl}$. In Voigt notation
($11 \to 1$, $22 \to 2$, $33 \to 3$, $23 \to 4$, $13 \to 5$, $12 \to 6$) the stiffness tensor of a
wurtzite crystal with $z \parallel c$ is

$$
C_{wz} =
\begin{pmatrix}
C_{11} & C_{12} & C_{13} & & & \\
C_{12} & C_{11} & C_{13} & & & \\
C_{13} & C_{13} & C_{33} & & & \\
 & & & C_{44} & & \\
 & & & & C_{44} & \\
 & & & & & C_{66}
\end{pmatrix}, \qquad C_{66} = \tfrac12\,(C_{11} - C_{12}). \tag{3.2}
$$

## 3.2 Pseudomorphic layer on a c-plane substrate

A layer grown coherently on a thick substrate adopts the substrate's in-plane lattice constant
$a_{sub}$. For a c-plane layer this fixes the in-plane strain and leaves no shear:

$$
\varepsilon_{xx} = \varepsilon_{yy} = \frac{a_{sub} - a_0}{a_0}, \qquad
\varepsilon_{xy} = \varepsilon_{xz} = \varepsilon_{yz} = 0 , \tag{3.3}
$$

with $a_0(x,y)$ the layer's own relaxed lattice constant from (2.1). The surface is free, so the
stress along the growth axis vanishes:

$$
\sigma_{zz} = C_{13}\,(\varepsilon_{xx} + \varepsilon_{yy}) + C_{33}\,\varepsilon_{zz} = 0 . \tag{3.4}
$$

Solving for the out-of-plane strain:

$$
\varepsilon_{zz} = -\frac{2\,C_{13}}{C_{33}}\,\varepsilon_{xx} . \tag{3.5}
$$

(3.3) and (3.5) are the complete strain state. For a laterally uniform stack this analytic result
is the exact minimum of the elastic energy, so no numerical strain solve is needed.

The in-plane stress that results is not an output of the tool, but follows directly and is useful
for comparison with wafer-curvature data:

$$
\sigma_{xx} = \left(C_{11} + C_{12} - \frac{2\,C_{13}^2}{C_{33}}\right)\varepsilon_{xx} . \tag{3.6}
$$

Sign convention: $\varepsilon_{xx} > 0$ is tensile (AlGaN on GaN), $\varepsilon_{xx} < 0$ is
compressive (InGaN on GaN, GaN on AlN).

## 3.3 Strain options

| Setting | In-plane strain used |
|---|---|
| default | (3.3) with $a_{sub}$ = lattice constant of the bottom layer |
| `relaxed=True` | $\varepsilon_{xx} = 0$ |
| `custom_strain_xx = s` | $\varepsilon_{xx} = s$ |

In every case $\varepsilon_{zz}$ follows from (3.5); it is never set independently.

> [!WARNING]
> There is no critical-thickness or relaxation model. A layer is coherent at any thickness unless
> the user marks it relaxed or supplies a measured strain. A 250 nm GaN layer on AlN will be
> computed as fully strained, which a real layer of that thickness is not.

## 3.4 Band-edge shifts

Strain shifts the conduction band edge through the deformation potentials $a_1$, $a_2$ and the
valence bands through $D_1 \dots D_4$ (table in section 2.7).

**Conduction band.**

$$
\Delta E_c = a_{cz}\,\varepsilon_{zz} + a_{ct}\,(\varepsilon_{xx} + \varepsilon_{yy}), \qquad
a_{cz} = a_1 + D_1, \quad a_{ct} = a_2 + D_2 . \tag{3.7}
$$

$a_1$, $a_2$ are the interband (gap) deformation potentials; adding $D_1$, $D_2$ converts them to
absolute conduction-band potentials on the same scale as the valence shifts below.

**Valence bands.** The strain terms that enter the zone-centre Hamiltonian (2.7)-(2.8) are

$$
\lambda_\varepsilon = D_1\,\varepsilon_{zz} + D_2\,(\varepsilon_{xx} + \varepsilon_{yy}), \qquad
\theta_\varepsilon = D_3\,\varepsilon_{zz} + D_4\,(\varepsilon_{xx} + \varepsilon_{yy}). \tag{3.8}
$$

For the biaxial state (3.3):

$$
\lambda_\varepsilon = D_1\,\varepsilon_{zz} + 2 D_2\,\varepsilon_{xx}, \qquad
\theta_\varepsilon = D_3\,\varepsilon_{zz} + 2 D_4\,\varepsilon_{xx}. \tag{3.9}
$$

The shift of the top valence band is the change of the largest eigenvalue:

$$
\Delta E_v = \max\left(E_{HH}, E_\pm\right)\Big|_{\varepsilon} - \max\left(E_{HH}, E_\pm\right)\Big|_{\varepsilon = 0}. \tag{3.10}
$$

Strain also changes the separations $\delta_b$ of (2.9), and can change which band is on top.

**Strained band edges.** The edges that enter the electrostatics are

$$
E_{c,0}(z) = E_c^{abs} + \Delta E_c, \qquad E_{v,0}(z) = E_v^{abs} + \Delta E_v , \tag{3.11}
$$

up to a common constant fixed by the boundary condition (section 5.6). The strained gap is
$E_g + \Delta E_c - \Delta E_v$.

| Example | $\varepsilon_{xx}$ | Change of gap |
|---|---|---|
| GaN on AlN | $-2.4\%$ | $+0.18$ eV |
| Al$_{0.6}$Ga$_{0.4}$N on AlN | $-1.0\%$ | $+0.16$ eV (4.80 to 4.97 eV) |

The shifts can be switched off with `include_strain_band_shift=False`, which leaves the
piezoelectric polarization in place and uses unstrained band edges.
