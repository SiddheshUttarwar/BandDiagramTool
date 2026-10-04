# 4. Polarization

Wurtzite nitrides carry a built-in electric polarization along the c-axis. Its discontinuities at
heterointerfaces are fixed charges of order $`10^{13}`$ cm<sup>−2</sup>, which is what creates
two-dimensional electron and hole gases in undoped structures and tilts every quantum well.

The total polarization is the sum of a spontaneous (pyroelectric) and a piezoelectric part:

```math
\mathbf{P} = \mathbf{P}_{sp} + \mathbf{P}_{pz}. \tag{4.1}
```

## 4.1 Spontaneous polarization

Wurtzite lacks inversion symmetry along c, so even the unstrained crystal is polarized. The
polarization is directed along $`-c`$, that is, negative in the tool's convention. It is
interpolated with pairwise bowing (2.2):

```math
P_{sp}(x,y) = -0.090\,x - 0.042\,y - 0.034\,z_{Ga} + 0.021\,x\,z_{Ga} + 0.037\,y\,z_{Ga} + 0.070\,x\,y
\quad [\text{C/m}^2]. \tag{4.2}
```

Temperature dependence (the pyroelectric effect proper):

```math
P_{sp}(T) = P_{sp}(300\,\text{K}) + (T - 300\,\text{K})\,\left[x\,p_{AlN} + (1 - x)\,p_{GaN}\right]. \tag{4.3}
```

Spontaneous polarization can be switched off for the whole device with
`include_spontaneous_polarization=False`.

## 4.2 Piezoelectric polarization

Strain adds a polarization through the piezoelectric tensor, $`P_{pz,i} = e_{ijk}\,\varepsilon_{jk}`$.
For wurtzite with $`z \parallel c`$, in Voigt notation:

```math
\begin{pmatrix} P_x \\ P_y \\ P_z \end{pmatrix}_{pz} =
\begin{pmatrix}
0 & 0 & 0 & 0 & e_{15} & 0
\\ 0 & 0 & 0 & e_{15} & 0 & 0
\\ e_{31} & e_{31} & e_{33} & 0 & 0 & 0
\end{pmatrix}
\begin{pmatrix}
\varepsilon_{xx} \\ \varepsilon_{yy} \\ \varepsilon_{zz} \\ 2\varepsilon_{yz} \\ 2\varepsilon_{xz} \\ 2\varepsilon_{xy}
\end{pmatrix}. \tag{4.4}
```

For the pseudomorphic c-plane strain state (3.3) the shear components vanish, so only $`P_z`$ survives:

```math
P_{pz} = e_{31}\,(\varepsilon_{xx} + \varepsilon_{yy}) + e_{33}\,\varepsilon_{zz}. \tag{4.5}
```

Inserting (3.5):

```math
P_{pz} = 2\,\varepsilon_{xx}\left(e_{31} - e_{33}\,\frac{C_{13}}{C_{33}}\right). \tag{4.6}
```

The bracket is negative for GaN, AlN and InN. Therefore:

| Strain | $`P_{pz}`$ | Relative to $`P_{sp}`$ |
|---|---|---|
| tensile ($`\varepsilon_{xx} \gt 0`$), e.g. AlGaN on GaN | negative | adds |
| compressive ($`\varepsilon_{xx} \lt 0`$), e.g. InGaN on GaN | positive | opposes |

## 4.3 Interface charge

Polarization is a bound-charge distribution, $`\rho_{pol} = -\nabla\cdot\mathbf{P}`$. In one dimension:

```math
\rho_{pol}(z) = -\frac{dP}{dz}. \tag{4.7}
```

At an abrupt interface between a lower material 1 and an upper material 2 this is a sheet charge

```math
\sigma_{pol} = P_1 - P_2 . \tag{4.8}
```

| Interface (upper on lower) | Sign of $`\sigma_{pol}`$ | Mobile charge it attracts |
|---|---|---|
| AlGaN on GaN | $`+`$ | electrons (2DEG) |
| InAlN on GaN | $`+`$ | electrons |
| GaN on AlGaN | $`-`$ | holes (2DHG) |
| GaN on AlN | $`-`$ | holes |
| InGaN on GaN | $`-`$ | holes |

**Example.** Al<sub>0.3</sub>Ga<sub>0.7</sub>N on relaxed GaN.

| Quantity | GaN | Al<sub>0.3</sub>Ga<sub>0.7</sub>N |
|---|---|---|
| $`\varepsilon_{xx}`$ | 0 | $`+0.0073`$ |
| $`P_{sp}`$ (C/m²) | $`-0.0340`$ | $`-0.0464`$ |
| $`P_{pz}`$ (C/m²) | 0 | $`-0.0114`$ |
| $`P`$ (C/m²) | $`-0.0340`$ | $`-0.0578`$ |

$`\sigma_{pol} = 0.0238`$ C/m² $`= 1.49\times10^{13}\,q`$/cm².

## 4.4 Volume charge in graded layers

Where the composition varies continuously, (4.7) gives a volume charge. For a linear grade from
$`P_{start}`$ to $`P_{end}`$ over a thickness $`d`$:

```math
\rho_{pol} = -\frac{P_{end} - P_{start}}{d}. \tag{4.9}
```

This is polarization doping. Grading GaN up to Al<sub>0.3</sub>Ga<sub>0.7</sub>N over 100 nm gives a uniform
positive charge of $`1.5\times10^{18}\,q`$/cm³, compensated by a three-dimensional slab of electrons
with no donors present. Grading in the opposite direction gives a hole slab.

## 4.5 Discrete form

Applying the rule (1.3) with flux $`-P`$:

```math
\rho_{pol,i}\; w_i = -\tfrac12\left(P_{i+1} - P_{i-1}\right). \tag{4.10}
```

Summing (4.10) over the nodes around an interface gives exactly $`P_1 - P_2`$ whatever the local
grid spacing, so the sheet charge (4.8) is reproduced exactly and the 2DEG density does not depend
on the mesh. The same formula covers abrupt and graded regions; no interface has to be identified
for the solver.

> [!NOTE]
> The polarization charge is computed once, before the self-consistent loop, and enters Poisson's
> equation as a fixed charge. It does not depend on $`\phi`$.

## 4.6 Charges that are not polarization

Two further fixed or semi-fixed charges can be placed at an interface by the user:

| Object | Charge | Depends on $`E_F`$? |
|---|---|---|
| `SurfaceCharge` | trap states, section 5.4 | yes |
| `InterfaceDipole` | two sheets $`\pm\sigma`$ separated by $`d`$; potential step $`\sigma d / \varepsilon_0\varepsilon_r`$ | no |

## 4.7 Scope

- Metal-polar c-plane only. N-polar, semi-polar and non-polar orientations are not available.
- Linear piezoelectricity; no second-order terms.
- No screening of $`\sigma_{pol}`$ by interface traps unless a `SurfaceCharge` is added.
