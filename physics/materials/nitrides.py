"""
Material classes for the III-nitride alloys the tool models:

    AlGaN    Al(x)Ga(1-x)N                    (x_Al)
    InGaN    In(y)Ga(1-y)N                    (x_In)
    InAlGaN  Al(x)In(y)Ga(1-x-y)N quaternary  (x_Al, x_In) -- also covers
             AlInN (x_Al + x_In = 1), e.g. lattice-matched Al0.83In0.17N

Each class owns its composition, validation and display name, and returns
its material parameters. All three are views onto ONE physical parameter
set (physics.materials.algan.get_nitride_params): the binaries GaN/AlN/InN
plus pairwise bowing. That keeps the band alignment between neighbouring
layers of different classes consistent -- e.g. InAlGaN(x_Al=0.3, x_In=0)
is exactly AlGaN(0.3), and InAlGaN(0, 0.15) is exactly InGaN(0.15).

Example
-------
    from physics.materials.nitrides import AlGaN, InGaN, InAlGaN
    InGaN(0.15).band_gap()            # 2.806 eV
    InAlGaN(0.83, 0.17).name          # 'Al0.83In0.17N'
    AlGaN(0.3).params().Psp           # spontaneous polarization [C/m^2]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from physics.materials.algan import (
    AlGaNParams, get_nitride_params_T, nitride_name,
)


class NitrideMaterial:
    """Common interface of AlGaN / InGaN / InAlGaN."""

    kind: str = ''

    @property
    def composition(self) -> Tuple[float, float]:
        """(x_Al, x_In) of the quaternary representation."""
        raise NotImplementedError

    @property
    def x_Al(self) -> float:
        return self.composition[0]

    @property
    def x_In(self) -> float:
        return self.composition[1]

    @property
    def x_Ga(self) -> float:
        return 1.0 - self.x_Al - self.x_In

    @property
    def name(self) -> str:
        return nitride_name(*self.composition)

    def params(self, T: float = 300.0) -> AlGaNParams:
        """Full material parameter set at temperature T [K]."""
        return get_nitride_params_T(self.x_Al, self.x_In, T)

    def band_gap(self, T: float = 300.0) -> float:
        """Unstrained band gap to the top valence band [eV]."""
        return self.params(T).Eg

    def lattice_constant(self) -> Tuple[float, float]:
        """Relaxed (a, c) [Angstrom]."""
        p = self.params()
        return p.a0, p.c0

    def __str__(self) -> str:
        return self.name


def _frac(name: str, v: float) -> float:
    v = float(v)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f'{name} must be in [0, 1], got {v}')
    return v


@dataclass(frozen=True)
class AlGaN(NitrideMaterial):
    """Al(x)Ga(1-x)N, x = x_Al in [0, 1] (x=0 GaN, x=1 AlN)."""
    x_Al_frac: float = 0.0
    kind = 'AlGaN'

    def __post_init__(self):
        _frac('x_Al', self.x_Al_frac)

    @property
    def composition(self):
        return (float(self.x_Al_frac), 0.0)


@dataclass(frozen=True)
class InGaN(NitrideMaterial):
    """In(y)Ga(1-y)N, y = x_In in [0, 1] (y=0 GaN, y=1 InN)."""
    x_In_frac: float = 0.0
    kind = 'InGaN'

    def __post_init__(self):
        _frac('x_In', self.x_In_frac)

    @property
    def composition(self):
        return (0.0, float(self.x_In_frac))


@dataclass(frozen=True)
class InAlGaN(NitrideMaterial):
    """Al(x)In(y)Ga(1-x-y)N with x + y <= 1. x + y = 1 is AlInN."""
    x_Al_frac: float = 0.0
    x_In_frac: float = 0.0
    kind = 'InAlGaN'

    def __post_init__(self):
        _frac('x_Al', self.x_Al_frac)
        _frac('x_In', self.x_In_frac)
        if self.x_Al_frac + self.x_In_frac > 1.0 + 1e-9:
            raise ValueError(f'x_Al + x_In = {self.x_Al_frac + self.x_In_frac:.3f} > 1')

    @property
    def composition(self):
        return (float(self.x_Al_frac), float(self.x_In_frac))


MATERIALS = {'AlGaN': AlGaN, 'InGaN': InGaN, 'InAlGaN': InAlGaN}


def make_material(kind: str, x_Al: float = 0.0, x_In: float = 0.0) -> NitrideMaterial:
    """Build a material of the given class from (x_Al, x_In), rejecting a
    composition the class cannot represent (In in AlGaN, Al in InGaN)."""
    if kind not in MATERIALS:
        raise ValueError(f"unknown material '{kind}'; choose one of {list(MATERIALS)}")
    if kind == 'AlGaN':
        if x_In:
            raise ValueError('AlGaN cannot contain In; use InAlGaN')
        return AlGaN(x_Al)
    if kind == 'InGaN':
        if x_Al:
            raise ValueError('InGaN cannot contain Al; use InAlGaN')
        return InGaN(x_In)
    return InAlGaN(x_Al, x_In)


def infer_material_kind(x_Al: float, x_In: float) -> str:
    """Simplest class that can represent the composition."""
    if x_In > 0 and x_Al > 0:
        return 'InAlGaN'
    if x_In > 0:
        return 'InGaN'
    return 'AlGaN'
