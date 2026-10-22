"""
One entry point for every material the tool models, in both crystal systems:

    wurtzite   (c-plane)   Al(x)In(y)Ga(1-x-y)N              physics.materials.algan
    zincblende ((001))     Al(x)In(y)Ga(1-x-y)As(1-v)P(v)    physics.materials.zincblende
    monoclinic ((010))     beta-(Al(x)Ga(1-x))2O3            physics.materials.gallium_oxide

A composition is the triple (x_Al, x_In, x_P): the group-III fractions of Al
and In (the rest is Ga) and, for the zincblende family, the P fraction of
the group-V sublattice (the rest is As). x_P is unused for the nitrides.

Material classes -- the names used by devices.layer and the GUI -- say which
fractions are free; the others are fixed:

    AlGaN      x_Al                AlGaAs     x_Al
    InGaN      x_In                InGaAs     x_In
    InAlGaN    x_Al, x_In          AlGaInAs   x_Al, x_In   (x_Al + x_In = 1: InAlAs)
                                   GaAsP      x_P
                                   InGaP      x_In         (x_P = 1; x_In = 1: InP)
                                   AlGaInP    x_Al, x_In   (x_P = 1)
                                   InGaAsP    x_In, x_P
                                   AlGaInAsP  x_Al, x_In, x_P
    Ga2O3      x_Al  (beta-(Al(x)Ga(1-x))2O3; x_Al = 0: beta-Ga2O3)

A device is built from one crystal system: wurtzite, zincblende and
monoclinic layers cannot be mixed in a stack.

Example
-------
    from physics.materials.alloys import make_material
    make_material('InGaAs', x_In=0.53).name           # 'In0.53Ga0.47As'
    make_material('InGaP', x_In=1.0).band_gap()       # InP, 1.353 eV
    make_material('AlGaN', x_Al=0.3).params().Psp
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from physics.materials import nitrides as _nitrides
from physics.materials.algan import (
    get_nitride_params, get_nitride_params_T, nitride_name,
    donor_ionization_energy as _nitride_Ed, acceptor_ionization_energy as _nitride_Ea,
)
from physics.materials.zincblende import get_zincblende_params, zincblende_name
from physics.materials.gallium_oxide import get_gallium_oxide_params, gallium_oxide_name

WURTZITE = 'wurtzite'
ZINCBLENDE = 'zincblende'
MONOCLINIC = 'monoclinic'

# kind -> (crystal, free fractions, fixed x_P or None)
KINDS = {
    'AlGaN':     (WURTZITE, ('Al',), None),
    'InGaN':     (WURTZITE, ('In',), None),
    'InAlGaN':   (WURTZITE, ('Al', 'In'), None),
    'AlGaAs':    (ZINCBLENDE, ('Al',), 0.0),
    'InGaAs':    (ZINCBLENDE, ('In',), 0.0),
    'AlGaInAs':  (ZINCBLENDE, ('Al', 'In'), 0.0),
    'GaAsP':     (ZINCBLENDE, ('P',), None),
    'InGaP':     (ZINCBLENDE, ('In',), 1.0),
    'AlGaInP':   (ZINCBLENDE, ('Al', 'In'), 1.0),
    'InGaAsP':   (ZINCBLENDE, ('In', 'P'), None),
    'AlGaInAsP': (ZINCBLENDE, ('Al', 'In', 'P'), None),
    'Ga2O3':     (MONOCLINIC, ('Al',), None),
}
NITRIDE_KINDS = tuple(k for k, v in KINDS.items() if v[0] == WURTZITE)
ZINCBLENDE_KINDS = tuple(k for k, v in KINDS.items() if v[0] == ZINCBLENDE)
OXIDE_KINDS = tuple(k for k, v in KINDS.items() if v[0] == MONOCLINIC)


def _kind(kind: str):
    if kind not in KINDS:
        raise ValueError(f"unknown material '{kind}'; choose one of {list(KINDS)}")
    return KINDS[kind]


def crystal_of(kind) -> str:
    """Crystal system of a material class; None (inferred nitride) is wurtzite."""
    return WURTZITE if kind is None else _kind(kind)[0]


def free_fractions(kind: str) -> Tuple[str, ...]:
    """Which of 'Al', 'In', 'P' the material class lets the user set."""
    return _kind(kind)[1]


def normalize_composition(kind: str, x_Al: float = 0.0, x_In: float = 0.0, x_P: float = 0.0):
    """(x_Al, x_In, x_P) with the fractions the class does not use set to
    their fixed values -- for converting a layer from one class to another."""
    crystal, free, fixed_p = _kind(kind)
    x_Al = float(x_Al) if 'Al' in free else 0.0
    x_In = float(x_In) if 'In' in free else 0.0
    if crystal != ZINCBLENDE:
        x_P = 0.0
    elif 'P' not in free:
        x_P = float(fixed_p)
    else:
        x_P = float(x_P)
    if x_Al + x_In > 1.0:
        x_In = max(0.0, 1.0 - x_Al)
    return x_Al, x_In, x_P


def get_params(crystal: str, x_Al: float = 0.0, x_In: float = 0.0, x_P: float = 0.0,
               T: float = 300.0):
    """Material parameters at temperature T [K] in either crystal system."""
    if crystal == ZINCBLENDE:
        return get_zincblende_params(x_Al, x_In, x_P, T)
    if crystal == MONOCLINIC:
        return get_gallium_oxide_params(x_Al, T)
    if T == 300.0:
        return get_nitride_params(x_Al, x_In)
    return get_nitride_params_T(x_Al, x_In, T)


def alloy_name(crystal: str, x_Al: float = 0.0, x_In: float = 0.0, x_P: float = 0.0,
               digits: int = 2) -> str:
    if crystal == ZINCBLENDE:
        return zincblende_name(x_Al, x_In, x_P, digits)
    if crystal == MONOCLINIC:
        return gallium_oxide_name(x_Al, digits)
    return nitride_name(x_Al, x_In, digits)


def dopant_ionization_energies(crystal: str, x_Al: float, x_In: float = 0.0, x_P: float = 0.0,
                               T: float = 300.0) -> Tuple[float, float]:
    """(Ed, Ea) [eV]: shallow donor below the conduction band, acceptor above
    the valence band. Nitrides: Si and Mg (physics.materials.algan).
    Zincblende: hydrogenic donor and a typical shallow acceptor. Ga2O3:
    hydrogenic donor (Si) and the deep Mg acceptor."""
    if crystal in (ZINCBLENDE, MONOCLINIC):
        p = get_params(crystal, x_Al, x_In, x_P, T)
        return float(p.Ed), float(p.Ea)
    return float(_nitride_Ed(x_Al, x_In)), float(_nitride_Ea(x_Al, x_In))


def mobilities(crystal: str, x_Al: float, x_In: float = 0.0, x_P: float = 0.0) -> Tuple[float, float]:
    """Low-field (mu_n, mu_p) [cm^2/Vs] at 300 K. Nitrides: linear in the Al
    fraction between GaN (300 / 10) and AlN (25 / 2)."""
    if crystal in (ZINCBLENDE, MONOCLINIC):
        p = get_params(crystal, x_Al, x_In, x_P)
        return float(p.mu_n), float(p.mu_p)
    return 300.0 * (1.0 - x_Al) + 25.0 * x_Al, 10.0 * (1.0 - x_Al) + 2.0 * x_Al


RECOMBINATION_KEYS = ('tau_n', 'tau_p', 'B_rad', 'C_n', 'C_p')


def recombination_coefficients(crystal: str, x_Al: float, x_In: float = 0.0, x_P: float = 0.0) -> dict:
    """{'tau_n', 'tau_p' [s], 'B_rad' [cm^3/s], 'C_n', 'C_p' [cm^6/s]} of
    the SRH, radiative and Auger recombination models. Nitrides: one set
    for every composition (physics.drift_diffusion.DEFAULT_RECOMBINATION)."""
    if crystal in (ZINCBLENDE, MONOCLINIC):
        p = get_params(crystal, x_Al, x_In, x_P)
        return {'tau_n': p.tau_srh, 'tau_p': p.tau_srh, 'B_rad': p.B_rad, 'C_n': p.C_aug, 'C_p': p.C_aug}
    from physics.drift_diffusion import DEFAULT_RECOMBINATION
    return dict(DEFAULT_RECOMBINATION)


@dataclass(frozen=True)
class ZincblendeMaterial:
    """An arsenide / phosphide alloy of a given material class."""
    kind: str
    x_Al: float = 0.0
    x_In: float = 0.0
    x_P: float = 0.0
    crystal = ZINCBLENDE

    @property
    def composition(self) -> Tuple[float, float, float]:
        return (self.x_Al, self.x_In, self.x_P)

    @property
    def name(self) -> str:
        return zincblende_name(self.x_Al, self.x_In, self.x_P)

    def params(self, T: float = 300.0):
        return get_zincblende_params(self.x_Al, self.x_In, self.x_P, T)

    def band_gap(self, T: float = 300.0) -> float:
        """Unstrained gap to the lowest conduction valley [eV]."""
        return self.params(T).Eg

    def lattice_constant(self) -> Tuple[float, float]:
        p = self.params()
        return p.a0, p.c0

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class OxideMaterial:
    """beta-(Al(x)Ga(1-x))2O3."""
    kind: str
    x_Al: float = 0.0
    x_In: float = 0.0
    x_P: float = 0.0
    crystal = MONOCLINIC

    @property
    def composition(self) -> Tuple[float, float, float]:
        return (self.x_Al, 0.0, 0.0)

    @property
    def name(self) -> str:
        return gallium_oxide_name(self.x_Al)

    def params(self, T: float = 300.0):
        return get_gallium_oxide_params(self.x_Al, T)

    def band_gap(self, T: float = 300.0) -> float:
        return self.params(T).Eg

    def lattice_constant(self) -> Tuple[float, float]:
        p = self.params()
        return p.a0, p.c0

    def __str__(self) -> str:
        return self.name


def make_material(kind: str, x_Al: float = 0.0, x_In: float = 0.0, x_P: float = 0.0):
    """Material object of class `kind`, rejecting a composition the class
    cannot represent (e.g. In in AlGaAs, x_P != 1 in InGaP)."""
    crystal, free, fixed_p = _kind(kind)
    if crystal == WURTZITE:
        if x_P:
            raise ValueError(f'{kind} is a nitride and takes no x_P')
        return _nitrides.make_material(kind, x_Al, x_In)
    if crystal == MONOCLINIC:
        if x_In or x_P:
            raise ValueError(f'{kind} contains only Al and Ga')
        if not 0.0 <= float(x_Al) <= 1.0:
            raise ValueError(f'x_Al must be in [0, 1], got {x_Al}')
        return OxideMaterial(kind, float(x_Al))
    for el, v in (('Al', x_Al), ('In', x_In), ('P', x_P)):
        if not 0.0 <= float(v) <= 1.0:
            raise ValueError(f'x_{el} must be in [0, 1], got {v}')
    if x_Al and 'Al' not in free:
        raise ValueError(f'{kind} cannot contain Al')
    if x_In and 'In' not in free:
        raise ValueError(f'{kind} cannot contain In')
    if 'P' not in free and abs(float(x_P) - fixed_p) > 1e-9:
        raise ValueError(f'{kind} has x_P = {fixed_p:g}, got {x_P}')
    if x_Al + x_In > 1.0 + 1e-9:
        raise ValueError(f'x_Al + x_In = {x_Al + x_In:.3f} > 1')
    return ZincblendeMaterial(kind, float(x_Al), float(x_In), float(x_P))
