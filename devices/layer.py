"""
Layer and contact dataclasses for building III-nitride device structures.

Every physical layer is one of three materials (physics.materials.nitrides):
  'AlGaN'    Al(x)Ga(1-x)N          composition x_Al      (x_In must be 0)
  'InGaN'    In(y)Ga(1-y)N          composition x_In      (x_Al must be 0)
  'InAlGaN'  Al(x)In(y)Ga(1-x-y)N   composition x_Al, x_In (x + y <= 1;
             x + y = 1 is AlInN, e.g. lattice-matched Al0.83In0.17N)
`material` defaults to the simplest class that fits the given fractions,
so existing AlGaN-only code and saved projects behave exactly as before.
  x_Al = 0, x_In = 0  ->  GaN;  x_Al = 1 -> AlN;  x_In = 1 -> InN

Supported grading profiles:
  'abrupt'    - constant composition (same as AbruptLayer)
  'linear'    - x varies linearly from x_Al_start to x_Al_end
  'parabolic' - x varies as a quadratic function
  'stepped'   - n_steps discrete uniform sub-layers, x linearly spaced

In addition to physical layers (AbruptLayer, GradedLayer), the device stack
can contain zero-thickness *interface layers* inserted at any position:
  QuantumRegionMarker - marks where the Schrodinger-solved quantum region
                         starts/ends (an explicit alternative to the
                         solver's automatic undoped-span heuristic)
  SurfaceCharge       - one or more donor-/acceptor-like trap states at an
                         interface, ionizing self-consistently with the
                         local Fermi level (areal analogue of bulk doping)
  InterfaceDipole     - a fixed structural dipole at an interface: two
                         equal-and-opposite sheet charges separated by a
                         user-defined distance
These consume no thickness/grid points of their own; their position in the
device is simply their position in the `layers` list, between whichever
physical layers precede and follow them.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from physics.materials.nitrides import (
    NitrideMaterial, make_material, infer_material_kind,
)


def _resolve_material(layer, pairs):
    """Infer (if None) and validate layer.material against every
    (x_Al, x_In) composition the layer spans (both ends if graded)."""
    if layer.material is None:
        kinds = {infer_material_kind(a, i) for a, i in pairs}
        layer.material = ('InAlGaN' if len(kinds) > 1 or 'InAlGaN' in kinds
                          else kinds.pop())
    for a, i in pairs:
        make_material(layer.material, a, i)   # raises ValueError if invalid


@dataclass
class AbruptLayer:
    """Single AlGaN layer with uniform composition."""
    x_Al: float            # Al composition [0, 1]
    thickness_nm: float    # Layer thickness [nm]
    n_doping: float = 0.0  # n-type (donor) doping concentration [cm^-3]
    p_doping: float = 0.0  # p-type (acceptor) doping concentration [cm^-3]
    relaxed: bool = False  # True if this layer is strain-relaxed (not pseudomorphic)
    custom_strain_xx: Optional[float] = None  # user-specified in-plane
        # biaxial strain (dimensionless; negative = compressive, positive =
        # tensile), overriding both the pseudomorphic-to-substrate
        # calculation and `relaxed` for this layer. eps_zz is still derived
        # from this via the usual elastic (Poisson) relation, not set
        # independently -- see physics.polarization.eps_zz_from_eps_xx.
        # None (default) = use `relaxed` / substrate-pseudomorphic as before.
    series_resistance: float = 0.0  # this layer's own contribution to the
        # device's total series resistance [Ohm*cm^2] (devices.device.
        # AlGaNDevice.total_series_resistance sums every layer's value,
        # since resistances of layers stacked in series add). 0.0 (default,
        # "no value given") means this layer contributes none.
    dx_nm: Optional[float] = None  # grid spacing for this layer [nm]; None = use the device default
    x_In: float = 0.0              # In fraction (InGaN / InAlGaN layers)
    material: Optional[str] = None  # 'AlGaN' | 'InGaN' | 'InAlGaN'; None = infer

    def __post_init__(self):
        _resolve_material(self, [(self.x_Al, self.x_In)])

    @property
    def nitride(self) -> NitrideMaterial:
        """The layer's material object (physics.materials.nitrides)."""
        return make_material(self.material, self.x_Al, self.x_In)

    @classmethod
    def from_material(cls, mat: NitrideMaterial, thickness_nm: float, **kw) -> "AbruptLayer":
        """AbruptLayer.from_material(InGaN(0.15), 3.0, n_doping=0)."""
        return cls(x_Al=mat.x_Al, thickness_nm=thickness_nm, x_In=mat.x_In,
                   material=mat.kind, **kw)


@dataclass
class GradedLayer:
    """
    AlGaN layer with a spatially varying Al composition.
    Composition runs from x_Al_start (substrate side) to x_Al_end (surface side).
    """
    x_Al_start: float      # Al composition at bottom of layer [0, 1]
    x_Al_end: float        # Al composition at top of layer [0, 1]
    thickness_nm: float    # Layer thickness [nm]
    n_doping: float = 0.0
    p_doping: float = 0.0
    profile: str = 'linear'  # 'linear' | 'parabolic' | 'stepped' | 'abrupt'
    n_steps: int = 10        # number of discrete steps for 'stepped' profile
    relaxed: bool = False
    custom_strain_xx: Optional[float] = None  # see AbruptLayer.custom_strain_xx
    series_resistance: float = 0.0  # see AbruptLayer.series_resistance
    dx_nm: Optional[float] = None  # grid spacing for this layer [nm]; None = use the device default
    x_In_start: float = 0.0        # In fraction at bottom of layer
    x_In_end: float = 0.0          # In fraction at top of layer
    material: Optional[str] = None  # 'AlGaN' | 'InGaN' | 'InAlGaN'; None = infer

    def __post_init__(self):
        _resolve_material(self, [(self.x_Al_start, self.x_In_start),
                                 (self.x_Al_end, self.x_In_end)])

    @classmethod
    def from_materials(cls, start: NitrideMaterial, end: NitrideMaterial,
                       thickness_nm: float, **kw) -> "GradedLayer":
        """Graded layer between two materials of the same family, e.g.
        GradedLayer.from_materials(InGaN(0.0), InGaN(0.1), 20)."""
        kind = start.kind if start.kind == end.kind else 'InAlGaN'
        return cls(x_Al_start=start.x_Al, x_Al_end=end.x_Al, thickness_nm=thickness_nm,
                   x_In_start=start.x_In, x_In_end=end.x_In, material=kind, **kw)


@dataclass
class QuantumRegionMarker:
    """
    Zero-thickness marker inserted into the device stack indicating where
    the quantum (Schrodinger-solved) region starts or ends. Insert one
    'start' marker and one 'end' marker at the desired positions; the
    quantum region used by the solver spans 2nm before the start marker to
    2nm after the end marker. Without both markers present, the solver
    falls back to its automatic undoped-span heuristic.
    """
    boundary: str = 'start'   # 'start' | 'end'


@dataclass
class SurfaceState:
    """One donor-like or acceptor-like trap energy state at an interface."""
    density_cm2: float         # areal trap density [cm^-2]
    energy_eV: float = 0.1     # activation energy: depth below Ec (donor) or height above Ev (acceptor) [eV]
    state_type: str = 'donor'  # 'donor' | 'acceptor'


@dataclass
class SurfaceCharge:
    """
    Zero-thickness interface/surface charge: one or more donor-like or
    acceptor-like trap states at a single interface, each ionizing
    self-consistently according to Fermi-Dirac occupation relative to the
    local Fermi level -- the same physics as bulk donor/acceptor ionization
    (see physics.self_consistent), but as a 2D areal density concentrated
    at one grid point rather than a 3D volume density spread over a layer.
    A donor-like state is neutral when occupied by an electron and +q when
    ionized (empty); an acceptor-like state is neutral when empty and -q
    when ionized (occupied).
    """
    states: List[SurfaceState] = field(default_factory=list)


@dataclass
class InterfaceDipole:
    """
    Zero-thickness interface dipole: two fixed, equal-and-opposite sheet
    charges +/-sigma separated by `separation_nm`, producing a built-in
    potential step across the interface. Unlike SurfaceCharge this is a
    fixed structural charge (e.g. interface reconstruction), not
    Fermi-level-dependent.
    """
    sheet_charge_C_m2: float    # magnitude of each sheet's areal charge [C/m^2]; sign sets polarity
    separation_nm: float = 0.5  # distance between the +/- sheets [nm]


@dataclass
class Contact:
    """Metal contact on one side of the device."""
    position: str       # 'bottom' (substrate side) | 'top' (surface side)
    contact_type: str   # 'ohmic' | 'schottky'
    metal: str          # e.g. 'Ni', 'Ti', 'Au', 'Al', 'Pt'
    barrier_eV: Optional[float] = None  # Schottky only: explicit barrier
        # height Ec - Ef at the metal [eV] (measured / surface-pinned value);
        # None = Schottky-Mott estimate work_function(metal) - chi.


# Union type alias
PhysicalLayer = AbruptLayer | GradedLayer
InterfaceLayer = QuantumRegionMarker | SurfaceCharge | InterfaceDipole
Layer = PhysicalLayer | InterfaceLayer
