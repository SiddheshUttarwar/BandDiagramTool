"""
Layer and contact dataclasses for building AlGaN device structures.

All layers are AlxGa(1-x)N with x_Al in [0, 1].
  x_Al = 0.0  →  GaN
  x_Al = 1.0  →  AlN

Supported grading profiles:
  'abrupt'    - constant composition (same as AbruptLayer)
  'linear'    - x varies linearly from x_Al_start to x_Al_end
  'parabolic' - x varies as a quadratic function
  'stepped'   - n_steps discrete uniform sub-layers, x linearly spaced
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AbruptLayer:
    """Single AlGaN layer with uniform composition."""
    x_Al: float            # Al composition [0, 1]
    thickness_nm: float    # Layer thickness [nm]
    n_doping: float = 0.0  # n-type (donor) doping concentration [cm^-3]
    p_doping: float = 0.0  # p-type (acceptor) doping concentration [cm^-3]
    relaxed: bool = False  # True if this layer is strain-relaxed (not pseudomorphic)
    dx_nm: Optional[float] = None  # grid spacing for this layer [nm]; None = use the device default


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
    dx_nm: Optional[float] = None  # grid spacing for this layer [nm]; None = use the device default


@dataclass
class Contact:
    """Metal contact on one side of the device."""
    position: str       # 'bottom' (substrate side) | 'top' (surface side)
    contact_type: str   # 'ohmic' | 'schottky'
    metal: str          # e.g. 'Ni', 'Ti', 'Au', 'Al', 'Pt'


# Union type alias
Layer = AbruptLayer | GradedLayer
