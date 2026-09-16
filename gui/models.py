"""
DeviceModel: the single source of truth for the device being edited in the
GUI (layer stack, contacts, and solve settings), with a simple
observer/callback mechanism so panels can react to edits without being
tightly coupled to each other.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Union

from devices.layer import (
    AbruptLayer, GradedLayer, Contact,
    QuantumRegionMarker, SurfaceCharge, InterfaceDipole,
)

Layer = Union[AbruptLayer, GradedLayer, QuantumRegionMarker, SurfaceCharge, InterfaceDipole]


@dataclass
class SolveSettings:
    T: float = 300.0
    dx_nm: float = 0.2
    quantum: bool = True
    include_spontaneous_polarization: bool = False
    n_states_e: int = 16
    n_states_h: int = 16
    max_iter: int = 5000
    tol: float = 1e-6
    alpha: float = 0.15

    # Bias / sweep
    sweep_mode: bool = False
    V_applied: float = 0.0
    V_start: float = 0.0
    V_stop: float = 1.0
    n_steps: int = 10


class DeviceModel:
    """
    Holds the device-under-edit: an ordered layer list (bottom -> top, same
    convention as devices.device.AlGaNDevice), two contacts, and solve
    settings. Fires registered listeners on every change so the GUI can
    refresh views / debounce a re-solve without each panel polling.
    """

    def __init__(self) -> None:
        self.layers: List[Layer] = []
        self.bottom_contact = Contact(position='bottom', contact_type='ohmic', metal='Ti')
        self.top_contact = Contact(position='top', contact_type='ohmic', metal='Ni')
        self.settings = SolveSettings()
        self._listeners: List[Callable[[], None]] = []

    # ------------------------------------------------------------------
    def add_listener(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)

    def notify(self) -> None:
        for cb in list(self._listeners):
            cb()

    # ------------------------------------------------------------------
    # Layer operations (all in bottom->top index space, matching
    # devices.device.AlGaNDevice's `layers` convention)
    # ------------------------------------------------------------------
    def add_layer(self, layer: Layer, index: Optional[int] = None) -> None:
        if index is None:
            self.layers.append(layer)
        else:
            self.layers.insert(index, layer)
        self.notify()

    def remove_layer(self, index: int) -> None:
        del self.layers[index]
        self.notify()

    def move_layer(self, from_index: int, to_index: int) -> None:
        if from_index == to_index:
            return
        layer = self.layers.pop(from_index)
        self.layers.insert(to_index, layer)
        self.notify()

    def replace_layer(self, index: int, layer: Layer) -> None:
        self.layers[index] = layer
        self.notify()

    # ------------------------------------------------------------------
    @property
    def contacts(self) -> List[Contact]:
        return [self.bottom_contact, self.top_contact]

    def set_bottom_contact(self, contact: Contact) -> None:
        self.bottom_contact = contact
        self.notify()

    def set_top_contact(self, contact: Contact) -> None:
        self.top_contact = contact
        self.notify()

    # ------------------------------------------------------------------
    def is_solvable(self) -> bool:
        return len(self.layers) > 0

    def clone_settings(self) -> SolveSettings:
        return replace(self.settings)
