"""
Device templates offered on the start page: complete, solvable structures a
new user can open, run and then edit. Each builds a fresh DeviceModel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

from devices.layer import AbruptLayer, Contact, QuantumRegionMarker
from gui.models import DeviceModel


@dataclass(frozen=True)
class Template:
    key: str
    title: str
    description: str
    tags: Tuple[str, ...]
    build: Callable[[], DeviceModel]


def _hemt() -> DeviceModel:
    m = DeviceModel()
    m.layers = [AbruptLayer(x_Al=0.0, thickness_nm=180, n_doping=1e16),
                QuantumRegionMarker('start'),
                AbruptLayer(x_Al=0.0, thickness_nm=20, dx_nm=0.1),
                AbruptLayer(x_Al=0.3, thickness_nm=25, dx_nm=0.1),
                QuantumRegionMarker('end')]
    m.bottom_contact = Contact('bottom', 'ohmic', 'Ti')
    # free surface: Fermi level pinned 0.84 + 1.3 x eV below Ec (Ambacher 2000)
    m.top_contact = Contact('top', 'schottky', 'Ni', barrier_eV=1.23)
    m.settings.dx_nm = 0.5
    m.settings.quantum = True
    return m


def _stark_well() -> DeviceModel:
    m = DeviceModel()
    m.layers = [AbruptLayer(x_Al=1.0, thickness_nm=60),
                QuantumRegionMarker('start'),
                AbruptLayer(x_Al=1.0, thickness_nm=8, dx_nm=0.05),
                AbruptLayer(x_Al=0.0, thickness_nm=2.6, dx_nm=0.02),
                AbruptLayer(x_Al=1.0, thickness_nm=8, dx_nm=0.05),
                QuantumRegionMarker('end'),
                AbruptLayer(x_Al=1.0, thickness_nm=30)]
    # insulating AlN on both sides: Fermi level near mid-gap
    m.bottom_contact = Contact('bottom', 'schottky', 'Ni', barrier_eV=3.0)
    m.top_contact = Contact('top', 'schottky', 'Ni', barrier_eV=3.0)
    m.settings.dx_nm = 0.2
    m.settings.quantum = True
    return m


def _pn_diode() -> DeviceModel:
    m = DeviceModel()
    m.layers = [AbruptLayer(x_Al=0.0, thickness_nm=300, n_doping=2e18),
                AbruptLayer(x_Al=0.0, thickness_nm=300, p_doping=2e19)]
    m.bottom_contact = Contact('bottom', 'ohmic', 'Ti')
    m.top_contact = Contact('top', 'ohmic', 'Ni')
    m.settings.dx_nm = 1.0
    m.settings.quantum = False
    m.settings.V_applied = 2.9
    return m


def _uv_led() -> DeviceModel:
    m = DeviceModel()
    layers: List = [AbruptLayer(x_Al=0.65, thickness_nm=195, n_doping=1e19),
                    QuantumRegionMarker('start')]
    for _ in range(5):                                   # multiple quantum wells
        layers += [AbruptLayer(x_Al=0.53, thickness_nm=1.0),
                   AbruptLayer(x_Al=0.70, thickness_nm=9.9)]
    layers += [QuantumRegionMarker('end'),
               AbruptLayer(x_Al=0.68, thickness_nm=13),
               AbruptLayer(x_Al=0.72, thickness_nm=6, p_doping=1e19)]      # electron-blocking layer
    for _ in range(5):                                   # p-type superlattice
        layers += [AbruptLayer(x_Al=0.65, thickness_nm=2.0, p_doping=3e19),
                   AbruptLayer(x_Al=0.55, thickness_nm=2.0, p_doping=3e19)]
    layers.append(AbruptLayer(x_Al=0.0, thickness_nm=10.0, p_doping=1e20))  # p-GaN contact
    m.layers = layers
    m.bottom_contact = Contact('bottom', 'ohmic', 'Ti')
    m.top_contact = Contact('top', 'ohmic', 'Ni')
    m.settings.dx_nm = 1.0
    m.settings.quantum = True
    m.settings.max_iter = 500
    return m


TEMPLATES: List[Template] = [
    Template("hemt", "AlGaN/GaN HEMT",
             "25 nm Al₀.₃Ga₀.₇N barrier on GaN. Polarization-induced "
             "two-dimensional electron gas with quantized subbands.",
             ("2DEG", "Schrödinger–Poisson"), _hemt),
    Template("stark", "GaN/AlN quantum well",
             "2.6 nm GaN well in AlN. Built-in field of several MV/cm, "
             "Stark-shifted transition and electron–hole overlap.",
             ("QCSE", "Optics"), _stark_well),
    Template("uvled", "AlGaN deep-UV LED",
             "Five-period multiple quantum well, electron-blocking layer "
             "and p-type superlattice on n-Al₀.₆₅Ga₀.₃₅N.",
             ("MQW", "LED"), _uv_led),
    Template("pn", "GaN p–n diode",
             "Abrupt junction at 2.9 V forward bias. Drift-diffusion with "
             "quasi-Fermi level splitting and conserved current.",
             ("Transport", "Bias"), _pn_diode),
]


def get(key: str) -> Template:
    return next(t for t in TEMPLATES if t.key == key)
