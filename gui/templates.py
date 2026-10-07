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


def _xhemt() -> DeviceModel:
    """AlN/GaN/AlN XHEMT on single-crystal AlN: Kim et al., arXiv:2506.16670
    (2025). 500 nm AlN buffer, 20 nm coherently strained GaN channel with a
    sheet of 5e13 cm^-2 Si donors 1 nm above its lower interface, 6 nm AlN
    barrier, 1 nm GaN cap. The surface barrier is not given in the paper;
    1.0 eV is assumed."""
    m = DeviceModel()
    m.layers = [AbruptLayer(x_Al=1.0, thickness_nm=480),
                QuantumRegionMarker('start'),
                AbruptLayer(x_Al=1.0, thickness_nm=20, dx_nm=0.2),
                AbruptLayer(x_Al=0.0, thickness_nm=1, dx_nm=0.1),
                AbruptLayer(x_Al=0.0, thickness_nm=1, n_doping=5e20, dx_nm=0.1),   # Si delta-doping, 5e13 cm^-2
                AbruptLayer(x_Al=0.0, thickness_nm=18, dx_nm=0.1),
                AbruptLayer(x_Al=1.0, thickness_nm=6, dx_nm=0.1),
                AbruptLayer(x_Al=0.0, thickness_nm=1, dx_nm=0.1),
                QuantumRegionMarker('end')]
    # undoped AlN below: Fermi level near mid-gap
    m.bottom_contact = Contact('bottom', 'schottky', 'Ni', barrier_eV=3.0)
    m.top_contact = Contact('top', 'schottky', 'Ni', barrier_eV=1.0)
    m.settings.dx_nm = 1.0
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


def _gaas_hemt() -> DeviceModel:
    """Modulation-doped Al0.3Ga0.7As/GaAs heterostructure: Si delta-doping
    behind a 10 nm spacer, GaAs cap, Schottky gate."""
    m = DeviceModel()
    m.layers = [AbruptLayer(x_Al=0.0, thickness_nm=400, material='AlGaAs', p_doping=1e14),
                QuantumRegionMarker(boundary='start'),
                AbruptLayer(x_Al=0.0, thickness_nm=40, material='AlGaAs', dx_nm=0.2),
                AbruptLayer(x_Al=0.3, thickness_nm=10, material='AlGaAs', dx_nm=0.2),      # spacer
                QuantumRegionMarker(boundary='end'),
                AbruptLayer(x_Al=0.3, thickness_nm=2, material='AlGaAs', n_doping=2.5e19, dx_nm=0.2),   # Si delta, 5e12 cm^-2
                AbruptLayer(x_Al=0.3, thickness_nm=25, material='AlGaAs', dx_nm=0.5),
                AbruptLayer(x_Al=0.0, thickness_nm=5, material='AlGaAs', dx_nm=0.5)]       # GaAs cap
    m.bottom_contact = Contact('bottom', 'ohmic', 'Ti')
    m.top_contact = Contact('top', 'schottky', 'Au', barrier_eV=0.8)
    m.settings.dx_nm = 2.0
    m.settings.quantum = True
    m.settings.max_iter = 500
    return m


def _inp_well() -> DeviceModel:
    """8 nm In0.53Ga0.47As well lattice-matched to InP: 1.55 um emission."""
    m = DeviceModel()
    m.layers = [AbruptLayer(x_Al=0.0, x_In=1.0, thickness_nm=100, material='InGaP'),
                QuantumRegionMarker(boundary='start'),
                AbruptLayer(x_Al=0.0, x_In=1.0, thickness_nm=10, material='InGaP', dx_nm=0.1),
                AbruptLayer(x_Al=0.0, x_In=0.53, thickness_nm=8, material='InGaAs', dx_nm=0.1),
                AbruptLayer(x_Al=0.0, x_In=1.0, thickness_nm=10, material='InGaP', dx_nm=0.1),
                QuantumRegionMarker(boundary='end'),
                AbruptLayer(x_Al=0.0, x_In=1.0, thickness_nm=100, material='InGaP')]
    m.bottom_contact = Contact('bottom', 'ohmic', 'Ti')
    m.top_contact = Contact('top', 'ohmic', 'Au')
    m.settings.dx_nm = 1.0
    m.settings.quantum = True
    m.settings.max_iter = 500
    return m


def _gaas_led() -> DeviceModel:
    """Al0.3Ga0.7As / GaAs / Al0.3Ga0.7As double heterostructure under
    forward bias: where the carriers recombine, and by which mechanism."""
    m = DeviceModel()
    m.layers = [AbruptLayer(x_Al=0.3, thickness_nm=200, material='AlGaAs', n_doping=1e18),
                AbruptLayer(x_Al=0.0, thickness_nm=100, material='AlGaAs'),
                AbruptLayer(x_Al=0.3, thickness_nm=200, material='AlGaAs', p_doping=1e18)]
    m.bottom_contact = Contact('bottom', 'ohmic', 'Ti')
    m.top_contact = Contact('top', 'ohmic', 'Au')
    m.settings.dx_nm = 1.0
    m.settings.quantum = False
    m.settings.V_applied = 1.3
    m.settings.max_iter = 500
    return m


TEMPLATES: List[Template] = [
    Template("hemt", "AlGaN/GaN HEMT",
             "25 nm Al₀.₃Ga₀.₇N barrier on GaN. Polarization-induced "
             "two-dimensional electron gas with quantized subbands.",
             ("2DEG", "Schrödinger–Poisson"), _hemt),
    Template("xhemt", "AlN/GaN/AlN XHEMT",
             "20 nm strained GaN channel between AlN on an AlN substrate, with Si "
             "δ-doping that removes the hole gas (Kim et al., 2025).",
             ("2DEG", "AlN substrate"), _xhemt),
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
    Template("gaas_hemt", "AlGaAs/GaAs HEMT",
             "Modulation-doped Al₀.₃Ga₀.₇As/GaAs heterostructure with Si δ-doping "
             "behind a 10 nm spacer. Two-dimensional electron gas without polarization.",
             ("2DEG", "Arsenides"), _gaas_hemt),
    Template("inp_well", "InGaAs/InP quantum well",
             "8 nm In₀.₅₃Ga₀.₄₇As well lattice-matched to InP. Confined states "
             "and the 1.55 µm transition.",
             ("Optics", "Phosphides"), _inp_well),
    Template("gaas_led", "AlGaAs/GaAs double heterostructure",
             "GaAs active layer between doped Al₀.₃Ga₀.₇As at 1.3 V forward bias. "
             "SRH, radiative and Auger recombination and the radiative efficiency.",
             ("Recombination", "Bias"), _gaas_led),
]


def get(key: str) -> Template:
    return next(t for t in TEMPLATES if t.key == key)
