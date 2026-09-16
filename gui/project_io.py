"""
JSON save/load for a DeviceModel (layers + contacts + settings).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict

from devices.layer import (
    AbruptLayer, GradedLayer, Contact,
    QuantumRegionMarker, SurfaceCharge, SurfaceState, InterfaceDipole,
)
from gui.models import DeviceModel, SolveSettings

_LAYER_TYPES = {
    'AbruptLayer': AbruptLayer,
    'GradedLayer': GradedLayer,
    'QuantumRegionMarker': QuantumRegionMarker,
    'InterfaceDipole': InterfaceDipole,
}


def _layer_to_dict(layer) -> Dict[str, Any]:
    d = asdict(layer)
    d['type'] = type(layer).__name__
    return d


def _layer_from_dict(d: Dict[str, Any]):
    d = dict(d)
    kind = d.pop('type')
    # Drop fields no longer part of the dataclass (e.g. quantum_region_start/
    # end, removed when quantum-region markers became their own layer type),
    # so older project files still load rather than hard-failing.
    if kind == 'SurfaceCharge':
        states = [SurfaceState(**s) for s in d.get('states', [])]
        return SurfaceCharge(states=states)
    cls = _LAYER_TYPES.get(kind)
    if cls is None:
        raise ValueError(f"Unknown layer type in project file: {kind!r}")
    known = {f for f in cls.__dataclass_fields__}
    kwargs = {k: v for k, v in d.items() if k in known}
    return cls(**kwargs)


def model_to_dict(model: DeviceModel) -> Dict[str, Any]:
    return {
        'version': 1,
        'layers': [_layer_to_dict(l) for l in model.layers],
        'bottom_contact': asdict(model.bottom_contact),
        'top_contact': asdict(model.top_contact),
        'settings': asdict(model.settings),
    }


def save_project(model: DeviceModel, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(model_to_dict(model), f, indent=2)


def load_project(path: str) -> DeviceModel:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model = DeviceModel()
    model.layers = [_layer_from_dict(d) for d in data.get('layers', [])]
    if 'bottom_contact' in data:
        model.bottom_contact = Contact(**data['bottom_contact'])
    if 'top_contact' in data:
        model.top_contact = Contact(**data['top_contact'])
    if 'settings' in data:
        # Ignore unknown/removed keys, keep defaults for missing ones, so
        # older/newer project files still load rather than hard-failing.
        known = {f for f in SolveSettings.__dataclass_fields__}
        kwargs = {k: v for k, v in data['settings'].items() if k in known}
        model.settings = SolveSettings(**kwargs)
    return model
