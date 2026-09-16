"""
CSV export/reload for a solved device: writes the per-grid-point profiles
that drive the Band Diagram / Carriers / Fields / Polarization / Strain
panels to a CSV file, then reloads them back into a SolverResult copy so
those panels plot from the saved file rather than straight from the
in-memory solve.

Wavefunction/QCSE-specific fields (psi_e, psi_h, E_e, E_h, per-subband QCSE
diagnostics) are intentionally NOT round-tripped through the CSV -- they're
per-subband arrays, not a single per-grid-point profile, so they don't fit
this flat table; the Wavefunctions/QCSE tabs keep using the original
in-memory result for those.
"""

from __future__ import annotations

import csv
import dataclasses
import re
from typing import List

import numpy as np

from physics.self_consistent import SolverResult
from physics.materials.algan import get_AlGaN_params

_INVALID_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|]')


def result_filename(project_name: str, V_applied: float, T: float) -> str:
    """Build the '<inputFileName>_Bias_<X>V_Temp_<Y>K' stem (no extension)
    for one solved result: X is the bias value, or 'Equi' at V=0; Y is the
    temperature, or 'RoomTemp' at T=300 K."""
    bias = "Equi" if abs(V_applied) < 1e-9 else f"{V_applied:g}"
    temp = "RoomTemp" if abs(T - 300.0) < 1e-9 else f"{T:g}"
    stem = f"{project_name}_Bias_{bias}V_Temp_{temp}K"
    return _INVALID_FILENAME_CHARS.sub('_', stem)

# Column order in the CSV == the band diagram's own curve order (vacuum,
# Ec, Efc [electron quasi-Fermi], Efv [hole quasi-Fermi], Ei, Ev), plus the
# other per-point profiles the remaining panels need.
_FIELDS = [
    "x_nm", "x_Al",
    "Vacuum_eV", "Ec_eV", "Efc_eV", "Efv_eV", "Ei_eV", "Ev_eV",
    "Ev_lh_eV", "Ev_so_eV",
    "phi_V", "E_field_V_per_m",
    "n_cm-3", "p_cm-3",
    "Psp_C_per_m2", "Ppz_C_per_m2", "P_total_C_per_m2",
    "eps_xx", "eps_zz",
]


def save_result_csv(result: SolverResult, path: str) -> None:
    """Write `result`'s per-grid-point profiles to `path` as CSV, with a
    leading comment block of scalar metadata (# key,value lines, skippable
    by any standard CSV reader)."""
    chi_arr = np.array([get_AlGaN_params(xi).chi for xi in result.x_Al])
    vacuum = result.Ec + chi_arr
    # LH/SO valence band edges (decoupled 3-band effective-mass model, see
    # physics.materials.algan.AlGaNParams.valence_band_structure); fall back
    # to Ev (== Ev_hh) for a SolverResult built without them.
    Ev_lh = result.Ev_lh if getattr(result, 'Ev_lh', None) is not None else result.Ev
    Ev_so = result.Ev_so if getattr(result, 'Ev_so', None) is not None else result.Ev

    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# V_applied_V,{result.V_applied}\n")
        f.write(f"# V_internal_V,{getattr(result, 'V_internal', result.V_applied)}\n")
        f.write(f"# T_K,{result.T}\n")
        f.write(f"# converged,{result.converged}\n")
        writer = csv.writer(f)
        writer.writerow(_FIELDS)
        for i in range(len(result.x_nm)):
            writer.writerow([
                repr(float(result.x_nm[i])), repr(float(result.x_Al[i])),
                repr(float(vacuum[i])), repr(float(result.Ec[i])),
                repr(float(result.Efn[i])), repr(float(result.Efp[i])),
                repr(float(result.Ei[i])), repr(float(result.Ev[i])),
                repr(float(Ev_lh[i])), repr(float(Ev_so[i])),
                repr(float(result.phi[i])), repr(float(result.E_field[i])),
                repr(float(result.n[i])), repr(float(result.p[i])),
                repr(float(result.Psp[i])), repr(float(result.Ppz[i])),
                repr(float(result.P_total[i])),
                repr(float(result.eps_xx[i])), repr(float(result.eps_zz[i])),
            ])


def load_result_csv(path: str) -> dict:
    """Read back the per-point columns written by save_result_csv, as a
    dict of numpy arrays keyed by SolverResult field name (x_nm, x_Al, Ec,
    Efn, Efp, Ei, Ev, Ev_lh, Ev_so, phi, E_field, n, p, Psp, Ppz, P_total,
    eps_xx, eps_zz) -- Vacuum_eV is derived on replot from Ec/x_Al, so it's
    not a SolverResult field and isn't returned here despite being in the
    file. Ev_hh is restored from the same Ev_eV column (Ev *is* the HH
    edge -- see physics.self_consistent.SolverResult)."""
    rows: List[List[float]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            break
        reader = csv.reader([line] + f.readlines())
        header = next(reader)
        col = {name: idx for idx, name in enumerate(header)}
        for row in reader:
            if not row:
                continue
            rows.append([float(v) for v in row])

    arr = np.array(rows, dtype=float).T
    return {
        "x_nm": arr[col["x_nm"]], "x_Al": arr[col["x_Al"]],
        "Ec": arr[col["Ec_eV"]], "Efn": arr[col["Efc_eV"]],
        "Efp": arr[col["Efv_eV"]], "Ei": arr[col["Ei_eV"]],
        "Ev": arr[col["Ev_eV"]], "Ev_hh": arr[col["Ev_eV"]],
        "Ev_lh": arr[col["Ev_lh_eV"]], "Ev_so": arr[col["Ev_so_eV"]],
        "phi": arr[col["phi_V"]], "E_field": arr[col["E_field_V_per_m"]],
        "n": arr[col["n_cm-3"]], "p": arr[col["p_cm-3"]],
        "Psp": arr[col["Psp_C_per_m2"]], "Ppz": arr[col["Ppz_C_per_m2"]],
        "P_total": arr[col["P_total_C_per_m2"]],
        "eps_xx": arr[col["eps_xx"]], "eps_zz": arr[col["eps_zz"]],
    }


def save_and_reload_csv(result: SolverResult, path: str) -> SolverResult:
    """Save `result` to `path`, then return a copy of `result` with its
    per-grid-point fields replaced by what was just read back from that
    file -- so the plots that follow are driven by the saved CSV, not the
    original in-memory arrays. Non-per-point fields (subband wavefunctions,
    QCSE diagnostics, metadata) pass through unchanged."""
    save_result_csv(result, path)
    data = load_result_csv(path)
    return dataclasses.replace(result, **data)
