"""
BandDiagramTool — main entry point.

Define your device here and run:
    python main.py

The device structure is fully user-defined via the layer list.
See examples/example_led.py for a detailed worked example.
"""

import matplotlib
matplotlib.use('TkAgg')   # change to 'Agg' if no display is available
import matplotlib.pyplot as plt

from devices.layer  import AbruptLayer, GradedLayer, Contact
from devices.device import AlGaNDevice
from visualization.plotter import plot_all, plot_band_diagram, plot_wavefunctions

# ──────────────────────────────────────────────────────────────────────────────
# Define your device (bottom layer first)
# ──────────────────────────────────────────────────────────────────────────────

layers = [
    # Layer format: AbruptLayer(x_Al, thickness_nm, n_doping, p_doping)
    #               GradedLayer(x_Al_start, x_Al_end, thickness_nm, ..., profile='linear')
    AbruptLayer(x_Al=0.0,  thickness_nm=200, n_doping=1e17, p_doping=0.0),  # n-GaN buffer
    AbruptLayer(x_Al=0.15, thickness_nm=10,  n_doping=0.0,  p_doping=0.0),  # AlGaN barrier
    AbruptLayer(x_Al=0.0,  thickness_nm=3,   n_doping=0.0,  p_doping=0.0),  # GaN quantum well
    AbruptLayer(x_Al=0.15, thickness_nm=20,  n_doping=0.0,  p_doping=0.0),  # AlGaN EBL
    AbruptLayer(x_Al=0.0,  thickness_nm=100, n_doping=0.0,  p_doping=3e17), # p-GaN contact
]

contacts = [
    Contact(position='bottom', contact_type='ohmic',    metal='Ti'),
    Contact(position='top',    contact_type='schottky', metal='Ni'),
]

# ──────────────────────────────────────────────────────────────────────────────
# Create and solve
# ──────────────────────────────────────────────────────────────────────────────

device = AlGaNDevice(layers=layers, contacts=contacts, T=300, dx_nm=0.2)
print(device)

result = device.solve(
    V_applied=0.0,   # change to e.g. 3.0 for forward bias
    quantum=True,
    verbose=True,
)

print(f"\nConverged: {result.converged}  ({result.n_iterations} iterations)")
print(f"Efn range = [{result.Efn.min():.4f}, {result.Efn.max():.4f}] eV,  "
      f"Efp range = [{result.Efp.min():.4f}, {result.Efp.max():.4f}] eV")

if result.qcse_transition_eV is not None:
    ie, ih = result.qcse_pair if result.qcse_pair is not None else (0, 0)
    if result.qcse_in_well:
        lo, hi = result.qw_window_nm
        where = f"in the quantum well ({lo:.1f}-{hi:.1f} nm)"
    elif result.qw_window_nm is not None:
        lo, hi = result.qw_window_nm
        where = (f"global ground state -- no solved subband is confined in the "
                 f"detected well ({lo:.1f}-{hi:.1f} nm)")
    else:
        where = "global ground state -- no quantum well layer detected"
    print(f"\nQCSE:  e{ie+1}-h{ih+1} transition energy = {result.qcse_transition_eV:.4f} eV   "
          f"e-h overlap = {result.qcse_overlap * 100:.2f}%  ({where})")

# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────

fig = plot_all(result)
plt.show()
