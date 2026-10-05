"""
GaN p-n diode under forward bias: drift-diffusion current.

    python examples/04_pn_diode_under_bias.py

Prints the current density at a few forward voltages together with the
current-conservation diagnostic (max J - min J) / max |J| across the device,
which is zero for an exact steady state.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from devices.device import AlGaNDevice  # noqa: E402
from devices.layer import AbruptLayer, Contact  # noqa: E402

if __name__ == '__main__':
    layers = [AbruptLayer(x_Al=0.0, thickness_nm=300, n_doping=2e18),
              AbruptLayer(x_Al=0.0, thickness_nm=300, p_doping=2e19)]
    contacts = [Contact('bottom', 'ohmic', 'Ti'), Contact('top', 'ohmic', 'Ni')]
    device = AlGaNDevice(layers, contacts, dx_nm=1.0)

    print('V [V]   |J| [A/cm^2]    conservation error    converged')
    for v in (2.6, 2.9, 3.2):
        r = device.solve(V_applied=v, quantum=False)
        print(f'{v:5.1f} {abs(r.J_total):13.3e} {r.current_conservation_error:18.1e} {str(r.converged):>12}')
