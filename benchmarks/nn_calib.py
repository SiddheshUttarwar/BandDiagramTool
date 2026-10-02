"""Extract nextnano++ intrinsic (unstrained, 300 K) band edges vs Al fraction.

Intrinsic edge = output band edge + electrostatic potential (E = E0 - q*phi).
"""
import os
import numpy as np
import nextnanopy as nn
from nngrid import xgrid_block

HERE = os.path.dirname(os.path.abspath(__file__))
xs = np.round(np.arange(0, 1.0001, 0.1), 2)
regs, pts = [], [0.0, 110.0]
for i, x in enumerate(xs):
    a, b = i * 10, (i + 1) * 10
    if x == 0:
        mat = 'binary{ name = "GaN" }'
    elif x == 1:
        mat = 'binary{ name = "AlN" }'
    else:
        mat = f'ternary_constant{{ name = "Al(x)Ga(1-x)N" alloy_x = {x} }}'
    regs.append(f'    region{{ {mat} line{{ x = [{a}, {b}] }} }}')
    pts += [a + 1, a + 5, a + 9]
pts = np.array(sorted(set(pts)), float)

s = f'''global{{ simulate1D{{}} crystal_wz{{ x_hkl = [0, 0, 1] y_hkl = [1, 0, 0] }} substrate{{ name = "GaN" }} temperature = 300 }}
contacts{{ ohmic{{ name = "c1" bias = 0.0 }} ohmic{{ name = "c2" bias = 0.0 }} }}
structure{{
{chr(10).join(regs)}
    region{{ contact{{ name = "c1" }} line{{ x = [0, 0.5] }} }}
    region{{ contact{{ name = "c2" }} line{{ x = [109.5, 110] }} }}
}}
grid{{
{xgrid_block(pts)}
}}
classical{{ Gamma{{}} HH{{}} LH{{}} SO{{}} output_bandedges{{ averaged = no }} output_carrier_densities{{}} }}
poisson{{ output_potential{{}} }}
run{{ poisson{{}} }}
'''
path = os.path.join(HERE, 'nextnano_inputs', 'calib.in')
open(path, 'w').write(s)
nn.InputFile(path).execute(show_log=False, convergenceCheck=False)
od = r'C:\nextnano_opt\output\calib\bias_00000'
be = np.loadtxt(os.path.join(od, 'bandedges.dat'), skiprows=1)
po = np.loadtxt(os.path.join(od, 'potential.dat'), skiprows=1)
out = []
for i, x in enumerate(xs):
    j = np.argmin(abs(be[:, 0] - (i * 10 + 5)))
    Ec, HH, LH, SO = be[j, 1:5] + np.interp(be[j, 0], po[:, 0], po[:, 1])
    out.append((x, Ec, HH, LH, SO))
    print(f"x={x:.1f} Ec0={Ec:.4f} HH={HH:.4f} LH={LH:.4f} SO={SO:.4f} "
          f"Eg(top)={Ec - max(HH, LH, SO):.4f}")
np.savetxt(os.path.join(HERE, 'nn_intrinsic_edges.txt'), np.array(out),
           header='x Ec HH LH SO  (eV, nextnano intrinsic, 300K, unstrained)')
