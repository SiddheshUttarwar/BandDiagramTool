"""Run every benchmark device in nextnano++ and our tool; save metrics + plots."""
import json
import os
import sys
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from bench import HERE, compare, run_nextnano, run_ours, interfaces
from devices_bench import DEVICES

OUT = os.path.join(HERE, 'results')
os.makedirs(OUT, exist_ok=True)

MODES = [('def', 'Ours: defaults (updated physics)', '#d62728', '--'),
         ('nnp', 'Ours: nextnano params', '#1f77b4', ':')]


def plot(dev, nn, runs, path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 2]})
    a1.plot(nn['x'], nn['Ec'], 'k-', lw=2.2, label='nextnano++')
    a1.plot(nn['x'], nn['Ev'], 'k-', lw=2.2)
    a1.axhline(0, color='gray', lw=0.8, ls='-')
    for m, lab, c, ls in MODES:
        r = runs.get(m)
        if r is None:
            continue
        a1.plot(r['x'], r['Ec'], color=c, ls=ls, lw=1.6, label=lab)
        a1.plot(r['x'], r['Ev'], color=c, ls=ls, lw=1.6)
    for z in interfaces(dev)[1:-1]:
        a1.axvline(z, color='gray', lw=0.4, alpha=0.5)
        a2.axvline(z, color='gray', lw=0.4, alpha=0.5)
    a1.set_ylabel('Energy (eV), E_F = 0')
    a1.set_title(f'{dev.name}\n{dev.ref}', fontsize=9)
    a1.legend(fontsize=8, loc='best')
    a2.semilogy(nn['xn'], np.maximum(nn['n'], 1e8), 'k-', lw=2, label='n (nextnano)')
    a2.semilogy(nn['xn'], np.maximum(nn['p'], 1e8), color='gray', lw=2, label='p (nextnano)')
    for m, lab, c, ls in MODES:
        r = runs.get(m)
        if r is None:
            continue
        a2.semilogy(r['x'], np.maximum(r['n'], 1e8), color=c, ls=ls, lw=1.3)
        a2.semilogy(r['x'], np.maximum(r['p'], 1e8), color=c, ls=ls, lw=1.0, alpha=0.7)
    a2.set_ylim(1e10, 1e21)
    a2.set_xlabel('z (nm), substrate -> surface')
    a2.set_ylabel('density (cm$^{-3}$)')
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(keys=None):
    summary = {}
    for dev in DEVICES:
        if keys and dev.key not in keys:
            continue
        print(f'=== {dev.key}', flush=True)
        entry = dict(name=dev.name, ref=dev.ref, expected=dev.expected)
        try:
            nn = run_nextnano(dev)
        except Exception as e:
            entry['error'] = 'nextnano: ' + str(e)[-400:]
            summary[dev.key] = entry
            print(entry['error'])
            continue
        entry['nn'] = dict(points=len(nn['pts']), time=nn['time'], converged=nn['converged'])
        runs = {}
        for m, *_ in MODES:
            try:
                r = run_ours(dev, m)
                runs[m] = r
                c = compare(dev, nn, r)
                c.update(converged=r['converged'], time=r['time'],
                         native_barrier=r['native_barrier'])
                entry[m] = c
                sh = {k: f"{v['nn']:.3g}/{v['ours']:.3g} ({v['rel']:+.0%})"
                      for k, v in c['sheets'].items()}
                print(f"  {m}: conv={r['converged']} t={r['time']:.1f}s "
                      f"rmsEc={c['rms_Ec']*1e3:.0f}meV maxEc={c['max_Ec']*1e3:.0f} "
                      f"rmsEv={c['rms_Ev']*1e3:.0f} maxEv={c['max_Ev']*1e3:.0f} {sh}", flush=True)
            except Exception as e:
                entry[m] = dict(error=repr(e))
                print(f'  {m}: ERROR {e!r}')
                traceback.print_exc()
        plot(dev, nn, runs, os.path.join(OUT, f'{dev.key}.png'))
        summary[dev.key] = entry
    path = os.path.join(OUT, 'summary.json')
    old = json.load(open(path)) if (keys and os.path.exists(path)) else {}
    old.update(summary)
    json.dump(old, open(path, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main(sys.argv[1:] or None)
