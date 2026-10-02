"""Benchmark device set (Ga-face c-plane, layers listed substrate -> surface).

Our tool only models Al(x)Ga(1-x)N, so InGaN-based LEDs/LDs cannot be built;
the optoelectronic entries are AlGaN deep-UV devices instead.
Schottky barriers follow Ambacher et al. JAP 87, 334 (2000): e*phi_B = 0.84 + 1.3x eV.
"""
from bench import Dev, L


def phiB(x):
    return 0.84 + 1.3 * x


DEVICES = [
    Dev('pn_GaN', 'GaN p-n homojunction (sanity)',
        'Textbook abrupt junction (Sze); no polarization interfaces',
        [L(0.0, 200, ND=1e18), L(0.0, 200, NA=1e19)],
        top=('ohmic',), fine=0.5,
        sheet_windows=[], expected={}),

    Dev('hemt_AlGaN', 'Al0.3Ga0.7N(25nm)/GaN HEMT',
        'Ambacher et al., JAP 85, 3222 (1999); JAP 87, 334 (2000)',
        [L(0.0, 400, ND=1e16), L(0.3, 25)],
        top=('schottky', phiB(0.3)), fine=0.1,
        sheet_windows=[('2DEG', 350, 425, 'n')],
        expected={'2DEG': (1.0e13, 1.4e13,
                           'Ambacher 1999/2000: n_s ~1.0-1.3e13 cm^-2 measured for x~0.3, d~25-30 nm')}),

    Dev('hemt_AlN', 'AlN(5nm)/GaN HEMT',
        'Cao & Jena, APL 90, 182112 (2007); Zimmermann et al., EDL 29, 661 (2008)',
        [L(0.0, 400, ND=1e16), L(1.0, 5)],
        top=('schottky', phiB(1.0)), fine=0.08,
        sheet_windows=[('2DEG', 350, 405, 'n')],
        expected={'2DEG': (2.0e13, 3.5e13,
                           'Cao & Jena 2007: n_s ~2-3.5e13 cm^-2 for 4-6 nm AlN barriers')}),

    Dev('hemt_AlN_interlayer', 'Al0.3GaN(24nm)/AlN(1nm)/GaN HEMT',
        'Shen et al., IEEE EDL 22, 457 (2001)',
        [L(0.0, 400, ND=1e16), L(1.0, 1.0), L(0.3, 24)],
        top=('schottky', phiB(0.3)), fine=0.08,
        sheet_windows=[('2DEG', 350, 425, 'n')],
        expected={'2DEG': (1.1e13, 1.5e13,
                           'Shen 2001: n_s ~1.2-1.4e13 cm^-2 with 1 nm AlN interlayer')}),

    Dev('polar_graded_pn', 'Polarization-doped graded AlGaN p-n (0->0.3->0)',
        'Simon et al., Science 327, 60 (2010); Jena et al., APL 81, 4395 (2002)',
        [L(0.0, 50, ND=1e18), L(0.0, 100, x1=0.3), L(0.3, 100, x1=0.0), L(0.0, 30, NA=1e19)],
        top=('ohmic',), fine=0.4,
        sheet_windows=[('3DEG (n-graded)', 50, 150, 'n'), ('3DHG (p-graded)', 150, 250, 'p')],
        expected={}),

    Dev('duv_led', 'AlGaN DUV LED ~280 nm, 3x MQW',
        'Representative of Hirayama et al. / Kneissl et al., Nat. Photon. 13, 233 (2019)',
        [L(0.6, 120, ND=3e18),
         L(0.55, 8), L(0.45, 2.5), L(0.55, 8), L(0.45, 2.5), L(0.55, 8), L(0.45, 2.5), L(0.55, 8),
         L(0.8, 15, NA=1e19), L(0.6, 40, NA=1e19), L(0.0, 30, NA=2e19)],
        top=('ohmic',), fine=0.12,
        sheet_windows=[('n in MQW', 120, 159.5, 'n'), ('p in MQW', 120, 159.5, 'p')],
        expected={}),

    Dev('uvc_ld', 'UVC laser diode with polarization-doped graded p-clad',
        'Representative of Zhang et al., Appl. Phys. Express 12, 124003 (2019)',
        [L(1.0, 10, ND=3e18), L(0.7, 100, ND=2e18), L(0.63, 50), L(0.45, 3), L(0.63, 50),
         L(1.0, 120, x1=0.7), L(0.0, 20, NA=2e19)],
        top=('ohmic',), fine=0.15,
        sheet_windows=[('n in QW', 160, 163, 'n'), ('holes, graded clad', 213, 333, 'p')],
        expected={}),

    Dev('hbt', 'AlGaN/GaN npn HBT (Al0.1 emitter)',
        'Shelton et al., IEEE TED 48, 490 (2001); Makimoto et al., APL 79, 380 (2001)',
        [L(0.0, 100, ND=5e18), L(0.0, 300, ND=1e16), L(0.0, 100, NA=1e19),
         L(0.1, 60, ND=5e17), L(0.0, 50, ND=5e18)],
        top=('ohmic',), fine=0.3,
        sheet_windows=[], expected={}),
]
