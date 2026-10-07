"""
Point-defect control by illumination during growth ("defect quasi-Fermi
level control", dQFL), for light of any wavelength.

Shining light that the layer absorbs on a doped layer while it grows creates excess
minority carriers and splits the quasi-Fermi levels. A charged compensating
defect then no longer forms against the equilibrium Fermi level: its
formation energy rises, and fewer of it are incorporated. The method was
developed for MOCVD of GaN and AlGaN at North Carolina State University
(Bryan et al., J. Electron. Mater. 42, 815; Appl. Phys. Lett. 105, 222101
(2014); Reddy et al., J. Appl. Phys. 120, 185704 (2016); Klump et al.,
J. Appl. Phys. 127, 045702 (2020)).

Model. For each layer, taken as the uniform, neutral film at the growth
surface at the growth temperature T:

  1. Carriers in the dark, n0 and p0, from charge neutrality with the
     layer's donors and acceptors (incompletely ionized) and Eg(T).
  2. Excess carriers from the light, for the given wavelength. The photon
     energy is E_ph = hc / lambda and the flux is the power density divided
     by E_ph. A layer absorbs only if E_ph exceeds its gap at the growth
     temperature: the absorption coefficient is zero below the gap and
     rises as sqrt(E_ph - Eg) to the given value 0.1 eV above it (a direct
     gap). One pair is generated per absorbed photon, G(z) = alpha * flux *
     exp(-alpha z) below the surface. The carriers do not stay where they
     are made: the minority carriers diffuse over L = sqrt(D tau) before
     recombining, and some recombine at the surface with velocity S. The
     steady state of  D d2(dn)/dz2 - dn/tau + G = 0  with  D d(dn)/dz = S dn
     at the surface gives, at the surface where the layer is growing,

         dn = dp = alpha flux tau * (D/L) / [(1 + alpha L) (S + D/L)].

     For S = 0 this is the local value alpha*flux*tau divided by
     (1 + alpha L): the carriers made within 1/alpha are spread over L.
     D = mu kT/q of the minority carrier, with the solver's mobilities
     (300 and 10 cm^2/Vs in GaN, 25 and 2 in AlN) scaled as (300 K / T)^1.5
     for phonon scattering. A layer whose gap is wider
     than the photon energy therefore gets no excess carriers and no change
     in its defects, which is how the same lamp can act on the GaN layers
     of a stack and not on its AlGaN layers.
  3. Quasi-Fermi levels E_Fn, E_Fp of n0 + dn and p0 + dp; the photovoltage
     is E_p = E_Fn - E_Fp.
  4. A defect in charge state q with transition level E_t exchanges
     carriers with both bands, by capture and by thermal emission. The
     carrier chemical potential it forms against is the rate-weighted mix
     (Alberi and Scarpulla, J. Appl. Phys. 123, 185702 (2018), Eq. 7)

         mu = w_n E_Fn + w_p E_Fp,
         w_n = (r_cn + r_en) / R,   w_p = (r_cp + r_ep) / R,
         r_cn = s_n n,   r_en = s_n Nc exp[-(Ec - E_t)/kT],
         r_cp = s_p p,   r_ep = s_p Nv exp[-(E_t - Ev)/kT],

     with s = (thermal velocity) x (capture cross-section). A charged
     defect attracts one carrier type and repels the other, so its two
     cross-sections differ: s for the attracted carrier (holes for a
     negative acceptor, electrons for a positive donor) is taken
     `capture_asymmetry` times the other. The formation energy changes by
     dE = q (mu - E_F)  and the concentration of that charge state by the
     factor exp(-dE / kT).

This reproduces the two limits quoted for the dQFL process: a compensating
level near the minority-carrier band edge follows the minority quasi-Fermi
level and gains the full photovoltage (shallow H+ in Mg-doped GaN), while a
level near the majority band edge follows the majority one and gains
nothing (the deep (+3/+) level of the nitrogen vacancy).

The capture asymmetry is not known for these defects (attractive and
repulsive centres differ by 10^2 to 10^4 and more in general). The default
is 1, which is Alberi and Scarpulla's model as published, with no adjusted
parameter. The result is sensitive to it wherever the level is far from the
minority band edge: for Si-doped Al0.65Ga0.35N the predicted reduction goes
from a few per cent at 1 to more than a hundredfold at 100, while the
experiments report changes of two to ten times. Treat those cases as
bounds, not predictions.

What it is not: the diffusion formula is for a thick uniform layer (no
heterointerface or field within a diffusion length of the surface); light below the gap (absorbed by defects or the Urbach
tail) is taken as not absorbed at all, and the lamp is one wavelength, not
a spectrum; the excess density is generation x lifetime, not a solution
of the continuity equations with diffusion and surface recombination; the
layer is treated as flat-band (no surface band bending or photovoltage
across a depletion layer); the result is a ratio of the charged state with
and without light, not an absolute concentration; and neutral forms of the
same impurity (the Mg-H complex, for one) are untouched, so a total
measured by SIMS falls by less than the factor given here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import brentq

from physics.materials.algan import (
    acceptor_ionization_energy, donor_ionization_energy, get_nitride_params, get_nitride_params_T,
    nitride_name,
)

_KB_EV = 8.617333262e-5          # eV / K
_Q = 1.602176634e-19             # J / eV


@dataclass(frozen=True)
class Compensator:
    """A compensating defect: its charge when ionized and its transition
    level, measured up from the valence band (ref='VB') or down from the
    conduction band (ref='CB')."""
    name: str
    charge: int
    level_eV: float
    ref: str = 'VB'
    note: str = ''

    def level_above_vb(self, prm) -> float:
        """Transition level above the valence band of the alloy `prm`. The
        level is given for GaN and is held fixed relative to the vacuum
        level as the bands move with composition and temperature (Reddy et
        al., Appl. Phys. Lett. 116, 032102 (2020))."""
        gan = get_nitride_params(0.0, 0.0)
        in_gan = self.level_eV if self.ref == 'VB' else gan.Eg - self.level_eV
        level_vac = -(gan.chi + gan.Eg) + in_gan            # below vacuum
        e = level_vac + (prm.chi + prm.Eg)                  # above this alloy's valence band
        return float(min(max(e, 0.0), prm.Eg))


# Indicative levels, given for GaN at room temperature and carried to other
# compositions at a fixed energy below vacuum. In p-type material the compensators are donors,
# in n-type material acceptors. H and V_N as taken in Klump et al. (2020):
# H a shallow donor, V_N deep, about 3 eV below the conduction band (0.5 eV
# above the valence band; Lyons and Van de Walle). C_N: (0/-) 0.9 eV above
# the valence band. The cation-vacancy complex stands for V_III-O_N and
# V_III-nSi; its level varies with the complex and with Al content.
COMPENSATORS_P = (
    Compensator('H (interstitial donor)', +1, 0.0, 'CB', 'passivates Mg as Mg-H'),
    Compensator('V_N (+3/+)', +1, 0.5, 'VB', 'self-compensation; deep, near the valence band'),
)
COMPENSATORS_N = (
    Compensator('C_N (0/-)', -1, 0.9, 'VB', 'carbon on the nitrogen site'),
    Compensator('V_III complex (-/2-)', -2, 1.1, 'VB', 'cation vacancy with O or Si; indicative level'),
)


@dataclass
class DefectChange:
    compensator: Compensator
    w_n: float                 # weight of the electron quasi-Fermi level
    w_p: float                 # weight of the hole quasi-Fermi level
    dE_eV: float               # rise of the formation energy
    factor: float              # concentration with light / without light
    level_eV: float = 0.0      # transition level above this layer's valence band


@dataclass
class LayerIllumination:
    index: int                 # layer number, counted from the substrate (1-based)
    name: str
    z0_nm: float
    z1_nm: float
    doping: str                # 'n', 'p' or 'undoped'
    Eg_eV: float               # at the growth temperature
    ni: float                  # cm^-3
    n0: float
    p0: float
    excess: float              # dn = dp, cm^-3
    Ep_eV: float               # photovoltage, E_Fn - E_Fp
    absorption_cm: float = 0.0 # at the lamp's photon energy; 0 if the layer is transparent to it
    diffusion_length_nm: float = 0.0   # of the minority carrier at the growth temperature
    excess_local: float = 0.0  # alpha*flux*tau, what the excess would be without diffusion
    defects: List[DefectChange] = field(default_factory=list)


@dataclass
class IlluminationResult:
    T_K: float
    power_W_cm2: float
    wavelength_nm: float
    photon_eV: float
    lifetime_s: float
    absorption_cm: float
    capture_asymmetry: float
    surface_velocity_cm_s: float = 0.0
    diffusion: bool = True
    layers: List[LayerIllumination] = field(default_factory=list)


def _carriers(prm, T, Nd, Na, Ed, Ea):
    """n0, p0, E_F - E_v for a neutral layer (Boltzmann bands, donors with
    degeneracy 2, acceptors with degeneracy 4)."""
    kT = _KB_EV * T
    Nc, Nv, Eg = prm.Nc(T), prm.Nv(T), prm.Eg

    def neutrality(ef):                     # ef measured from the valence band
        n = Nc * np.exp(-(Eg - ef) / kT)
        p = Nv * np.exp(-ef / kT)
        nd = Nd / (1.0 + 2.0 * np.exp((ef - (Eg - Ed)) / kT)) if Nd > 0 else 0.0
        na = Na / (1.0 + 4.0 * np.exp((Ea - ef) / kT)) if Na > 0 else 0.0
        return p + nd - n - na

    ef = brentq(neutrality, -1.0, Eg + 1.0, xtol=1e-12)
    return Nc * np.exp(-(Eg - ef) / kT), Nv * np.exp(-ef / kT), ef


def defect_change(comp: Compensator, prm, Nc, Nv, n, p, efn, efp, ef0, kT,
                  capture_asymmetry: float = 1.0) -> DefectChange:
    """Change of one compensator's formation energy in the alloy `prm`;
    energies from the valence band. capture_asymmetry = s(attracted
    carrier) / s(repelled carrier)."""
    Eg = prm.Eg
    et = comp.level_above_vb(prm)
    s_n, s_p = (capture_asymmetry, 1.0) if comp.charge > 0 else (1.0, capture_asymmetry)
    r_cn, r_en = s_n * n, s_n * Nc * np.exp(-(Eg - et) / kT)
    r_cp, r_ep = s_p * p, s_p * Nv * np.exp(-et / kT)
    total = r_cn + r_en + r_cp + r_ep
    w_n, w_p = (r_cn + r_en) / total, (r_cp + r_ep) / total
    mu = w_n * efn + w_p * efp
    dE = comp.charge * (mu - ef0)
    return DefectChange(comp, float(w_n), float(w_p), float(dE), float(np.exp(-dE / kT)), level_eV=et)


def absorption(photon_eV: float, Eg: float, above_gap_cm: float, onset_eV: float = 0.1) -> float:
    """Band-to-band absorption coefficient [1/cm] of a direct-gap layer:
    zero below the gap, sqrt(E - Eg) above it, reaching `above_gap_cm`
    at `onset_eV` above the gap and constant beyond."""
    if photon_eV <= Eg:
        return 0.0
    return float(above_gap_cm * min(1.0, np.sqrt((photon_eV - Eg) / onset_eV)))


def surface_excess(alpha: float, flux: float, tau: float, T: float, x_al: float, doping: str,
                   surface_velocity_cm_s: float = 0.0, diffusion: bool = True):
    """Excess carrier density [cm^-3] at the illuminated surface of a thick
    layer. Returns (with diffusion and surface recombination, the local
    value alpha*flux*tau, the minority-carrier diffusion length [cm])."""
    kT = _KB_EV * T
    local = alpha * flux * tau
    # minority carrier: electrons in p-type, holes in n-type and undoped layers
    mu300 = (300.0 * (1 - x_al) + 25.0 * x_al) if doping == 'p' else (10.0 * (1 - x_al) + 2.0 * x_al)
    D = mu300 * (300.0 / T) ** 1.5 * kT             # cm^2 / s
    L = float(np.sqrt(D * tau)) if tau > 0 else 0.0
    if diffusion and L > 0 and alpha > 0:
        return local * (D / L) / ((1.0 + alpha * L) * (surface_velocity_cm_s + D / L)), local, L
    return local, local, L


def simulate_illumination(layers: Sequence, T_growth_C: float = 1040.0, power_W_cm2: float = 1.0,
                          wavelength_nm: float = 300.0,
                          lifetime_ns: float = 0.3, absorption_cm: float = 3e5,
                          capture_asymmetry: float = 1.0,
                          surface_velocity_cm_s: float = 0.0, diffusion: bool = True,
                          compensators_n: Optional[Sequence[Compensator]] = None,
                          compensators_p: Optional[Sequence[Compensator]] = None) -> IlluminationResult:
    """Effect of illumination at `wavelength_nm` during growth on the
    compensating defects of every doped layer in `layers` (devices.layer
    objects, substrate first). Defaults follow the growth conditions of
    Klump et al. (2020): 1040 C, 1 W/cm^2, lifetime 0.1-1 ns, absorption
    1e5-1e6 /cm above the gap; 300 nm (4.13 eV) is above the gap of GaN
    and of AlGaN up to about 65 % Al at that temperature."""
    T = float(T_growth_C) + 273.15
    kT = _KB_EV * T
    tau = float(lifetime_ns) * 1e-9
    comps_n = COMPENSATORS_N if compensators_n is None else tuple(compensators_n)
    comps_p = COMPENSATORS_P if compensators_p is None else tuple(compensators_p)
    if wavelength_nm <= 0:
        raise ValueError("The wavelength must be positive.")
    photon_eV = 1239.841984 / float(wavelength_nm)
    out = IlluminationResult(T, float(power_W_cm2), float(wavelength_nm), photon_eV, tau, float(absorption_cm), float(capture_asymmetry),
                             float(surface_velocity_cm_s), bool(diffusion))

    z, number = 0.0, 0
    for layer in layers:
        number += 1
        thickness = getattr(layer, 'thickness_nm', None)
        if thickness is None:
            continue                                    # marker, surface states, dipole
        if getattr(layer, 'crystal', 'wurtzite') != 'wurtzite':
            raise ValueError("The defect quasi-Fermi-level model covers the III-nitrides only "
                             "(its compensating defects are H, V_N, C_N and cation-vacancy complexes).")
        if hasattr(layer, 'x_Al'):
            x_al, x_in = layer.x_Al, getattr(layer, 'x_In', 0.0)
        else:                                           # graded: evaluate at mid-composition
            x_al = 0.5 * (layer.x_Al_start + layer.x_Al_end)
            x_in = 0.5 * (getattr(layer, 'x_In_start', 0.0) + getattr(layer, 'x_In_end', 0.0))
        prm = get_nitride_params_T(float(x_al), float(x_in), T)
        Nd, Na = float(layer.n_doping), float(layer.p_doping)
        n0, p0, ef0 = _carriers(prm, T, Nd, Na,
                                float(donor_ionization_energy(x_al, x_in)),
                                float(acceptor_ionization_energy(x_al, x_in)))
        Nc, Nv, Eg = prm.Nc(T), prm.Nv(T), prm.Eg

        flux = power_W_cm2 / (photon_eV * _Q)           # photons / cm^2 / s
        alpha = absorption(photon_eV, Eg, absorption_cm)
        doping = 'n' if Nd > Na else ('p' if Na > Nd else 'undoped')
        excess, excess_local, L = surface_excess(alpha, flux, tau, T, float(x_al), doping,
                                                 surface_velocity_cm_s, diffusion)
        n, p = n0 + excess, p0 + excess
        efn = Eg + kT * np.log(n / Nc)
        efp = -kT * np.log(p / Nv)

        rec = LayerIllumination(number, nitride_name(float(x_al), float(x_in)), z, z + thickness, doping,
                                float(Eg), float(np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * kT))),
                                float(n0), float(p0), float(excess), float(efn - efp), absorption_cm=alpha,
                                diffusion_length_nm=L * 1e7, excess_local=float(excess_local))
        for comp in (comps_n if doping == 'n' else comps_p if doping == 'p' else ()):
            rec.defects.append(defect_change(comp, prm, Nc, Nv, n, p, efn, efp, ef0, kT, capture_asymmetry))
        out.layers.append(rec)
        z += thickness
    return out
