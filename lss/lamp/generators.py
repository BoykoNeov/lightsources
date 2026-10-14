"""Generators: lamp + model spec → EmissionProfile (design §4).

Each generator produces bare emission plus whatever it already folds in
(``includes``):

* ``reference`` — Level 0: the measured or tabulated spectrum itself.
* ``planck`` — thermal Level 1: a blackbody at the filament temperature.
* ``tungsten`` — thermal Level 2: tungsten's measured selective emissivity × Planck.
* ``led_two_band`` — LED Level 1: a blue pump band plus one asymmetric phosphor band,
  both shaped in photon energy, with fitted shapes.
"""

from lss.lamp.profile import EmissionProfile, Zone
from lss.spectral.continuum import PlanckContinuum
from lss.spectral.spectrum import Spectrum


def _reference(lamp, spec) -> EmissionProfile:
    from lss.accuracy import references

    if lamp.reference is None:
        raise ValueError(f"{lamp.id}: level {spec.level} needs a reference")
    ref = references.get(lamp.reference)
    return EmissionProfile(
        zones=(Zone(ref.spectrum, name="whole lamp"),),
        power=None,  # the tables are relative
        includes=ref.includes,
        metadata={
            "generator": "reference",
            "level": spec.level,
            "reference": ref.id,
            "params": {},  # measured data: no model parameters
            "valid_range_nm": ref.valid_range,
            "data_versions": [ref.data_version],
        },
    )


def _planck(lamp, spec) -> EmissionProfile:
    t = lamp.parameter("filament_temperature", spec)
    if t.unit != "K":
        raise ValueError("filament_temperature must be in K")
    spectrum = Spectrum.of(PlanckContinuum(t.value), quantity="W m-2 sr-1 nm-1", name=f"Planck {t.value:g} K")
    return EmissionProfile(
        zones=(Zone(spectrum, name="filament"),),
        power=None,  # radiance only: no filament area in the physical description yet
        includes=frozenset(),
        metadata={
            "generator": "planck",
            "level": spec.level,
            "params": {"filament_temperature": t.as_dict()},
            "data_versions": ["CODATA 2018 constants"],
        },
    )


def _tungsten(lamp, spec) -> EmissionProfile:
    from lss.physics.thermal import tungsten

    t = lamp.parameter("filament_temperature", spec)
    if t.unit != "K":
        raise ValueError("filament_temperature must be in K")
    tab = tungsten.table()
    data = tab.data_version
    if tab.extrapolated(t.value):
        lo, hi = tab.measured_range_k
        data += f"; extrapolated linearly in T to {t.value:g} K (measured {lo:g}-{hi:g} K)"
    spectrum = Spectrum.of(tungsten.TungstenContinuum(t.value), quantity="W m-2 sr-1 nm-1",
                           name=f"tungsten {t.value:g} K")
    return EmissionProfile(
        zones=(Zone(spectrum, name="filament"),),
        power=None,  # radiance only, and ε is unknown outside 300-2500 nm
        includes=frozenset(),
        metadata={
            "generator": "tungsten",
            "level": spec.level,
            "params": {"filament_temperature": t.as_dict()},
            "valid_range_nm": tab.wavelength_range_nm,
            "data_versions": ["CODATA 2018 constants", data],
        },
    )


def _nm(lamp, spec, name):
    p = lamp.parameter(name, spec)
    if p.unit != "nm":
        raise ValueError(f"{name} must be in nm")
    return p


def _led_two_band(lamp, spec) -> EmissionProfile:
    from lss.physics.led.bands import EnergyBand

    pump_peak, pump_fwhm = _nm(lamp, spec, "pump_peak"), _nm(lamp, spec, "pump_fwhm")
    ph_peak = _nm(lamp, spec, "phosphor_peak")
    ph_short = _nm(lamp, spec, "phosphor_half_width_short")
    ph_long = _nm(lamp, spec, "phosphor_half_width_long")
    ratio = lamp.parameter("phosphor_to_pump", spec)
    if ratio.unit != "1":
        raise ValueError("phosphor_to_pump is a power ratio, unit 1")
    pump = EnergyBand.symmetric(pump_peak.value, pump_fwhm.value)
    phosphor = EnergyBand.split(ph_peak.value, ph_short.value, ph_long.value, power=ratio.value)
    used = {"pump_peak": pump_peak, "pump_fwhm": pump_fwhm, "phosphor_peak": ph_peak,
            "phosphor_half_width_short": ph_short, "phosphor_half_width_long": ph_long,
            "phosphor_to_pump": ratio}
    return EmissionProfile(
        zones=(Zone(Spectrum.of(pump, phosphor, name="LED two-band"), name="package"),),
        power=None,  # relative: no drive current or efficiency at this level
        includes=frozenset({"phosphor"}),  # the phosphor band is the converted light
        metadata={
            "generator": "led_two_band",
            "level": spec.level,
            "params": {k: p.as_dict() for k, p in used.items()},
            "data_versions": ["analytic band shapes, no tabulated data"],
        },
    )


GENERATORS = {"reference": _reference, "planck": _planck, "tungsten": _tungsten, "led_two_band": _led_two_band}


def run(lamp, spec) -> EmissionProfile:
    try:
        gen = GENERATORS[spec.generator]
    except KeyError:
        raise KeyError(f"unknown generator {spec.generator!r}; known: {sorted(GENERATORS)}") from None
    return gen(lamp, spec)
