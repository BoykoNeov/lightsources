"""Generators: lamp + model spec → EmissionProfile (design §4).

Each generator produces bare emission plus whatever it already folds in
(``includes``):

* ``reference`` — Level 0: the measured or tabulated spectrum itself.
* ``planck`` — thermal Level 1: a blackbody at the filament temperature.
* ``tungsten`` — thermal Level 2: tungsten's measured selective emissivity × Planck.
* ``tungsten_coil`` — thermal Level 3: the same, with the coil's cavity effect on the
  emissivity (one open-area fraction standing for the coil's geometry).
* ``tungsten_resistance`` — thermal Level 4: the coil of Level 3 at the temperature its
  hot resistance (voltage over current) and cold resistance give, through tungsten's
  measured resistivity.
* ``tungsten_balance`` — thermal Level 4, second step: the same coil at the temperature
  where the current's electrical power equals the radiated power plus the gas loss, so
  the current alone sets the lamp and the voltage is predicted, not read.
* ``led_two_band`` — LED Level 1: a blue pump band plus one asymmetric phosphor band,
  both shaped in photon energy, with fitted shapes.
* ``led_phosphor_layer`` — LED Level 2: the die's blue light through a layer of a Ce³⁺
  garnet and a red Eu²⁺ nitride (Beer–Lambert absorption, re-emission, re-absorption),
  one spectrum per emission-angle zone.
* ``fluorescent_triband`` — fluorescent Level 1: the visible Hg lines in measured
  ratios plus the rare-earth tri-phosphor blend, with shapes from the shared phosphor
  library and this lamp's own mix.
"""

from lss.lamp import phosphors
from lss.lamp.profile import EmissionProfile, Zone
from lss.spectral.continuum import PlanckContinuum, TabulatedContinuum
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


def _tungsten(lamp, spec, coil: bool = False, electrical: bool = False, balance: bool = False) -> EmissionProfile:
    from lss.physics.thermal import resistance, tungsten

    used, derived, data_versions = {}, {}, ["CODATA 2018 constants"]
    if balance:
        # Level 4, second step: the current alone sets the temperature (design §4.1 L4).
        from lss.physics.thermal import balance as power_balance

        current = _unit(lamp, spec, "lamp_current", "A", used)
        bal = power_balance.FilamentBalance(
            cold_resistance_ohm=_unit(lamp, spec, "cold_resistance", "ohm", used),
            wire_area_m2=_unit(lamp, spec, "filament_wire_area", "m2", used),
            open_fraction=_unit(lamp, spec, "coil_open_fraction", "1", used),
            gas_share=_unit(lamp, spec, "gas_loss_share", "1", used),
            gas_exponent=_unit(lamp, spec, "gas_loss_exponent", "1", used),
        )
        op = bal.operating_point(current)
        t_value = op["temperature_k"]
        derived = {
            "filament_temperature": {"value": t_value, "unit": "K", "from": f"power balance at {current:g} A"},
            "lamp_voltage": {"value": op["voltage_v"], "unit": "V", "from": "I x R_filament(T); leads not included"},
            "radiated_power": {"value": op["radiated_w"], "unit": "W", "from": "coil, total emissivity"},
            "gas_loss": {"value": op["gas_loss_w"], "unit": "W", "from": "empirical share and exponent"},
        }
        data_versions += [resistance.table().data_version, power_balance.total_emissivity_table().data_version]
    elif electrical:
        # Level 4: the temperature follows from the operating point and the cold resistance.
        current = _unit(lamp, spec, "lamp_current", "A", used)
        voltage = _unit(lamp, spec, "lamp_voltage", "V", used)
        cold = _unit(lamp, spec, "cold_resistance", "ohm", used)
        t_value = resistance.temperature(voltage / current, cold)
        derived = {"filament_temperature": {"value": t_value, "unit": "K",
                                            "from": f"R_hot/R_cold = {voltage / current:.5g}/{cold:.5g} ohm"}}
        data_versions.append(resistance.table().data_version)
    else:
        t = lamp.parameter("filament_temperature", spec)
        if t.unit != "K":
            raise ValueError("filament_temperature must be in K")
        used["filament_temperature"] = t
        t_value = t.value
    tab = tungsten.table()
    data = tab.data_version
    if tab.extrapolated(t_value):
        lo, hi = tab.measured_range_k
        data += f"; extrapolated linearly in T to {t_value:g} K (measured {lo:g}-{hi:g} K)"
    data_versions.append(data)
    if coil and balance:
        f = used["coil_open_fraction"]
        cont = tungsten.TungstenCoilContinuum(t_value, open_fraction=f.value)
        name = f"tungsten coil {t_value:g} K, open fraction {f.value:g}"
    elif coil:
        f = lamp.parameter("coil_open_fraction", spec)
        if f.unit != "1":
            raise ValueError("coil_open_fraction is an area ratio, unit 1")
        used["coil_open_fraction"] = f
        cont = tungsten.TungstenCoilContinuum(t_value, open_fraction=f.value)
        name = f"tungsten coil {t_value:g} K, open fraction {f.value:g}"
    else:
        cont, name = tungsten.TungstenContinuum(t_value), f"tungsten {t_value:g} K"
    meta = {
        "generator": ("tungsten_balance" if balance else "tungsten_resistance" if electrical
                      else "tungsten_coil" if coil else "tungsten"),
        "level": spec.level,
        "params": {k: p.as_dict() for k, p in used.items()},
        "valid_range_nm": tab.wavelength_range_nm,
        "data_versions": data_versions,
    }
    if derived:
        meta["derived"] = derived
    return EmissionProfile(
        zones=(Zone(Spectrum.of(cont, quantity="W m-2 sr-1 nm-1", name=name), name="filament"),),
        power=None,  # radiance only, and ε is unknown outside 300-2500 nm
        includes=frozenset(),
        metadata=meta,
    )


def _tungsten_coil(lamp, spec) -> EmissionProfile:
    return _tungsten(lamp, spec, coil=True)


def _tungsten_resistance(lamp, spec) -> EmissionProfile:
    return _tungsten(lamp, spec, coil=True, electrical=True)


def _tungsten_balance(lamp, spec) -> EmissionProfile:
    return _tungsten(lamp, spec, coil=True, balance=True)


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


def _unit(lamp, spec, name, unit, used):
    p = lamp.parameter(name, spec)
    if p.unit != unit:
        raise ValueError(f"{name} must be in {unit}")
    used[name] = p
    return p.value


#: Reflectance of the die and package under the phosphor layer. A perfect mirror at this
#: rung: with relative spectra a lower value mainly trades against the optical depths.
DIE_REFLECTANCE = 1.0


def _led_phosphor_layer(lamp, spec) -> EmissionProfile:
    from lss.physics.led import layer

    used = {}

    def val(name, unit):
        return _unit(lamp, spec, name, unit, used)

    t_j = val("junction_temperature", "K")
    pump = layer.die_emission(val("die_peak", "nm"), val("die_fwhm", "nm"), t_j)
    # The phosphor runs at the junction temperature at this rung (Level 3 heats each part).
    g_abs, g_em = layer.ce_garnet(val("garnet_zero_phonon", "cm-1"), val("garnet.huang_rhys", "1"),
                                  val("garnet.phonon_energy", "cm-1"), t_j,
                                  val("garnet.inhomogeneous_fwhm", "cm-1"), val("garnet.ground_splitting", "cm-1"),
                                  val("garnet.ratio_2f7_2f5", "1"))
    n_abs, n_em = layer.eu_nitride(val("nitride.emission_peak", "nm"), val("nitride.emission_fwhm", "nm"),
                                   val("nitride.absorption_edge", "nm"), val("nitride.edge_width", "cm-1"))
    phosphors_ = [layer.Phosphor(g_abs, g_em, val("garnet_optical_depth", "1")),
                  layer.Phosphor(n_abs, n_em, val("nitride_optical_depth", "1"))]
    out = layer.transfer(pump, phosphors_, die_reflectance=DIE_REFLECTANCE)
    zones = tuple(
        Zone(Spectrum.of(TabulatedContinuum(layer.GRID_NM, out.power_per_nm(i)), name=f"LED layer {lo:g}-{hi:g} deg"),
             shape=f"polar {lo:g}-{hi:g} deg", name=f"emission {lo:g}-{hi:g} deg")
        for i, (lo, hi) in enumerate(out.zones_deg))
    return EmissionProfile(
        zones=zones,
        power=None,  # relative: no drive current, and the quantum yields are taken as 1
        includes=frozenset({"phosphor"}),
        metadata={
            "generator": "led_phosphor_layer",
            "level": spec.level,
            "params": {k: p.as_dict() for k, p in used.items()},
            "shared": sorted(k for k in used if lamp.reads_from_library(k, spec)),
            "zones": "polar angle inside the layer; refraction at its surface not modelled",
            "data_versions": [phosphors.LIBRARY_PATH],
        },
    )


def _ratio(lamp, spec, name):
    p = lamp.parameter(name, spec)
    if p.unit != "1":
        raise ValueError(f"{name} is a power ratio, unit 1")
    return p


def _fluorescent_triband(lamp, spec) -> EmissionProfile:
    from lss.physics.discharge import mercury

    used = {}

    def read(name):
        used[name] = lamp.parameter(name, spec)
        return used[name]

    # Powers relative to the green phosphor's: the scores normalise each lamp to equal
    # luminous flux, so its overall scale is not a parameter.
    hg, blue, red = (_ratio(lamp, spec, n) for n in ("hg_to_green", "blue_to_green", "red_to_green"))
    t_gas = lamp.parameter("gas_temperature", spec)
    if t_gas.unit != "K":
        raise ValueError("gas_temperature must be in K")
    sources = mercury.visible_lines(hg.value, t_gas.value)
    for pid, power in (("BAM", blue.value), ("LAP", 1.0), ("YOX", red.value)):
        sources += phosphors.emission(pid, read, power)
    used = {"hg_to_green": hg, "blue_to_green": blue, "red_to_green": red, "gas_temperature": t_gas, **used}
    return EmissionProfile(
        zones=(Zone(Spectrum.of(*sources, name="fluorescent tri-band"), name="tube"),),
        power=None,  # relative: no discharge energy budget at this level
        includes=frozenset({"phosphor"}),  # the phosphor bands are the converted light
        metadata={
            "generator": "fluorescent_triband",
            "level": spec.level,
            "params": {k: p.as_dict() for k, p in used.items()},
            "shared": sorted(k for k in used if phosphors.is_shared(k)),
            "data_versions": [mercury.table().source, phosphors.LIBRARY_PATH],
        },
    )


GENERATORS = {"reference": _reference, "planck": _planck, "tungsten": _tungsten, "tungsten_coil": _tungsten_coil,
              "tungsten_resistance": _tungsten_resistance, "tungsten_balance": _tungsten_balance,
              "led_two_band": _led_two_band, "led_phosphor_layer": _led_phosphor_layer,
              "fluorescent_triband": _fluorescent_triband}


def run(lamp, spec) -> EmissionProfile:
    try:
        gen = GENERATORS[spec.generator]
    except KeyError:
        raise KeyError(f"unknown generator {spec.generator!r}; known: {sorted(GENERATORS)}") from None
    return gen(lamp, spec)
