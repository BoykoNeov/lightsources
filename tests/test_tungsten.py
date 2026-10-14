"""Tungsten selective emitter (design §4.1 L2, §12 "Tungsten")."""

import numpy as np
import pytest

from lss.lamp import lamp as lamps
from lss.physics.thermal import tungsten
from lss.spectral import colorimetry as cm
from lss.spectral.continuum import PlanckContinuum
from lss.spectral.grids import analysis_grid
from lss.spectral.spectrum import Spectrum

# Independent readings of De Vos's curves, used only to check the adopted table.
# TPRC vol. 7 (1970) table 234 curve 18 (De Vos 1600 K), as served by EKHI.
TPRC_1600K = {300: 0.482, 400: 0.481, 500: 0.469, 600: 0.456, 700: 0.444, 800: 0.432, 900: 0.412,
              1000: 0.390, 1600: 0.263, 1800: 0.234, 2000: 0.210, 2200: 0.190}
# Kravchenko & Mamuta (2023) table 2, De Vos re-digitised and extrapolated to 3000 K.
KRAVCHENKO_3000K = {300: 0.452, 500: 0.445, 600: 0.430, 700: 0.415, 800: 0.396, 900: 0.380, 1000: 0.366,
                    1100: 0.350, 1400: 0.315, 1600: 0.296, 1800: 0.281, 2000: 0.265, 2200: 0.253, 2500: 0.237}


def test_table_obeys_tungsten_physics():
    # Short wavelengths: ε falls with T. Long: it rises. One crossover, near 1.25 µm.
    tab = tungsten.table()
    slope = np.diff(tab.values, axis=1)  # change between successive temperatures
    sign = np.sign(slope.sum(axis=1))
    assert np.all(sign[:, None] * slope >= 0), "ε must change monotonically with T at each wavelength"
    nonzero = sign[sign != 0]
    assert np.count_nonzero(np.diff(nonzero)) == 1, "exactly one crossover"
    crossover = tab.wavelengths_nm[sign == 0]
    assert np.all((crossover > 1100) & (crossover < 1400))
    # Above 0.4 µm every curve falls with wavelength.
    vis_ir = tab.wavelengths_nm >= 400
    assert np.all(np.diff(tab.values[vis_ir], axis=0) < 0)


def test_agrees_with_independent_tprc_reading():
    lam = np.array(list(TPRC_1600K))
    np.testing.assert_allclose(tungsten.emissivity(lam, 1600.0), list(TPRC_1600K.values()), atol=0.002)


def test_linear_extrapolation_agrees_with_kravchenko_3000k():
    lam = np.array(list(KRAVCHENKO_3000K))
    np.testing.assert_allclose(tungsten.emissivity(lam, 3000.0), list(KRAVCHENKO_3000K.values()), atol=0.002)


def test_temperature_and_wavelength_limits():
    tab = tungsten.table()
    assert not tab.extrapolated(2800.0) and tab.extrapolated(2856.0)
    with pytest.raises(ValueError):
        tungsten.emissivity(550.0, 1500.0)
    with pytest.raises(ValueError):
        tungsten.TungstenContinuum(3300.0)
    assert np.isnan(tungsten.emissivity(250.0, 2000.0))
    assert tungsten.TungstenContinuum(2000.0).density(np.array([250.0, 2600.0])).tolist() == [0.0, 0.0]


def test_continuum_is_epsilon_times_planck():
    lam = np.array([400.0, 555.0, 1234.0])
    c = tungsten.TungstenContinuum(2400.0)
    expected = tungsten.emissivity(lam, 2400.0) * PlanckContinuum(2400.0).density(lam)
    np.testing.assert_allclose(c.density(lam), expected, rtol=1e-12)


def _cct_and_uv(continuum):
    binned = Spectrum.of(continuum).bin(analysis_grid())
    return cm.cct_duv(binned)[0], cm.uv_prime(binned)


def test_selective_tungsten_is_bluer_than_its_temperature():
    # Design §12: tungsten at 2856 K has a CCT above 2856 K; a greybody does not.
    cct_w, _ = _cct_and_uv(tungsten.TungstenContinuum(2856.0))
    cct_p, uv_p = _cct_and_uv(PlanckContinuum(2856.0))
    _, uv_grey = _cct_and_uv(PlanckContinuum(2856.0, scale=0.45))
    assert cct_w > cct_p + 30.0
    np.testing.assert_allclose(uv_grey, uv_p, atol=1e-12)


def test_level_2_preset_uses_true_temperature_and_matches_illuminant_a_colour():
    lamp = lamps.load("incandescent_cie_a")
    l1, l2 = lamp.generate(1), lamp.generate(2)
    assert l1.metadata["params"]["filament_temperature"]["value"] == 2856
    t2 = l2.metadata["params"]["filament_temperature"]
    assert t2["empirical"] and t2["value"] < 2856
    assert l2.empirical_count == 1 and l1.empirical_count == 0
    assert l2.metadata["valid_range_nm"] == (300.0, 2500.0)
    assert not any("extrapolated" in d for d in l2.metadata["data_versions"])
    cct, duv = cm.cct_duv(l2.spectrum.bin(analysis_grid()))
    assert cct == pytest.approx(2855.7, abs=1.0)
