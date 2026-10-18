"""Soot as a flame's light (design §4.5 L2)."""

import numpy as np
import pytest

from lss.lamp import lamp as lamps
from lss.physics.flame import soot
from lss.spectral import colorimetry as cm
from lss.spectral.continuum import PlanckContinuum
from lss.spectral.grids import analysis_grid
from lss.spectral.spectrum import Spectrum

# Chang & Charalampopoulos (1990) Table 3, Kramers–Kronig values 8 mm above the burner,
# the height their eqs. 19 were fitted at: (λ µm, n, k).
CHANG_TABLE3_8MM = ((0.40, 1.60, 0.66), (0.54, 1.73, 0.58), (0.71, 1.76, 0.56), (1.00, 1.82, 0.54),
                    (3.89, 2.18, 1.11), (6.24, 2.39, 1.45))


def test_dispersion_agrees_with_the_papers_values_as_it_states():
    # "agree, on the average, to within 5 % for n and 6 % for k" at 8 mm. (Their Table 4,
    # the per-entry differences, cannot be reproduced from Table 3 and eqs. 19 as printed: at
    # 1 µm eq. 19a gives 1.811 exactly against 1.82, 0.5 %, where Table 4 says 2.0 %.)
    m = soot.refractive_index(np.array([r[0] for r in CHANG_TABLE3_8MM]) * 1000.0)
    n, k = np.array([r[1] for r in CHANG_TABLE3_8MM]), np.array([r[2] for r in CHANG_TABLE3_8MM])
    dn, dk = np.abs(m.real - n) / n, np.abs(-m.imag - k) / k
    assert dn.mean() < 0.05 and dk.mean() < 0.06
    assert dn.max() < 0.06 and dk.max() < 0.08  # worst: n at 0.40 µm, k at 1.00 µm
    assert m[3] == pytest.approx(1.811 - 0.5821j)  # ln λ = 0 at 1 µm: the constant terms


def test_absorption_function_and_its_slope():
    e = soot.absorption_function(soot.refractive_index(np.array([450.0, 550.0, 850.0])))
    assert e == pytest.approx([0.261, 0.238, 0.218], abs=0.001)
    # κ ~ E/λ: an exponent about 1.28 over 450–850 nm, between the two bounds (1 and 1.39).
    lam = np.linspace(450.0, 850.0, 41)
    alpha = -np.polyfit(np.log(lam), np.log(soot.relative_absorption(lam)), 1)[0]
    assert 1.25 < alpha < 1.31
    assert soot.relative_absorption(np.array([425.0]), "flat")[0] == pytest.approx(550.0 / 425.0)
    assert soot.relative_absorption(np.array([425.0]), "hottel_broughton")[0] == pytest.approx((550 / 425) ** 1.39)
    for name in soot.sets():
        assert soot.relative_absorption(np.array([550.0]), name)[0] == pytest.approx(1.0)
        assert np.isnan(soot.relative_absorption(np.array([350.0]), name)[0])  # outside every range


def test_thin_and_thick_limits():
    lam = np.linspace(450.0, 850.0, 9)
    b = PlanckContinuum(1900.0).density(lam)
    thin = soot.SootContinuum(1900.0, 1e-6).density(lam)
    np.testing.assert_allclose(thin, 1e-6 * soot.relative_absorption(lam) * b, rtol=1e-6)
    thick = soot.SootContinuum(1900.0, 50.0).density(lam)
    np.testing.assert_allclose(thick, b, rtol=1e-12)
    assert soot.SootContinuum(1900.0, 0.3).density(np.array([300.0, 31000.0])).tolist() == [0.0, 0.0]


def test_thin_soot_is_bluer_than_a_grey_body_at_the_same_temperature():
    # Soot's emissivity falls with wavelength, like tungsten's (design §4.5).
    def cct(c):
        return cm.cct_duv(Spectrum.of(c).bin(analysis_grid()))[0]

    planck = cct(PlanckContinuum(1900.0))
    thin, deep = cct(soot.SootContinuum(1900.0, 1e-6)), cct(soot.SootContinuum(1900.0, 3.0))
    assert planck == pytest.approx(1900.0, abs=2.0)
    assert 2050.0 < thin < 2150.0
    assert planck < deep < thin


def test_bad_inputs_fail_clearly():
    with pytest.raises(ValueError):
        soot.SootContinuum(1900.0, 0.0)
    with pytest.raises(KeyError):
        soot.SootContinuum(1900.0, 0.1, "graphite")
    with pytest.raises(ValueError):
        soot.refractive_index(550.0, "flat")


def test_candle_preset_is_unscored_and_flags_its_placeholders():
    lamp = lamps.load("candle_paraffin")
    assert lamp.reference is None
    profile = lamp.generate(2)
    assert profile.metadata["generator"] == "soot"
    assert profile.empirical_count == 2
