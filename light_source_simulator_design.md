# Light Source Simulator — Design Document

Physics-first desktop tool for modelling lamps as spectral emitters, rendering their effect on a scene with a spectral ray tracer, and comparing sources. Intended to be built incrementally with Claude Code; each section ends with concrete, testable work items.

**Revision 2 (2026-10-14).** Rewritten after a design review. Main changes from revision 1:
- Maximum physical accuracy is the stated north star, and there is a new section on the path to it (§11): reference data, an accuracy report, parameter fitting, no hidden fudge factors.
- Two spectral representations. The physics layer works line by line on fine local grids; rendering uses coarse bins. Optically thick lines can no longer be binned before radiative transfer.
- κ is computed from line data and ε from Kirchhoff, which fixes a unit error. Stage C's radiation term gets an explicit source: a net emission coefficient first, the interior radiation field later. Total power comes from a wide-range or analytic integral, not the 300–1000 nm grid.
- Emission is a sum of fixed spatial/angular *zones*, each with its own spectrum. Zone shapes never depend on physics parameters. Light returning to the emitter is the one stated approximation. The renderer renders only unit-spectrum lighting maps per zone, so every spectrum edit is instant, and that now includes arc lamps whose spectrum varies across the tube.
- Clear split: generators produce bare emission (including phosphor conversion). Envelopes, reflectors and filters are traced by the renderer.
- The fluorescence claim is removed: scene materials must not fluoresce.
- The first usable version is fixed scenes with instant re-colouring. Drag-the-lamp progressive rendering moves to a later milestone.
- The roadmap is reordered: physics and lamp sheets first, then a renderer spike, then the renderer.

---

## 1. Goals and non-goals

**North star.** Predict a lamp's spectrum, flux, angular distribution and efficacy from its *physical* parameters (dimensions, fill, current, phosphor layer, temperatures) as accurately as physics allows, for every lamp family. Day one need not be maximally accurate. Every design choice must keep the road to higher accuracy open and incremental: add a physics rung, measure the improvement, keep the rest working.

**Goals**
- Model light sources as a ladder of models per family, from measured data and simple fits up to a self-consistent 1-D arc plasma, with instant switching between models.
- Make accuracy measurable: every model is scored automatically against reference measurements (§11).
- Represent a lamp as an assembly of swappable components (emitter, envelope, coating, reflector, filter).
- Render fixed scenes spectrally and make every spectrum-side edit (lamp model, parameters, dimming, lamp-side filters) instant.
- Measure: spectrum at a point, illuminance, CCT/Duv, CRI/TM-30, efficacy, UV/IR quantities; compare sources and model levels side by side.

**Non-goals (for now)**
- Interactive drag-the-lamp rendering in the first usable version. The design is kept (§9.6) for a later milestone.
- Photorealistic materials beyond what lighting studies need.
- Fluorescent scene materials (optical brighteners in paper or cards, fluorescent dyes). They would break re-colouring (§9.3), so the scene library avoids them.
- Time-resolved behaviour (AC modulation, flicker, warm-up), 3-D arc effects (convection, electrodes as geometry), non-LTE. These are later physics rungs, each with a defined plug-in point (§5.4).

---

## 2. Core abstraction

The renderer never knows what a lamp is. Every lamp, at every model level, produces an `EmissionProfile`:

```
EmissionProfile
  geometry   : emitting surface(s) in lamp-local coordinates
  zones[k]   : fixed grey spatial/angular basis functions  shape_k(x, ω)   (dimensionless)
  spectra[k] : spectrum of each zone  S_k(λ)   [W·m⁻²·sr⁻¹·nm⁻¹], as line list + continuum
  radiance(x, ω, λ) = Σ_k shape_k(x, ω) · S_k(λ)
  power      : total radiant flux over ALL wavelengths (from the generator's energy budget)
  includes   : which components are already folded in {phosphor, envelope, reflector, luminaire}
  metadata   : model level, parameters, data versions, derived metrics
```

Why zones:
- **They keep re-colouring exact for any lamp.** The zones are fixed per emitter *geometry*, not per parameter set, so the renderer renders one lighting map per zone once (§9.3). A physics change only changes the `S_k`, which are applied after rendering.
- **They capture real spatial and angular colour variation:**
  - an arc tube's spectrum varies across its width (self-reversal) and with viewing angle, so zones are bands in impact parameter b and angle to the tube axis θ;
  - an LED's colour varies over angle (more blue leaks straight ahead, the "yellow ring" at wide angles), so zones are bands in emission angle, each with its own spectrum;
  - a filament is cooler near its supports, so it has zones along its length.
- **Zone rule: a zone's shape never depends on a physics parameter.** Zones are fixed spatial/angular *bins* (bands in b, θ, emission angle, position along the coil). Everything parameter-dependent goes into the zone spectra. A "pump zone" and a "phosphor zone" with their own angular patterns would break this: those patterns change with phosphor thickness, which would invalidate the lighting maps.
- **Accuracy is controlled by zone count.** Refining the zones until the rendered result stops changing is a test (§12). A filament or a fluorescent tube can start with a single zone.
- **Light returning to the emitter is the one non-linear coupling.** Reflectors and outer bulbs send light back onto the emitter.
  - An arc is opaque at line cores and transparent elsewhere, and that transmission changes with arc parameters.
  - An LED phosphor re-absorbs and re-emits returning blue.
  - **v1 approximation:** the emitter boundary is treated as a fixed surface (arc: transparent; LED/filament/phosphor tube: a fixed grey-diffuse reflector), and that choice is part of the emitter geometry.
  - **Consequence:** for lamps with strong back-reflection (reflector HID, reflector LED) "instant re-colouring" is exact only within this approximation. The error is bracketed by a test (§12).
  - The accurate rung is an emitter whose boundary transmission or reflection comes from the generator (arc κ along the chord, phosphor absorption). It costs one extra map set per returning-light bounce class.

Consequences:
- Switching model level means selecting a different cached set of `S_k` for the same geometry. It is instant.
- An expensive model (arc solve) runs once per parameter set, is cached on disk by parameter hash, and never blocks the UI.
- A half-finished high-level model is still usable, because it only has to produce zone spectra.
- `includes` prevents double counting. A measured whole-lamp SPD already contains its envelope and phosphor, and an IES/LDT file describes a whole luminaire in the far field, so neither is passed through those components again. IES data is applied as the angular distribution of a *small* source, never smeared over an extended emitter surface.

---

## 3. Spectral representation

There are two representations, because physics accuracy and rendering cost pull in opposite directions.

**3.1 Physics representation (exact).**
- Line spectra are line lists (centre, strength, profile model) plus a continuum on a wavelength grid. Lines are never sampled at their centre onto a coarse grid.
- Radiative transfer and anything nonlinear in κ runs **line by line on an adaptive fine grid local to each line**. The resolution is set by the narrowest feature: Doppler cores are picometres wide, and HPS self-reversal dips are a few nm. Only the *outgoing* radiance is integrated into analysis bins. Binning ε or κ before transfer is forbidden, because exp(−∫κ ds) does not commute with bin averaging.
- Filters and envelopes act on line lists exactly: T(λ) is evaluated over each line profile, not via a binned value. A filter edge inside a bin is then handled correctly.

**3.2 Analysis grid (colorimetry and lamp sheets).**
- 360–830 nm at 1 nm, matching the CIE tables. Colour metrics come from bin integrals of the exact spectrum.
- Extended band for UV and IR quantities: 200–3000 nm, variable spacing. It is used for UV-A/UV-B dose and IR load.
- **Total power is never a grid integral.** Thermal sources use analytic band integrals of the Planck function (and, for a wavelength-dependent emissivity, a dedicated wide-range integral of ε(λ)·B_λ), since at 2856 K about 75 % of the power lies beyond 1000 nm. Discharges use the generator's own energy budget (Stage C), which includes every line at every wavelength.

**3.3 Render grid (lighting maps).**
- Coarse box bins over the renderer's wavelength range. The default is 10 nm; the bin width is a per-scene setting.
- Justification: lighting maps carry the *scene's* spectral response (smooth reflectances, glass), not the lamp's spikes. The spiky lamp spectrum is applied exactly afterwards: `pixel_XYZ = Σ_bins map_bin · ∫_bin S(λ)·cmf(λ) dλ`. This assumes the map is constant within a bin.
- A scene containing a sharp spectral feature (dichroic, narrow-band gel) needs finer bins. The bin-halving convergence test (§12) decides.

**3.4 Units and colour.**
- Radiometric units are the source of truth. Photometric values come from V(λ) with K_m = 683 lm/W. Mesopic, melanopic and S/P quantities use their own action spectra.
- Colour is derived at display time only: spectrum → XYZ (CIE 1931 2°, optionally 2006 10°) → linear sRGB → exposure/tone map → display.
- Colour science uses the **`colour-science`** package (CMFs, CCT/Duv, CRI, TM-30, CAMs). Our code wraps it; tests check against CIE published values.

---

## 4. Lamp physics models

Each family has a ladder. **Level 0 is the measured reference** (tabulated SPD, optionally with angular and electrical data). It is not "lowest fidelity": it is ground truth, and every other level is scored against it (§11). Levels 1, 2, … add physics. Each rung replaces a fitted or empirical parameter with a physical mechanism wherever possible.

Every lamp preset has one **physical description** shared by all levels: geometry, materials, fill, electrical operating point. Each level derives its inputs from it, or from documented calibration where the physics is not yet modelled. Without this, "the same lamp at another level" is undefined.

### 4.1 Incandescent / halogen

- **L1 — Planck** at filament temperature T.
- **L2 — Tungsten selective emitter.** ε(λ, T) from De Vos (1954), as read from his curves in NASA TN D-1088 (1961) and cross-checked against two independent readings (`lss/data/tungsten_emissivity.yaml`). Measured 1600–2800 K, 300–2500 nm; extrapolated linearly in T to 3200 K and flagged when it is. Its temperature is the *true* filament temperature, which is not the distribution temperature that L1 uses: Illuminant A's colour needs about 2800 K (fitted with `lss fit --objective colour`). Tungsten's emissivity falls with wavelength across the visible, and *that* is why its CCT exceeds its true temperature. (A true greybody, with constant ε, has exactly the chromaticity of Planck at the same T.)
- **L3 — Coil geometry.** Coiled-coil cavity effect.
  - First step: ε_eff = ε / (ε + (1−ε)·f_open).
  - Accurate step: ray-trace the coil mesh in the renderer as an emitting tungsten surface with spectral BRDF 1−ε(λ, T) (virtual instrument, §9.5). This replaces f_open with geometry.
  - *Built (step 2), first step:* generator `tungsten_coil` (`lss/physics/thermal/tungsten.py`). It takes De Vos's ε(λ, T) and applies the one-zone cavity formula above. That formula assumes an isothermal cavity with diffuse walls; f_open is the ratio of the opening's area to the radiating wall area (1 is a flat ribbon, which is L2; 0 is a closed cavity, a blackbody). Because the gain is largest where ε is smallest, a coil is greyer, so redder, than a ribbon at the same temperature. Two empirical numbers: temperature and f_open.
  - *Scored against a measured lamp* (§11: NIST-calibrated 1000 W FEL halogen lamps, coiled-coil filament, 300–2400 nm). The fit weights each of the table's wavelengths by NIST's stated uncertainty there (`lss fit --objective uncertainty`), and the report gives χ² per node: the mean squared error in units of that uncertainty, each node's total uncertainty treated as independent. About 1 means within the total stated uncertainty point by point; that is generous as a test of shape (see below). Fitted to lamp F-455: T = 3091 K, f_open = 0.30; χ² per node 0.71, NRMS 1.0 %, Δu′v′ 1.7×10⁻⁴ (the floor from interpolating the table is 1.4×10⁻⁴), ΔCCT −5 K. The ladder climbs as it should on real tungsten: χ² per node 73 for Planck, 34 for flat tungsten (L2), 0.71 for the coil (L3). The ranking holds under every treatment of the uncertainty below. The second lamp, F-456, scored with F-455's values and not refitted: χ² per node 1.1, Δu′v′ 2.9×10⁻⁴. Both lamps are one model from one batch, so this shows the fit carries to another lamp of the same design, not to other designs.
  - *Why the uncertainty weighting.* The default `spectral` objective weights every wavelength by the mean level of the table. Then the bright near IR (≈ 90 % of the mean at 800–1300 nm) outvotes the dim violet and UV (2–20 %). Fitted that way the coil lands at T = 3075 K, f_open = 0.35, NRMS 0.68 %, but 2–3.5 % low at 300–400 nm, three to six times the stated uncertainty there; about two thirds of its squared error is then above 780 nm, where it is within the uncertainty. Weighting by the uncertainty trades 0.35 % of NRMS for a fit within the total stated uncertainty at every node, and the colour error falls from 6.4×10⁻⁴ to 1.7×10⁻⁴.
  - *Finding (likely, not settled): the one-zone formula probably misses the shape by a few tenths of a per cent.* Most of NIST's 0.9–4.4 % is systematic; the random, point-to-point part is only 0.1–0.2 % in the visible (k = 2). The report does not say how the systematic parts correlate between wavelengths, so the verdict depends on an assumption (`lss/data/nist_fel.yaml` holds the parts; `tests/test_nist_fel.py` reproduces the table). Coil χ² per degree of freedom over 300–2400 nm, and the open fraction fitted separately to 300–800 and 800–2400 nm:

    | Uncertainty treatment | χ²/dof | f_open, 300–800 vs 800–2400 nm | Gap |
    |---|---|---|---|
    | total, independent per node (the report's χ² per node) | 0.79 | 0.28 ± 0.05 vs 0.42 ± 0.07 | 1.7σ |
    | random parts only | 4.6 | 0.29 ± 0.01 vs 0.45 ± 0.06 | 2.9σ |
    | random, plus each systematic part fully correlated across wavelength | 4.0 | 0.27 ± 0.01 vs 0.46 ± 0.06 | 3.2σ |

    The 300–800 nm window does not fit well even alone (χ²/dof ≈ 4–5 under the last two treatments); scaling each window's errors by √(χ²/dof) leaves gaps of 2.5σ and 2.8σ. The model's own input is uncertain too: De Vos's emissivity, read off his curves to about ±0.002, is 0.1–0.5 % after the cavity formula, as large as NIST's random part. Taken generously (that spread as one standard deviation, independent per node) it brings χ²/dof to 1.3 but leaves the gap at 2.3σ. So the shape miss is probably real but its cause is open: the emissivity data and its extrapolation from 2800 K to 3091 K; a filament not at one temperature (L5: hot parts dominate the blue, the average the IR, as the short windows' higher temperature suggests); or the formula itself (a coil's outer faces radiate as bare tungsten, only its inner faces as a cavity). *History:* an unweighted split first suggested this (0.18 against 0.46); commit c5d20e0 withdrew it after weighting by the total uncertainty as independent; separating the random part from the systematic part brings it back.
  - *Not fitted against Illuminant A.* Tried once (2026-10-14, a copy of the A preset with a bounded f_open): both objectives drive f_open to its lower bound, 0.05, a nearly closed cavity, and `lss fit` flags it. A is a blackbody by definition, so that is an artefact of the definition, not a finding. The CIE A preset stops at L2.
- **L4 — Temperature from electrical input.** Solve P_elec = P_rad(T) + P_cond(T) + P_lead. Inputs:
  - P_rad from tungsten's *total* hemispherical emissivity with the coil's cavity effect (De Vos's spectral data cannot be integrated over all λ: it stops at 2.5 µm, short of about 13 % of the power at 3100 K);
  - P_cond from a fill-gas (Langmuir sheath) model;
  - tungsten resistivity ρ(T) linking V, I and T;
  - the empirical lamp laws (T ∝ V^0.42 …) only as the documented low rung.
  - *Built (step 2), first step: temperature from resistance.* Generator `tungsten_resistance` (`lss/physics/thermal/resistance.py`). The hot resistance is voltage over current. Tungsten's resistivity (Desai et al., J. Phys. Chem. Ref. Data 13, 1069 (1984), Table 5; `lss/data/tungsten_resistivity.yaml`) gives the temperature at which R(T)/R(293 K) equals hot over cold resistance. The column *uncorrected* for thermal expansion is used, because that is what a wire's resistance ratio measures. The spectrum is then Level 3's coil at that temperature. Current and voltage are measured inputs in the lamp's physical description; the cold resistance is one empirical number, and it replaces the temperature. This is how metrology labs take an FEL lamp's temperature (Kärhä et al., NEWRAD 2011). With both voltage and current known the power balance is not needed for the temperature, but without it the model cannot say what the lamp does at another current; the second step below adds it. The presets' Level 4 is now that second step. This first step stays as generator `tungsten_resistance`, and it is how the second step's filament surface is solved.
  - *The cold resistance is fitted, so lamp F-455 tests nothing.* Neither NIST report, nor any FEL datasheet found (2026-10-15), gives the cold resistance or the wire's dimensions. Fitted to F-455 it is 0.7331 Ω, which puts the filament at Level 3's 3091.2 K: on F-455, Level 4 is Level 3 by construction.
  - *Finding: one lamp's cold resistance does not carry to the other.* F-456 draws the same 8.200 A at 109.8 V, against F-455's 109.4 V: 0.37 % more resistance. Resistance rises as T^1.21 here, so with F-455's cold resistance F-456 comes out 9.3 K hotter (3100.5 K). Its spectrum, with F-455's open fraction and only the temperature fitted, says 3.2 K hotter (3094.4 K); the ratio of the two lamps' tables, in which the systematic errors they share cancel, says 2.4 K (§11). At the resistance temperature its shape score worsens, χ² per node from 1.07 (F-455's temperature carried over, Level 3) to 2.29, although its colour error falls from 2.9×10⁻⁴ to 0.7×10⁻⁴ in u′v′: colour alone would have called this an improvement. The gap is 0.24 % of F-456's hot resistance: about 32 mΩ, or 0.26 V on its voltage. Rounding does not explain it (each voltage is given to 0.1 V, so their difference is uncertain by about ±0.1 V, ±2 K). Two causes fit equally well, and the data cannot tell them apart: the filaments differ (0.24 % in cold resistance, e.g. 0.12 % in wire diameter, is ordinary lamp-to-lamp spread), or the voltages were read differently. NIST recorded them "for reference only" and does not say where on the lamp they were taken, so how much lead, seal and contact resistance each includes is unknown. Either way, Level 4 needs each lamp's own measured cold resistance and a voltage taken at the lamp itself (four-wire); a shared cold resistance transfers no better than a shared temperature.
  - *A measured cold resistance would not be precise either.* Desai states ±2 % for the resistivity up to 2500 K and ±3 % above. If the errors at 293 K and at 3000 K were independent, the temperature would carry about 3 %, ~90 K; if fully correlated, they cancel. Desai does not say which. A fitted cold resistance avoids this, because only the local slope of the table matters then.
  - *Built (step 2), second step: the power balance.* Generator `tungsten_balance` (`lss/physics/thermal/balance.py`). The current alone sets the temperature, where I²·R_cold·ρ(T)/ρ(293 K) = P_rad(T) + P_gas(T); the voltage is then a prediction, not an input. The spectrum is Level 3's coil at that temperature.
    - **Radiation:** P_rad = A·f·ε_eff·σT⁴, with A the wire's surface, f the coil's open fraction (the power leaves through the opening f·A) and ε_eff the one-zone cavity formula applied to tungsten's total hemispherical emissivity: Forsythe & Watson, J. Opt. Soc. Am. 24, 114 (1934), Table I, to 3000 K, and Forsythe & Adams, J. Opt. Soc. Am. 35, 108 (1945), above (`lss/data/tungsten_total_emissivity.yaml`). Applying the cavity formula to a total rather than wavelength by wavelength is a first step.
    - **Gas loss:** one term for the heat carried off by the fill gas and conducted into the supports, P_gas ∝ T^m, set by its share of the input at 3000 K. Langmuir's sheath model would compute it, but the fill gas, its pressure and the coil's dimensions are unknown for FEL lamps. Share 0.10 and m = 2 (the sheath model's slope near 3100 K for conductivity rising as T^0.75 and a wall near 900 K) are both `empirical`; the share is a round guess.
    - **Leads:** the resistance of the leads and seals is in series and carries the same current, so on a current supply it changes the voltage read across the lamp, not the filament's own power. It has no term.
    - **The surface is solved, not measured.** At F-455's 8.200 A and 109.4 V the first step gives the temperature (3091.2 K, Level 3's), and the surface is the one whose losses there equal I·V = 897.1 W: 8.37 cm². On F-455 Level 4 is therefore still Level 3 by construction, now with 5 empirical numbers instead of 2. The open fraction is held at Level 3's value (with the surface fixed it alone would swing the filament from 2550 K to past 3400 K), so `lss fit --level 4` fits only the cold resistance, and returns the same 0.7331 Ω. With the cold resistance the surface implies a wire 0.290 mm thick and 0.918 m long (0.239 mm and 0.62 m as a straight wire), a plausible 1 kW, 110 V filament; no datasheet found gives it. The surface is the hot one, about 3 % larger than at room temperature (tungsten expands about 1.6 % in length by 3000 K), while the cold resistance is at room-temperature dimensions, so the implied size is 1–2 % off.
    - **F-456 looks better than it is.** With F-455's constants and the same current it comes out at F-455's temperature and voltage: 109.4 V against its own 109.8 V. Its scores are then Level 3's with F-455's temperature carried over (χ² per node 1.07, Δu′v′ 2.9×10⁻⁴), better in shape than the first step's 2.29, but only because its voltage is no longer read. The accuracy report's flags for this change (Δu′v′ and ΔCCT up on F-456, empirical numbers 2 → 5) were accepted for that reason.
  - *Check against NIST's measured response: a sanity check, not a verdict.* Walker & Thompson, J. Res. NIST 99, 255 (1994), §1: "By way of comparison, measurements made on our FEL-type standard spectral irradiance lamps operating at 8.000 A give a value of about 0.06% per 1 mA in irradiance at 655 nm." Those are NIST's own FEL lamps, not F-455 or F-456. The check does not depend on the filament's size, only on how the losses and the resistance scale with temperature (tested: twice the surface at the same temperature gives the same response per fractional change of current). The figure has one digit, so the band 0.05–0.07 %/mA was set before the model was run. F-455's model, run at 8.000 A (3036 K, 104.4 V):

    | Gas share at 3000 K | 0 | 0.05 | 0.10 | 0.15 | 0.20 |
    |---|---|---|---|---|---|
    | Coil, %/mA | 0.060 | 0.062 | 0.065 | 0.068 | 0.070 |
    | Straight wire (f = 1), %/mA | 0.057 | 0.059 | 0.061 | 0.064 | 0.067 |

    The response rises with the gas share. Where it leaves the band depends on how NIST's one digit is read: about 0.19 (coil) if "about 0.06" means 0.05–0.07, about 0.10 if it means 0.055–0.065. So the check does not usefully bound the gas loss. The total emissivity's slope with temperature is poorly fixed by the table's three figures right where the lamp runs: the rows give d ln ε/d ln T = 0.58 below 3000 K and 0.32 above, and the 1934 table's own slope column 0.53 at 3000 K. The low slope above 3000 K rests only on the 1945 table's two rows there, whose origin is not stated and which nothing cross-checks. That spread moves the coil's 0.065 between 0.063 and 0.065, the straight wire's 0.061 between 0.058 and 0.062. So the check passes whatever the unknowns are. Passing shows the balance behaves sensibly, not that its physics is right. A coil follows the current more steeply than a straight wire, because the cavity flattens the emissivity's rise with temperature.
  - *Transcription check of the emissivity data.* The 1934 table gives the radiated power per area beside each emissivity. Their ratio over T⁴ is one constant (5.71×10⁻¹² W cm⁻² K⁻⁴, the radiation constant of the time) on every row within 0.3 %. The 1945 table's values at 1200–3000 K are the 1934 values reprinted, so the two papers are not independent measurements; above 3000 K the 1945 values' origin is not stated. The same 1934 table's resistivity ratio ρ(3000 K)/ρ(293 K), 16.4, is 6 % below Desai's 17.6; its slope agrees with Desai's within 1.5 %, and only the slope matters while the cold resistance is fitted.
  - *Next step:* a lamp measured at several currents (spectrum, current and a four-wire voltage at each) would test the balance properly: the voltage and spectral change it predicts at another current, against measurement, with the gas share fitted at one current and tested at the others. Langmuir's sheath model can replace the gas share once a lamp's fill and coil dimensions are known.
- **L5 — Non-uniform filament.** Temperature profile along the coil, cooler at supports and leads, from 1-D heat conduction. Gives zones along the filament. Envelope (quartz vs borosilicate, frosting) is traced by the renderer.

### 4.2 LED

- **L1 — Two-band fit.**
  - Pump: Gaussian *in energy*, ~450 nm, FWHM 18–25 nm, with an optional low-energy tail.
  - Phosphor: YAG:Ce, asymmetric, fitted with a split Gaussian or the sum of two Gaussians.
  - *Built (step 2):* generator `led_two_band` (`lss/physics/led/bands.py`). Both bands are Gaussians in photon energy; the phosphor has a separate width on each side. Widths are in nm, measured on the energy shape (peak to half maximum). Six parameters, all empirical and fitted: pump peak and width (shared by every level), phosphor peak, two half widths and phosphor-to-pump power ratio (Level 1's own).
  - *Finding:* one phosphor band fits a cool-white LED (CIE LED-B5: Δu′v′ 7×10⁻⁴, ΔRa −0.8, no parameter at a bound). It does not fit a neutral-white one (LED-B3: Δu′v′ 2.3×10⁻³, ΔRa −5.2). B3's fitted band is lopsided the wrong way, wider on the blue side (80 nm, at its bound) than the red (71 nm), while a single Ce³⁺ band, like B5's (66 vs 77 nm), has its longer tail to the red. B3 is a blend of phosphors. A pump with a red tail did not help. A second free phosphor band gave unstable fits with peaks at their bounds, so the blend belongs in L2 (built: garnet plus red nitride, below).
  - The fitted pump widths (24–26 nm) sit at or above the wide end of the 18–25 nm usual for InGaN. Table blur is already taken out: models are scored through the reference's 5 nm bandpass (§11), which took about 1 nm off each width. The rest is a pump shape that a Gaussian in energy does not match; a red tail on the pump did not help either. *Explained at L2 (below): absorption in the layer plus the die's blue hot-carrier tail.*
- **L2 — Phosphor conversion.** Beer–Lambert absorption of the pump (thickness × concentration), measured excitation and emission spectra, quantum efficiency, Stokes loss. Blue leakage emerges from thickness, including its angle dependence: the path through the layer grows as 1/cos θ. Multi-phosphor (red nitride) for high-CRI parts. Output: one spectrum per emission-angle zone.
  - *Built (step 2):* generator `led_phosphor_layer` (`lss/physics/led/layer.py`). Counted in photons on a wavenumber grid, so the Stokes loss follows from photon energies.
    - **Die:** a Gaussian in energy (alloy disorder) convolved with the Boltzmann tail exp(−ΔE/kT) of hot carriers, which gives the blue band a longer short-wavelength side (Schubert, *Light-Emitting Diodes*, §5.1). T is the tail's slope written as a temperature; the fits show it is a shape parameter, not the junction temperature (findings below). Per lamp, in the physical description: peak, width, tail temperature.
    - **Garnet (Ce³⁺):** a configurational-coordinate model with one phonon (Henderson & Imbusch 1989, ch. 5). The 4f→5d₁ absorption peaks S·ħω above the zero-phonon line and the emission S·ħω below it, both Gaussian with σ² = S(ħω)² coth(ħω/2kT) plus an inhomogeneous spread. T is the phosphor's own temperature, a separate input from the die's tail since 2026-10-15. The emission ends on ²F₅/₂ and ²F₇/₂, so it is two such bands. ħω = 200 cm⁻¹ is measured (Bachmann et al., Chem. Mater. 21, 2077 (2009)). S, the inhomogeneous width, the ²F splitting and the two bands' ratio are shared by every lamp (`lss/data/phosphors.yaml`, `led_phosphors`). Each lamp sets its own zero-phonon energy, which stands for its garnet's composition (Lu and Ga raise it, Gd lowers it).
    - **Nitride (Eu²⁺):** phenomenological and shared. One Gaussian emission band in energy, and an absorption edge (an error function) that is flat across the blue and falls off through the green.
    - **Layer:** a flat slab with no scattering. The die lights it from below as a Lambertian source, and the phosphors absorb by Beer–Lambert along each path (τ/cos θ). Each lamp sets two optical depths. Phosphor light is emitted isotropically; light going down is reflected by the die (reflectance 1 at this rung). Light the layer absorbs again is re-emitted by whichever phosphor took it, so the nitride takes part of the garnet's green. Quantum yields are 1, so the output is relative.
    - **Output:** three emission-angle zones (0–30°, 30–60°, 60–90°, angles inside the layer; refraction at its surface is not modelled). More blue leaks straight ahead than at wide angles, because the oblique path is longer.
    - The photon balance closes exactly (every pump photon leaves or is lost), and the leaked pump equals the analytic 2E₃(τ) of a Lambertian source.
  - *Scores* (`lss fit led_cie_b1 led_cie_b2 led_cie_b3 led_cie_b4 led_cie_b5 --level 2`). Five CIE phosphor-converted blue LEDs, 2650–6600 K. Level 1, fitted to each lamp alone with 6 numbers: NRMS 8.0/6.7/8.1/6.5/6.1 %. Level 2, in-sample: 1.7/2.1/2.8/5.1/4.7 %, Δu′v′ 0.2–0.7×10⁻³, |ΔRa| ≤ 0.6. 15 empirical numbers per lamp, 8 of them shared; 6 are fitted per lamp, as at Level 1, and the phosphor temperature is held (findings below).
  - *Out of sample* (shared constants fitted to the other four lamps, then only the lamp's own six fitted; `--hold-out <lamp>`): B1 2.9 %, Δu′v′ 1.7×10⁻³; B2 2.6 %, 8.6×10⁻⁴; B3 4.6 %, 1.2×10⁻³; B4 6.0 %, 6.3×10⁻⁴; B5 5.3 %, 6.9×10⁻⁴. So Level 2 predicts a lamp it never saw better than Level 1 fits it. **Limits of this test:** the five come in similar pairs (B1/B2 warm, B4/B5 cool), so a held-out lamp usually has a sibling among the four. B3, between the pairs, is the strictest case (8.1 % → 4.6 %). The shared constants are still learned from the same family of tables they are scored on. The real test is a measured phosphor spectrum, or LEDs from another source.
  - *Findings:*
    - L1's too-wide pump (24–26 nm) is explained. The die widths come out at 18–22 nm, inside InGaN's usual range. The garnet's absorption reshapes the leaked blue (its peak sits on the pump's centre in the warm LEDs, ~459 nm, and on its red side in the cool ones, ~478 nm), and the die's tail fills the short-wavelength side. The widths do not hinge on what the tail stands for: with the tail and the phosphor temperature both free (below) they stay at 18.6–22.8 nm. That the tail is hot carriers holds at most for the cool LEDs; the warm ones want a tail shorter than room temperature gives.
    - The fit picks a greener garnet for the warm LEDs (zero-phonon line ~21 100 cm⁻¹, a Lu/Ga-type garnet) and one close to YAG for the cool ones (~20 300 cm⁻¹, YAG 20 450). The two warm LEDs also carry 6–8 × the cool ones' nitride (optical depth 0.56–0.64 against 0.08–0.09; B3 0.34). This matches how warm and high-CRI parts are usually made: a green garnet plus a red nitride.
    - The fitted coupling S ≈ 3.3 is about half YAG:Ce's measured S ≈ 6. The model garnet's Stokes shift (2Sħω ≈ 1300 cm⁻¹) is therefore about half the measured 2400 cm⁻¹, and its width comes mostly from the inhomogeneous term. A 5 nm table sees the garnet's absorption only through the shape of the leaked blue, so S is weakly constrained; a measured excitation spectrum would fix it.
    - *The one temperature was the die tail's shape, not the junction's* (checked 2026-10-15). Level 2 first used one temperature for the die's tail and for the phosphors' phonon widths; the joint fit put it at the 300 K bound for the three warm LEDs and at 360–380 K for the cool ones, and in the hold-out fits it drifted to either bound. Moving each part alone by +80 K from the fitted value, without refitting:

      | NRMS (Δu′v′ in 10⁻³) | B1 | B2 | B3 | B4 | B5 |
      |---|---|---|---|---|---|
      | as fitted | 1.7 % (0.7) | 2.1 % (0.6) | 2.8 % (0.3) | 5.1 % (0.2) | 4.7 % (0.6) |
      | die tail +80 K | 2.6 % (0.9) | 3.3 % (1.3) | 5.0 % (1.3) | 8.4 % (3.8) | 8.9 % (3.5) |
      | phosphors +80 K | 1.9 % (0.1) | 2.2 % (1.3) | 2.9 % (0.9) | 5.4 % (1.5) | 4.8 % (0.8) |

      The spectral score rests on the tail. With the two temperatures free and fitted separately (bounds 200–500 K, each lamp's own six numbers refitted, shared constants held), the tail goes to 200 K (the bound) in B1 and B2, 282 K in B3, and 356–387 K in B4 and B5; the phosphor temperature lands at 284, 436, 310, 200 and 500 K, so the tables do not fix it. The gain is at most 0.25 % NRMS. A tail at or below room temperature is not a junction temperature, so it is now the die's own shape parameter (`die_tail_temperature`) and the phosphors have their own input (`phosphor_temperature`), held at the old fitted value; every score is unchanged. Level 3 heats the phosphors and moves the die's peak; it leaves the tail's shape at Level 2's value.
    - The ²F splitting (3120 cm⁻¹) and the nitride's width (132 nm) sit near the top of their ranges. The cool LEDs keep ~5 % error, mostly a cyan-valley shape (480–530 nm) that the garnet's short-wavelength edge does not follow; scattering, which lengthens paths and sharpens self-absorption, is the likely missing piece (L4).
    - *A measured LED outside the CIE set does not fit with these constants* (checked 2026-10-15). Wu et al. (Sci. Rep. 16, 7751 (2026)) measured two LEDs they made from a blue die and two commercial phosphors, "LCG525F2" (green) and "LCN650" (red), which could not be identified from public sources (`lss leddrive`, below). Level 2 fitted to sample 1 at 150 mA, over 430–715 nm: with the CIE constants and the lamp's own six numbers refitted, NRMS 19 % (Δu′v′ 2×10⁻²); with every shared constant freed as well, 3.1 % (Δu′v′ 1.2×10⁻⁴), but five of them run to their bounds (coupling S at 10, inhomogeneous width near 0, ²F splitting at 1500 cm⁻¹, band ratio at 0.1, zero-phonon line at 22 000 cm⁻¹), the nitride's emission peak sits near its lower bound (586 nm) and the die widens to 32 nm. The shape can be matched, but not with constants that mean anything, so the green phosphor is probably not a Ce³⁺ garnet like the CIE LEDs'. Widening the bounds to make it fit was not done.
- **L3 — Thermal.** Junction temperature from thermal resistance × dissipated power (the die's tail stays a shape parameter; see the L2 findings). Effects: pump peak shift (~0.03–0.05 nm/K), pump efficiency droop, phosphor thermal quenching.
  - *Built (step 2), first step:* generator `led_thermal` (`lss/physics/led/thermal.py`).
    - **Junction temperature:** T_j = T_board + R_th·(I·V − P_opt), the heat left in the package through one effective thermal resistance. Current, voltage and radiant power are the operating point's measured values (as incandescent L4's first step took the voltage); an efficiency model would predict them. The phosphors run at T_j: an assumption, since a phosphor in silicone also heats from its own Stokes loss.
    - **Relative to Level 2.** Level 2's die peak, die width and phosphor constants were fitted at its `phosphor_temperature`; Level 3 shifts them from there by T_j minus that temperature, so at that temperature it is Level 2 exactly (tested on all five CIE LEDs, which have no drive data). Level 2's quantum yields are 1, so a Level 3 yield is relative to that state; Level 3 refuses to run below it. The die's tail keeps Level 2's shape.
    - **Die:** peak +0.0389 nm/K and width +0.0466 nm/K, measured on a blue GaInN LED at 460–464 nm (Chhajed et al., J. Appl. Phys. 97, 054506 (2005), as tabulated by Cengiz, Azarifar, Arik, Micromachines 13, 1615 (2022); Lin et al. (2012) give 0.032 and 0.053 nm/K).
    - **Phosphors:** the phonon widths follow T_j (Level 2's model), and each yield quenches over one barrier, η = 1/(1 + C·exp(−ΔE/kT)), with C set by the fraction kept at 150 °C: 90.3 % for the garnet, from a Ga-substituted YAG (Y₂.₉₄Al₄GaO₁₂:Ce³⁺; Qiang et al., Ceram. Int. 42, 767 (2016)), standing in for every garnet here since YAG itself quenches only near 600 K (Ueda et al., J. Phys. Chem. C 119, 25003 (2015)); 94 % for the nitride (CaAlSiN₃:Eu²⁺; Kim et al., J. Alloys Compd. 633, 97 (2015)). The barrier, 0.3 eV for both, is assumed (`empirical`): the sources give only the end points.
    - **Only what changes colour.** The output is relative. A die that loses efficiency as it heats, or droops with current, dims every part of the spectrum alike, so it is left out.
    - New shared constants (`lss/data/phosphors.yaml`): the two die coefficients and two quenching numbers per phosphor, five of them cited, the two barriers assumed. New per-lamp inputs: current, voltage, radiant power, board temperature and the thermal resistance (`empirical`). No preset has Level 3 yet: the CIE tables come without drive data.
  - *Data:* Wu et al. (Sci. Rep. 16, 7751 (2026)): two LEDs (about 2900 K and 3400 K, CRI about 95), each measured three times in quick succession (about 2.5 s apart) at 150–500 mA in 50 mA steps, 25 °C ambient, integrating sphere, with current, voltage and radiant power. The supplement is CC BY-NC-ND 4.0, so `lss leddrive` downloads it from the publisher into a local cache (`LSS_CACHE`, default `~/.cache/lss`), checks its SHA-256, and writes nothing derived from it into the repository; the tests that need it skip without it.
    - The instrument zeroes weak signal, more of it at low current: at 150 mA the spectra end near 427 and 730 nm, at 500 mA near 413 and 752 nm, and even at 700–730 nm the 150 mA spectra sit well below the 500 mA ones. Any colour computed from them depends on the window. From 150 to 500 mA, v′ falls by 4.0×10⁻³ over 430–715 nm but by 1.3–1.7×10⁻³ over 470–650 nm, and Level 3's sign of Δv′ on warming flips between those windows. So the study does not compare chromaticities. It compares each 10 nm band's share of 450–680 nm, where every current is well above the floor.
    - **The LED warms while it is measured.** Within every triplicate the voltage falls (by 0.01–0.08 V) and the radiant power falls (0.1–1.9 %), and the light turns bluer, in every window tried. At a fixed current this isolates temperature. The floor's effect on these drifts, estimated from the 150-vs-500 mA difference scaled by the 1–2 % fall in signal, is at most 0.09 % in the blue bands, where the drifts are 1–3 %, and at most 0.04 % elsewhere, where they are 0.1–0.9 %.
  - *Findings* (drift at 300–500 mA, per 0.07 V of voltage drop, which is 4–9 K for 5–6 dies in series at 1.6–3.0 mV/K each, the usual range for blue GaInN in Cengiz et al.; the paper states neither):

    | Change of share (%) | blue 450–490 | green 500–570 | red 600–680 |
    |---|---|---|---|
    | measured, sample 1 | +1.56 | −0.46 | −0.11 |
    | measured, sample 2 | +2.18 | −0.80 | −0.11 |
    | Level 3 (B1–B3 phosphors), at the warming below, assumed barrier | +0.5 to +1.0 | −0.07 to −0.16 | −0.01 to −0.09 |
    | the same, steepest quenching the cited points allow | +0.9 to +1.4 | −0.07 to −0.21 | −0.06 to −0.19 |

    - *The die's cited shift fits.* Fitting the measured drift as Level 3's die-only change plus its phosphor-only change gives a warming of 3.5–3.9 K (sample 1) and 5.5–6.1 K (sample 2), through the cited 0.0389 nm/K, inside the 4–9 K the voltage gives, without assuming a die count. The fit leaves 37–60 % of the change unexplained (RMS), as expected with B1–B3's phosphors standing in for this LED's, and the dip a red-shifting die should leave near 440–450 nm is missing in the data.
    - *This LED's green dims several times faster than any cited curve allows; its red agrees.* One-barrier curves through the cited 150 °C points cannot lose more than about 0.12 %/K (garnet) and 0.07 %/K (nitride) at room temperature, the chord between 25 and 150 °C; the assumed 0.3 eV gives 0.014 %/K for the garnet. Even at the steepest, at the warming the die gives, the green share falls 3.5–7 times less than measured, while the red falls by about the measured amount. Even at 9 K, the top of the voltage's range, the green falls 1.5–4.8 times less, and the model's blue and red then change more than measured. A phosphor layer hotter than the die would dim both alike, so the likelier cause is a green phosphor much less stable than a garnet (orthosilicate greens, for example, are known to quench strongly; LCG525F2 could not be identified). This does not test the garnet's cited quenching, only shows this LED is outside it.
    - *Part of the loss with current leaves the colour alone.* Per fractional loss of radiant power per amp, v′ moves 0.037–0.050 within triplicates but 0.022–0.023 across currents (430–715 nm). The ratio depends on the window (at 470–650 nm, 0.011–0.019 against 0.008–0.009), so this is qualitative: some of the loss with current is droop (or the current's own blue shift of the die), which a heat-only Level 3 cannot predict. A droop term (the ABC model) is the next step for light output against current.
- **L4 — Volumetric phosphor.** Phosphor as a scattering, absorbing, re-emitting medium. This runs *inside the generator* (its own small Monte-Carlo on the package geometry), not in the scene renderer, because the scene renderer does no fluorescence. It predicts the per-angle-zone spectra from first principles.

### 4.3 Low-pressure discharge (fluorescent, LPS)

- **L1 — Line list + phosphor.**
  - Fluorescent: Hg lines (365, 405, 436, 546, 578 nm) with fixed ratios, plus tri-phosphor (Eu²⁺ ~450, Tb³⁺ ~543, Eu³⁺ ~611 nm) or halophosphate bands.
  - LPS: Na D 589.0/589.6 nm.
  - *Built (step 2), fluorescent tri-band:* generator `fluorescent_triband`.
    - Hg: the visible lines (404.7–579.1 nm) in the ratios measured in a low-pressure Hg–Ar pencil lamp (Sansonetti, Salit & Reader, Appl. Opt. 35, 74 (1996), Table 2), each with its Doppler width. The UV lines are left to L2: the phosphor coat absorbs and converts them.
    - Phosphors: blue BAM:Eu²⁺, green LaPO₄:Ce,Tb (LAP), red Y₂O₃:Eu (YOX). Their shapes belong to the phosphor, so they live in a shared library (`lss/data/phosphors.yaml`) that every tri-phosphor lamp reads. Each band is a Gaussian in photon energy (split, where one side is visibly longer). A lamp sets only its Hg, blue and red power relative to green: equal-flux scoring makes the overall scale invisible.
    - The shapes are *learned*, not measured: no tabulated phosphor spectra were found (2026-10-14). `lss fit` fits them jointly to CIE FL10, FL11 and FL12, which share the blend in different mixes (red/green ×1.75 and blue/green ÷3.6 from FL10 to FL12). A synthetic test with mixes that alike recovers every shape and mix. Band centres and the widths a 5 nm table can see are fitted; five weak bands keep a stated 2 nm width (1 or 3 nm moves a score by < 0.5 %).
    - The tube is soda-lime glass on the quick path (§5).
  - *Scores:* in-sample (the shapes were fitted to these tables): FL10/FL11/FL12 NRMS 6.6/6.0/7.4 %, Δu′v′ 1.6/0.6/1.4×10⁻³, ΔRa +0.9/+0.3/+0.3. 45 empirical numbers per lamp, 42 of them shared. **Out of sample** (shapes fitted to the other two lamps, only the lamp's own mix fitted): FL10 NRMS 8.9 %, Δu′v′ 2.1×10⁻³, ΔRa +1.3; FL11 7.0 %, 6.8×10⁻⁴, +0.0; FL12 11.9 %, 3.3×10⁻³, +0.7 (`lss fit fluorescent_cie_fl10 fluorescent_cie_fl11 fluorescent_cie_fl12 --level 1 --hold-out <lamp>`). FL11, between the other two mixes, is predicted almost as well as it is fitted; the end mixes, extrapolated, worse. **What this does and does not show:** all three tubes are the same three phosphors in different amounts, so a held-out tube is close to a blend of the other two, and almost any shapes that fit two will fit it. A good held-out score means the shapes *blend* well, not that they are the phosphors' true spectra; anything the three tables share (the smooth floor, the raised 555–600 nm valley) is baked into the shapes, as the oversized YOX ⁵D₀→⁷F₁ band shows. The band set itself was also chosen from errors recurring in all three lamps, so the held-out lamp shaped the model; leave-one-out tests only the fitted numbers. The scores do not depend on the fit's start (the documented command, starting from the calibrated library, gives the same numbers as a run from earlier values). The real test is a measured spectrum of a single phosphor, or a tube with a different blend (another phosphor set, or a measured lamp).
  - *Findings* (what L1 cannot do, pointing at L2 and at better references):
    - Violet Hg is too strong against blue: the model overshoots at 405 nm in all three lamps and undershoots at 436 nm in FL11 and FL12 (a free factor on the 435.8 nm line comes out 0.986 / 1.054 / 1.050 for FL10 / FL11 / FL12). The pencil lamp's 405:436 ratio is 0.44; the tables imply less, as expected if the phosphor coat absorbs violet before it leaves the tube. The ratio stays fixed at L1. *Checked (2026-10-15, under L2 below):* the shortfall is 5–11 %, and the three tables cannot tell coat absorption from a loss every tube shares, so they neither support nor rule it out.
    - A smooth floor of about 3–10 % of the mean level at 380–415 nm and 720–780 nm, the same in all three tables and following none of the phosphors. Likely stray light in the CIE measurement or a weak discharge continuum. L1 has no such term; it is worth about 2–3 % NRMS.
    - The 555–600 nm valley sits higher in the tables than any band explains. The YOX ⁵D₀→⁷F₁ band fills it: its width runs to the 20 nm bound and its share (~0.42 of the 611 nm line) is several times what that Eu³⁺ group carries.
    - The YOX 611 nm width runs to its 6 nm bound: the 615 nm node holds more than one narrow line at 612 nm gives it, so further ⁵D₀→⁷F₂ components are unresolved.
    - Hg 546 nm sits on the green phosphor's main peak, and the three mixes barely move Hg against green, so the fixed 546:436 ratio decides how the 545 nm feature is split between them. Scaling the 546 nm line by 0.8 or 1.2 moves the green phosphor's main-band share of its power by ±2.3 % (0.658 / 0.643 / 0.629) and each lamp's Hg power by about ±10 %; the scores move by < 0.3 % NRMS (slightly better at 1.2). The split is set by the measured ratio, not by the data.
    - Measured phosphor emission spectra (L2) would replace every learned shape and test them; until then the shapes are calibrations.
- **L2 — UV budget.** The 254/185 nm resonance radiation pumps the phosphor (excitation spectrum, QE, Stokes loss). Visible Hg line fraction and 254 nm output depend on Hg vapour pressure via the cold-spot temperature. Electrode and ballast losses are an explicit term. This gives efficacy from the energy budget.
  - *What it can and cannot change (2026-10-15).* At L1 each lamp sets three power ratios over the shared phosphor shapes. Writing them instead as each phosphor's share of the absorbed UV, times its quantum efficiency and its Stokes loss, reaches exactly the same set of spectra, and the scores compare shapes at equal luminous flux. So the UV budget itself cannot move a score against the CIE tables. What it adds is an efficacy prediction, which the tables cannot check (they are relative and name no lamp wattage); a datasheet efficacy of a 4000 K tri-phosphor tube would be a sanity check only. It also adds empirical numbers with no change in score, so the accuracy report will flag it for that reason.
  - *Checked before building: does the coat absorb the violet Hg line?* The hypothesis: if BAM:Eu²⁺'s excitation band reaches 405 nm (not checked against a source), a coat absorbing violet light on its way out would take more of the 404.7/407.8 nm lines in a tube with more BAM. That is the one part of L2 the tables could test. With the shared shapes fixed, one factor on the violet lines was fitted per lamp together with the lamp's own mix (`tests/test_fluorescent.py`):
    - FL10 (blue/green 0.74): 0.916 ± 0.022, NRMS 6.65 → 6.54 %;
    - FL11 (0.53): 0.888 ± 0.017, 5.98 → 5.69 %;
    - FL12 (0.25): 0.950 ± 0.023, 7.38 → 7.34 %.
    The errors take the residual scatter as noise; model errors are correlated, so they are rough. The loss is small and gains at most 0.3 % NRMS. FL11 loses slightly more than FL10, which has 40 % more BAM, but the difference (0.028) is about one combined standard error. A loss in proportion to BAM (one shared constant: χ² 4.8 on 2 degrees of freedom) and the same loss in every tube (χ² 4.7) fit equally well, and neither fits well: three tables cannot tell coat absorption from a cause the tubes share, and do not rule either out. That shared cause could be the tube's own discharge, whose line ratios differ from the 15 mA pencil lamp's, or something in the CIE tables. The 405 nm node also holds BAM's short tail and the smooth floor (L1 findings), and the shared BAM shape was fitted with the violet line at full strength. **Not built:** with these references it would be a fitting knob, not a tested mechanism. A measured excitation spectrum of BAM, or tubes measured with their violet line resolved, would decide it.
  - *Deferred to L3:* the cold-spot temperature. It sets the Hg vapour pressure, and through it the UV output and the visible line fraction. Turning vapour pressure into output is the positive column's physics (L3), and no reference here varies it.
- **L3 — Positive column model** (later). Electron energy balance and radiation trapping of the resonance lines (Holstein) give UV output vs current and pressure.

### 4.4 High-pressure discharge (HP Hg, HPS, metal halide) — the deep end

Built as separable stages, each producing a usable result, each with plug-in points for the next accuracy rung.

**Stage A — Local emission and absorption under LTE.**
For each line k→i (NIST ASD data):

```
upper/lower populations  n_k = n · g_k · exp(−E_k / k_B T) / U(T)     (n_i likewise)
absorption coefficient   κ_λ = (λ⁴ / 8πc) · A_ki · (g_k / g_i) · n_i · φ_λ(λ) · (1 − exp(−hc / λ k_B T))
emission coefficient     ε_λ = κ_λ · B_λ(T)                               (Kirchhoff, LTE)
```

- φ_λ is normalised per unit wavelength (∫φ_λ dλ = 1). The λ⁴/(8πc) prefactor is the wavelength-units form; revision 1 had a factor of λ²/c missing.
- Deriving ε from κ makes ε/κ consistency hold by construction. The stimulated-emission factor is already in κ.
- Composition from Saha (ionisation) and partition functions U(T), closed at given pressure and T.
- Continuum: electron–ion recombination and bremsstrahlung (per unit frequency ~flat × exp(−hν/kT); per unit wavelength ∝ λ⁻²), and for Hg the quasi-molecular Hg₂ bands.

**Line profile — a plug-in interface, because this is where HPS accuracy lives.**
- **Rung 1: impact-theory Voigt.** Doppler ⊗ Lorentzian, with resonance (same-species; dominant for Na D), van der Waals (foreign gas: Hg, Xe) and Stark (electrons) widths and shifts. This reproduces the self-reversal dip and its pressure trend.
- **Rung 2: quasi-static far wings from interatomic potentials** (Na–Na, Na–Hg, Na–Xe), including satellite bands such as the one in the blue wing near 551 nm.
  - Most of an HPS lamp's visible output is in the far wings, which are asymmetric, so this rung is **required for quantitative HPS** spectra, CRI and efficacy.
  - The final form is a unified line-shape theory joining the impact core to the quasi-static wings.
- The interface is `profile(λ, line, T, densities) → φ_λ`. Profiles are swapped per species and per line.

**Stage B — Radiative transfer through the arc.**
- Cylindrical arc of radius R with radial profile T(r): prescribed at first (parabolic/Gaussian, 4000–6000 K axis, ~1200–1500 K wall for HPS), solved in Stage C.
- For each line-local fine-grid wavelength, impact parameter b and angle θ to the tube axis, integrate along the chord:
  ```
  dI_λ/ds = ε_λ(T(s)) − κ_λ(T(s)) · I_λ
  ```
  Chord length at angle θ scales as 1/sin θ, so optically thin parts brighten at grazing angles while thick line cores saturate. That is why θ is a coordinate and not an afterthought.
- Output, mode 1 (for the renderer): I_λ(b, θ) at the arc boundary, averaged into the arc's zones (bands in b and θ) to give the zone spectra S_k(λ).
- Output, mode 2 (for Stage C): the net radiative loss u_rad(r) at interior radii, which needs the mean intensity J_λ(r) *inside* the arc (see Stage C). Two rungs:
  - **Rung 1: net emission coefficient** (Lowke-style). u_rad(r) ≈ 4π·ε_N(T(r), p, R_eff) from precomputed tables of escape through an isothermal sphere or cylinder. This is cheap and standard for HID modelling.
  - **Rung 2: full J_λ(r).** On a radial grid, integrate over all directions at each interior point (or use a discrete-ordinates/P_N scheme in cylindrical geometry). The cost is lines × fine-grid points × radii × directions, *on every Stage C outer iteration*. This is the main computational load of the project; Numba or Dr.Jit is the likely place to vectorise it.
- This gives the **self-reversed** optically thick lines: a broad emission from the hot core, centre absorbed by the cooler mantle, two peaks around a dip. For HPS the D-line structure spans tens of nm.
- Decomposition: the arc physics computes outgoing radiance at the arc boundary. The renderer takes it from there and traces the arc-tube wall (fused quartz; translucent PCA for HPS) as geometry. Renderer and plasma never mix, which keeps the hardest problem contained.

**Stage C — Self-consistent temperature profile (Elenbaas–Heller).**

```
(1/r) d/dr ( r κ_th(T) dT/dr ) + σ_el(T) E² − u_rad(r) = 0,     I = 2π E ∫₀ᴿ σ_el r dr
```

- κ_th: thermal conductivity including reactive (ionisation/dissociation) parts.
- σ_el: electrical conductivity from n_e(T) (Saha).
- u_rad(r) = 4π ∫ κ_λ (B_λ(T(r)) − J_λ(r)) dλ: the **net** radiative loss per volume, i.e. emission minus reabsorbed radiation, from Stage B mode 2. Optically thick line radiation absorbed in the mantle is deposited heat there.
- E: axial field, fixed by the current.
- Boundary conditions: dT/dr = 0 at r = 0, T(R) = T_wall.
- Solver:
  - finite-volume discretisation with Newton iteration and pseudo-transient continuation (more robust than shooting or `solve_bvp`, since σ_el is exponential in T);
  - outer loop: Stage B radiation → u_rad → Stage C → T(r) → Stage B, iterated to a self-consistent solution.
- Outputs: T(r), arc voltage, radiated spectrum, efficacy. These are predictions, not fits.
- **Electrode losses** are a plug-in term (cathode/anode fall voltage × current, with a documented default). This makes whole-lamp efficacy comparable with catalogue values.

*Built (step 3, first slice, 2026-10-15): Stages A and B for the sodium resonance lines, rung 1* (`lss/physics/hid/`). Not yet a generator: no lamp uses it, and no measured spectrum has scored it.
- **Data** (`lss/data/na_atomic.yaml`): Na I levels, ionisation limit and the D lines from the NIST ASD (v5.12, retrieved 2026-10-15; A_ki graded AA, ≤ 1 %).
  - *Added (2026-10-15): the lines ending on 3p.* 3p–3d (818.3/819.5 nm), 3p–4d (568.3/568.8), 3p–5d (497.9/498.3), 3p–5s (615.4/616.1), 3p–6s (514.9/515.3), 3p–7s (474.8/475.2) and 3p–4s (1138/1140, outside the visible), 17 lines with A_ki graded A or A+ (≤ 3 %), except 6s (B+, ≤ 7 %) and 7s (B, ≤ 10 %). The set is every multiplet ending on 3p that STARK-B tabulates at 10¹⁶ cm⁻³, an HPS axis's electron density; higher ones (3p–6d at 466.9 nm and up) stop being isolated lines below that density.
- **Composition** (`composition.py`): uniform total pressure, each element's share of the heavy particles the same at every radius (no demixing), sodium ionised by Saha's equation with a Debye–Hückel lowering (Griem 1964). At an HPS-like 4000 K, 13 kPa Na, 70 kPa Hg, 20 kPa Xe, 3 % of the sodium is ionised and nₑ = 7×10²¹ m⁻³, inside the 6×10²¹–10²² m⁻³ that Mioković, Balković and Veza measured in Na–Hg discharges from Stark shifts (arXiv physics/0410146, 2004).
- **Widths** (`broadening.py`), Voigt in frequency:
  - Doppler;
  - natural (the line's own A_ki; for a line ending on 3p this leaves out the lower level's 6×10⁷ s⁻¹, against a resonance width of at least 6.9×10¹⁰ rad/s at every radius of the arc below);
  - resonance broadening by ground-state sodium, measured (Huennekens & Gallagher, Phys. Rev. A 27, 1851 (1983): FWHM = k·N with k = 4.67×10⁻⁷ cm³/s for D2, 3.07×10⁻⁷ for D1, ±15 %). It is the 3p level's width: a ground-state perturber couples resonantly to the 3p level only, and each fine-structure level keeps its own rate (their §II). So a line ending on 3p₃/₂ takes D2's constant and one ending on 3p₁/₂ D1's; the paper notes resonance broadening has been measured on such lines for that reason;
  - van der Waals broadening and shift by Hg and Xe from Lindholm–Foley with Unsöld's C₆ and the polarizabilities of Schwerdtfeger & Nagle (2019). This last is an estimate, known to run about 2× low; it adds 11–16 % to the resonance width between 1500 and 4500 K at HPS pressures. Jongerius (J. Phys. B 20, 3345 (1987)) measured the xenon broadening in HPS arcs; it is not adopted yet (not openly available), and it cannot explain the red wing below: the term is symmetric and 11–16 % of the width.
  - Stark broadening and shift by electron impacts (added 2026-10-15), from STARK-B (Sahal-Bréchot, Dimitrijević & Moreau, stark-b.obspm.fr, retrieved 2026-10-15; semiclassical tables of Dimitrijević & Sahal-Bréchot, JQSRT 34, 149 (1985); JQSRT 44, 421 (1990)). FWHM and shift per multiplet at 10¹⁴, 10¹⁵ and 10¹⁶ cm⁻³ and 2500, 5000 and 10 000 K, interpolated per electron in log N and log T: the 3p–nd values are not proportional to the density (per electron, 9 % (4d) and 26 % (5d) less at 10¹⁶ than at 10¹⁵ cm⁻³). Griem states about 30 % for such calculations (as quoted by Mioković et al.); for the D lines, Konjević et al. (J. Phys. Chem. Ref. Data 31, 819 (2002)) report measured widths 13 % above these values, and these shifts 40 % above measured ones. Mioković et al. (2004) measured the 5s, 6s, 7s, 5d and 6d → 3p lines in Na–Hg and Na–Cd arcs (±25 %), taking their electron density from the 7s shift with these same values, so that shift agrees by construction. Independently of it, their 7s widths scatter around these values and their 5d shifts agree within the error, while 5s and 6s differ from all the calculations by a small systematic amount. On the HPS axis it adds 8 % (D2) and 12 % (D1) to the resonance width; it is 0.65–1.0× the resonance width for 3p–3d, 1.5–2.3× for 3p–5s, 9–14× for 3p–4d and 26–39× for 3p–5d. 3p–5d sits at its isolated-line limit there (STARK-B's C/w: 1.07×10²² m⁻³, against nₑ = 7.2×10²¹ m⁻³), so its profile is the least certain; the forbidden 3p–5f, 5g components that appear near that limit are not modelled.
  - *Not included:* broadening by the ions' quasi-static fields, which Mioković et al. put at up to about 25 % of the width of the nd → 3p lines in such arcs; STARK-B gives no ion parameter for Na I.
- **Transfer** (`arc.py`): shells of constant temperature, each chord crossing solved exactly, at any angle θ to the axis; output is the radiance per impact parameter on a line-local grid (geometric offsets from each line centre), and its mean over the arc's width.
- **What it shows.** The conditions here are assumed, chosen to look like an HPS arc, not taken from a source or a lamp: T falling parabolically from 4000 K on the axis to 1500 K at a 3.8 mm wall; 13 kPa sodium, 70 kPa mercury, 20 kPa xenon. The D lines self-reverse: two peaks at 585.0 and 593.4 nm, unchanged from 60 to 480 shells, with the line centres absorbed to under 0.1 % of them. With 120 shells or more a faint bump appears between the two lines, where the mantle absorbs least, still under 1 % of the peaks. The dark centre's own level is set by the coolest shell, so it converges slowly with the shell count (−19 % from 60 to 120 shells); everything else converges (§12). An isothermal arc instead saturates to the blackbody with no dip, and the peaks spread as sodium pressure rises (tests below). These D-line figures are for the D lines alone, now with their Stark width; they did not move when it was added.
  - *With the lines ending on 3p* (same assumed arc, 1 nm analysis bins, which converge to < 0.1 % (§12)): the 819 nm bin is the spectrum's maximum, 2.7 times the D peaks' bins; 568.8 nm is 1.2 times them; 616 and 498 nm 0.4 times; 515 nm 0.1. The 3p–3d lines self-reverse as the D lines do, but less deeply (on the fine grid their centres sit at 27–33 % of their peaks, unchanged from 60 to 480 shells). 3p–4d, 3p–5d and the 3p–ns lines show Stark red shifts of 0.02–0.11 nm at the axis.
- **Against nine measured street lamps** (2026-10-15; `lss streethps`, `lss/studies/street_hps.py`; LICA-UCM, §15). The model is the assumed arc above with every line, nothing fitted to the lamps, passed through the spectrograph's response (§15) and sampled at the lamps' 1 nm. The lamps' wattages and arc tubes are unknown, so the model is compared with the range the nine span, not with one lamp.
  - *What falls among the lamps.* Each subordinate line's power over the D lines' (575–605 nm), above a straight baseline: 498 nm 0.039 (lamps 0.027–0.054), 515 nm 0.007 (0.005–0.011), 568.8 nm 0.125 (0.090–0.215), 616 nm 0.033 (0.018–0.037), 819 nm 0.535 (0.376–0.576). The D peaks sit 9.2 nm apart (lamps 7.8–19.2 nm; seven of the nine 10–14 nm). The far blue wing at 525–540 nm, as its mean over the D peak, is 0.010 (lamps 0.005–0.057; seven of the nine 0.005–0.016).
  - *What does not.* The red wing at 640–700 nm is 0.015 of the D peak in the model and 0.049–0.151 in the lamps: every lamp is 3.4–10.4 times brighter there. With the blue wing in agreement, the excess is one-sided, as the quasi-static red wing of rung 2 (Na–Na, Na–Xe, Na–Hg) would give, but the arc's conditions are assumed, so this shows the rung-1 wing falls short, not by how much at a given lamp's conditions. More sodium in the model would not close it alone: the lamp with the narrowest peak spacing (1920 K, 7.8 nm, against the model's 9.2) has half the model's far blue wing, yet 4.0 times its red wing. Between the D lines and 560 nm the lamps also show the Hg 546 nm line and a rise from about 548 nm that the model has no counterpart of; it is left out of the comparison.
  - *What does not depend on the response.* Through the mercury and the argon responses, and the mercury one shifted ±0.5 nm, the peak spacing moves by < 0.2 nm, each line power by < 10 %, and the red wing by < 10 %.
  - *Uncertainty in LICA's response correction.* A tilt across wavelengths would move the 819 nm power against the D lines' (§15). The halogen check bounds anything but a smooth tilt to < 1 %; a smooth one is degenerate with the filament's temperature, and ±100 K there is ±8 % on 819 against 589 nm, less than the lamps' own spread.
  - *The red wing, examined (2026-10-15).* In the model, 640–700 nm is the D lines' far wing alone (the other lines give < 1 %), and the arc is thin there: 0.004–0.011 optical depth across the diameter. So the shortfall is the wing's absorption times its source; 84 % of that light comes from shells above 3000 K. Against the frequency offset from the D lines, with 665–685 nm left out, the rung-1 wing falls as Δν^−1.84; six lamps fall at −1.8 to −2.3, the three of highest CCT at −0.8 to −1.4. Six lamps also show a broad rise at about 668–680 nm, some 12 nm wide, that the model has no counterpart of: 668–678 nm sits 16–41 % above a straight line from 656–662 to 690–696 nm (one more lamp 10 %, two not at all). *Traced (2026-10-15):* NaHg has red satellite bands S1–S4 at 669, 656, 648 and 640 nm in absorption (Schlejen, Woerdman & Pichler, J. Mol. Spectrosc. 128, 1 (1988), as quoted by Windholz et al. 1991); the sodium–mercury pairs below put the classical satellite at 668.7–678.4 nm, but too weak to make the rise.

*Rung 2, first part: sodium atom pairs (2026-10-15;* `lss/physics/hid/pairs.py`*,* `lss/data/na2_pairs.yaml`*).* Far from the D lines a sodium atom absorbs only while another ground-state sodium atom is close: the pair is briefly a molecule and absorbs at the gap between its curves at that distance. `ArcColumn(pairs=PairWings(...))` replaces the resonance share of the D lines' far wings with this absorption beyond a splice distance (the mercury, xenon and Stark widths stay); within it the lines keep their impact profile.
- **Method.** The classical limit of the Franck–Condon sum (the quasi-static wing, bound pairs included) over the four bands from two ground-state atoms: X→A and a→b to the red, X→B and a→c to the blue. LTE pair densities, ½ for identical atoms, spin weights ¼ (X) and ¾ (a). The sum is binned in wavenumber, so the satellites (551.5, 804, 880 nm) stay finite. It has no rovibrational structure, which is finer than the lamps' 1 nm; near the satellites, where the classical limit is least accurate, it is sharper than a quantum calculation.
- **Data.** X and a curves: Knoop et al. (Phys. Rev. A 83, 042704 (2011)), analytic, fitted to spectroscopy and ultracold data; the a state inside its fitted range from Chung et al.'s figure. Upper curves: Chung, Kirby & Babb (Phys. Rev. A 63, 032516 (2001)) difference potentials, digitised from the vector paths of their arXiv figure, smoothed (the plot comes in 20 cm⁻¹ steps) and joined to long-range forms (`scripts/digitise_na2_curves.py` reproduces them). The digitised X curve matches Knoop's within 27–66 cm⁻¹ rms over 5–24 a₀. Their sources (RKR tables, ab initio curves) are not openly available. Transition dipoles: Chung et al.'s long-range forms a + b/R³, used at every R, or held at a; the triplet bands take the singlet form of the same symmetry change, an assumption. One measurement, from its abstract (Ahmed et al., J. Chem. Phys. 124, 084308 (2006): 9.72 D, i.e. 3.82 a.u., at an R-centroid of 4.81 Å, 9.09 a₀), lies between the two forms there (3.97 and 3.59 a.u.). The single asymptote (16 956 cm⁻¹, 3p₁/₂) ignores the fine structure, 0.6 nm, inside every splice used here.
- **A check that needs no paper.** 5 nm from the lines, where the curves are nearly the long-range resonance term, the pairs give 1.37 (blue) and 1.99 (red) times the impact wing of Huennekens & Gallagher's measured widths: within the factor 2 set for it, the red side at its edge. So the counting (spin weights, Λ, identical atoms) is right to that level.
- **Against Chung et al.'s quantum calculation: missed.** The mark, set before running: within 30 % of their Fig. 6 (digitised) over 640–700 and 525–540 nm at 2000 and 3000 K. The pairs give 1.29 times theirs (525–540 nm, 2000 K), 1.42, 1.46 and 1.51: three of four miss. Against Woerdman & de Groot's value from an HPS discharge (500 nm, 2000 K: 5 ± 1 × 10⁻³⁷ cm⁵) they give 4.7: met.
  - *A limit of the reference explains most of the miss near the lines, though this is not certain.* Chung et al. include rotational levels up to J = 250 (stated for the bound and quasi-bound levels). Classically, at 2000–3000 K, many pairs 14 a₀ apart rotate faster. With the same cut (for the comparison only), the pairs come to 1.14–1.37 times theirs. Independently of this model, their triplet wing at 610–630 nm falls from 2000 to 3000 K to 0.74 of itself; an uncut free-pair wing would stay flat (here 0.99), and the cut predicts 0.76. The blue window at 3000 K stays at 1.37 with the cut, unexplained.
  - *The dipole.* At 1000 K, where bound pairs dominate and the cut does not matter, the long-range dipole form gives 1.66 times their 640–700 nm value and the constant form 0.92 times. So the long-range form overstates the inner X→A branch; the two forms bracket the truth there.
  - So the coefficient is not called validated. It is used as a generous estimate, with both dipole forms reported.
- **Against the nine lamps** (the same assumed arc; `lss streethps`). Over both dipole forms and splices of 3, 6 and 12 nm:
  - *Red wing:* 1.95–2.4 times rung 1's, 0.028–0.035 of the D peak. Every lamp is still 1.4–5.3 times brighter. The pairs also flatten the wing (slope −1.0 to −1.2, as above), where six lamps fall at −1.8 to −2.3: what the lamps have beyond the model lies nearer the lines than where the pairs put light.
  - *Blue wing:* 1.2–1.35 times rung 1's, 0.0125–0.014, still among seven of the lamps.
  - *Line powers:* every subordinate line's power over the D lines' stays within the lamps' range.
  - *Peak spacing:* depends on the splice: 9.0–9.2 nm for 6–12 nm splices, 10.8 nm for 3 nm, where the pairs absorb more than the impact wing near the peaks (lamps 7.8–19.2 nm).
  - *Held out, nothing tuned:* the pairs give a rise from 548 nm, which rung 1 lacks and the lamps show from 547–549 nm; the model's is sharper and peaks at 549–550 nm, where Chung et al.'s curves put the satellite (548 nm; measured at 551.5 nm).
  - *Not shown:* how much of the remaining 1.4–5.3 times is sodium–mercury or sodium–xenon pairs (the assumed arc has five times as many Na–Hg pairs as Na–Na), and how much is the arc's conditions, which are assumed. The 3000–4000 K core, where most of the red light is made, is beyond the reference's 3000 K.

*Rung 2, second part: sodium–mercury pairs (2026-10-15;* `lss/physics/hid/mercury_pairs.py`*,* `lss/data/nahg_pairs.yaml`*).* The assumed arc has five times as many Na–Hg pairs as Na–Na pairs, so far from the D lines a sodium atom absorbs mostly while a mercury atom is close. `ArcColumn(mercury_pairs=MercuryPairWings(...))` replaces the mercury van der Waals share of the D lines' far wings with this absorption beyond a splice distance (the mercury and xenon impact widths are now kept apart for this); it combines with `pairs`.
- **Why the curves' short-range shape is needed.** With Unsöld's ΔC₆ (685 au, or twice it), 640–700 nm needs the two atoms 6.2–7.8 a₀ apart, inside 8 a₀: a wing from the long-range form alone would be an extrapolation there.
- **Method.** As for sodium pairs (the classical limit, LTE pair densities), for unlike atoms: no ½, one lower state (X ²Σ⁺, weight 1), density n_Na(3s)·n_Hg. Three bands, Hund's case (c): I½ and I3/2 (from A ²Π) and II½ (from B ²Σ⁺ at short range), each with a third of the D lines' strength at long range (d_z = 2.49 e·a₀ from the D lines' A values). A check that needs no paper: for ΔV = A − C/R⁶ over a flat lower curve, the sum gives the analytic quasi-static wing within 1 %.
- **Data.** X: Thiel, Hotop & Meyer's CCSD(T) curve (J. Chem. Phys. 119, 9008 (2003), tabulated in Thiel's open thesis, 2004): Re 6.52 a₀, De 69 meV. Upper curves: the only source found with them, Windholz, Zerza, Pichler & Hess (Z. Phys. D 18, 373 (1991)), Fig. 8, digitised from a scan (`scripts/digitise_nahg_curves.py`, given a copy of the PDF; the repository holds only the numbers). Czuchaj et al. (Chem. Phys. Lett. 178, 246 (1991)), which also gives transition dipoles, was not obtained. Fig. 8 has computed points at 4, 5, 5.5, 6 and 8–10 a₀; around the minimum and between 6 and 8 a₀ the curve is the authors' drawing. Beyond 8 a₀ (the avoided crossing) a monotone curve through the computed points replaces the drawing, whose kinks include near-flat stretches and a dip of a few cm⁻¹, and the curves join their asymptotes with continuous slope: a flat stretch would make a spike in the classical sum, and a slope step a step.
- **The shift, fixed before any comparison.** The paper's curves lie 600–950 cm⁻¹ below the D lines at 10 a₀ (its own basis-set error, which it notes). One constant for all three: at 9.95 a₀ I3/2 (pure Π) sits on the 3p₃/₂ asymptote minus ΔC₆/R⁶, ΔC₆ Unsöld's times 1–2: +478 to +323 cm⁻¹. The paper's Fig. 7, taken at face value, binds its excited curves by about 700–1000 cm⁻¹ at 10 a₀ and 170–380 cm⁻¹ at 14–15 a₀, far more than a van der Waals tail (about 60 cm⁻¹ at 14 a₀); a rough reading by hand: the Π curve bound by about 700 cm⁻¹ at 10 a₀, less Thiel's X there (about 150 cm⁻¹), would make the shift about +80 cm⁻¹ and put the satellite near 690 nm.
- **Transition dipoles.** Not given beyond the A ← X oscillator strength falling 10× from 500 to 200 pm. Two forms: constant d_z², and that fall on the two I bands (II½ constant). A bracket, not validated.
- **The satellite: where it was predicted.** Mark set before comparing: the classical satellite at the II½ minimum (drawn 14 413 cm⁻¹ at 4.75 a₀; computed point 14 473 cm⁻¹ at 4.98 a₀) plus the shift lies at 668.5–678.3 nm; with the final curves 668.7–678.4 nm (the change beyond 8 a₀ moved the reference point by 4 cm⁻¹), and 667.8–679.4 nm with ±20 cm⁻¹ for digitising, which the mark included. The measured S1 satellite, 669 nm in absorption, is inside either way, 0.3 nm from the blue edge without the digitising error. A coarse test: the window is 10–11 nm wide. The undulations S1–S4 are quantum interference, which the classical limit cannot give.
- **Against the nine lamps** (the same assumed arc; `lss streethps`; splice 6 nm, plus 3 and 12 nm for one case). With sodium and mercury pairs together, over both dipole forms of each and both ends of the ΔC₆ range:
  - *Red wing:* 0.053–0.079 of the D peak (rung 1: 0.015; sodium pairs alone: 0.029–0.035), now within the lamps' 0.049–0.151, at their faint end: the lamps run from 0.62 to 2.9 times the model. Mercury pairs alone give 0.038–0.061.
  - *Its slope:* −1.34 to −2.09, where sodium pairs alone flattened it to −1.0 to −1.15. Six lamps fall at −1.74 to −2.21, three at −0.78 to −1.33. (The fit, now in the study code: log power against log offset from 589.3 nm over 640–700 nm without 665–685 nm. It gives −1.82 for rung 1, against −1.84 above.)
  - *Held out, the rise at 668–678 nm: missed.* The model gives −0.017 to −0.003 where six lamps give +0.16 to +0.41. The satellite is in place but weak: the II½ minimum sits about 3000 cm⁻¹ up the X curve's inner wall, and at 3000 K II½ carries under a third of the pair absorption at 650–670 nm and stops beyond its minimum, an edge more than a peak. The paper says its bond distances come out short; a minimum at larger R would sit lower on the wall and be stronger. Not shown.
  - *Blue wing:* 0.0113–0.0130 (beyond the splice the mercury width goes, and nothing replaces it on the blue side), among seven of the lamps.
  - *Line powers:* 498, 515, 568.8 and 819 nm stay within the lamps' range. The 616 nm measure is not usable: at 3000 K the pair absorption peaks 48 % and 62 % above its neighbours (±4 nm) at 609 and 620 nm with Unsöld's ΔC₆ (615 and 626 nm with twice it), from slow stretches of the curves at the avoided crossing between computed points 0.5 a₀ apart; in the observed spectrum, where they clear the 616 nm line, they are +13 to +25 %. They move with the shift and fall on that line's baseline windows; over all variants its power runs from −0.003 to 0.047 (lamps 0.018–0.037).
  - *Peak spacing:* 8.9 nm (Unsöld's ΔC₆) to 12.9 nm (twice it); 11.4 nm with a 3 nm splice (lamps 7.8–19.2).
  - *Not shown:* how much of the remaining distance to the brighter lamps is the arc's conditions, which are assumed, and how much is xenon pairs or the curves' uncertainty.

*Each lamp's arc fitted (2026-10-16;* `lss/studies/street_hps_fit.py`*,* `lss streethpsfit`*).* Each of the nine lamps gets its own axis temperature and sodium pressure; the radius (3.8 mm), wall temperature (1500 K), parabolic profile, mercury (70 kPa) and xenon (20 kPa) stay as assumed. With mercury held, more sodium also means fewer Na–Hg pairs per Na–Na pair.
- **Measured values to read the fits against.** J. J. de Groot and J. A. J. M. van Vliet, *The High-Pressure Sodium Lamp* (Philips Technical Library, Macmillan 1986, doi:10.1007/978-1-349-09196-6): axis temperature "between 4100 and 4300 K" in standard 400 W Na+Hg lamps, the consensus of six measurements (p. 133); about 4000 K without mercury (Table 3.5). Sodium from the peak spacing Δλ: p_Na·√R/Δλ ≈ 2.7 ± 0.5 kPa nm⁻¹ mm^½ by the spectroscopic method (eq. 3.27), 1.9 ± 0.5 from metal-bath experiments (eq. 3.28). Colour-improved lamps: Δλ about 25 and 45 nm (p. 301), wider than any of the nine (7.8–19.2 nm). The lamps' tube radius is not known: by the rule a fitted pressure scales as R^−½, so each is the pressure a 3.8 mm tube would need (a 3 mm tube: 12 % more). How the fitted temperature moves with the radius was not checked.
- **What is fitted, set before any fit.** The blue D peak's wavelength (σ 0.3 nm) and the power of the 498, 515 and 568.8 nm lines over the D lines' (σ 10 % each); 819 nm shown with and without; 616 nm left out (above). Not the peak spacing: in the model the red peak's place depends on the splice (3, 6 or 12 nm move it by several nm at 10–15 kPa); the blue peak's moves by under 0.6 nm below 16 kPa. Above that the blue peak meets the splice too: with 6 nm it jumps 0.8 nm as it crosses 583 nm (22–23 kPa). Held out: the spacing, the red wing's level and slope, the 668–678 nm rise and the blue wing. The fit exists to ask whether the red wing's shortfall is the assumed arc's conditions; a fit that saw the red wing could not answer that.
- **Method.** One table per variant: the central one (sodium and mercury pairs, constant dipoles, Unsöld's ΔC₆, 6 nm splice) and one per change from it (ΔC₆ × 2, falling Hg dipole, long-range Na dipole, splice 3 nm, splice 12 nm). Each holds the recorded spectrum (the instrument's response applied) over 3400–4800 K × 5–45 kPa (8 × 23 points, about 5 minutes on 16 workers), read by cubic splines in (T, log p). Checks: a direct run at every fitted point agrees with the table within 0.08 nm on the blue peak; halfway between nodes within 0.1 nm, except up to 0.3 nm at 21–24 kPa, where the peak jumps; a fit to a direct run between nodes recovers the temperature within 20 K and the pressure within 3 %. The two settings separate: sodium moves the blue peak, temperature the line powers. The D band's red half (above 590 nm), where the model's red peak sits elsewhere than some lamps', holds the same share of the band's power within 9 % at every fit. That share still matters: with the line powers over the band's blue half (below 590 nm) instead of the whole band, the central temperatures move by −64 to +103 K (checked later, with the mercury study below; at the lamps' own mercury the share is no longer within 9 %).
- **Fitted.** Temperature 3925–4484 K (central; −106 to +67 K across the tables): three lamps inside the book's 4100–4300 K, all within 200 K of it (with the line powers over the D band's blue half, seven of nine: 1732 K 287 K above, 2038 K 39 K below). χ² per target ≤ 0.7: the three lines agree on one temperature, and adding 819 nm moves it by ≤ 24 K. Sodium 7.6–24.6 kPa, within −8 to +1 % across the tables (one lamp, 2176 K, +16 %). None sits on the grid's edge. The pre-set ±0.5 nm wavelength offset moves each pressure by up to +24/−26 % (the low-pressure lamps; 4–7 % for the highest). The LPS lamp's D and 568.8 nm doublets put the offset near the D lines at −0.17 ± 0.06 nm beyond the mercury response's, which lowers each pressure by 0.3–0.8 kPa. The rank correlation of the fitted sodium with the lamps' CCT is +0.53 (p = 0.14, nine lamps): inconclusive.
- **The book's rule.** The model keeps it: at the fitted points its own p·√R/Δλ is 2.1–3.0 (one lamp at 2.09, just below the book's 2.7 ± 0.5). Applied to the lamps' own spacings it gives 1.31–1.69 times the fitted sodium for the five lamps below whose peaks are wider apart than the model's, and 1.02–1.13 times for the other four.
- **Held out** (from direct runs at the fitted points, over the six tables):
  - *Red wing: not closed.* Seven lamps are brighter than the model in every table, by 1.2–5.1 times (1732 K 3.3–5.1; 1819, 1869, 1895, 1920 and 2255 K 1.6–2.9; 2005 K 1.2–1.7); two (2038 and 2176 K) are 0.61–1.03 times it. With the one assumed arc it was 0.62–2.9 times. Within this model's two free settings, the arc's conditions are not what the red wing lacks.
  - *Spacing.* Five lamps' peaks are further apart than any table puts them at their fitted point (7.8–13.7 nm against at most 7.3–12.9), and the same five are among the red-wing shortfalls, by 1.69 times or more in every table. The spacing does not depend on the wavelength offset. Of the other four, 2255 K's red wing is short (1.6–2.3) without a spacing shortfall. Nine lamps show a pattern, not one mechanism: both point at red-side absorption the model lacks, near the lines and far from them. (The model's red peak here sits next to the splice, which makes a false second maximum, lower at 70 kPa; see the mercury study below.)
  - *Red slope:* within the tables' range for four lamps; 1732, 1819, 1895 and 1920 K are steeper than every table (by 0.11–0.36), 2255 K flatter (−0.78 against at least −1.34).
  - *The 668–678 nm rise:* none in any table.
  - *Blue wing:* 0.70–1.44 times the model for seven lamps; 1732 K 1.9–2.2 times, 2255 K 2.8–3.4.
  - *Not shown:* whether the missing red-side absorption is the mercury pressure (held at 70 kPa; the lamps' is not known), xenon pairs, the pair curves or the tube radius. The mercury pressure is taken up next.

*Each lamp's mercury, read off its own peaks (2026-10-16;* `lss/studies/street_hps_mercury.py`*,* `lss streethpshg`*).* The book gives a practical formula for the mercury-to-sodium pressure ratio from the two self-reversal peaks (de Groot & van Vliet 1986, eq. 5.7, p. 159, calibrated on 350 W arc tubes of known amalgam): p_Hg/p_Na = ((Δλ_R/Δλ_B)² − 1) / (0.057 (Δλ_R^½ − 0.85)), Δλ_B and Δλ_R the distances in nm from the D lines' centre to the blue and red peaks. Mercury widens the red side only, so it pushes the red peak out. The book gives its accuracy as 20–30 % and claims it only above 10; its calibrated lamps (Fig. 5.24) span 13–65, and it calls 10–20 the optimum for practical lamps (p. 288).
- **Set before any run.** *Input:* each lamp's ratio from its own peaks, so the spacing is no longer held out. *Fitted:* temperature and sodium, to the same targets as above, with mercury = ratio × sodium (one table over 3400–4800 K × 5–45 kPa × ratio 1–128 in steps of two, central variant only, about 40 minutes on 16 workers). *Held out:* the red wing's level and slope, the 668–678 nm rise and the blue wing. A rough prediction, written first: the red wing within about 1.5 times of every lamp; 1819, 1869 and 1895 K might overshoot, 2255 K stay short. Before the refit, but after a first unrefitted look (each lamp's 70 kPa fit, its own mercury: lamp/model 0.69–1.32), two limits were written down. The lamps' order in red light is not a test: wide spacing and bright red wing go together in the lamps, and the formula reads mercury from the spacing. The level is the test, and only conditionally: the formula puts all red-side asymmetry down to mercury, so anything else that pushes the red peak out (xenon pairs, other curves) would pass it too. As a control, every lamp is also fitted at one shared ratio, the median of the nine (20.6).
- **The ratios.** The measured peaks are first corrected for the instrument: blurring each lamp's spectrum once more with the response moves its blue peak 0.25–0.45 nm and its red peak 0.60–1.07 nm to the red, taken as the shift the instrument made; and the LPS offset (−0.17 ± 0.06 nm). The model's own peaks cannot give that shift (its red side is in question), and the model does not return its own mercury through the formula (it reads 14.1 for a true 8.4, 23.9 for 5.3). Each band spans both D-line centres (589.29 nm midpoint, 589.19 nm weighted), the offset ±2 standard errors, half or all of the blur shift, then the book's 30 %. Central ratios: 11.2–48.4 for six lamps (2005 K, 11.2, at the edge), mercury 143–335 kPa; 2.0–6.8 for 2038, 2176 and 2255 K, below the formula's range. Bands run from about half the central value to 2–3.4 times it.
- **Held out: the red wing's level, reached.** Lamp/model red wing, as the geometric mean over the nine (and the standard deviation of its logarithm): 1.62 (0.48) at 70 kPa; 0.99 (0.22) with each lamp's own mercury; 0.77 (0.64) at the shared ratio. Per lamp, own mercury: 0.68–1.46 (was 1.2–5.1 for seven lamps); 1732 K 1.46 (band 0.87–1.78), 2255 K 1.36, the rest 0.89–1.03 except 2038 K. The prediction held: every lamp within about 1.5 times, 1819, 1869 and 1895 K about 10 % over, 2255 K short. The shared ratio gets the level about as close, so what this shows is mercury at about 20 times the sodium pressure, as the book says lamps run; the per-lamp gain in scatter is the order, which is not a test. Unrefitted, the level was already reached (above), so it does not rest on the refit.
- **Held out: the misses.** *2038 K overshoots:* 0.68, below 1 across its whole band (0.36–0.93); its ratio (5.0) is outside the formula's range. *Red slope:* steeper by 0.07–0.12 than at 70 kPa, still 0.16–0.58 shallower than six lamps (−1.74 to −2.21 against −1.58 to −1.75); 2255 K stays flatter (−0.78 against −1.52). *The 668–678 nm rise:* none (−0.01 to +0.01 against the six lamps' +0.16 to +0.41). *Blue wing:* fainter with more mercury (beyond the splice the mercury width goes, and nothing replaces it on the blue side): lamp/model 0.86–1.8 for seven lamps, 1732 and 2255 K about 3 times.
- **Found in the run: a false red peak at the splice.** At the lamps' mercury the model's red side has two maxima, at 593–595 and 597–599 nm (eight lamps; 2176 K one, at 601 nm); each lamp has one, at 595–600 nm. The outer one follows the splice: at Toledo's fitted point it moves to 604 nm with a 12 nm splice, and with 3 nm the two merge at 596 nm. So it is the join between the impact profile and the pair wings, which more mercury makes stronger. At 70 kPa it is there but lower (0.63–0.81 of the inner maximum in the three lamps looked at), so the red peak read is the inner one. The model's spacing jumps between the two maxima (Toledo 6.5 nm at its central ratio, 12.2 at the high end; the lamp 10.2), so it is not reported; an earlier remark from the unrefitted look, that the spacing widened, was this jump.
- **Found in the run: the temperature, and why it rose.** Fitted as set beforehand, the temperatures rose by 43–282 K for the six lamps in the formula's range (all nine 3924–4766 K; Toledo's high end on the grid's 4800 K edge). The false peak raises the D band's red half above the lamps' share by up to 19 % (0.52–0.66 against 0.52–0.62), which lowers the line powers over the whole band and so raises the fitted temperature. A check added after the run, with the line powers over the band's blue half (below 590 nm): the lamps' own mercury then moves the temperature by −13 to +74 K from the same check at 70 kPa, and the red wing's level stays (0.97, scatter 0.22; shared ratio 0.76). The temperatures by that check are 3861–4661 K: three lamps inside the book's 4100–4300 K, seven within 200 K of it, Toledo 361 K above.
- **Not shown:** that the lamps hold this much mercury (no independent measure here; the ratios of the six are 11–48, the book's optimum 10–20); whether xenon pairs or the pair curves would do the same; the red slope and the 668–678 nm rise, which mercury at these pressures does not make; and the spectrum without the false peak, which needs the join made smooth.

*The join made smooth (2026-10-16;* `lss/physics/hid/arc.py` *`join_weight`).* Two faults, taken apart.
- **A hole at each splice.** Each D line dropped its share beyond the splice on its own, but the pairs were added only where both lines were past it: for 0.6 nm at each splice (in air about 583.0–583.6 and 595.0–595.6 nm) κ fell to about a third. Each line's share of the pairs (its share of the D lines' strength, D1 ⅓, D2 ⅔) now comes in where its wing goes. This was the blue peak's jump at 21–24 kPa (4100 K, 6 nm splice: 0.38 nm per 0.5 kPa there, now 0.2–0.26 throughout). It was not the false red peak: Toledo still had two.
- **A step at the splice.** The impact wing and the quasi-static wing never agree. For a C₆ interaction the quasi-static wing exceeds the Lindholm–Foley one by about 1.6√(Δω/ω_W), ω_W the Weisskopf detuning (0.36–0.47 nm for Na–Hg at 2500–4000 K with Unsöld's C₆). So the impact wing is the wrong form beyond about ω_W, and at 6 nm the Na–Hg pairs were 6–7 times the mercury share they replaced. On the red side κ stepped up about 3.3 times there, which the self-reversed line showed as the second maximum. Measured on the model: Na–Hg pairs over the impact wing 1.9–2.2 at 0.2 nm, 3–3.5 at 0.7 nm, 6–7 at 6 nm.
- **The hand-over.** Each line hands its resonance and mercury shares to its own part of the pairs by a weight that is 0 inside half the join, 1 beyond twice it, and a smooth step (3x² − 2x³ in log offset) between, so κ and its slope stay continuous. The join's centre is 0.8 nm (about twice ω_W) for both kinds of pair. The Na–Na curves have one 3p asymptote: each line now takes its share of the table at its own upper level (far out the two average to the 3p centroid, 11.5 cm⁻¹ above 3p₁/₂ where the table sat before). With one asymptote the pairs were 6 times the resonance impact wing 0.2 nm red of D1 and 0.14 times 0.2 nm blue of D2; with the split, 1.5–2.0 on both sides at 0.4–6 nm. The Na–Hg bands sit on their own lines already (I½ on 3p₁/₂; I3/2 and II½ on 3p₃/₂).
- **Checks.** κ falls monotonically from 0.3 to 2.5 nm on both sides at 1600–4500 K. The 2 cm⁻¹ pair table, read as the arc reads it, keeps the analytic C₆ wing within 2 % at 0.3 nm and 0.5 % from 0.5 nm. Beyond about 3 nm the pair tables' own structure (the digitised curves) is not monotonic, as before. At three points (4100–4300 K, 12–15 kPa, Hg/Na 5–30) the observed spectrum has one red maximum, at 596–602 nm; moving the join to 0.4 or 1.6 nm changes it by at most 0.35 % of the peak beyond 3 nm and moves neither peak.
- **Set before the studies are rerun.** Both sides are treated alike: beyond 1.6 nm the mercury impact width is gone on the blue side too (it went at 6 nm before), where the Na–Hg curves give no absorption. The fitted targets, held-out quantities and variants stay as they were, with the joins 0.4 and 1.6 nm in place of the 3 and 12 nm splices. What κ does, from the assumed arc's fill (13 kPa sodium): at 2–5 nm, blue ×1.23–1.41 at 70 kPa mercury and ×0.95–1.11 at twenty times the sodium; red ×1.9–2.3 and ×2.5–3.4; unchanged beyond 8 nm. Expected, therefore: at 70 kPa the blue peak further out at a given sodium, so fitted sodium lower, by roughly 10–15 %; at the lamps' own mercury little change in sodium. The red peak further out everywhere, so the model's spacing wider, closing part of the spacing shortfall of five lamps; and the spacing reported again in the mercury study. The red wing (640–700 nm) is unchanged in κ; it moves only through the fits and its normalisation. More of the D band's power on the red side may raise the temperatures fitted over the whole band; the blue-half check will show it.
- **The reruns, against that** (`lss streethps`, `lss streethpsfit`, `lss streethpshg`; the earlier numbers in this section are from the 6 nm cut and stand as the record of that run).
  - *Assumed arc:* as expected, the far wings barely move. Sodium pairs alone: red wing 0.028–0.034 of the D peak, slope −0.97 to −1.14, blue wing 0.013–0.014. With mercury pairs: red wing 0.051–0.078, slope −1.33 to −2.09, blue wing 0.011–0.013; mercury pairs alone 0.038–0.060. The spacing no longer depends on where the wings join: 11.0 nm with sodium pairs at all three joins (it was 3 nm wider with a 3 nm splice than a 12 nm one), 11.8–12.9 nm with both kinds (was 8.9–12.9).
  - *Arc fit at 70 kPa, sodium: held.* 9–16 % lower for eight lamps, 5 % for 2176 K (6.8–23.4 kPa, was 7.6–24.6). Across the tables within 0–2 % (was up to 16 %).
  - *Arc fit, temperature: no rise.* Central temperatures moved by −35 to +46 K (3907–4520 K). Four lamps inside the book's 4100–4300 K (was three), eight within 200 K of it; Toledo 220 K above. Across the tables −19 to +70 K (was −106 to +67, most of it the splice variants).
  - *Spacing: held, in part.* The five lamps whose peaks were further apart than the model's are still, but by 0.9–3.3 nm at their fitted points (was 1.9–4.8); their model spacing widened by 1.0–1.5 nm. The other four lamps' model spacing narrowed with their lower sodium, now within 0.0–0.9 nm of the lamps (was up to 2.1). Across the tables the spacing spreads by at most 1.4 nm (was 3.9).
  - *Red wing at 70 kPa:* fainter with less sodium. Lamp/model level 1.96 (scatter 0.49; was 1.62); seven lamps 1.5–6.0 times brighter than the model in every table (was 1.2–5.1), 2038 and 2176 K 0.78–1.14. The red slope keeps its pattern: 1732, 1819, 1895 and 1920 K steeper than every table (by 0.06–0.31), 2255 K flatter, and now 2038 K by 0.01.
  - *The book's rule:* at the fitted points the model's p·√R/Δλ is now 1.9–2.3 (was 2.1–3.0): at the low end of the spectroscopic 2.7 ± 0.5, inside the metal-bath 1.9 ± 0.5. Applied to the lamps' own spacings it gives 1.5–1.9 times the fitted sodium for the five lamps whose peaks are further apart than the model's, 1.1–1.3 for the other four (was 1.31–1.69 and 1.02–1.13).
  - *Own mercury, sodium: did not hold.* Against the first run's own-mercury fits sodium moved by −15 to +13 %. What changed is that the blue peak no longer depends on mercury: beyond 1.6 nm nothing mercury does reaches the blue side, so each lamp's sodium is now the same as at 70 kPa within 0.2 kPa.
  - *Own mercury, temperature:* the change from the 70 kPa fit is −6 to +143 K (was +43 to +282 K): it was mostly the false peak. The D band's red share is still above the lamps' (model/lamp up to 1.14, was 1.19); with the line powers over the band's blue half, own mercury and 70 kPa agree within 15 K (was −13 to +74).
  - *Own mercury, red wing: the level stays.* Lamp/model 1.09 (scatter 0.19; was 0.99 and 0.22); blue-half check 1.08 (0.20); shared ratio 0.86 (0.64). Per lamp 0.91–1.79: 2038 K no longer overshoots (0.96, band 0.43–1.37; was 0.68), 2255 K is shorter (1.79, was 1.36), Toledo 1.17 (was 1.46).
  - *One red maximum* at every lamp's fitted point, within 1 nm of the lamp's (seven at the same node). With each lamp's mercury from eq. 5.7 the model's spacing lands within 0.4 nm of the lamp's. That is not a test (the formula reads the spacing); it suggests the model and the formula now roughly agree (with the 6 nm cut the model read back 14.1 for a true 8.4), but the model's own read-back was not rerun.
  - *Unchanged:* no 668–678 nm rise (0.00–0.01); the red slope still 0.15–0.58 shallower than six lamps; the blue wing fainter than the lamps (Toledo 2.5 times, 2255 K 4.1, the rest 1.0–1.8).

**Metal halide** adds chemistry: dissociation equilibria of NaI, TlI, ScI₃, DyI₃ … by Gibbs-energy minimisation, giving radially varying free-metal densities. It is a plug-in providing n_species(T, p) to Stage A. HP-Hg and HPS come first.

### 4.5 Later physics rungs (each a defined plug-in point)

| Rung | Plugs into |
|------|-----------|
| Unified line-shape theory, more interatomic potentials | Line profile interface |
| Transport coefficients computed from collision cross-sections | Transport-data interface (Stage C) |
| Metal-halide chemistry, wall reactions | Composition interface (Stage A) |
| Axial non-uniformity, electrode regions | Arc geometry (Stage B/C) |
| Convection / arc bowing (2-D) | Replaces Stage C solver |
| Non-LTE near the wall | Population model (Stage A) |
| AC operation, warm-up | Time-dependent Stage C |
| PCA as a scattering medium | Renderer material |
| Volumetric phosphor (LED L4) | LED generator |

### 4.6 Data the physics layer needs (each with citation and stated uncertainty)

- NIST ASD line lists for Hg, Na, Sc, Tl, Dy, In, Xe: curated CSV subsets.
- Partition functions (Irwin polynomials or tabulated).
- Broadening data: resonance (from oscillator strengths), van der Waals (C₆), Stark (Griem tables), interatomic potentials for wing profiles.
- Transport coefficients σ_el, κ_th for Hg/Na/Xe mixtures (Elenbaas, Waymouth, de Groot & van Vliet). Literature tables at first.
- Tungsten emissivity (spectral, and total for the power balance) and resistivity; glass, quartz and PCA transmission; phosphor excitation/emission/QE spectra.
- CIE CMFs, V(λ), illuminants, CRI and TM-30 samples (via `colour-science`).
- **Reference lamps** (Level 0 and the accuracy report): SPDs (LSPDD, manufacturer data, own measurements), plus electrical data and angular data where available.

---

## 5. Lamp as a component pipeline

```
Lamp
  physical     : the shared physical description (§4)
  emitter      : a generator from §4 → EmissionProfile (bare emission, incl. phosphor conversion)
  envelope[]   : geometry + spectral material, traced by the renderer
  reflector    : geometry + spectral BRDF, traced by the renderer
  external[]   : filters/diffusers outside the lamp, also placeable anywhere in the scene
```

The responsibilities are split one way only:
- **Generators** produce the light leaving the emitting surface. Anything that moves energy between wavelengths belongs here, because the scene renderer does no fluorescence: phosphor layers on LEDs and on fluorescent tubes, and volumetric phosphor.
- **The renderer** traces everything after that as ordinary geometry: arc-tube walls, outer bulbs, coatings that only transmit, scatter or reflect, reflectors including angle-dependent dichroics, and filters. These are linear and do not change wavelength, so they are captured exactly in the lighting maps (§9.3).

**Quick path vs instrument path for lamp sheets.**
- *Quick path (no renderer, milliseconds):* envelope transmission applied as an angle-averaged spectral factor to the line list and continuum. Reflectors are treated as an efficiency factor.
- *Instrument path (renderer):* the full lamp is placed in a virtual integrating sphere or goniophotometer (§9.5). This gives total flux and angular distribution including every geometric effect, and it is the accurate number.
- A test keeps the two paths consistent where the quick path's assumptions hold.
- *Built (step 2): envelope on the quick path* (`lss/lamp/envelope.py`, `lss/lamp/quickpath.py`). A preset may name an envelope: a glass with tabulated n and k (clear soda-lime from Rubin 1985 so far) and a wall thickness. Its internal transmittance exp(−4πk·d/λ) is applied at normal incidence; surface reflection is left out (flat to 0.2 %, and on a coated tube it returns to the phosphor). Lamp sheets, the accuracy report and `lss fit` all use it, so a model is scored against a reference measured through its envelope like with like. It is not neutral: iron in the glass takes ~3 % at 780 nm and ~1 % in the green, which moves a 4000 K source by Δu′v′ ≈ 4×10⁻⁴.

Swapping a component recomputes only what depends on it. A generator change touches only the zone spectra. An envelope, reflector or filter change invalidates the lighting maps that include it.

---

## 6. Model-level switching and comparison

- A `ProfileCache` maps `(lamp_id, level, parameter_hash, data_version)` → zone spectra on disk.
- Changing level shows the cached result instantly. Otherwise it runs the generator as a background job and keeps showing the previous level with a "computing level N…" badge.
- Levels can be pinned side by side to see what the physics changed.
- **Comparison normalisation is explicit**, because it changes conclusions completely:
  - equal electrical power (only between levels that model electrical input);
  - equal luminous flux;
  - equal radiant flux;
  - absolute as computed.

  The current choice is shown on every compare view.

---

## 7. Measurement and comparison

- **Pixel probe:** spectrum arriving at a pixel (from the lighting maps × zone spectra), luminance, chromaticity, CCT.
- **Surface probe:** irradiance spectrum at a point. Probes are rendered as per-zone unit-spectrum irradiance values, so they re-colour instantly too. Derived quantities: illuminance, CCT/Duv, CRI Ra and R9, TM-30 Rf/Rg, S/P, melanopic EDI, UV dose.
- **Lamp sheet:** SPD (zone-summed and per zone), radiant and luminous flux, efficacy, CCT/Duv, CRI, TM-30 colour-vector graphic, the parameters and data versions that produced it, and its accuracy score against the reference if one exists.
  - *Built (step 2):* CRI R1–R14 and TM-30's per-hue numbers (local fidelity, chroma shift and hue shift for each of the 16 hue bins), in text and JSON. They are the colour-vector graphic's data; the graphic itself comes with the viewer (step 6).
- **No-render studies (instant, no 3D):** colour chart and test samples under a lamp, metameric pairs that match in daylight and split under HPS, CRI/TM-30. These are spectral products of lamp SPD and reflectance.
  - *Built (step 2): colour chart* (`lss chart`, `lss/studies/`). It shows how far each sample's colour moves under the lamp, compared with the same sample under TM-30's reference light at the same CCT. The method is TM-30's: 10° observer, CIECAM02 with full adaptation, CAM02-UCS. So the chart and the lamp sheet's Rf measure the same thing; with TM-30's own 99 samples (`--samples ces`) the chart's mean gives Rf to 0.05. Each shift is split into lightness, chroma and hue. Display colours (`--svg`, lamp over reference) use the 2° observer and CAT02 adaptation to D65.
  - Sample sets (`lss/data/reflectances.yaml`, read from colour-science datasets like the reference SPDs): the ColorChecker (BabelColor average of 30 charts, 380–730 nm at 10 nm), the CRI's TCS01–14 and TM-30's 99 samples. Tables are read piecewise linear and held at their end values beyond their range. That hold is an extrapolation, reported with each chart; the colour-matching functions are below 1 % of their peak there.
  - First result: under CIE HP1 (HPS), the ColorChecker's mean shift is 8.7, four times the LED-B3 model's 2.1, and the red patch loses 19 of its chroma.
  - *Built (step 2): metameric pair* (`lss metamers`, `lss/studies/metamers.py`). **The pair is constructed, not measured.** The second sample is a ColorChecker patch plus a reflectance change that leaves XYZ under D65 unchanged for both the 2° and the 10° observer (a "metameric black"). The match is imposed in the same arithmetic the studies use, so it holds to 10⁻¹¹ in XYZ. The rule: base the mid-grey patch (neutral 6.5); change a cosine series over 380–780 nm up to order 8; direction the one with the largest XYZ change under CIE A per unit reflectance change (daylight-to-tungsten is the CIE's standard change of illuminant for daylight-matched pairs), so it is not tuned to the lamps shown; size the largest that keeps the partner between 0.02 and 0.98.
  - Results (CAM02-UCS ΔE′ between the two samples): D65 0, A 1.7, FL2 7.0, FL11 15.0, LED-B3 L2 5.4, HP1 12.8. The spiky three-band tube splits the pair most. A, the light the direction was chosen under, splits it least: A is smooth like D65, so even this change barely separates under it, while spiky lamps pick out the change's wiggles.
  - **What this does and does not show.** How far the pair splits is set by the construction, not by any real material: it demonstrates the effect and ranks lamps for this one pair; it is not a finding about real paints. The smoothness limit matters most. Order 6 (no feature under a 67 nm half-period) gives splits below 0.4 under every lamp; order 10, 1.2–1.5× those of order 8. Order 8 was chosen after that scan, with its splits in view, but by a test that does not involve the lamps: the partner is then about as steep as the steepest ColorChecker patch (0.024 against 0.0237 per nm, within 2 %). A narrower Gaussian basis (σ = 15 nm) was tried first and rejected: it made the partner three times steeper than any patch. CAM02-UCS is also a weak scale under a light as orange as HPS, so compare lamps rather than read the HPS value exactly.
- **Compare:** A/B split or slider over the same scene, difference image, difference of probe spectra, across lamps or across model levels.
- **False colour:** illuminance, UV dose, "exceeds X lux" isolines.

---

## 8. Renderer

**Engine:** Mitsuba 3 (3.9.1, August 2026, ships Windows wheels for Python 3.9–3.14). Spectral variants:
- `scalar_spectral` for debugging;
- `llvm_ad_spectral` for multi-core CPU;
- `cuda_ad_spectral` if the spike (§8.3) confirms the RTX 5090 works.

Mitsuba gives a spectral path tracer, BVH, BSDFs, a Python API and differentiability.

**8.1 How lamps enter the scene.**
- Each zone of an `EmissionProfile` becomes an emitter with a **flat unit spectrum** and the zone's grey spatial/angular shape. The renderer never sees a lamp spectrum.
- Shapes that depend on direction need a small custom emitter plugin written in Python on Dr.Jit:
  - arc zones: impact parameter b and axis angle θ are computed from the surface point and outgoing direction;
  - LED angular patterns.
- Mitsuba's `bitmap` texture is RGB-only (spectral variants upsample it), so the design does not rely on per-texel spectra at all.
- Envelopes, reflectors and filters are ordinary shapes with spectral BSDFs.

**8.2 Scene.**
- A reference "lighting lab": ColorChecker with measured spectral reflectances, grey and white cards (non-fluorescent), matte and chrome spheres, a long wall for falloff, a metameric pair, and a camera.
- Materials: spectral reflectance curves; diffuse, rough-dielectric, rough-conductor (spectral complex IOR for metals).
- Scene objects and lamps are positioned per study (fixed scenes, §9).

**8.3 Renderer spike (before any renderer-dependent work).** Small scripts that answer, with evidence:
1. **Custom emitter:** a Python/Dr.Jit emitter with (b, θ) zone masks works in `llvm_ad_spectral`; measure the cost.
2. **Spectral output:** `specfilm` with N box channels (10 nm over the visible) works; measure noise per bin vs spp.
3. **Wavelength range:** can the spectral variants sample outside ~360–830 nm?
   - If yes, UV and IR maps are rendered directly.
   - If no, use the **proxy-wavelength** trick. The scene has no dispersion beyond its material curves, so a UV pass can be rendered by loading each material's UV reflectance into the visible slots and relabelling the output.
4. **GPU:** `cuda_ad_spectral` runs on the RTX 5090 (Blackwell). Dr.Jit builds its GPU kernels at run time, so this is likely fine but unverified.
5. **Recompilation cost** when resolution or max depth changes. This only matters for §9.6.

---

## 9. Rendering pipeline: fixed scenes with instant re-colouring

### 9.1 Principle

Light transport is linear in emission and, without fluorescence, does not mix wavelengths. For each lamp ℓ and zone k, render a **lighting map** `M_ℓk(x, y, λ_bin)` with that zone emitting a flat unit spectrum. Then for any zone spectra:

```
Image(x, y, λ) = Σ_ℓ Σ_k M_ℓk(x, y, λ_bin) · S_ℓk(λ)
```

Because the zones are fixed per emitter geometry, the following are **instant with no re-render**:
- every lamp-physics edit (model level, filament temperature, LED phosphor, arc pressure or current, including the arc's spatially varying spectrum);
- dimming;
- swapping between lamps that share an emitter geometry.

Exact, with one stated exception: light that returns to the emitter (§2). For lamps with strong back-reflection this holds within the v1 fixed-boundary approximation, and the size of that error is measured, not assumed (§12).

This is the main physics-exploration loop, and it is the most valuable optimisation in the project.

### 9.2 Fixed scenes (first usable version)

- A **study** fixes the camera, scene objects and lamp placements. Its lighting maps are rendered once at high quality as a background job, which may take minutes, and stored on disk.
- After that, the viewer is a pure post-process. Exposure, tone mapping, white point and false colour are free; re-colouring is instant.
- Moving a lamp, an object or the camera, or changing an envelope/reflector/filter material, makes the affected lighting maps stale. They are re-rendered as a background job with honest progress (§9.4). The viewer keeps showing the stale result with a visible "stale" state until the new maps land.

### 9.3 What the lighting maps capture, and their limits

- **Captured exactly:** any linear, wavelength-preserving transport, including glass transmission, dichroic reflectors, dispersion and in-scene filters. They are in the scene geometry, so they are in the maps.
- **Not captured:** fluorescence, which moves energy between wavelengths and would need a λ×λ transport matrix. The scene library therefore contains no fluorescent materials. Fluorescence that belongs to a lamp (phosphors) lives in the generator.
- **Accuracy knobs, each with a convergence test (§12):**
  - render bin width (§3.3);
  - zone count;
  - spp, controlled by the noise criterion.
- **Memory:** 1024 × 768 × 47 bins × float32 ≈ 150 MB per zone map. A lamp with 8 zones is about 1.2 GB. That is acceptable on the 64 GB target machine; maps live on disk and are memory-mapped.

### 9.4 Background render jobs with honest progress

- A **render worker process** (not a thread) owns the Mitsuba scene. This gives hard cancellation and crash isolation.
- Rendering is progressive: the map is the running mean of independent passes. Each pass is time-bounded (~200 ms, adaptive spp), so a cancel or a scene change is honoured quickly.
- Variance is tracked per pixel with Welford's algorithm over pass means, on luminance (to save memory).
- **Convergence:** luminance-weighted relative standard error below a threshold (default 1 %), computed per tile. A plain mean of per-pixel relative errors is dominated by dark pixels that never converge.
- **Progress message** (after every pass and on a heartbeat):
  `{study, map, pass_index, spp_done, elapsed_s, noise, threshold, eta_s, worker_alive}`.
- **Honest ETA and bar.** Noise falls as 1/√spp, so the spp still needed is `spp_done · ((noise/threshold)² − 1)`. That gives a real ETA, and a bar of `(threshold/noise)²` that grows linearly in time.
- **States:** Queued · Rendering (map j / M) · Converged · Stale · Paused · Worker unresponsive · Error.
- **Liveness watchdog:** two signals.
  - A heartbeat thread in the worker shows the *process* is alive.
  - Pass completions within a multiple of the expected pass time show the *render* is progressing.
  - Missing heartbeat → "Worker unresponsive" with a Restart button (kill by the worker's PID, respawn, replay the study). Heartbeat present but no passes → "Stalled".
- Optional denoiser (OIDN) only for preview display, badged. Stored lighting maps are never denoised, since denoising is not linear and would break the physics.

### 9.5 Virtual instruments

The renderer doubles as a lab:
- **Integrating sphere:** total flux of a full lamp, including envelope and reflector losses.
- **Goniophotometer:** angular intensity distribution I(ω, λ), exportable as IES/LDT.
- **Coil cavity:** effective emissivity of a ray-traced filament coil (§4.1 L3).

These feed lamp sheets on the instrument path (§5) and are cross-checked against the quick path.

### 9.6 Later milestone: interactive drag with progressive preview

Kept from revision 1, with fixes, for when geometric interactivity becomes a goal:
- Quality tiers: Preview (¼ res, 1 spp, depth 3), Draft (full res, 4 spp), Refine (8 spp/pass until converged).
- Generation counter and scene-state diffs applied via `mi.traverse`.
- Shared-memory framebuffer.

The spike result on recompilation cost (§8.3 item 5) decides whether "preview < 100 ms" is achievable or needs fixed-size kernels.

---

## 10. Cheap vs expensive edits (what the user feels)

| Edit | Cost |
|------|------|
| Exposure, tone map, false colour, white point | Instant (post-process) |
| Lamp model level, any lamp-physics parameter, dimming | Instant (re-colouring), after the generator runs; for reflector lamps, within the fixed-boundary approximation (§2) |
| Generator run: analytic models | Milliseconds |
| Generator run: Stage A/B arc | Seconds (background job) |
| Generator run: Stage C arc | Seconds to minutes (background job) |
| Colour chart, metameric pairs, CRI/TM-30 | Instant (no render) |
| Spectral probe at a pixel or surface point | Instant |
| Move lamp/object/camera, change envelope/reflector/filter | Background re-render of affected maps (minutes), honest progress |
| New emitter geometry or zone layout | New lighting maps (background job) |

---

## 11. Path to accuracy

This section is what makes "maximum accuracy" a direction rather than a slogan.

- **Reference library.** Measured lamps with SPD, and electrical and angular data where available. Each has provenance (source, instrument, date) and stated uncertainty. Level 0 of every family draws from it.
  - *Built (step 2): first measured lamps* (`lss/data/nist_fel.yaml`, references `nist-fel-f455` and `nist-fel-f456`). Two 1000 W quartz-halogen lamps (FEL type, coiled-coil tungsten), calibrated by NIST in 1997 for absolute spectral irradiance at 50 cm, 250–2400 nm, run at 8.200 A DC (109.4 and 109.8 V). The uncertainty is 0.9–4.4 % (k = 2), stated per wavelength. The values were transcribed by hand from a scanned report. A check that the two lamps differ only by a smooth trend (a 2.4 K temperature difference and a scale) leaves every node within half its uncertainty, where a misread digit would stand out by ~1 %. The report asks not to be reproduced except in full without NIST's approval: only the values and their citation are kept, not the report.
  - **Uneven tables are scored at their own wavelengths.** The table has 35 nodes, 10 nm apart in the UV and up to 300 nm in the IR, with only nine in the visible. Spectral errors are scored at those nodes, over the overlap with the model's own valid range (300–2400 nm for De Vos's data, 250–2400 nm for Planck), never at interpolated points. Colour, and the luminous normalisation, need the whole visible, so the table is filled in through its ratio to a blackbody, interpolated by a monotone cubic (`lss/accuracy/interpolation.py`). Linear interpolation of that ratio, or NIST's recommended blackbody × fifth-order polynomial, move the reference's chromaticity by at most 1.4×10⁻⁴ and its CCT by 2 K. That is the floor under colour scores against these tables. The quartz envelope is applied from fused-silica data (Franta 2016), which only absorbs in its water bands here (≤ 1.3 % at the table's wavelengths).
- **Accuracy report** (`lss accuracy`). For each reference lamp and each model level it reports:
  - spectral error: normalised RMS on the 1 nm grid, plus per-band error;
  - Δu′v′, ΔCCT, ΔRa/ΔRf;
  - Δflux and Δefficacy where electrical data exists.

  **At the reference's resolution:** a table recorded with a spectroradiometer bandpass (the CIE 5 nm tables: a triangle of 5 nm FWHM, which the FL11 table's Hg lines bear out) spread every lamp line over its neighbouring nodes. Before scoring, the model is sampled through the same triangular bandpass (`lss/spectral/bandpass.py`), so a sharp line is not scored against a smeared one. Each reference declares its `bandpass_nm`; a definition such as Illuminant A has none. Lamp sheets never see this step.

  **Like is compared with like:** a generator's bare output is first passed through whatever the reference already includes (its `includes` flags: envelope, phosphor, reflector), via the quick path or the instrument path (§5), and only then scored. The path used is recorded in the report.

  It runs in CI-style on every physics change; results are stored as golden files. An error increase beyond tolerance fails the run. Improvements are visible as numbers.
- **Parameter fitting** (`lss fit`). Fits a model's free *physical* parameters to a reference with `scipy.optimize.least_squares`, within physically plausible bounds. It reports the fitted values and how close they sit to the bounds. A fit that only works with implausible parameters is the signal of missing physics, and it points at which rung to build next. *Built (step 2):* the free parameters are those a level reads that have `bounds` in the preset. The default objective uses the report's own residuals, so a fit minimises exactly the NRMS the report shows; `--objective colour` matches chromaticity instead. `--objective uncertainty`, for a measured reference that states its uncertainty per wavelength, weights each wavelength by it (scale free, nodes treated as independent), and minimises the χ² per node the report shows for such references. It ignores the correlation of the systematic parts between wavelengths, which calibration reports rarely state; a fit with a full covariance is a possible later step. A model lacking a component its reference includes is refused. The tool prints the fitted value as preset YAML tagged `empirical`; it never edits presets.
- **No hidden fudge factors.** Every parameter is either a physical quantity with units and a source, or explicitly tagged `empirical` with a reason. Climbing the ladder means replacing `empirical` parameters with mechanisms. The count of `empirical` parameters per model is part of the accuracy report.
- **Uncertainty propagation** (later rung). Monte-Carlo over input-data uncertainties (broadening coefficients, transport data, potentials) gives error bands on spectra and efficacy. A disagreement with a measurement larger than these bands means the model is missing physics, not just short of data.
- **Provenance everywhere.** Every lamp sheet and render records model level, parameters, data versions and code version.

---

## 12. Validation (tests Claude Code can run)

Run all tests at below-normal process priority, so the machine stays usable. On Windows, from a POSIX shell:
`cmd //v:on //c "start /belownormal /b /wait python -m pytest & exit !errorlevel!"`.
This form keeps the real exit code. The shorter `start /belownormal` form without `exit !errorlevel!` reports success even when tests fail.

**Spectral and colour core (pure NumPy, run on every change)**
- **Planck:**
  - ∫B_λ dλ = σT⁴/π within 0.1 %, integrated on a wide log grid (50 nm–1 mm) or analytically;
  - band fractions (e.g. 360–830 nm) match the analytic series;
  - peak at Wien's wavelength.
- **Colour:**
  - CIE Illuminant A → CCT 2856 K, Ra ≈ 100;
  - D65 → (0.3127, 0.3290);
  - E → (1/3, 1/3);
  - our wrappers agree with `colour-science`;
  - TM-30 for CIE FL2 matches the IES TM-30-18 calculator: Rf, Rg and all 16 per-hue fidelity, chroma-shift and hue-shift values.
- **Tungsten** *(passes, step 2)*: tungsten's selective ε(λ, T) at 2856 K gives a CCT above 2856 K (about 2916 K), while a constant-ε greybody gives exactly the Planck chromaticity. The emissivity table itself must pass a physics check: ε changes monotonically with T at each wavelength, falling at short wavelengths and rising at long ones, with one crossover near 1.25 µm.
- **Coiled tungsten and the measured halogen lamps** *(passes, step 2)*:
  - the cavity formula gives ε at f_open = 1 and 1 as f_open → 0, raises ε most where it is smallest, and a coil at f_open = 0.35 is more than 20 K redder in CCT than a ribbon at the same temperature;
  - the NIST tables load in SI units, the interpolated spectrum passes through every node, and the two lamps differ only smoothly (every node within half its uncertainty of a scale times a temperature difference);
  - the interpolation floor: three methods agree within 2×10⁻⁴ in u′v′ and 3 K in CCT;
  - against lamp F-455, NRMS and χ² per node fall from Planck to flat tungsten to coil; only the coil has χ² per node below 1.5, with Δu′v′ < 3×10⁻⁴; F-456, not refitted, stays below χ² 1.5 and Δu′v′ 5×10⁻⁴;
  - the uncertainty treatments of §4.1 L3: with each node's total uncertainty independent, the coil has χ²/dof < 1.2 and the 300–800 and 800–2400 nm open fractions differ by < 2σ; with the random parts only, or with the systematic parts fully correlated, χ²/dof exceeds 2.5 and the gap exceeds 2.5σ; Planck and flat tungsten stay far worse under all three;
  - a table node on the edge of a model's valid range (De Vos's 300 nm) is scored inside that range, not averaged with the zero beyond it;
  - node residuals are exactly the reported NRMS, and the reported χ² per node is exactly what the uncertainty objective minimises.
- **Temperature from resistance (incandescent L4, first step)** *(passes, step 2)*:
  - Desai's resistivity table, as interpolated, agrees with his own polynomials within 0.1 % from 293 to 3600 K, rises at every step, and gives d ln ρ / d ln T of 1.1–1.3 near 3000 K;
  - temperature → resistance ratio → temperature round-trips; a ratio below 1 or past 3600 K is refused;
  - on F-455 (where the cold resistance was fitted) the first step gives Level 3's temperature within 0.05 K and the same χ² per node, with the same count of empirical numbers;
  - on F-456, with F-455's cold resistance, the resistance temperature is 8–11 K above F-455's and more than 4 K above what F-456's spectrum says, χ² per node is more than 1.5× Level 3's, and the cold resistance F-456 would need is 0.1–0.5 % above F-455's (or its voltage reading that much high).
- **Power balance (incandescent L4, second step)** *(passes, step 2)*:
  - the total emissivity table is read consistently: radiated power over ε·T⁴ is one constant on every row within 0.3 %, and the 1945 table repeats the 1934 values;
  - the balance closes (input = radiation + gas loss) and reproduces F-455's 3091.2 K and 109.4 V at 8.200 A; temperature and voltage rise with current;
  - at 8.000 A the response at 655 nm is inside NIST's 0.05–0.07 %/mA for gas shares 0–0.15, for a straight wire, and at either end of the emissivity slope's spread, and it rises with the gas share;
  - the response per fractional change of current does not depend on the filament's surface;
  - the implied wire is 0.25–0.33 mm thick and 0.8–1.05 m long;
  - on the presets, F-455's Level 4 equals Level 3 (χ² per node within 0.01), and F-456 comes out at F-455's temperature, 0.4 V below its own voltage.
- **LED phosphor layer** *(passes, step 2)*:
  - photon balance: every pump photon leaves the layer or is lost (die reflectance < 1, quantum yield < 1);
  - an empty layer passes the pump unchanged, and a flat absorber of depth τ passes 2E₃(τ) of the Lambertian pump;
  - blue leakage falls with emission angle, and with layer thickness;
  - Ce³⁺ absorption and emission peak S·ħω either side of the zero-phonon line, with widths growing as √coth(ħω/2kT).
- **Fluorescent violet line** *(passes, step 2; records a finding, §4.3 L2)*: with a free factor on the violet Hg lines, each CIE tube's factor is 0.85–0.97 (σ < 0.03) and gains < 0.5 % NRMS; a loss in proportion to BAM and one loss shared by every tube differ by < 2 in χ².
- **Colour chart** *(passes, step 2)*:
  - a Planck lamp below 4500 K is its own TM-30 reference: every patch shifts by < 0.02;
  - with TM-30's 99 samples the chart's fidelity equals the lamp sheet's Rf within 0.1 (FL2, HP1, LED-B3 L2);
  - a perfect white displays as sRGB white under any lamp (full adaptation);
  - sample tables sit on their nodes exactly and are held flat beyond them.
- **Metameric pair** *(passes, step 2)*:
  - the pair matches under D65 through the studies' own evaluation path, for both observers (ΔE′ < 10⁻⁶), and under a scaled D65;
  - it splits (ΔE′ > 1) under A, HP1, FL11 and LED-B3;
  - the partner stays within 0.02–0.98, reaches one of those limits (the size rule), and is within 2 % of the steepest ColorChecker patch's steepness.
- **Lines:**
  - rasterising a line list into bins conserves power for any bin width;
  - applying a sharp-edged filter to a line list gives the exact result, independent of the grid.
- **Voigt profile:**
  - reduces to Gaussian or Lorentzian in the limits;
  - FWHM matches Olivero–Longbothum;
  - ∫φ_λ dλ = 1.

**HID** *(passes for the sodium D lines and the lines ending on 3p, step 3: `tests/test_hid_arc.py`, except where marked)*
- **Stage A, composition:** the particles sum to the fill pressure, the electrons satisfy Saha's equation, each element keeps its share, and sodium is a few per cent ionised at an HPS axis.
- **Stage A, absolute strength:** the optically thin Na D emission at given n, T matches a hand calculation from A_ki and g-values (Σ hν₀/4π · A · n_k times the chord, within 0.2 %); ∫κ_ν dν gives the line strength from A_ki and the g-values.
- **Stage A, absolute strength, lines ending on 3p:** the same thin-arc check for the 3p–3d lines from the 3d population, within 0.2 %.
- **Stage A, units:** κ has dimension m⁻¹, by explicit bookkeeping (the checks above are in SI).
- **Widths from their sources:** the Stark FWHM and shift equal STARK-B's Å values converted by hand (2πc·w/λ²), at a table point and between points (per electron, linear in log N and log T); the shift moves a Stark-dominated line (3p–5d on an HPS axis) to the red by the profile's shift; every line ending on 3p takes its level's D-line resonance width; every line is inside STARK-B's isolated-line limit at an HPS axis.
- **Stage B, transfer integrator:**
  - an optically thin column gives I = ε·L;
  - an optically thick isothermal column saturates to B_λ(T).
- **Stage B, grid convergence:** with every line, refining the line-local grid or the shells 2× changes each analysis bin by < 0.1 % of itself outside the D lines' dark centre (585–594 nm), and by < 0.1 % of the peak bin everywhere. The self-reversed 819 nm lines need no exclusion: their centres stay at about a third of their peaks. The dark centre's own level is set by the coolest shell and converges slowly with the shell count.
- **Stage B, HPS self-reversal** (the D lines alone):
  - Na D at HPS conditions shows a central minimum between two maxima, and an isothermal arc shows none;
  - the peak separation increases with Na pressure. *Not yet:* matching the published peak-separation vs Na-pressure relation used as an HPS diagnostic.
- **Stage B, angle:** a thin line's radiance scales with 1/sin θ. *Not yet:* a thick line's core does not.
- **Measured street lamps** *(passes, step 3: `tests/test_street_hps.py`, skipped until `lss streethps` has downloaded the data; `tests/test_instrument.py` without data)*:
  - the response: a kernel fitted to lines sampled every 1 nm is recovered exactly; observed through it, a narrow line keeps its power (within 0.5 %) and its centroid moves by the offset plus the tail; LICA's mercury lines give 1.55–1.75 nm FWHM (rms residual < 2.5 % of the peak), and the same response, held fixed, fits the low-pressure sodium D and 568.8 nm doublets within 3 % and 2 % rms (819 nm 8 %);
  - LICA's response correction leaves the halogen lamp within 1 % of a tungsten spectrum in every 50 nm band from 450 to 850 nm;
  - the comparisons move by < 0.2 nm and < 10 % between responses;
  - *finding:* every subordinate line's power, the peak spacing and the far blue wing fall within the nine lamps' range, and every lamp's red wing is more than three times the model's, including the lamp whose peaks are closer together than the model's.
- **Stage C:**
  - HP-Hg at known current and pressure reproduces published axis temperature (~5500–6000 K) and field strength within ~15 %;
  - with the electrode-loss term, a 400 W HP-Hg lamp lands at 50–60 lm/W.
- **HPS efficacy:** 400 W HPS in 100–130 lm/W. This is expected to pass **only with Rung 2 line profiles**. With Voigt it is a tracked known failure, not a hidden one.

**Renderer (slower, on demand)**
- **Furnace:** a closed diffuse box (albedo ρ, uniform emission L_e) converges to L_e/(1−ρ).
- **Inverse square:** probe illuminance falls as 1/d² for a small emitter.
- **Re-colouring:** maps × spectrum equals a direct render with that spectrum, within noise.
- **Direct-calculation check:** a flat patch under a distant source renders to the XYZ computed directly from SPD × reflectance (ΔE₀₀ < 0.5). This replaces "matches published ColorChecker sRGB", which varies between charts.
- **Convergence of the accuracy knobs:**
  - halving the render bin width changes ΔE₀₀ by < 0.3 on the reference scene;
  - doubling the zone count changes it by < 0.3.
- **Instruments:**
  - integrating-sphere flux of a bare emitter equals its `power` integrated over the *renderer's* wavelength range. Total `power` covers all wavelengths, so in-band is compared with in-band;
  - the quick-path and instrument-path lamp sheets agree where the quick path's assumptions hold.
- **Returning-light bracket:** for each reflector lamp preset, render the virtual sphere and the reference scene with the emitter boundary transparent and with it opaque. The difference bounds the error of the fixed-boundary approximation (§2). It is reported, and flagged when above ΔE₀₀ 1 or 2 % flux.

---

## 13. Stack and project layout

- Python 3.12+ (Mitsuba 3.9.1 wheels exist up to 3.14).
- NumPy, SciPy, **colour-science**, Mitsuba 3 (spectral variants).
- PySide6 for the desktop UI, pyqtgraph for spectra and plots.
- `multiprocessing` for the render worker; memory-mapped lighting maps.
- `pytest`.
- Numba as an option, only if profiling shows the line-by-line transfer needs it.
- Alternative considered: Rust plus a custom path tracer. Rejected for now: the physics is the point, and Mitsuba removes most renderer work. Revisit if the spike (§8.3) finds a blocking limitation.

```
lss/
  spectral/      line lists, continua, filters, grids (analysis, extended, render), units,
                 generic line profiles (Gaussian, Lorentzian, Voigt),
                 colorimetry (thin wrappers over colour-science)
  physics/
    thermal/     planck.py, tungsten.py, filament.py (T from power, non-uniform T)
    led/         bands.py, phosphor.py, thermal.py
    lowpressure/ lines.py, uv_budget.py
    hid/         sodium.py (NIST levels and lines, partition function), composition.py (Saha),
                 broadening.py (pressure-broadening widths; later quasistatic wings),
                 arc.py (Stages A and B), later: Stage C, electrodes.py, chemistry.py (optional)
  lamp/          physical description, components, EmissionProfile (zones), ProfileCache
  accuracy/      reference library, fit.py, report.py
  studies/       no-render studies: reflectance samples, appearance, colour chart, metameric pairs
  scene/         study description, reference lab, materials, probes
  render/        mitsuba adapter, emitter plugins, worker, lighting maps,
                 re-colouring, progress, instruments (sphere, goniophotometer)
  ui/            viewer, lamp editor, compare view, probe panels, lamp sheet
  data/          line lists, broadening/potentials, transport, emissivities, glass/PCA,
                 phosphor spectra, reference SPDs, ColorChecker (all with citations)
  cli.py         lampsheet / accuracy / fit / render / probe (headless)
tests/           pytest suite (outside the package)
```

CLI first:
- `lss lampsheet hps_400w.yaml`
- `lss accuracy`
- `lss fit hps_400w.yaml --level 2 --ref lspdd:...`
- `lss render study_lab.yaml`
- `lss probe … --at x,y,z`

The GUI wraps the same library.

---

## 14. Roadmap

1. **Spectral and colour core.** *Done (2026-10-14).*
   - Line lists, the three grids, units, `colour-science` wrappers.
   - Reference library with provenance; Level-0 lamps.
   - CLI lamp sheet; accuracy-report scaffolding.
   - Core tests in §12 pass, except the tungsten test, which needs the De Vos data and moves to step 2.
2. **Thermal, LED and fluorescent models.** *In progress: incandescent L2–L4 (L3 as the one-zone cavity estimate; L4 as the temperature from resistance, with a fitted cold resistance, then as the power balance from the current alone, with a solved filament surface and an empirical gas loss, 2026-10-15), the first measured references (NIST FEL halogen lamps), `lss fit`, LED L1–L3 (L3 as the heat-only first step from cited coefficients, checked against a measured LED's warming, 2026-10-15), fluorescent L1 (tri-band; for L2, the coat's violet absorption was checked against the tables and not built, 2026-10-15) and the no-render studies (colour chart, constructed metameric pair, CRI R1–R14 and TM-30 per hue) done (2026-10-14).* Incandescent L1–L4, LED L1–L3, fluorescent L1–L2, each with `fit` against references and scores in the accuracy report. No-render studies: colour chart, metameric pairs, CRI/TM-30.
3. **HID Stage A + B on line-local fine grids.** *In progress: Stages A and B for the sodium resonance lines at rung 1 (Saha composition, measured self-broadening, estimated Hg/Xe broadening, exact shell-by-shell transfer at any angle), checked against physical limits only (2026-10-15); the lines ending on 3p, with STARK-B's Stark widths and shifts on every line (2026-10-15); against nine measured street lamps through their spectrograph's response, which shows the rung-1 red wing 3–10 times too faint (2026-10-15); rung 2's sodium-pair wings, which missed the mark set against a quantum calculation (most of the miss traced, not certainly, to its rotational limit) and, as a generous estimate, double the red wing without closing it (2026-10-15). rung 2's sodium–mercury pairs, from the only excited curves found (digitised from a 1991 figure, shifted by a rule fixed beforehand), which put the predicted satellite where it is measured (coarsely) and bring the red wing and its slope among the lamps', at their faint end, but do not make the lamps' rise at 668–678 nm (2026-10-15). Each lamp's arc temperature and sodium pressure fitted to its blue peak and three lines (2026-10-16): temperatures within 200 K of the book's measured 4100–4300 K (seven of nine if the line powers are taken over the D band's blue half), but the red wing still 1.2–5 times too faint for seven lamps in every variant, so the arc's temperature and sodium do not explain it. Each lamp's mercury read off its own peaks by the book's formula, the arc refitted with it (2026-10-16): the red wing's level reached (lamp/model 0.99 on average, 0.68–1.46 per lamp), as it is with one shared mercury ratio, conditional on the formula's putting all red-side asymmetry down to mercury; the red slope still too shallow, no 668–678 nm rise; and a false red peak where the pair wings are spliced on, which more mercury brings out. The join made smooth (2026-10-16): a 0.6 nm hole at each splice filled (it was the blue peak's jump), the step replaced by a gradual hand-over near the Weisskopf detuning, the Na–Na table split onto each D line; one red maximum, the join's place no longer matters, sodium no longer depends on mercury, the red wing's level with own mercury 1.09. Candidates for next, not chosen: sodium–xenon pairs; the Czuchaj et al. (1991) curves and dipoles (the copy at hand is their 1992 CdHg paper, not it); the red slope and the 668–678 nm rise, which nothing so far makes.*
   - Voigt profiles, (b, θ) chords, zone spectra.
   - HP-Hg and HPS presets; self-reversal tests.
   - The headline result needs no renderer, so it arrives here.
4. **Renderer spike** (§8.3). It decides the CPU or GPU variant, the emitter plugin approach, and direct vs proxy-wavelength UV/IR.
5. **Renderer adapter.**
   - Zone emitters, lighting maps, reference lab, CLI render.
   - Probes and virtual instruments.
   - Renderer tests pass.
6. **Viewer UI: first usable version.**
   - Fixed studies with instant re-colouring.
   - Lamp editor with model-level switch and normalisation choice.
   - Probes, compare view, lamp sheet, TM-30 graphics, false colour.
   - Background re-render jobs with honest progress.
7. **HID accuracy.**
   - Stage C (Elenbaas–Heller) with electrode losses: first with the net-emission-coefficient radiation term, then with full interior J_λ(r).
   - Rung-2 quasi-static wing profiles for HPS; Hg₂ continuum.
   - The HPS efficacy test turns green.
8. **Uncertainty propagation**, plus the LED volumetric phosphor (L4) and the fluorescent positive-column model (L3).
9. **Later:**
   - interactive drag with progressive preview (§9.6);
   - metal-halide chemistry;
   - PCA as a scattering medium;
   - AC operation and warm-up;
   - non-LTE;
   - the other rungs in §4.5.

---

## 15. Open questions

- **Renderer limits:** wavelength range and RTX 5090 support. Answered by the spike (§8.3).
- **Reference measurements:** which lamps, and from where (LSPDD, manufacturers)? Is a spectrometer available for our own measurements? Measured SPD plus electrical data for the same lamp is the most valuable input to the accuracy loop.
  - **Incandescent needs a measured reference.** *Partly answered (2026-10-14):* two NIST-calibrated FEL halogen lamps (§11) now score L1–L3 against real tungsten, with current and voltage. Against Illuminant A, a blackbody by definition, better physics can still score worse (L2's residual there is about 2 % NRMS at matched CCT), so A stays a colour calibration. *L4 partly answered (2026-10-15):* tungsten's resistivity is Desai et al. (1984), and L4 now takes the temperature from resistance, but with the cold resistance fitted to F-455 (§4.1 L4). Still missing: each lamp's own measured cold resistance (or wire diameter and length), which neither report gives and which the F-456 check shows cannot be shared between lamps. The power balance (built 2026-10-15, with Forsythe & Watson's total emissivity) needs a lamp measured at several currents, with a four-wire voltage, to be tested beyond NIST's one-digit 0.06 %/mA; and the fill gas and coil dimensions, to replace the empirical gas loss with Langmuir's sheath model. A lamp of another design (a general-service or vacuum lamp) would also test L3 beyond one batch.
  - **Needed before step 3.** The starter references (CIE 15:2018 tables) are 5 nm, 380–780 nm, relative and representative rather than measured. They check chromaticity, CCT and colour rendering, but not line shapes: the HPS self-reversal dip is a few nm wide and a 5 nm table smears it out. HID line-shape work needs finely resolved (≤ 1 nm) real spectra of HPS and HP-Hg lamps, ideally with electrical data. Every reference records its `sampling_nm` so the accuracy report shows what it can and cannot check.
    *Partly answered (2026-10-15):* the LICA-UCM lamp database (C. Tapia, A. Sánchez de Miguel & J. Zamorano, UCM eprint 40930 (2015); https://guaix.fis.ucm.es/lamps_spectra) has 9 HPS, 3 mercury-vapour and 13 metal-halide street lamps, and one low-pressure sodium lamp. It also has an Oriel mercury pencil lamp. They were measured with an Ocean Optics Jaz spectrograph, sampled at 1 nm over 300–900 nm.
    - **Resolution.** The pencil lamp's lines, which are far narrower than the instrument, show its real response: about 2 nm wide, with a longer tail on the red side; the 546.075 nm line's centroid reads 546.79 nm, 0.7 nm to the red (tail and any wavelength-scale offset together).
    - **What it can test.** The HPS peaks (about 10 nm apart) are resolved. The depth of the dip between them is not: the response fills it in.
    - **What it lacks.** There is no electrical data, wattage, arc-tube size or fill: these are lamps in the street.
    - **The response, fitted (2026-10-15;** `lss/spectral/instrument.py`, `lss/accuracy/lica.py`**).** Every file used states the same spectrograph (Jaz serial JAZA2310) and LICA's corrections for dark signal, non-linearity and spectral response. A Gaussian core convolved with an exponential red tail, fitted to the pencil lamp's three isolated lines (404.7, 435.8, 546.1 nm), has 1.64 nm FWHM and puts a line's peak 0.33 nm to the red; held fixed, it fits the low-pressure sodium lamp's D and 568.8 nm doublets within 2.7 % and 1.7 % rms of the peak, with no further shift. It fits the 819 nm doublet less well (8 %): two argon lines of LICA's thorium–argon lamp (738.4, 763.5 nm; another lamp) give 1.41 nm FWHM and the peak 0.13 nm to the blue, and fit the 819 nm doublet worse (15 %). So the two lamps' responses differ and neither fits 819 nm within 8 %; whether wavelength, lamp or session causes it is not shown. The neon lamp's lines are too crowded at 1 nm steps to fit. The street lamps' features are chosen so this does not matter (§4.4).
    - **LICA's response correction, checked.** The halogen lamp against a coiled tungsten filament (temperature, open fraction and scale free) fits at 2906 K within 0.6 % in every 50 nm band from 450 to 850 nm, so the correction has no structure beyond a smooth tilt. The open fraction goes to its bound (0.05, near a blackbody), so a small tilt may be absorbed there. The mercury lines' powers do not settle it either: against Reader et al.'s pencil lamp the 365 nm group is 1.5–1.6 times and 404.7 nm 1.2–1.3 times as strong relative to 546.1 nm (depending on the response fitted alongside), but all three end on metastable 6p levels and can self-absorb by an amount that depends on the lamp, so this cannot tell the correction from the lamp.
    - **Terms.** The site asks to be contacted before the data are used. The repository therefore does not include the files: it downloads them on first use into a cache outside git, checked by checksum, and keeps only the citation, the links and the numbers derived from them (decision, 2026-10-15).
- **Sodium–mercury curves.** *Partly answered (2026-10-15):* the upper curves come from one figure (Windholz et al. 1991), digitised from a scan, with no transition dipoles. Czuchaj, Rebentrost, Stoll & Preuss (Chem. Phys. Lett. 178, 246 (1991)) tabulates curves and dipoles and would replace both the drawing and the dipole bracket. Measured Na–Hg far-wing absorption (Schlejen, Woerdman & Pichler 1988; Woerdman et al., J. Phys. B 18, 4205 (1985)) would test the wing's strength, as Chung et al.'s calculation did for sodium pairs; none of these is openly available.
- **Metameric pairs.** The study's pair is constructed (§7). The CIE's standard virtual pairs (ISO/CIE 23603, the successor of CIE 51.2's metamerism index: pairs matched under D55, D65 or D75 for the 10° observer) are distributed only on the standard's companion disk, not openly (checked 2026-10-14). Those pairs, or a measured pair of real samples, would make the split a standard or a real result rather than a demonstration.
- **LED phosphor spectra.** No tabulated excitation or emission spectra of LED garnets or nitrides were found (2026-10-14; papers, patents and maker leaflets show them as plots only). LED Level 2's shared phosphor constants are therefore fitted to the five CIE LED tables they are scored on (§4.2). A measured spectrum of a single garnet or nitride, or measured LEDs outside the CIE set, would make the scores a real test and pin the garnet's coupling S.
- **LED drive data.** *Partly answered (2026-10-15):* the first measured LEDs at several currents (Wu et al. 2026, §4.2 L3) give the voltage and radiant power, but not the board temperature, the thermal resistance, the die count or the phosphors, and their spectra are clipped where the signal is weak. They show the die's cited peak shift is consistent and that their green phosphor quenches far faster than the cited garnet, but cannot test Level 3's quenching of the garnet and nitride it models. A real test needs an LED of known phosphors (or their measured quenching curves) measured at several currents and board temperatures, with the board's temperature and the forward voltage at each, and spectra without a signal floor.
- **Transport coefficients for Hg/Na/Xe:** literature tables first. Computing them from collision cross-sections is the accurate rung, but it is a project in itself.
- **Interatomic potentials** for the HPS wing profiles (Na–Na, Na–Hg, Na–Xe): which published sets to adopt. *Na–Na partly answered (2026-10-15):* Knoop et al. (2011) for the lower curves, Chung, Kirby & Babb (2001) digitised for the upper ones (§4.4). Still missing: open tables of the Na₂ upper curves and of the X–A, X–B transition dipoles inside 12 a₀ (Stevens et al. 1977 and the RKR sources are not open), and any usable Na–Hg or Na–Xe curves.
- **Default comparison normalisation** in the UI (equal luminous flux is the likely default).
