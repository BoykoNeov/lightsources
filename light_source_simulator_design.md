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
  - *Scored against a measured lamp* (§11: NIST-calibrated 1000 W FEL halogen lamps, coiled-coil filament, 300–2400 nm). The fit weights each of the table's wavelengths by NIST's stated uncertainty there (`lss fit --objective uncertainty`), and the report gives χ² per node: the mean squared error in units of that uncertainty, about 1 when a model matches within the measurement. Fitted to lamp F-455: T = 3091 K, f_open = 0.30; χ² per node 0.71, NRMS 1.0 %, Δu′v′ 1.7×10⁻⁴ (the floor from interpolating the table is 1.4×10⁻⁴), ΔCCT −5 K. The ladder climbs as it should on real tungsten: χ² per node 73 for Planck, 34 for flat tungsten (L2), 0.71 for the coil (L3). Only the coil matches the lamp within its measurement. The second lamp, F-456, scored with F-455's values and not refitted: χ² per node 1.1, Δu′v′ 2.9×10⁻⁴. Both lamps are one model from one batch, so this shows the fit carries to another lamp of the same design, not to other designs.
  - *Why the uncertainty weighting.* The default `spectral` objective weights every wavelength by the mean level of the table. Then the bright near IR (≈ 90 % of the mean at 800–1300 nm) outvotes the dim violet and UV (2–20 %). Fitted that way the coil lands at T = 3075 K, f_open = 0.35, NRMS 0.68 %, but 2–3.5 % low at 300–400 nm, three to six times the stated uncertainty there; about two thirds of its squared error is then above 780 nm, where it is within the uncertainty. Weighting by the uncertainty trades 0.35 % of NRMS for a fit that is within the measurement everywhere, and the colour error falls from 6.4×10⁻⁴ to 1.7×10⁻⁴.
  - *One f_open fits the whole range.* Fitted separately (weighted as above), 300–800 nm gives f_open = 0.28 ± 0.05 at 3094 K and 800–2400 nm gives 0.42 ± 0.07 at 3054 K: 1.7 standard errors apart, not a significant difference (`tests/test_nist_fel.py`). An earlier unweighted split (0.18 against 0.46) had looked like a wrong spectral shape; with the stated uncertainty it is not one. So this table does not show that the one-zone formula is wrong. With χ² near 1 it cannot test finer physics either: a non-uniform filament temperature (L5), the coil's outer faces radiating as bare tungsten, or De Vos's ε(T) extrapolated from 2800 K to 3091 K would all have to show up below the measurement's 1–4 %.
  - *Not fitted against Illuminant A.* Tried once (2026-10-14, a copy of the A preset with a bounded f_open): both objectives drive f_open to its lower bound, 0.05, a nearly closed cavity, and `lss fit` flags it. A is a blackbody by definition, so that is an artefact of the definition, not a finding. The CIE A preset stops at L2.
- **L4 — Temperature from electrical input.** Solve P_elec = P_rad(T) + P_cond(T) + P_lead. Inputs:
  - P_rad from L2/L3 integrated analytically over all λ;
  - P_cond from a fill-gas (Langmuir sheath) model;
  - tungsten resistivity ρ(T) linking V, I and T;
  - the empirical lamp laws (T ∝ V^0.42 …) only as the documented low rung.
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
    - **Die:** a Gaussian in energy (alloy disorder) convolved with the Boltzmann tail exp(−ΔE/kT) of hot carriers, which gives the blue band a longer short-wavelength side (Schubert, *Light-Emitting Diodes*, §5.1). T is the junction temperature. Per lamp, in the physical description: peak, width, junction temperature.
    - **Garnet (Ce³⁺):** a configurational-coordinate model with one phonon (Henderson & Imbusch 1989, ch. 5). The 4f→5d₁ absorption peaks S·ħω above the zero-phonon line and the emission S·ħω below it, both Gaussian with σ² = S(ħω)² coth(ħω/2kT) plus an inhomogeneous spread. The emission ends on ²F₅/₂ and ²F₇/₂, so it is two such bands. ħω = 200 cm⁻¹ is measured (Bachmann et al., Chem. Mater. 21, 2077 (2009)). S, the inhomogeneous width, the ²F splitting and the two bands' ratio are shared by every lamp (`lss/data/phosphors.yaml`, `led_phosphors`). Each lamp sets its own zero-phonon energy, which stands for its garnet's composition (Lu and Ga raise it, Gd lowers it).
    - **Nitride (Eu²⁺):** phenomenological and shared. One Gaussian emission band in energy, and an absorption edge (an error function) that is flat across the blue and falls off through the green.
    - **Layer:** a flat slab with no scattering. The die lights it from below as a Lambertian source, and the phosphors absorb by Beer–Lambert along each path (τ/cos θ). Each lamp sets two optical depths. Phosphor light is emitted isotropically; light going down is reflected by the die (reflectance 1 at this rung). Light the layer absorbs again is re-emitted by whichever phosphor took it, so the nitride takes part of the garnet's green. Quantum yields are 1, so the output is relative.
    - **Output:** three emission-angle zones (0–30°, 30–60°, 60–90°, angles inside the layer; refraction at its surface is not modelled). More blue leaks straight ahead than at wide angles, because the oblique path is longer.
    - The photon balance closes exactly (every pump photon leaves or is lost), and the leaked pump equals the analytic 2E₃(τ) of a Lambertian source.
  - *Scores* (`lss fit led_cie_b1 led_cie_b2 led_cie_b3 led_cie_b4 led_cie_b5 --level 2`). Five CIE phosphor-converted blue LEDs, 2650–6600 K. Level 1, fitted to each lamp alone with 6 numbers: NRMS 8.0/6.7/8.1/6.5/6.1 %. Level 2, in-sample: 1.7/2.1/2.8/5.1/4.7 %, Δu′v′ 0.2–0.7×10⁻³, |ΔRa| ≤ 0.6. 14 empirical numbers per lamp, 8 of them shared; 6 are the lamp's own, as at Level 1.
  - *Out of sample* (shared constants fitted to the other four lamps, then only the lamp's own six fitted; `--hold-out <lamp>`): B1 2.9 %, Δu′v′ 1.7×10⁻³; B2 2.6 %, 8.6×10⁻⁴; B3 4.6 %, 1.2×10⁻³; B4 6.0 %, 6.3×10⁻⁴; B5 5.3 %, 6.9×10⁻⁴. So Level 2 predicts a lamp it never saw better than Level 1 fits it. **Limits of this test:** the five come in similar pairs (B1/B2 warm, B4/B5 cool), so a held-out lamp usually has a sibling among the four. B3, between the pairs, is the strictest case (8.1 % → 4.6 %). The shared constants are still learned from the same family of tables they are scored on. The real test is a measured phosphor spectrum, or LEDs from another source.
  - *Findings:*
    - L1's too-wide pump (24–26 nm) is explained. The die widths come out at 18–22 nm, inside InGaN's usual range. The garnet's absorption reshapes the leaked blue (its peak sits on the pump's centre in the warm LEDs, ~459 nm, and on its red side in the cool ones, ~478 nm), and the hot-carrier tail fills the short-wavelength side.
    - The fit picks a greener garnet for the warm LEDs (zero-phonon line ~21 100 cm⁻¹, a Lu/Ga-type garnet) and one close to YAG for the cool ones (~20 300 cm⁻¹, YAG 20 450). The two warm LEDs also carry 6–8 × the cool ones' nitride (optical depth 0.56–0.64 against 0.08–0.09; B3 0.34). This matches how warm and high-CRI parts are usually made: a green garnet plus a red nitride.
    - The fitted coupling S ≈ 3.3 is about half YAG:Ce's measured S ≈ 6. The model garnet's Stokes shift (2Sħω ≈ 1300 cm⁻¹) is therefore about half the measured 2400 cm⁻¹, and its width comes mostly from the inhomogeneous term. A 5 nm table sees the garnet's absorption only through the shape of the leaked blue, so S is weakly constrained; a measured excitation spectrum would fix it.
    - Junction temperature is poorly determined. The three warm LEDs sit at the 300 K bound; the cool ones at 360–380 K. In the hold-out fits it drifts to either bound. The CIE tables may have been measured pulsed (junction near room temperature), but the tables do not say, so the tail slope acts as a shape parameter here. Level 3 should compute it from the drive.
    - The ²F splitting (3120 cm⁻¹) and the nitride's width (132 nm) sit near the top of their ranges. The cool LEDs keep ~5 % error, mostly a cyan-valley shape (480–530 nm) that the garnet's short-wavelength edge does not follow; scattering, which lengthens paths and sharpens self-absorption, is the likely missing piece (L4).
- **L3 — Thermal.** Junction temperature from thermal resistance × dissipated power. Effects: pump peak shift (~0.03–0.05 nm/K), pump efficiency droop, phosphor thermal quenching.
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
    - Violet Hg is too strong against blue: the model overshoots at 405 nm and undershoots at 436 nm in all three lamps. The pencil lamp's 405:436 ratio is 0.44; the tables imply less, as expected if the phosphor coat absorbs violet before it leaves the tube. The ratio stays fixed at L1; L2's phosphor absorption should produce it.
    - A smooth floor of about 3–10 % of the mean level at 380–415 nm and 720–780 nm, the same in all three tables and following none of the phosphors. Likely stray light in the CIE measurement or a weak discharge continuum. L1 has no such term; it is worth about 2–3 % NRMS.
    - The 555–600 nm valley sits higher in the tables than any band explains. The YOX ⁵D₀→⁷F₁ band fills it: its width runs to the 20 nm bound and its share (~0.42 of the 611 nm line) is several times what that Eu³⁺ group carries.
    - The YOX 611 nm width runs to its 6 nm bound: the 615 nm node holds more than one narrow line at 612 nm gives it, so further ⁵D₀→⁷F₂ components are unresolved.
    - Hg 546 nm sits on the green phosphor's main peak, and the three mixes barely move Hg against green, so the fixed 546:436 ratio decides how the 545 nm feature is split between them. Scaling the 546 nm line by 0.8 or 1.2 moves the green phosphor's main-band share of its power by ±2.3 % (0.658 / 0.643 / 0.629) and each lamp's Hg power by about ±10 %; the scores move by < 0.3 % NRMS (slightly better at 1.2). The split is set by the measured ratio, not by the data.
    - Measured phosphor emission spectra (L2) would replace every learned shape and test them; until then the shapes are calibrations.
- **L2 — UV budget.** The 254/185 nm resonance radiation pumps the phosphor (excitation spectrum, QE, Stokes loss). Visible Hg line fraction and 254 nm output depend on Hg vapour pressure via the cold-spot temperature. Electrode and ballast losses are an explicit term. This gives efficacy from the energy budget.
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
- Tungsten emissivity and resistivity; glass, quartz and PCA transmission; phosphor excitation/emission/QE spectra.
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
- **Parameter fitting** (`lss fit`). Fits a model's free *physical* parameters to a reference with `scipy.optimize.least_squares`, within physically plausible bounds. It reports the fitted values and how close they sit to the bounds. A fit that only works with implausible parameters is the signal of missing physics, and it points at which rung to build next. *Built (step 2):* the free parameters are those a level reads that have `bounds` in the preset. The default objective uses the report's own residuals, so a fit minimises exactly the NRMS the report shows; `--objective colour` matches chromaticity instead. `--objective uncertainty`, for a measured reference that states its uncertainty per wavelength, weights each wavelength by it (scale free), and minimises the χ² per node the report shows for such references: the statistically right fit to a measurement. A model lacking a component its reference includes is refused. The tool prints the fitted value as preset YAML tagged `empirical`; it never edits presets.
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
  - weighted by the stated uncertainty, one f_open fits 300–2400 nm with χ² per degree of freedom < 1.2, and the 300–800 and 800–2400 nm halves agree within two standard errors;
  - a table node on the edge of a model's valid range (De Vos's 300 nm) is scored inside that range, not averaged with the zero beyond it;
  - node residuals are exactly the reported NRMS, and the reported χ² per node is exactly what the uncertainty objective minimises.
- **LED phosphor layer** *(passes, step 2)*:
  - photon balance: every pump photon leaves the layer or is lost (die reflectance < 1, quantum yield < 1);
  - an empty layer passes the pump unchanged, and a flat absorber of depth τ passes 2E₃(τ) of the Lambertian pump;
  - blue leakage falls with emission angle, and with layer thickness;
  - Ce³⁺ absorption and emission peak S·ħω either side of the zero-phonon line, with widths growing as √coth(ħω/2kT).
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

**HID**
- **Stage A, absolute strength:** the optically thin Na D emission at given n, T matches a hand calculation from A_ki and g-values.
- **Stage A, units:** κ has dimension m⁻¹. This is a unit test using a units library or explicit dimensional bookkeeping.
- **Stage B, transfer integrator:**
  - an optically thin slab gives I = ε·L;
  - an optically thick isothermal slab saturates to B_λ(T).
- **Stage B, grid convergence:** analysis-bin radiances change < 0.1 % when the line-local grid is refined 2×.
- **Stage B, HPS self-reversal:**
  - Na D at HPS conditions shows a central minimum between two maxima;
  - the peak separation increases with Na pressure and matches the published peak-separation vs Na-pressure relation used as an HPS diagnostic.
- **Stage B, angle:** a thin line's radiance scales with 1/sin θ, and a thick line's core does not.
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
    hid/         lines.py (NIST import), profiles/ (pressure-broadening widths, quasistatic.py),
                 lte.py (Saha, partition fns), transfer.py (Stage B),
                 arc.py (Stage C), electrodes.py, chemistry.py (optional)
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
2. **Thermal, LED and fluorescent models.** *In progress: incandescent L2–L3 (L3 as the one-zone cavity estimate), the first measured references (NIST FEL halogen lamps), `lss fit`, LED L1–L2, fluorescent L1 (tri-band) and the no-render studies (colour chart, constructed metameric pair, CRI R1–R14 and TM-30 per hue) done (2026-10-14).* Incandescent L1–L4, LED L1–L3, fluorescent L1–L2, each with `fit` against references and scores in the accuracy report. No-render studies: colour chart, metameric pairs, CRI/TM-30.
3. **HID Stage A + B on line-local fine grids.**
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
  - **Incandescent needs a measured reference.** *Partly answered (2026-10-14):* two NIST-calibrated FEL halogen lamps (§11) now score L1–L3 against real tungsten, with current and voltage. Against Illuminant A, a blackbody by definition, better physics can still score worse (L2's residual there is about 2 % NRMS at matched CCT), so A stays a colour calibration. Still missing for L4 (temperature from electrical input): the filament's dimensions (wire diameter and length) or its cold resistance, which neither report gives, plus tungsten's resistivity ρ(T) and total hemispherical emissivity with citations. A lamp of another design (a general-service or vacuum lamp) would also test L3 beyond one batch.
  - **Needed before step 3.** The starter references (CIE 15:2018 tables) are 5 nm, 380–780 nm, relative and representative rather than measured. They check chromaticity, CCT and colour rendering, but not line shapes: the HPS self-reversal dip is a few nm wide and a 5 nm table smears it out. HID line-shape work needs finely resolved (≤ 1 nm) real spectra of HPS and HP-Hg lamps, ideally with electrical data. Every reference records its `sampling_nm` so the accuracy report shows what it can and cannot check.
- **Metameric pairs.** The study's pair is constructed (§7). The CIE's standard virtual pairs (ISO/CIE 23603, the successor of CIE 51.2's metamerism index: pairs matched under D55, D65 or D75 for the 10° observer) are distributed only on the standard's companion disk, not openly (checked 2026-10-14). Those pairs, or a measured pair of real samples, would make the split a standard or a real result rather than a demonstration.
- **LED phosphor spectra.** No tabulated excitation or emission spectra of LED garnets or nitrides were found (2026-10-14; papers, patents and maker leaflets show them as plots only). LED Level 2's shared phosphor constants are therefore fitted to the five CIE LED tables they are scored on (§4.2). A measured spectrum of a single garnet or nitride, or measured LEDs outside the CIE set, would make the scores a real test and pin the garnet's coupling S.
- **Transport coefficients for Hg/Na/Xe:** literature tables first. Computing them from collision cross-sections is the accurate rung, but it is a project in itself.
- **Interatomic potentials** for the HPS wing profiles (Na–Na, Na–Hg, Na–Xe): which published sets to adopt.
- **Default comparison normalisation** in the UI (equal luminous flux is the likely default).
