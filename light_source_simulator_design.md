# Light Source Simulator — Design Document

Physics-first desktop tool for modelling lamps as spectral emitters, rendering their effect on a scene with a spectral ray tracer, and comparing sources. Intended to be built incrementally with Claude Code; each section ends with concrete, testable work items.

---

## 1. Goals and non-goals

**Goals**
- Model light sources at several fidelity levels, from tabulated spectra to a 1-D arc plasma solved from first principles, with instant toggling between levels.
- Represent a lamp as an assembly of swappable components (emitter, envelope, coating, reflector, filter).
- Render a scene spectrally and let the user move the lamp, scene objects, filters, and camera.
- Keep the UI responsive: a coarse render appears immediately, the high-fidelity render converges in the background with honest progress reporting.
- Measure: spectrum at a point, illuminance, CCT, CRI/TM-30, efficacy; compare sources side by side.

**Non-goals (for now)**
- Photorealistic materials beyond what is needed for lighting studies.
- Time-resolved behaviour (flicker, warm-up). Flagged as a later extension.
- 3-D arc simulation (convection, electrode physics). The 1-D cylindrical arc is the ceiling.

---

## 2. Core abstraction

The renderer never knows what a lamp is. Every lamp, at every fidelity level, produces the same thing:

```
EmissionProfile
  geometry      : emitting surface(s) as mesh(es), in lamp-local coordinates
  radiance(x, ω, λ) : spectral radiance leaving point x in direction ω  [W·m⁻²·sr⁻¹·nm⁻¹]
  power         : total radiant flux, for sanity checks and normalisation
  metadata      : model level, parameters, derived metrics (CCT, CRI, lm/W)
```

In practice `radiance` is tabulated, not evaluated analytically: a spectral texture over the emitter surface (W × H × N_λ) plus an angular model (Lambertian, measured IES/LDT distribution, or a per-texel cosine-power lobe). Physics models are *generators* that fill this table; the renderer only ever reads it. Consequences:

- Toggling fidelity = selecting a different cached `EmissionProfile` for the same lamp. Instant.
- An expensive model (arc solve) runs once per parameter set, is cached on disk keyed by a parameter hash, and never blocks the renderer.
- A half-finished high-fidelity model is still usable, because it only has to produce the table.

---

## 3. Spectral representation

- Grid: 300–1000 nm at 5 nm (141 bins) internally; 380–780 nm is the display sub-range. UV and near-IR matter for halogen/HID studies (UV dose, IR load).
- All spectral quantities are stored on this grid as `float64` NumPy arrays. Resampling happens once at import, never during rendering.
- Radiometric units are the source of truth: W, W·sr⁻¹·nm⁻¹, W·m⁻²·nm⁻¹. Photometric values (lm, lx, cd) are derived via V(λ) with K_m = 683 lm/W.
- Colour is derived at display time only: spectrum → XYZ (CIE 1931 2°, optionally 2006 10°) → linear sRGB → exposure/tone map → display. Rendering stays spectral; the framebuffer is spectral (see §8.4).
- Line spectra are stored as line lists (centre, strength, profile parameters) *and* rasterised onto the grid. Rasterisation integrates the line profile over each bin rather than sampling the centre, so narrow lines conserve power.

---

## 4. Lamp physics models

Each lamp family has a ladder of levels. Level 0 is always "tabulated measured spectrum + angular distribution". Higher levels are listed below with the physics they add.

### 4.1 Incandescent / halogen

- **L1 — Planck.** Blackbody at filament temperature T. Parameter: T (or power, via a resistance–temperature relation for tungsten).
- **L2 — Tungsten greybody.** Multiply Planck by spectral, temperature-dependent emissivity ε(λ, T) (De Vos 1954 tables). Visible emissivity ~0.43–0.46, falling into the IR, which is why tungsten is "bluer" than a blackbody of the same T.
- **L3 — Coil geometry.** A coiled-coil filament has an effective emissivity higher than flat tungsten because of inter-turn cavity reflections. Model as ε_eff = ε / (ε + (1−ε)·f_open) with an openness factor f_open, or ray-trace the coil mesh directly once the renderer exists (the renderer can compute it: emitting tungsten coil with spectral metallic BRDF).
- **L4 — Envelope and fill.** Quartz (halogen) vs borosilicate transmission; gas-fill conduction losses affect the T(P) relation. Frosted/inside-white coating → diffuser component.
- Temperature from power: solve P_elec = P_rad(T) + P_cond(T) with P_rad = ε_tot(T)·σ·A·T⁴ and P_cond from a fill-gas model, or simply use the empirical tungsten lamp laws (T ∝ V^0.42 etc.) at low fidelity.

### 4.2 LED

- **L1 — Two-Gaussian.** Blue pump (InGaN, ~450 nm, FWHM ~18–25 nm; Gaussian in energy, not wavelength) + phosphor band (YAG:Ce, peak ~560 nm, FWHM ~110 nm, asymmetric; a split-Gaussian or sum of two Gaussians fits well). Parameters: pump peak, phosphor peak/width, blue-to-phosphor ratio.
- **L2 — Phosphor conversion.** Blue absorption by the phosphor layer via Beer–Lambert (thickness × concentration), Stokes-shift energy loss, quantum efficiency, re-emission isotropic. Blue leakage ratio emerges from the thickness rather than being a free parameter. Multi-phosphor (red nitride) for high-CRI variants.
- **L3 — Thermal.** Junction temperature shifts the pump peak (~0.03 nm/K) and reduces phosphor QE. Needs a thermal resistance parameter.
- **L4 — Volumetric phosphor.** Phosphor as a participating medium with wavelength-dependent absorption and re-emission in the renderer. Mostly useful for packaging/optics studies; low priority.

### 4.3 Low-pressure discharge (fluorescent, LPS)

- **L1 — Line list + phosphor.** Hg lines (365, 405, 436, 546, 578 nm) with fixed relative intensities plus phosphor bands: tri-phosphor (Eu²⁺ ~450, Tb³⁺ ~543, Eu³⁺ ~611 nm, narrow) or halophosphate (two broad bands). LPS: Na D doublet at 589.0/589.6 nm, essentially monochromatic.
- **L2 — UV budget.** 254 nm (and 185 nm) resonance radiation as the pump; phosphor conversion efficiency and Stokes loss give visible output; visible Hg line fraction tied to fill pressure and cold-spot temperature (vapour pressure curve of Hg). Gives a realistic efficacy without an arc model.

### 4.4 High-pressure discharge (HP Hg, metal halide, HPS) — the deep end

This is where the physics gets interesting and where scope discipline matters. Build it as three separable stages that each produce a usable result.

**Stage A — Line emission under LTE at a single temperature.**
For each line k→i from a line list (NIST ASD):

```
emission coefficient   ε_λ = (h c / 4π λ) · A_ki · n_k · φ(λ)
upper-level population n_k = n · (g_k / U(T)) · exp(−E_k / k_B T)
absorption coefficient κ_λ = (λ² / 8π) · A_ki · (g_k / g_i) · n_i · φ(λ) · (1 − exp(−h c / λ k_B T))
```

Line profile φ(λ): Voigt = Doppler (Gaussian, ∝ √(T/M)) ⊗ Lorentzian from pressure broadening. Pressure broadening is the whole story at high pressure: resonance broadening (same-species, dominant for Na D in HPS), van der Waals (foreign gas, Hg buffer), Stark (electrons; matters in Hg arcs). Each gives a width ∝ density with a known coefficient; parameters: species densities, T.

Also include continuum: electron–ion recombination and bremsstrahlung (∝ n_e² / √T, weak λ dependence) and, for Hg, the quasi-molecular continuum. Partition functions U(T) and ionisation via Saha close the system for given pressure and T.

Output: ε_λ(T, p) and κ_λ(T, p) on the spectral grid. Already produces recognisable HP-Hg spectra and the Na D broadening trend.

**Stage B — Radiative transfer through a prescribed temperature profile.**
Assume a cylindrical arc of radius R with a radial profile T(r) (start with a parabolic or Gaussian profile peaking at 4000–6000 K on axis, ~1200–1500 K at the wall for HPS). For each wavelength and each chord through the cylinder at impact parameter b, integrate

```
dI_λ/ds = ε_λ(T(s)) − κ_λ(T(s)) · I_λ
```

This yields the **self-reversed** profile of optically thick lines: the hot core emits a broad line, the cooler mantle absorbs the centre, leaving two wings and a central dip. For HPS at 10–30 kPa Na the D-line profile becomes tens of nm wide with the characteristic dip at 589 nm; the asymmetric far wings come from the quasi-static (collision-time) broadening and a satellite near 551 nm. This is the signature result of the project and is tractable: a few hundred lines of NumPy.

Output: I_λ(b) — apparent radiance as a function of position across the tube. This maps *directly* onto the `EmissionProfile` texture of the arc-tube cylinder: the renderer sees a tube whose radiance varies across its width and with wavelength, and never has to trace through the plasma. The decomposition "arc physics computes outgoing radiance at the tube surface; renderer takes it from there" is what keeps the tar pit contained.

**Stage C — Solve for the temperature profile (Elenbaas–Heller).**
Instead of prescribing T(r), solve the arc energy balance in the cylindrical, LTE, no-convection limit:

```
(1/r) d/dr ( r κ_th(T) dT/dr ) + σ_el(T) E² − U_rad(T) = 0
```

with κ_th thermal conductivity (including reactive conductivity from ionisation/dissociation), σ_el electrical conductivity from n_e(T) (Saha), U_rad net radiative loss (from Stage A/B, iterated), E the axial field set by the current I = 2π E ∫ σ_el r dr. Boundary: dT/dr = 0 at r = 0, T = T_wall at r = R. Solve with shooting or a boundary-value solver (SciPy `solve_bvp`), iterate with the radiation term. Outputs T(r), arc voltage, efficacy — real predictions, not fits.

For metal halide add the chemistry: dissociation equilibria of NaI, TlI, ScI₃, DyI₃ etc. versus temperature (Gibbs-energy minimisation), giving radially varying free-metal densities. This is the genuine tar pit; keep it as an optional plug-in that provides n_species(T, p) to Stage A, and ship HP-Hg and HPS first.

**What is deliberately ignored:** electrode regions, axial gradients, convection (arc bowing in horizontal lamps), AC modulation, non-LTE near the wall, and the radiative transfer *inside* the renderer (covered by Stage B). Each is documented as a possible extension.

### 4.5 Data the physics layer needs

- NIST Atomic Spectra Database line lists for Hg, Na, Sc, Tl, Dy, In, Xe (curated subsets shipped as CSV, with citation).
- Partition functions (Irwin polynomials or tabulated).
- Broadening coefficients: resonance (from oscillator strength), van der Waals (C₆ estimates), Stark (Griem tables for Hg).
- Transport coefficients for Hg/Na vapour: σ_el, κ_th vs T (tabulated from literature, e.g. Elenbaas, Waymouth, de Groot & van Vliet).
- Tungsten emissivity (De Vos), glass/quartz/PCA transmission (manufacturer curves).
- CIE CMFs, V(λ), illuminants, colour-rendering test samples (CRI 14 samples, TM-30 99 CES).
- Measured lamp SPDs (LSPDD, manufacturer data) for Level 0 and for validation.

---

## 5. Lamp as a component pipeline

```
Lamp
  emitter      : one of the generators in §4 → EmissionProfile
  envelope[]   : ordered list of SpectralTransform + optional geometry
  reflector    : optional geometry with spectral BRDF
  external[]   : filters/diffusers placed outside the lamp (also placeable in the scene)
```

`SpectralTransform`: transmission or reflectance T(λ) (optionally angle-dependent), scattering model (none / diffuse / measured haze), geometry (mesh or parametric). Examples:

- Arc tube: fused quartz (UV cut at ~170 nm, HP-Hg) or polycrystalline alumina (HPS; translucent → treated as a diffusing emitter surface with ~95 % total transmission; later as a scattering medium).
- Outer bulb: borosilicate, clear or diffuse-coated; UV absorption.
- Phosphor coating: a transform that absorbs UV/blue and re-emits — the only transform that moves energy between wavelengths, so it is a special `ConvertingTransform` with an excitation→emission matrix.
- Reflector: aluminium or dichroic (wavelength- and angle-dependent reflectance, which is why halogen "cool beam" reflectors leak IR backwards).
- Filters: gel/dichroic/neutral density/Wratten, as transmission curves.

Swapping a component rebuilds the pipeline; only the stages downstream of the change recompute. The final `EmissionProfile` is what the scene uses.

---

## 6. Fidelity toggling

Every lamp carries `level ∈ {0, 1, 2, …}` per family. A `ProfileCache` maps `(lamp_family, level, parameter_hash)` → `EmissionProfile` on disk. UI rule: changing level shows the cached profile instantly if present, otherwise launches the generator as a background job (same job system as rendering, §8) and shows the next-lower cached level with a "computing level N…" badge until done. The user can also pin a level for comparison: render the scene under level 1 and level 3 side by side to see what the physics changed.

---

## 7. Renderer

**Engine:** Mitsuba 3, spectral variants (`scalar_spectral` for debugging, `llvm_ad_spectral` for multi-core CPU, `cuda_ad_spectral` if an NVIDIA GPU exists). Reasons: spectral by construction (hero-wavelength sampling, 4 wavelengths per path), Python API, mesh emitters with spectral textures, `irradiancemeter` and `radiancemeter` sensors for probes, participating media available later, differentiable if parameter fitting to measured spectra ever becomes interesting.

**Mapping lamps into the scene:** the `EmissionProfile` geometry becomes mesh emitters whose radiance is a spectral texture (Mitsuba `bitmap` spectral textures, or a custom emitter plugin written in Python/Dr.Jit for the angular model). Envelope and reflector geometry become ordinary shapes with spectral BSDFs. Arc tubes use the Stage-B radiance texture; filaments use coil meshes with emissive tungsten.

**Scene:** a "lighting lab" reference scene — Macbeth ColorChecker (measured spectral reflectances), grey and white cards, matte and chrome spheres, a long wall for falloff, a pair of metameric samples that match under daylight and split under HPS, and a calibrated camera. Scene objects are movable; a filter is a scene object with a transmissive spectral BSDF, so it can be placed anywhere between lamp and target.

**Materials:** spectral reflectance curves on the §3 grid; diffuse, rough-dielectric, rough-conductor (spectral complex IOR for metals).

---

## 8. Interactive, progressive rendering with honest progress

### 8.1 Principle

Path tracing is embarrassingly progressive: the image is the running mean of independent passes. "Coarse first, refine in the background" is therefore not two renderers but one accumulation with changing resolution and a well-defined notion of completion.

### 8.2 Quality tiers

| Tier | Resolution | spp per pass | Max depth | Purpose |
|------|-----------|--------------|-----------|---------|
| Preview | 1/4 (e.g. 320×240) | 1 | 3 | Appears < 100 ms after any change, updated continuously while dragging |
| Draft | full | 4 | 6 | First full-resolution frame, ~1 s |
| Refine | full | 8 per pass | 12+ | Repeated until converged or budget hit |

Preview upscales to the viewport with nearest-neighbour and is drawn with a visible "PREVIEW" badge and a slight desaturated border so it can never be mistaken for final output.

### 8.3 Job system

- The UI thread never renders. A **render worker process** (not a thread: it gives hard cancellation, isolates crashes, and avoids GIL contention with the UI even though Mitsuba releases the GIL) owns the Mitsuba scene.
- Scene state in the UI is a plain, serialisable description (JSON-like dict). Every edit increments a `generation` counter and sends the *diff* to the worker. The worker applies the diff to its Mitsuba scene via `mi.traverse` parameter updates (cheap) or rebuilds the affected object (rare).
- The worker loop:
  1. Read latest generation. If changed since the current accumulation started: discard the accumulator, render a Preview pass, publish it, then a Draft pass, then start Refine passes.
  2. After each pass: check for a newer generation (abort early), publish the frame and a progress message.
  3. Between passes, if a convergence criterion is met, publish `Converged` and idle.
- Edits are debounced at ~30 ms while a slider is being dragged; the preview tier keeps up with dragging because each preview pass is tens of milliseconds.
- Framebuffer is a spectral accumulator (W × H × N_λ, see §8.4) in **shared memory** (`multiprocessing.shared_memory` + NumPy). The UI reads it on demand, converts to display RGB with the current exposure/white point without any round trip to the worker. Exposure, tone mapping, colour space, and false-colour mode are therefore free and instant.

### 8.4 Spectral framebuffer and the re-lighting cache

Because the framebuffer is spectral, the display is a pure post-process. More importantly, light transport is **linear in emission**. For each lamp ℓ, render a *transport image* `T_ℓ(x, y, λ)` with that lamp emitting a flat unit spectrum (the spatial/angular profile unchanged). Then for any new spectrum S_ℓ(λ):

```
Image(x, y, λ) = Σ_ℓ T_ℓ(x, y, λ) · S_ℓ(λ)
```

This makes the following edits **instant, with no re-render**: changing a lamp's spectrum (model level, filament temperature, LED phosphor mix, arc pressure), adding a filter *at the lamp*, dimming, and swapping lamps that share the same emitter geometry. Since this is the physics-exploration loop, it is the single most valuable optimisation in the project.

Limits, stated in the UI as a "cache valid / cache invalidated" indicator:
- Any geometric change (move lamp, move object, move camera, place a filter in the scene) invalidates the transport image; the job system then re-renders with the tiered scheme.
- Wavelength-dependent transport (dispersion, fluorescence, strongly wavelength-dependent IOR) is captured exactly only if the transport image itself is rendered spectrally — which it is, since `T_ℓ` has a λ axis. Only *emission-dependent* effects (none in a linear renderer) would break it.
- Changing the *spatial* emission texture (e.g. the Stage-B radial profile changes with pressure) is a partial case: factor the profile as `shape(x) · S(λ)` where possible; when the shape changes, invalidate.
- Memory: 1024 × 768 × 81 bins (5 nm over 380–780) × float32 ≈ 255 MB per lamp. Use 41 bins (10 nm) for the cache if memory is tight; allow the 141-bin grid for the final render only.

### 8.5 Progress the user can trust

Progress messages, sent after every pass and on a 500 ms heartbeat:

```
Progress {
  generation, tier, pass_index, spp_done, spp_target,
  elapsed_s, eta_s,            # ETA from an exponential moving average of seconds/pass
  noise_estimate,              # mean relative standard error of pixel luminance
  denoised: bool,
  worker_alive: heartbeat timestamp
}
```

UI representation:
- A state label with exactly these states: **Previewing · Refining (pass k / N) · Converged · Stale – change pending · Paused · Worker unresponsive · Error**.
- A determinate bar for spp_done/spp_target, plus elapsed and ETA text. When running "until converged" with no fixed target, show the noise estimate against the threshold instead (bar = threshold / noise, capped at 1), so the bar still moves.
- A per-pass tick on a small sparkline of seconds-per-pass: if ticks stop arriving the user *sees* it stall before the watchdog does.
- Watchdog: three missed heartbeats (1.5 s) → "Worker unresponsive" with a Restart button that kills and respawns the worker and replays the current scene state. A crash in the worker therefore never takes the UI down and never leaves the user guessing.
- Convergence: per-pixel variance is tracked via Welford's algorithm in the accumulator (mean and M2 per pixel per wavelength, or per-pixel luminance only to save memory). "Converged" means mean relative standard error below a user-set threshold (default 1 %) — a definition, not a feeling.
- Optional denoiser (Intel Open Image Denoise, with Mitsuba albedo/normal AOVs) applied to Draft and early Refine frames. Denoised frames carry a "denoised" badge; the final converged frame is shown undenoised by default so the user can judge the real noise level.

### 8.6 Cheap vs expensive edits (what the user feels)

| Edit | Cost |
|------|------|
| Exposure, tone map, false colour, white point | Instant (display post-process) |
| Lamp spectrum, model level, lamp-side filter, dimming | Instant (re-lighting cache) |
| Spectral probe at a pixel | Instant (read from spectral framebuffer) |
| Move/rotate lamp, move object, in-scene filter, camera | Preview < 100 ms, draft ~1 s, converged seconds–minutes |
| Change arc parameters needing a new Stage B/C solve | Background physics job (seconds), then instant re-light |
| Change emitter geometry (filament → arc tube) | Scene rebuild + full re-render |

---

## 9. Measurement and comparison

- **Pixel probe:** click → spectrum of radiance arriving at that pixel (from the framebuffer), with luminance, chromaticity, CCT.
- **Surface probe:** place an `irradiancemeter` at a point → irradiance spectrum, illuminance, CCT, CRI Ra, TM-30 Rf/Rg, S/P ratio, melanopic EDI, UV-A/UV-B dose. Several probes can be pinned.
- **Lamp sheet:** for each lamp, its SPD, radiant and luminous flux, efficacy, CCT/Duv, CRI, TM-30 colour-vector graphic, and the model-level parameters that produced it.
- **Compare:** A/B slider or split view over the same scene and camera; because both renders live in spectral framebuffers, also show a difference image and the difference of probe spectra. Compare across lamps *or* across model levels of one lamp.
- **False colour:** illuminance map, UV-dose map, "where would this exceed X lux" isolines.

---

## 10. Validation (tests Claude Code can run)

Physics core (pure NumPy, fast, run on every change):
- Planck: ∫B(λ,T)dλ = σT⁴/π within 0.1 % on the extended grid; peak at Wien's wavelength.
- Colour: CIE Illuminant A → CCT 2856 K, CRI 100; D65 → (0.3127, 0.3290); E → (1/3, 1/3).
- Tungsten greybody at 2856 K has CCT slightly above the blackbody value (emissivity falls with λ).
- Line rasterisation conserves power regardless of bin width.
- Voigt profile reduces to Gaussian/Lorentzian in the limits; FWHM matches the Olivero–Longbothum approximation.
- Stage B: an optically thin slab gives I = ε·L; an optically thick slab saturates to the Planck function (Kirchhoff) — checks ε/κ consistency.
- Stage B: Na D at HPS conditions shows a central minimum between two maxima; dip depth increases with Na pressure.
- Stage C: HP-Hg arc at known current/pressure reproduces published axis temperature (~5500–6000 K) and field strength within ~15 %; efficacy of a textbook 400 W HP-Hg lamp lands in 50–60 lm/W; HPS 400 W in 100–130 lm/W.

Renderer (slower, run on demand):
- Furnace test: closed diffuse box with albedo 0.5 and uniform emission converges to the analytic radiance.
- Inverse-square: illuminance at probes falls as 1/d² for a point-like emitter.
- Re-lighting cache equals a direct re-render within noise for a spectrum change.
- Colour checker under D65 renders to its published sRGB values within ΔE ≈ 2.

---

## 11. Stack and project layout

- Python 3.12, NumPy, SciPy; Mitsuba 3 (spectral variants); PySide6 for the desktop UI; pyqtgraph for spectra and sparklines (fast, Qt-native); `multiprocessing` + shared memory for the worker; optional `oidn` for denoising; `pytest`.
- Alternative considered: Rust + custom path tracer. Rejected for now — the physics is the point, and Mitsuba removes a year of renderer work. Revisit only if real-time GPU preview becomes a hard requirement.

```
lss/
  spectral/      grid, resampling, colour science (cmf, cct, cri, tm30), units
  physics/
    planck.py, tungsten.py, led.py, lowpressure.py
    hid/  lines.py (NIST import), broadening.py, lte.py (Saha, partition fns),
          transfer.py (Stage B), arc.py (Stage C), chemistry.py (optional)
  lamp/          components, pipeline, EmissionProfile, ProfileCache
  scene/         scene description, reference lab, materials, probes
  render/        mitsuba adapter, worker process, accumulator, relight cache, progress
  ui/            viewport, lamp editor, compare view, probe panels
  data/          CIE tables, line lists, emissivities, glass curves, SPDs, ColorChecker
  tests/
  cli.py         render / probe / lampsheet commands (headless)
```

CLI first: `lss lampsheet hps_400w.yaml`, `lss render --lamp … --scene lab --spp 256 --out img.exr`, `lss probe … --at x,y,z`. The GUI wraps the same library.

---

## 12. Roadmap

1. **Spectral core + colour science + Level-0 lamps** (tabulated SPDs, IES angular). Tests in §10 pass. CLI lamp sheet works.
2. **Incandescent L1–L2, LED L1–L2, fluorescent L1.** Lamp pipeline with envelopes and filters.
3. **Mitsuba adapter + reference lab + CLI render + probes.** Furnace and inverse-square tests pass.
4. **Worker process, tiered progressive rendering, progress protocol, watchdog.** Minimal PySide6 viewport with a lamp position gizmo. This milestone is about feel: dragging must stay fluid.
5. **Spectral framebuffer + re-lighting cache.** Spectrum edits become instant.
6. **HID Stage A** (LTE line emission, broadening) and **Stage B** (radiative transfer, self-reversal). Arc tube as a radiance texture. HP-Hg and HPS lamp presets.
7. **Compare view, lamp sheet UI, TM-30 graphics, false colour.**
8. **HID Stage C** (Elenbaas–Heller) — time-boxed; cylindrical HP-Hg and HPS only.
9. Later: metal-halide chemistry, translucent alumina as a scattering medium, volumetric phosphor, AC flicker, measured-spectrum parameter fitting via Mitsuba's differentiability.

---

## 13. Open questions

- Memory budget for the re-lighting cache on the target machine (decides 5 nm vs 10 nm cache bins and whether multi-lamp caches are kept).
- Extended-grid range: is 300–1000 nm enough, or should IR go to 2500 nm for halogen heat-load studies (cost: more bins everywhere)?
- Whether the angular model should be baked into the emitter texture (simple) or live as a custom Mitsuba emitter plugin (flexible, needed for measured IES on an extended source).
- Source for Hg/Na transport coefficients (σ_el, κ_th): hand-digitise from literature, or compute from cross-sections (another rabbit hole).
