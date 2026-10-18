# lightsources

Physics-first simulator of lamps as spectral emitters. See
[`light_source_simulator_design.md`](light_source_simulator_design.md) for the design.

## Setup

```
uv venv --python 3.14 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
```

The renderer (Mitsuba 3) is the optional `render` extra (`-e ".[dev,render]"`). Its GPU
variant needs only an NVIDIA driver. Its CPU variant needs an installed LLVM, named by
`DRJIT_LIBLLVM_PATH` (e.g. `C:\Program Files\LLVM\bin\LLVM-C.dll`), and, with LLVM 22
and Dr.Jit 1.5.0, at most 8 worker threads (`drjit.set_thread_count(8)`): at 16 it
crashes at random on large kernels (design doc §8.3).

## Use

```
lss refs                         # list the reference spectra
lss lampsheet incandescent_cie_a # lamp sheet for a preset (or ref:<id>)
lss accuracy                     # score models against references
lss fit incandescent_cie_a --level 2   # fit a level's bounded parameters to its reference
lss fit halogen_nist_fel_f455 --level 3 # coiled tungsten against a NIST-measured halogen lamp
lss fit halogen_nist_fel_f455 --level 4 --objective uncertainty
                                 # power balance (the current sets the temperature):
                                 # fits the cold resistance
lss fit fluorescent_cie_fl10 fluorescent_cie_fl11 fluorescent_cie_fl12 --level 1
                                 # joint fit of the shared phosphor shapes
lss fit fluorescent_cie_fl10 fluorescent_cie_fl11 fluorescent_cie_fl12 --level 1 \
    --hold-out fluorescent_cie_fl12  # out-of-sample score for one lamp
lss fit led_cie_b1 led_cie_b2 led_cie_b3 led_cie_b4 led_cie_b5 --level 2
                                 # LED phosphor layer: shared garnet and nitride constants
lss chart ref:cie-hp1 --svg hp1.svg   # ColorChecker under a lamp vs TM-30's reference light
lss chart led_cie_b3 --samples ces    # the 99 TM-30 samples (their mean shift gives Rf)
lss metamers                          # a CONSTRUCTED pair matched in daylight, split under lamps
lss leddrive                          # a measured LED at 150-500 mA (downloaded on first use; needs
                                      # pip install "lss[data]"), for LED Level 3
lss streethps                         # nine measured HPS street lamps (LICA-UCM, downloaded on first
                                      # use) against the sodium arc, through the spectrograph's response,
                                      # with and without sodium- and sodium-mercury-pair far wings
lss streethpsfit                      # each of them with its own arc temperature and sodium pressure,
                                      # fitted to its blue peak and three lines (six tables built on
                                      # first use, about half an hour on 16 cores); the red wing held out
```

## Tests

On Windows, from a POSIX shell, at below-normal priority:

```
cmd //v:on //c "start /belownormal /b /wait .venv\Scripts\python -m pytest & exit !errorlevel!"
```
