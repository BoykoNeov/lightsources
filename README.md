# lightsources

Physics-first simulator of lamps as spectral emitters. See
[`light_source_simulator_design.md`](light_source_simulator_design.md) for the design.

## Setup

```
uv venv --python 3.14 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
```

## Use

```
lss refs                         # list the reference spectra
lss lampsheet incandescent_cie_a # lamp sheet for a preset (or ref:<id>)
lss accuracy                     # score models against references
lss fit incandescent_cie_a --level 2   # fit a level's bounded parameters to its reference
```

## Tests

On Windows, from a POSIX shell, at below-normal priority:

```
cmd //v:on //c "start /belownormal /b /wait .venv\Scripts\python -m pytest & exit !errorlevel!"
```
