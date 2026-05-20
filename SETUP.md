# Setup Guide

Steps to reproduce the analysis on macOS (Apple Silicon) as of March 2026.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) package manager
- Git

## 1. Create a Python 3.10 virtual environment

The project was built for Python 3.7 with Kedro 0.17.6. Python 3.10 is the newest version that works reliably with these dependencies.

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
```

## 2. Install dependencies

The original `src/requirements.txt` has two issues:
- References a private PyPI index (`gitlab.sspa.local`) that is not accessible
- Pins PyYAML 5.4.1 which fails to build with modern compilers

Install with a PyYAML override:

```bash
uv pip install --override <(echo 'pyyaml>=6.0') -r src/requirements.txt
```

> **Note:** Ignore warnings about `pywin32` and `pywinpty` — those are Windows-only packages.

## 3. Patch Kedro to accept PyYAML 6+

Kedro 0.17.6 declares `PyYAML<6.0` in its metadata but works fine with PyYAML 6. Patch the constraint:

```bash
uv pip install --override <(echo 'pyyaml>=6.0') -r src/requirements.txt
sed -i '' 's/Requires-Dist: PyYAML (<6.0,>=4.2)/Requires-Dist: PyYAML (>=4.2)/g' "$KEDRO_META"
```

## 4. Install the project package

```bash
cd src && pip install -e . --no-deps && cd ..
```

## 5. Disable telemetry

Without this, kedro will hang waiting for interactive consent:

```bash
echo "consent: false" > .telemetry
```

## 6. Restore missing parameter files

Several parameter files were accidentally removed from the repository (git commit `8d888e1`). Restore them:

```bash
git show 8d888e1^:conf/base/parameters/lowpass.yml > conf/base/parameters/lowpass.yml
git show 8d888e1^:conf/base/parameters/motion_regression.yml > conf/base/parameters/motion_regression.yml
git show 8d888e1^:conf/base/parameters/prediction.yml > conf/base/parameters/prediction.yml
git show 8d888e1^:conf/base/parameters/force_regression.yml > conf/base/parameters/force_regression.yml
git show 8d888e1^:conf/base/parameters/join_runs.yml > conf/base/parameters/join_runs.yml
```

## 7. Download and prepare the wPCC data

Download the wPCC dataset from [Mendeley Data](https://data.mendeley.com/datasets/j5zdrhr9bf).

Copy the files into the raw data folder:

```bash
# Copy CSV test files into the wpcc data directory
cp "wPCC manoeuvring model tests/model tests/"*.csv data/01_raw/wpcc/
```

### Add missing ship metadata

The `ship_data.yml` in the Mendeley dataset is missing the `TWIN` field (twin propeller flag) required by the motion regression pipeline. Add it:

```bash
echo "TWIN: 1" >> data/01_raw/wpcc/ship_data.yml
```

### Adjust runs config to match available data

The Mendeley dataset only contains a subset of the test runs referenced in the config. Edit `conf/base/runs_globals.yml` and comment out any wPCC test IDs for which you don't have CSV files in `data/01_raw/wpcc/`. The available IDs in the Mendeley download are:

```
22611, 22635, 22639, 22765, 22770, 22772, 22773
```

## 8. Patch vessel_manoeuvring_models prime system

The `ship_data.yml` contains accelerometer position fields and other keys that don't have unit definitions in `vessel_manoeuvring_models`. Add them:

```bash
PRIME_SYSTEM=".venv/lib/python3.10/site-packages/vessel_manoeuvring_models/prime_system.py"
sed -i '' 's/r"Arr\/Ind\/Fri": "-",/r"Arr\/Ind\/Fri": "-",\
    "accelerometer1_x": "length",\
    "accelerometer1_y": "length",\
    "accelerometer1_z": "length",\
    "accelerometer2_x": "length",\
    "accelerometer2_y": "length",\
    "accelerometer2_z": "length",\
    "accelerometer3_x": "length",\
    "accelerometer3_y": "length",\
    "accelerometer3_z": "length",\
    "n_prop": "-",\
    "b_R": "length",\
    "c_r": "length",\
    "c_t": "length",\
    "y_R_port": "length",\
    "y_R_stbd": "length",\
    "y_p_port": "length",\
    "y_p_stbd": "length",\
    "z_p": "length",/' "$PRIME_SYSTEM"
```

## 9. Run the pipeline

The `kedro` CLI command doesn't load project commands properly due to plugin conflicts. Use `python -m` instead:

```bash
source .venv/bin/activate
PYTHONPATH=src:$PYTHONPATH python3 -m wPCC_pipeline --pipeline wpcc
```

Available pipelines:

| Pipeline | Description |
|----------|-------------|
| `wpcc` | Full wPCC analysis |
| `kvlcc2_hsva` | KVLCC2 HSVA analysis |
| `kvlcc2_hsva_create` | Preprocess KVLCC2 HSVA data |
| `plot_wpcc` | Plot wPCC simulation results |
| `plot_kvlcc2_hsva` | Plot KVLCC2 HSVA results |
| `lowpass_study` | Low-pass filter investigation |
| (no flag) | Run all pipelines (`__default__`) |

## Known issues

- **Deprecation warnings**: Many `DeprecationWarning` messages from `statsmodels`, `vessel_manoeuvring_models`, and `numpy` — safe to ignore.
- **Incomplete dataset**: The Mendeley download has 7 of the ~15 test runs used in the paper. Results will be based on the available subset.
- **`kedro` CLI**: The `kedro run` command doesn't work because `kedro-viz` plugin conflicts prevent project command loading. Always use `python -m wPCC_pipeline` instead.