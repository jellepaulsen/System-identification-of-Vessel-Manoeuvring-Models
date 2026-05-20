#!/usr/bin/env bash
#
# setup.sh — Fully automated setup for the wPCC vessel manoeuvring pipeline
#
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# Prerequisites:
#   - uv (https://docs.astral.sh/uv/)
#   - git
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

VMM_PKG=".venv/lib/python3.10/site-packages/vessel_manoeuvring_models"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Check prerequisites
# ─────────────────────────────────────────────────────────────────────────────
command -v uv  >/dev/null 2>&1 || error "uv is not installed. Install from https://docs.astral.sh/uv/"
command -v git >/dev/null 2>&1 || error "git is not installed."

info "Setting up in: $REPO_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Create Python 3.10 virtual environment
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    info "Creating Python 3.10 virtual environment..."
    uv venv --python 3.10 .venv
else
    info "Virtual environment already exists, skipping creation."
fi

# Activate
# shellcheck disable=SC1091
source .venv/bin/activate
info "Using Python: $(python3 --version) at $(which python3)"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Install dependencies (with PyYAML override for modern compilers)
# ─────────────────────────────────────────────────────────────────────────────
info "Installing Python dependencies..."
uv pip install --override <(echo 'pyyaml>=6.0') -r src/requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# 4. Upgrade numpy/scipy/statsmodels/scikit-learn for Apple Silicon
#    The pinned numpy 1.21.5 links against OpenBLAS at /opt/arm64-builds/lib
#    which doesn't exist on most machines — causes segfaults in OLS regression.
#    numpy 1.26.4 bundles its own BLAS and is the last 1.x release.
# ─────────────────────────────────────────────────────────────────────────────
info "Upgrading numpy stack for Apple Silicon compatibility..."
uv pip install 'numpy==1.26.4' 'scipy==1.12.0' 'statsmodels==0.13.5' 'scikit-learn==1.3.2' 'pandas==1.3.5' 'plotly==5.5.0' 'patsy==0.5.6'

# ─────────────────────────────────────────────────────────────────────────────
# 5. Patch Kedro metadata to accept PyYAML 6+
# ─────────────────────────────────────────────────────────────────────────────
KEDRO_META=".venv/lib/python3.10/site-packages/kedro-0.17.6.dist-info/METADATA"
if [ -f "$KEDRO_META" ]; then
    if grep -q 'PyYAML (<6.0' "$KEDRO_META"; then
        info "Patching Kedro metadata for PyYAML 6+ compatibility..."
        sed -i 's/Requires-Dist: PyYAML (<6.0,>=4.2)/Requires-Dist: PyYAML (>=4.2)/g' "$KEDRO_META"
    else
        info "Kedro PyYAML constraint already patched."
    fi
else
    warn "Kedro metadata not found at expected path — skipping PyYAML patch."
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. Install the project package (editable, no deps — they're already installed)
# ─────────────────────────────────────────────────────────────────────────────
info "Installing wPCC_pipeline package..."
(cd src && pip install -e . --no-deps)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Disable telemetry (prevents interactive consent prompt)
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -f ".telemetry" ]; then
    info "Disabling Kedro telemetry..."
    echo "consent: false" > .telemetry
else
    info "Telemetry file already exists."
fi

# ─────────────────────────────────────────────────────────────────────────────
# 8. Restore missing parameter files from git history
# ─────────────────────────────────────────────────────────────────────────────
PARAM_DIR="conf/base/parameters"
RESTORE_COMMIT="8d888e1"

restore_param() {
    local file="$1"
    if [ ! -f "$PARAM_DIR/$file" ]; then
        info "Restoring $file from git history..."
        git show "${RESTORE_COMMIT}^:${PARAM_DIR}/${file}" > "$PARAM_DIR/$file"
    else
        info "$file already exists."
    fi
}

mkdir -p "$PARAM_DIR"
restore_param "lowpass.yml"
restore_param "motion_regression.yml"
restore_param "prediction.yml"
restore_param "force_regression.yml"
restore_param "join_runs.yml"

# ─────────────────────────────────────────────────────────────────────────────
# 9. Prepare wPCC test data
# ─────────────────────────────────────────────────────────────────────────────
WPCC_RAW="data/01_raw/wpcc"
MENDELEY_DIR="wPCC manoeuvring model tests"

# Copy Mendeley data if the CSV files aren't in the raw data dir yet
if [ -d "$MENDELEY_DIR/model tests" ]; then
    CSV_COUNT=$(find "$WPCC_RAW" -maxdepth 1 -name "*.csv" -not -name "runs_meta_data.csv" -not -name "nomenclature.csv" -not -name "open_water_characteristics.csv" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$CSV_COUNT" -lt 7 ]; then
        info "Copying wPCC test data from Mendeley download..."
        cp "$MENDELEY_DIR/model tests/"*.csv "$WPCC_RAW/"
    else
        info "wPCC CSV test files already present ($CSV_COUNT files)."
    fi

    # Copy metadata files if missing
    for meta_file in nomenclature.csv runs_meta_data.csv ship_data.yml; do
        if [ ! -f "$WPCC_RAW/$meta_file" ] && [ -f "$MENDELEY_DIR/$meta_file" ]; then
            info "Copying $meta_file..."
            cp "$MENDELEY_DIR/$meta_file" "$WPCC_RAW/"
        fi
    done
else
    warn "Mendeley data directory not found: '$MENDELEY_DIR'"
    warn "Download from: https://data.mendeley.com/datasets/j5zdrhr9bf"
    warn "Place it in the project root as '$MENDELEY_DIR/'"
fi

# Add TWIN field to ship_data.yml if missing
SHIP_DATA="$WPCC_RAW/ship_data.yml"
if [ -f "$SHIP_DATA" ]; then
    if ! grep -q "TWIN" "$SHIP_DATA"; then
        info "Adding TWIN field to ship_data.yml..."
        echo "TWIN: 1" >> "$SHIP_DATA"
    else
        info "TWIN field already present in ship_data.yml."
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 10. Patch vessel_manoeuvring_models
#     The library was written for numpy 1.21 / sklearn 1.0 and needs several
#     fixes to work with the upgraded stack.
# ─────────────────────────────────────────────────────────────────────────────

# 10a. Prime system — add missing unit definitions for accelerometer fields
PRIME_SYSTEM="$VMM_PKG/prime_system.py"
if [ -f "$PRIME_SYSTEM" ]; then
    if ! grep -q "accelerometer1_x" "$PRIME_SYSTEM"; then
        info "Patching prime_system.py (accelerometer fields)..."
        sed -i 's/r"Arr\/Ind\/Fri": "-",/r"Arr\/Ind\/Fri": "-",\
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
    else
        info "prime_system.py already patched."
    fi
fi

# 10b. Parameters — get_feature_names was removed in scikit-learn 1.2
PARAMETERS="$VMM_PKG/parameters.py"
if [ -f "$PARAMETERS" ]; then
    if grep -q "get_feature_names(" "$PARAMETERS"; then
        info "Patching parameters.py (sklearn get_feature_names)..."
        sed -i 's/get_feature_names(df_\.columns)/get_feature_names_out(df_.columns)/' "$PARAMETERS"
    else
        info "parameters.py already patched."
    fi
fi

# 10c. substitute_dynamic_symbols — numpy 1.24+ rejects ragged arrays
#      The sympy-lambdified functions create np.array() with inhomogeneous
#      sub-arrays (e.g. coefficient vectors of different DOF lengths).
SUBST="$VMM_PKG/substitute_dynamic_symbols.py"
if [ -f "$SUBST" ]; then
    if ! grep -q "inhomogeneous" "$SUBST"; then
        info "Patching substitute_dynamic_symbols.py (numpy ragged array compat)..."
        python3 << 'PYEOF'
import re

with open(".venv/lib/python3.10/site-packages/vessel_manoeuvring_models/substitute_dynamic_symbols.py") as f:
    content = f.read()

old = '''    return function(*args)'''
# Only replace the one inside the run() function (after the args= line)
new = '''    try:
        return function(*args)
    except ValueError as e:
        if "inhomogeneous shape" in str(e) or "setting an array element with a sequence" in str(e):
            # numpy>=1.24 rejects ragged arrays. The lambdified sympy expressions
            # call array() internally with inhomogeneous sub-arrays. Temporarily
            # patch array() in the function's globals to allow dtype=object fallback.
            _orig_array = np.array
            def _lenient_array(*a, **kw):
                try:
                    return _orig_array(*a, **kw)
                except ValueError:
                    kw['dtype'] = object
                    return _orig_array(*a, **kw)
            np.array = _lenient_array
            if hasattr(function, '__globals__') and 'array' in function.__globals__:
                function.__globals__['array'] = _lenient_array
            try:
                return function(*args)
            finally:
                np.array = _orig_array
                if hasattr(function, '__globals__') and 'array' in function.__globals__:
                    function.__globals__['array'] = _orig_array
        raise'''

# Replace only the return inside run() — find the function and replace its return
content = content.replace(
    '    args = [kwargs[parameter] for parameter in parameters]\n    return function(*args)',
    '    args = [kwargs[parameter] for parameter in parameters]\n' + new
)

with open(".venv/lib/python3.10/site-packages/vessel_manoeuvring_models/substitute_dynamic_symbols.py", "w") as f:
    f.write(content)
PYEOF
    else
        info "substitute_dynamic_symbols.py already patched."
    fi
fi

# 10d. regression.py — expand covariance matrix for connected Y-N rudder params
REGRESSION="$VMM_PKG/models/regression.py"
if [ -f "$REGRESSION" ]; then
    if ! grep -q "B_cov_expanded" "$REGRESSION"; then
        info "Patching regression.py (covariance matrix for connected params)..."
        python3 << 'PYEOF'
with open(".venv/lib/python3.10/site-packages/vessel_manoeuvring_models/models/regression.py") as f:
    content = f.read()

old_cov = """        covs = run(
            A_lambda,
            A_coeff=self.model_X.cov_HC0,
            B_coeff=self.model_Y.cov_HC0,
            C_coeff=self.model_N.cov_HC0,
            **self.parameters,
            **self.added_masses,
            **ship_parameters_prime,
        )
        columns = (
            list(self.model_X.params.keys())
            + list(self.model_Y.params.keys())
            + list(self.model_N.params.keys())
        )"""

new_cov = """        # Expand model_Y covariance to include connected (zero-variance) parameters
        B_cov = self.model_Y.cov_HC0
        if len(self.connected_parameters_Y) > 0:
            n_y = len(self.model_Y.params)
            n_connected = len(self.connected_parameters_Y)
            n_total = n_y + n_connected
            B_cov_expanded = np.zeros((n_total, n_total))
            B_cov_expanded[:n_y, :n_y] = B_cov
            B_cov = B_cov_expanded

        covs = run(
            A_lambda,
            A_coeff=self.model_X.cov_HC0,
            B_coeff=B_cov,
            C_coeff=self.model_N.cov_HC0,
            **self.parameters,
            **self.added_masses,
            **ship_parameters_prime,
        )
        columns = (
            list(self.model_X.params.keys())
            + list(self.model_Y.params.keys())
            + list(self.connected_parameters_Y.keys())
            + list(self.model_N.params.keys())
        )"""

content = content.replace(old_cov, new_cov)

with open(".venv/lib/python3.10/site-packages/vessel_manoeuvring_models/models/regression.py", "w") as f:
    f.write(content)
PYEOF
    else
        info "regression.py already patched."
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 11. Clear cached .pyc files for patched modules
# ─────────────────────────────────────────────────────────────────────────────
info "Clearing cached bytecode for patched modules..."
find "$VMM_PKG" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# 12. Ensure data directories exist
# ─────────────────────────────────────────────────────────────────────────────
for d in 02_intermediate 03_primary 04_feature 05_model_input 06_models 07_model_output 08_reporting 09_tracking; do
    mkdir -p "data/$d"
done

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
echo ""
info "============================================"
info "  Setup complete!"
info "============================================"
echo ""
echo "  Activate the environment:"
echo "    source .venv/bin/activate"
echo ""
echo "  Run the wPCC pipeline:"
echo "    PYTHONPATH=src:\$PYTHONPATH python3 -m wPCC_pipeline --pipeline wpcc"
echo ""
echo "  Launch Jupyter:"
echo "    PYTHONPATH=src:\$PYTHONPATH jupyter lab"
echo ""
echo "  Available pipelines:"
echo "    wpcc             — Full wPCC analysis"
echo "    plot_wpcc        — Plot wPCC results"
echo "    kvlcc2_hsva      — KVLCC2 HSVA analysis (needs private data)"
echo "    lowpass_study    — Low-pass filter investigation"
echo "    (no --pipeline)  — Run all pipelines"
echo ""
