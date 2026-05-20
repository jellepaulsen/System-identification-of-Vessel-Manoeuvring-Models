"""Project settings."""
from wPCC_pipeline.hooks import ProjectHooks
from kedro.config import TemplatedConfigLoader  # NEU
from kedro_viz.integrations.kedro.sqlite_store import SQLiteStore
from pathlib import Path

# Instantiate and list your project hooks here
HOOKS = (ProjectHooks(),)

# NEU: Config Loader für Jinja2-Templates
CONFIG_LOADER_CLASS = TemplatedConfigLoader
CONFIG_LOADER_ARGS = {
    "globals_pattern": "*globals.yml",
}

# Tracking:
SESSION_STORE_CLASS = SQLiteStore
SESSION_STORE_ARGS = {"path": str(Path(__file__).parents[2] / "data")}