import logging

logging.basicConfig(level=logging.WARNING)

try:
    from kedro.extras.extensions.ipython import load_ipython_extension
    load_ipython_extension(get_ipython())
except Exception:
    pass
