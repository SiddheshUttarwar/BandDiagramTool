"""
pytest configuration: put the repo root on sys.path.

There's no top-level package/setup.py for this project (see examples/*.py,
which all do the same sys.path.insert), so this conftest makes `from
physics...` / `from devices...` imports resolve when pytest is run from the
repo root.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
