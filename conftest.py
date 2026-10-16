import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# tests/ too, so shared fixtures (tests/tiny.py) import by plain name from any
# test module regardless of which directory pytest was invoked from.
sys.path.insert(0, str(Path(__file__).parent / "tests"))

import os
import tempfile

# Keep the suite hermetic: every cache (pipeline, labels, remote) resolves
# under a throwaway directory instead of the user's real cache home. A test
# that wants a specific location passes cache_dir= or sets the variable itself.
if "MOTTLED_CACHE_DIR" not in os.environ:
    os.environ["MOTTLED_CACHE_DIR"] = tempfile.mkdtemp(prefix="mottled-test-cache-")
