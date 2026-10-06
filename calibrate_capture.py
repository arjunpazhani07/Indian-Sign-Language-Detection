"""Step 1 - record your own ISL samples for the 5-word calibration model."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration.capture_samples import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
