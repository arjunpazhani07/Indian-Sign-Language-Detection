"""Step 2 - train the 5-word calibration model (saved to models/calibration/)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration.train_calibration import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
