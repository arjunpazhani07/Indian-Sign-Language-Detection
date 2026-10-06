"""Step 3 - live demo GUI using ONLY the 5-word calibration model."""
import os
import sys
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    from gui.calibration_app import CalibrationISLApp
    root = tk.Tk()
    CalibrationISLApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
