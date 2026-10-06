# Live Calibration Mode (5 words): Hello, I, Need, Water, Thank You

A separate, personal classifier trained ONLY on samples you record with your own
webcam. It reuses the project's existing MediaPipe landmark extraction
(`src/hand_detector.py`) and 20-frame temporal features
(`src/feature_extractor.extract_clip_features`). The original 12-class model in
`models/` is never read or written by this mode.

```
webcam -> your real ISL sign -> MediaPipe landmarks -> 20-frame features
       -> calibration classifier (models/calibration/) -> smoothing + debounce
       -> sentence -> English + Tamil -> speech
```

## Steps (double-click, or run the commands)

| Step | Double-click | Command (in project folder) |
|---|---|---|
| 1. Record samples | `CALIB_1_CAPTURE.bat` | `.venv\Scripts\python.exe calibrate_capture.py` |
| 2. Train | `CALIB_2_TRAIN.bat` | `.venv\Scripts\python.exe calibrate_train.py` |
| 3. Live demo | `CALIB_3_DEMO.bat` | `.venv\Scripts\python.exe calibrate_demo.py` |

Capture keys: `A` auto-capture on/off, `SPACE` one sample, `N`/`P` next/previous word,
`1`-`6` jump to a word, `D` delete last sample, `Q` quit.

Data: `dataset/calibration/<Word>/sample_*.npy` (raw landmark sequences, one file per recording).
Model: `models/calibration/calib_model.joblib`, `calib_scaler.joblib`,
`calib_label_encoder.joblib`, `calib_metadata.json`, `calib_report.txt`.

Optional `_Neutral` class (word 6): hands visible but NOT signing (resting, moving
between signs). It is never shown or added to the sentence; it only helps reject
non-signs and re-arm the debounce. Used only if 10 or more samples exist.

Debounce: after a word is added, the same word is blocked until you lower your hands
out of view (about 6 frames) or the classifier sees `_Neutral`. A different word can
follow directly (minimum 0.8 s gap). Clear Sentence also resets the debounce.

Tuning: `calibration/calib_config.py` (CONFIDENCE_THRESHOLD, VOTE_*, ABSENCE_FRAMES_TO_RESET).
