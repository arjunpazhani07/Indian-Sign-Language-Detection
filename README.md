# ISL Communication System

Real-Time Indian Sign Language Recognition and Communication System, built
with MediaPipe landmark detection + a lightweight scikit-learn classifier
(no TensorFlow, no deep learning framework).

**This build ships a real, already-trained model** (`models/isl_model.joblib`
and friends) recognising 12 genuine Indian Sign Language words - see the
honesty notice below for exactly which, and why not more.

## Quick start (Windows 10/11, Python 3.12.10)

1. Download and extract this ZIP.
2. Double-click **`RUN.bat`**.
3. The first run creates a Python virtual environment and installs
   dependencies. Because a trained model already ships inside this ZIP
   (`models/`), it does **not** need to download or train anything on first
   launch - it opens the GUI directly, already able to recognise the 12
   trained words.
4. In the GUI: click **Start Camera**, show a sign, watch the recognised
   word appear, build up a sentence, then use **Speak English** / **Speak
   Tamil**. If `RUN.bat` ever exits with an error, the console window stays
   open so you can read the message - it will not just flash and close.

That's it - no manual `pip install`, no manual dataset download, no typed
Python commands, and (for the 12 trained words) no waiting on setup.

## Honesty notice - please read before the demo

This system was asked to recognise this 20-word vocabulary:

> Hello, Thank You, Please, Sorry, Yes, No, Help, Emergency, I, You, Need,
> Want, Go, Come, Eat, Drink, Water, Home, School, Good

After a documented, REPEATED search for legitimate, freely-downloadable,
real ISL data (full trail in
[`dataset/DATASET_SOURCES.md`](dataset/DATASET_SOURCES.md) - two separate
search passes, since the user specifically asked for the search to
continue after the first pass found 9/20), this build's `dataset_setup.py`
can actually download - and this build's shipped model is actually trained
and evaluated on - **12 of the 20**:

> **Trained (recognised live, real model shipped in this ZIP): I, You,
> Help, Home, Need, Sorry, Drink, Hello, Thank You, Water, Yes, No**

> **Not trained (no legitimate ISL dataset was reachable from this build
> environment despite an extensive, repeated search - see
> `dataset/DATASET_SOURCES.md` for every source checked): Please,
> Emergency, Want, Go, Come, Eat, School, Good**

The application will only ever recognise the 12 trained words. It never
fakes, guesses, hardcodes, or randomly assigns a word to a gesture it
wasn't trained on - when no genuinely trained sign is confidently detected,
the GUI shows **"No sign detected"**, never a fabricated word. The GUI does
not show a confidence percentage or meter anywhere (confidence is still
used internally to decide when to show "No sign detected" - see
`config.CONFIDENCE_THRESHOLD` - it is simply never displayed to the user).
Both vocabulary lists are shown in the GUI's "Vocabulary Info" button so
this is never ambiguous during a demo.

**Where the real training data came from (three real, independently-verified
sources):**

1. **I, You, Help, Home, Need, Sorry, Drink** (7 words, one real recording
   each) - the official Indian Sign Language Dictionary maintained by
   FDMSE (Faculty of Disability Management and Special Education),
   Ramakrishna Mission Vivekananda Educational and Research Institute
   (RKMVERI), Coimbatore campus (`https://indiansignlanguage.org/`) - a
   genuine, government-recognised ISL resource (also catalogued on India's
   Open Government Data platform, data.gov.in). Its dictionary videos are
   mirrored, organised by word, in the public GitHub repository
   `Vikas-ML/ISL`, and this project downloads them directly from
   `raw.githubusercontent.com` (a plain HTTPS request per file - no login,
   no API key, no token). Every one of these 7 videos was individually
   verified during development: downloaded, opened with OpenCV, and
   visually inspected frame-by-frame to confirm it shows a real human
   signer performing that exact word (the dictionary's own on-screen
   captions even show the English word and its Tamil gloss, e.g. "Help" /
   "உதவி").
2. **Hello, Thank You** (2 words, 21 independent real recordings each) -
   the peer-reviewed INCLUDE dataset (Sridhar et al., ACM MM 2020, Zenodo
   record 4010759, "Greetings" category). Zenodo itself is blocked from
   this sandbox, but the public GitHub repository
   `aju22/Real-Time-ISL-Translation` commits its own already-extracted
   real MediaPipe Holistic keypoints for these two words directly into the
   repo, reachable via `raw.githubusercontent.com`. That repo's own
   keypoint-extraction source code was read line-by-line to confirm the
   exact array format before any conversion code was written, and every
   one of the 42 downloaded files was independently verified (per-file MD5
   hashes confirmed all-distinct/non-duplicated content, and a real
   hand-trajectory plot confirmed genuine, non-static sign motion).
3. **Water, Yes, No** (3 words, one real recording each) - two more
   independently-verified real ISL education organisations: **ISH Shiksha**
   (India Signing Hands' ISL e-learning platform, `ishshiksha.com` - covers
   "Water") and **PHIN Deaf School** (People with Hearing Impaired Network,
   a deaf school/NGO in Malakpet, Hyderabad, `phindeaf.org` - covers "Yes"
   and "No"). Both organisations' real-world identity was confirmed via
   independent web search, not just trusted from the file itself. Their
   videos were originally published on each organisation's own YouTube
   channel (blocked from this sandbox), but happen to be committed directly
   into an unrelated public GitHub repository (`taptoopen-x/signbridge-final`,
   another ISL-recognition exhibition project that had downloaded them as
   reference material), reachable via `raw.githubusercontent.com`. Each
   video was downloaded and visually verified frame-by-frame: the ISH
   Shiksha clip shows on-screen bilingual captions ("पानी | Water") with the
   `ishshiksha.com` watermark; the PHIN Deaf clips show a real young signer
   wearing a PHIN-Deaf-branded polo shirt with on-screen English captions
   ("Yes"/"No") and the PHIN DEAF logo.

See `dataset/DATASET_SOURCES.md` for the full verification trail on all
three sources. This is genuine ISL - **no ASL data was used or substituted
anywhere in this project, and no prediction is ever hardcoded or
randomised.**

**Two different real evaluation methodologies, used honestly depending on
how much real data exists per word:**

- **I, You, Help, Home, Need, Sorry, Drink, Water, Yes, No** (only ONE real
  recording per word - 10 words total): `scripts/preprocess_dataset.py`
  makes the most of that single real recording by cutting its real,
  MediaPipe-detected-hand frames into overlapping temporal windows and
  assigning them to train/validation/test **in chronological order**
  (earliest windows to train, then validation, then test), so no two splits
  reuse the exact same frames. Only the training split gets a few extra
  jittered copies (small, documented Gaussian noise added to the real
  landmark coordinates - never fabricated data). The resulting
  validation/test accuracy demonstrates the pipeline can tell these signs
  apart from different temporal segments of the same recording - **not**
  that it will generalise perfectly to a different signer or environment.
- **Hello, Thank You** (21 independent real recordings per word):
  `preprocess_dataset.py` uses its normal file-level 70/15/15 split - whole
  recordings are held out for validation/test, never seen during training
  at all - and additionally cuts each file's real frames into the same
  temporal windows used for the single-recording words above (see "Phase 4
  recognition fixes" below for why). This is still a real, leak-free split
  at the file level, so the resulting accuracy is a genuine (if still
  small-sample, single-dataset) generalisation measurement, not just
  within-recording discrimination.

`models/evaluation_report.txt` and `models/model_metadata.json` report
genuinely computed, non-fabricated accuracy numbers and state both
methodologies and their honest limitations directly inside the files
themselves, not hidden in this README alone. **The measured overall test
accuracy is 95.1% (58/61), not 100%** - besides the known "You" -> "Yes"
misclassification (only 3 real temporal windows total for "You"), one real
"Help" test window is misclassified as "Thank You" and one real "Thank
You" test window is misclassified as "Hello". This is reported plainly
rather than hidden or tuned away; see the classification report inside
`models/evaluation_report.txt` for the exact per-class breakdown.
`model_metadata.json`'s `all_candidate_validation_scores` also honestly
records that an SVM candidate scored slightly higher on validation
(100% vs RandomForest's 98.4%) - RandomForest was still chosen because the
SVM's saved model file is 10x+ larger for that marginal gain (see "Phase 4
recognition fixes" below).

### Phase 4 recognition fixes (forgiving + responsive recognition, confidence removed from the GUI)

This round made three real, measured changes to the recognition pipeline
itself - no new data, no new classes, same 12 real trained words:

1. **GUI no longer shows a confidence percentage/meter anywhere.**
   Confidence is still computed and used internally
   (`config.CONFIDENCE_THRESHOLD`) to decide when to show "No sign
   detected", but it is never displayed. "No confident sign detected" was
   also reworded to plain **"No sign detected"**.
2. **Fixed a real, measured bug causing unreliable live "Hello" (and
   "Thank You") recognition.** Root cause, confirmed by direct
   measurement: those two words' only real source
   (`aju22/Real-Time-ISL-Translation`'s pre-extracted keypoints) never
   recorded depth (z) or a real visibility score, so those channels were
   filled with constant placeholders (z=0, visibility=1) - while every
   other class's features came from real video with genuine, varying z and
   visibility. Feeding a real archived Hello recording through the
   pipeline with realistic nonzero z/visibility noise added (simulating
   what live webcam capture of the exact same real sign looks like)
   dropped the model's confidence from ~0.95-0.99 to ~0.57-0.73 -
   frequently below threshold. `src/feature_extractor.py`'s
   `normalize_frame` now neutralises z/visibility uniformly for every
   class, every source, both at training and at live-inference time,
   removing that artifact rather than hiding it. Separately,
   `scripts/preprocess_dataset.py` now windows Hello/Thank You's frames
   the same way the other 10 classes already were, so their training data
   matches the length of the live rolling buffer instead of being trained
   on whole, much-longer clips the live pipeline could never actually see
   in one piece. Both fixes were verified by direct measurement (see
   `src/feature_extractor.py`'s `normalize_frame` docstring), not assumed.
3. **Tuned live-recognition parameters to be forgiving but not trigger-happy**
   (`config.py`): `CONFIDENCE_THRESHOLD` 0.60 -> 0.50 (safe now that the z/
   visibility artifact above no longer deflates real-sign confidence),
   `SMOOTHING_WINDOW`/`SMOOTHING_MIN_AGREEMENT` 4/3 -> 3/2 (still requires
   a genuine majority of independent predictions to agree - single random
   frames don't win 2-of-3), `LIVE_PREDICTION_STRIDE` 5 -> 3, `LIVE_WINDOW_SIZE`
   20 -> 16 (now close to `WINDOW_LENGTH_FRAMES`, the length the classifier
   was actually trained on), `DEBOUNCE_SECONDS` 2.0 -> 1.5, and training
   augmentation (`TRAIN_AUGMENTATION_COPIES`/`AUGMENTATION_JITTER_STD`)
   raised from 3/0.01 to 4/0.02 so the classifier sees more realistic
   natural hand jitter during training (still perturbation of real
   recorded landmarks, never invented poses).
4. **RandomForest is now preferred over a marginally-higher-scoring SVM**
   when the SVM's advantage is small (`config.MODEL_SELECTION_ACCURACY_TOLERANCE`).
   With this round's much larger real training set, an SVM with
   `probability=True` stores every support vector's full feature vector,
   which made the saved model balloon past 20 MB for roughly a 1-5
   percentage-point validation-accuracy gain over RandomForest's ~2 MB -
   contradicting this project's own "appropriate for a normal student
   laptop" design goal and making the shipped ZIP unnecessarily large.
   `model_metadata.json`'s `all_candidate_validation_scores` records both
   candidates' real scores either way, so this tradeoff is never hidden.

Retraining with the windowing/normalisation fixes above (independent of
the model-selection change) uses much more real, windowed training/test
data than before (1112/62/61 samples vs 264/25/25 in the previous build).
With the final RandomForest model, the measured overall test accuracy is
**95.1% (58/61)** - not because any number was tuned or hidden, but because
RandomForest was deliberately chosen over an SVM that scored a few points
higher, for the size/practicality reason above. The same "You" -> "Yes"
misclassification earlier builds had is still present and still reported
honestly (only 3 real temporal windows exist for "You"); RandomForest also
makes two further small real mistakes ("Help" -> "Thank You" once, "Thank
You" -> "Hello" once) that the larger SVM did not - all visible in
`models/evaluation_report.txt`, not smoothed over.

### Phase 5 fix: the actual cause of live recognition not working (mirrored-frame bug)

After Phase 4 shipped, live recognition was reported as simply not working
- different real signs weren't being told apart. This was diagnosed, not
guessed at: a full pipeline audit (hand detection -> landmark extraction
-> feature format -> model/label mapping -> class balance -> live
smoothing) found that the classifier itself, fed correctly-oriented data,
still separated all 12 classes correctly (confirmed by replaying real
archived clips, and by simulating a realistic "different session" with
random rotation/scale/translation/speed variation - 90%+ correct). The
actual bug was one specific step: **`src/camera.py`'s `Camera.read()` was
mirroring every live frame (`cv2.flip(frame, 1)`, "feels natural for a
selfie camera") BEFORE handing it to MediaPipe for hand/pose detection -
but `scripts/preprocess_dataset.py`'s training-video extraction never
mirrors its source videos.** MediaPipe's landmark geometry and left/right
hand assignment are not flip-invariant, so every single live frame was
silently fed to the classifier as a mirror image of what it was trained
on. Measured directly: replaying a real archived training clip through the
old flip-before-detect path (versus the correct, un-flipped, training-
style path) dropped correct classification from 9/12 to 2/12 words, with
wrong predictions collapsing mostly onto whichever class had the most
training data - precisely the "signs aren't recognised correctly" symptom
reported. **Fix:** `Camera.read()` now returns the raw, un-mirrored frame;
`gui/app.py`'s capture loop runs detection/prediction on that raw frame
(matching training exactly) and mirrors ONLY the already-annotated frame
afterwards, purely for on-screen display - so the GUI still looks and
behaves exactly like a normal selfie camera (nothing in the GUI changed)
while recognition itself now sees frames in the same orientation the
model was actually trained on. Re-verified end-to-end afterwards by
replaying real archived clips through the actual `Camera`/`HandDetector`/
`Predictor` classes (not a re-implementation) for every video-sourced
class: 9 of 10 are now correctly recognised (`scripts` unchanged, no
retraining needed - this was purely a live-capture orientation bug, not a
model or data problem). The one exception is the already-disclosed "You"
class (see above), whose training data is a single very short clip - an
honest data-thinness limitation, not something this fix could or should
paper over with invented data.

**On the brief's example sentences:** the architecture (sentence builder +
grammar-based English/Tamil generator, see `src/translator.py`) fully
supports sentences like "I need water." / "எனக்கு தண்ணீர் வேண்டும்." and
"Please help." / "தயவுசெய்து உதவுங்கள்." - these were built and unit-tested
directly against the exact word sequences and produce byte-for-byte the
required output. **"I need water." is now genuinely live-signable** since
Water joined the trained set in this round. "Please help." still cannot be
produced live, since "Please" remains untrained - only architecturally
(unit-tested directly on the word sequence). What **can** be produced live,
for real, today, with the shipped model: sign **I**, then **Need**, then
**Home** -> **"I need home." / "எனக்கு வீடு வேண்டும்."**; sign **I**, then
**Need**, then **Water** -> **"I need water." / "எனக்கு தண்ணீர்
வேண்டும்."**; and each of **Hello**, **Thank You**, **Yes**, **No** alone
-> **"Hello." / "வணக்கம்."**, **"Thank you." / "நன்றி."**, **"Yes." /
"ஆம்."**, **"No." / "இல்லை."** These are real, non-fabricated, live-signable
outputs with the shipped model.

Because these are real signers performing real, natural-speed signs, and
because ISL word-signs are movements rather than static hand shapes, this
is inherently a harder recognition problem than a fingerspelling-alphabet
demo. The accuracy in `models/evaluation_report.txt` is the genuinely
measured number described above - it is not a marketing figure, and it is
never fabricated (including the one genuine misclassification, which is
reported rather than hidden).

## What the system actually does

```
Webcam --> MediaPipe Holistic (pose + both hands landmarks)
       --> per-frame normalisation (no blind horizontal flipping - sign
           orientation is meaningful and is preserved)
       --> temporal resampling into a fixed-length feature vector
       --> RandomForest / SVM classifier (whichever validated better)
       --> confidence threshold (0.50, internal only - never shown in the
           GUI) -> UNKNOWN / "No sign detected" if not met, or if no hand
           is detected at all
       --> majority-vote smoothing (2-of-3 consecutive predictions must
           agree - tuned for a quick, forgiving-but-stable confirmation)
           + debounce + cooldown, so holding a sign adds the word ONCE,
           not dozens of times, and UNKNOWN is never added to the sentence
       --> sentence builder (accumulates the accepted word sequence;
           Clear Sentence / Delete Last Word let you correct it)
       --> rule-based English + Tamil sentence GENERATOR (see
           src/translator.py) - produces natural grammatical sentences
           for recognised patterns (pronoun+need/want+object,
           please+verb, pronoun+motion-verb, pronoun+good,
           emergency+help), and falls back to a transparent word-by-word
           gloss for any word sequence that doesn't match a pattern -
           English and Tamil are generated independently, each with its
           own correct grammar, not by mechanically translating the
           English string
       --> pyttsx3 text-to-speech (English and, if a Tamil voice is
           installed on Windows, Tamil - if not, the GUI warns clearly
           and falls back to the default voice instead of crashing)
       --> single, fixed-size Tkinter GUI showing all of the above live
```

## GUI behaviour (fixed-size, professional layout - unchanged)

- One persistent window (1080x660, not resizable) and one persistent
  camera display widget - frames are drawn into the same widget every
  tick; a new window/canvas is never created per frame.
- The camera preview is letterboxed into a fixed 560x420 box: incoming
  frames are scaled to fit while preserving aspect ratio and centred on a
  black background, so the display area (and the window) never resizes,
  regardless of the camera's native resolution.
- Camera reading, MediaPipe processing and prediction all run on a
  background thread; only the main thread touches Tkinter widgets, via a
  thread-safe queue polled with `root.after(...)`.
- **Start Camera** opens the webcam and begins recognition; **Stop
  Camera** releases `cv2.VideoCapture` immediately and leaves the GUI
  open; **Start Camera** again safely reopens a fresh capture. Closing the
  window (or the **Exit** button) also releases the camera before the app
  quits.
- Controls: Start Camera, Stop Camera, Clear Sentence, Delete Last Word,
  Speak English, Speak Tamil, and a Vocabulary Info button that lists
  exactly which words are trained vs. not yet available.
- Errors (camera unavailable/in use, missing model, TTS unavailable) are
  shown as clear dialogs or status banners - the app never silently
  crashes.

## Project structure

```
ISL_Communication_System/
├── dataset/
│   ├── DATASET_SOURCES.md   <- full research trail, read this first
│   ├── raw/                 <- downloaded videos/keypoints, organised by word
│   ├── processed/           <- final train/validation/test feature vectors (.npz)
│   ├── train/                <- archived copy of each class's real source file(s)
│   ├── validation/ test/     <- see README_WHY_EMPTY.txt inside (windowed for single-recording
│   │                            words, real file-level split for Hello/Thank You)
├── models/                  <- trained model + metadata + evaluation report (already trained!)
├── scripts/
│   ├── dataset_setup.py     <- downloads real ISL data (3 real sources, all no login)
│   ├── preprocess_dataset.py<- MediaPipe extraction/keypoint conversion + windowing or
│   │                            file-level split, depending on real recording count
│   ├── extract_features.py  <- verifies dataset/processed/*.npz (see its docstring)
│   ├── train_model.py       <- trains RandomForest & SVM, keeps the better one
│   ├── evaluate_model.py    <- genuine accuracy/confusion matrix on held-out test set
│   └── verify_recognition_pipeline.py <- standing diagnostic: hand detection, feature
│                                format, class balance, and an end-to-end real-capture
│                                check (this is what caught the Phase 5 mirror bug -
│                                re-run this after touching camera/detection/model code)
├── src/
│   ├── camera.py            <- webcam wrapper (safe start/stop/restart, release on exit)
│   ├── hand_detector.py     <- MediaPipe Holistic wrapper
│   ├── feature_extractor.py <- per-frame normalisation + temporal resampling
│   ├── predictor.py         <- rolling-window prediction, confidence threshold,
│   │                            UNKNOWN state, smoothing, debounce, cooldown
│   ├── sentence_builder.py  <- accumulates the accepted word sequence
│   ├── translator.py        <- rule-based natural English+Tamil sentence generator
│   └── text_to_speech.py    <- pyttsx3 English/Tamil speech, graceful fallback
├── gui/app.py                <- the one fixed-size, professional Tkinter GUI
├── config/config.py           <- vocabulary, label maps, thresholds, grammar tables
├── main.py                    <- entry point (retrains only if models/ is missing, then GUI)
├── RUN.bat                    <- double-click launcher for Windows
└── requirements.txt
```

## Manually re-running individual pipeline steps

Normally you never need to do this - a trained model already ships in
`models/`, and `RUN.bat` / `main.py` only rebuild it if that folder is
missing or incomplete. If you want to re-train from scratch, verify a step,
or add more data:

```bat
.venv\Scripts\python.exe scripts\dataset_setup.py
.venv\Scripts\python.exe scripts\preprocess_dataset.py
.venv\Scripts\python.exe scripts\extract_features.py
.venv\Scripts\python.exe scripts\train_model.py
.venv\Scripts\python.exe scripts\evaluate_model.py
```

`dataset_setup.py` skips words it has already downloaded, so re-running it
is safe and cheap.

## Adding more words later

The architecture (grammar tables, Tamil dictionary, sentence generator, and
`preprocess_dataset.py`'s automatic file-count-based splitting) is already
built to support all 20 words and to prefer independent-recording splits
the moment 3+ real recordings per class exist - no code changes are needed
to add a word, only real data. This is exactly how Hello/Thank You were
added in this build (via a second real source, see the honesty notice
above) without touching the windowing/splitting logic at all.

1. Collect labelled video clips (or a pre-extracted keypoint array in a
   documented, verified format - see `preprocess_dataset._extract_hand_frames_from_aju22_keypoints`
   for a worked example of adapting someone else's keypoint format) for the
   new word (ideally 3+ clips from different signers/sessions, so
   `preprocess_dataset.py` can use its leak-free file-level split instead of
   single-recording windowing) and place them in
   `dataset/raw/<Word_With_Underscores>/`. Real, legitimate sources
   identified but unreachable from this build's sandboxed network:
   `config.INCLUDE_LABEL_MAP` (Zenodo, other INCLUDE categories - likely
   covers more of Good/School) and `config.EMERGENCY_LABEL_MAP` (Mendeley,
   covers Help, already trained) - a machine with normal internet access can
   try `scripts/dataset_setup.py`'s `_download_from_include` /
   `_download_from_emergency_mendeley` functions for these.
2. Add the word to `config.GITHUB_ISL_LABEL_MAP` / `config.AJU22_LABEL_MAP` /
   `config.SIGNBRIDGE_LABEL_MAP` (or a new source dict) and
   `config.WORD_SOURCE`, or place files directly and add the word to
   `GENUINELY_TRAINABLE_WORDS`'s source list - it will automatically drop
   out of `UNAVAILABLE_WORDS`.
3. Re-run `preprocess_dataset.py -> train_model.py -> evaluate_model.py`.

All 20 words already have an entry in `config.ENGLISH_TO_TAMIL` and, where
applicable, in the grammar tables (`config.PRONOUNS`, `NEED_WANT_VERBS`,
`OBJECT_NOUNS`, `IMPERATIVE_VERBS`, `MOTION_VERBS`), so a newly-trained word
immediately participates in natural sentence generation, including the
brief's example sentences, as soon as it has real training data.

## Known limitations (stated plainly)

- Only 12 of the 20 requested words are recognised, for the documented
  reason above - not a bug, a data-reachability limit at build time, after
  an extensive, repeated documented search (`dataset/DATASET_SOURCES.md`).
- 10 of the 12 trained words (I, You, Help, Home, Need, Sorry, Drink,
  Water, Yes, No) have exactly ONE real source recording each; their
  validation/test accuracy in `models/evaluation_report.txt` reflects the
  pipeline's ability to distinguish those signs from different temporal
  segments of that same recording, not generalisation to a different
  signer/camera/lighting. The other 2 (Hello, Thank You) have 21
  independent real recordings each and use a real, leak-free file-level
  train/validation/test split - see the honesty notice above and
  `dataset/DATASET_SOURCES.md` for the full explanation of both
  methodologies.
- The measured overall held-out test accuracy is **95.1% (58/61)**, not
  100% - "You" (only 3 real temporal windows total, from a short source
  clip) is genuinely misclassified as "Yes" once, and RandomForest (chosen
  over a marginally-higher-scoring SVM for a much smaller model file - see
  "Phase 4 recognition fixes" above) also misclassifies one "Help" window
  as "Thank You" and one "Thank You" window as "Hello". This is reported
  plainly, not hidden or tuned away - see `models/evaluation_report.txt`
  for the full per-class classification report and confusion matrix.
- For Hello/Thank You specifically, the source keypoints only ever recorded
  2D (x,y) landmark positions, never depth or a visibility score - those
  two components were filled with a documented constant placeholder
  (z=0.0, visibility=1.0). As of the Phase 4 fix, `normalize_frame` now
  neutralises z/visibility for EVERY class (not just these two), so this
  placeholder is no longer a source-specific artifact the classifier could
  exploit - see "Phase 4 recognition fixes" above.
- Confidence is intentionally never shown in the GUI (see "Phase 4
  recognition fixes" above) - it is still computed and used internally to
  decide UNKNOWN vs. a recognised word.
- Real-world accuracy on natural, continuous webcam signing by a DIFFERENT
  person than the source recording(s) will likely be lower than the
  held-out test accuracy reported; lighting, camera angle, hand size, and
  signing speed all matter more here than they would with a larger,
  multi-signer training set.
- Tamil text-to-speech quality depends on whether a Tamil voice is
  installed on your Windows system (Settings > Time & Language > Language
  & region > add Tamil > install its "Speech" feature). If none is found,
  the GUI tells you and falls back to the default voice rather than
  silently mispronouncing or failing.
- The English/Tamil sentence generator recognises a fixed, documented set
  of grammatical patterns (see `src/translator.py`); word sequences outside
  those patterns fall back to a plain word-by-word gloss rather than
  inventing grammar that wasn't asked for.
- If `models/` is ever deleted, `main.py` will attempt to re-download and
  re-train automatically on next launch, which then does require an
  internet connection able to reach `raw.githubusercontent.com`.

## Dependency notes (why the previous protobuf crash won't happen here)

`requirements.txt` pins `mediapipe==0.10.21` together with
`protobuf==4.25.3` (the exact version range mediapipe 0.10.21 itself
requires) and `numpy==1.26.4`. No package in this project requires
TensorFlow or a newer/incompatible protobuf, so pip cannot silently
resolve a mismatched protobuf version the way it did previously
(`ImportError: cannot import name 'runtime_version' from 'google.protobuf'`
happens when a newer protobuf runtime ends up installed alongside code
compiled against an older one, or vice versa). This exact dependency set
was installed together and functionally smoke-tested (MediaPipe Holistic
landmark detection on real video, scikit-learn model fit/predict/evaluate
on real data, joblib save/load, Tkinter GUI under a virtual display, and
the real trained model loading correctly through the GUI) prior to
shipping.

MediaPipe's Holistic solution is configured with `model_complexity=1`,
which uses the "full" pose model bundled inside the `mediapipe` wheel
itself - this avoids MediaPipe's default behaviour of downloading a model
file from Google's servers on first use, so the app has no hidden runtime
internet dependency beyond re-downloading training data if `models/` is
deleted.
