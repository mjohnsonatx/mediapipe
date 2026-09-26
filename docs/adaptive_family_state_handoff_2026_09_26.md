# Adaptive pose family swap - retained per-graph state and the AAR changes to share it

## Summary

Adaptive pose detection swaps between Lite, Full, and Heavy by selecting a different,
already-loaded `PoseLandmarker`. Each landmarker is its own MediaPipe graph with its own temporal
state, left over from the last frame that graph processed. On a swap, the incoming graph resumes
from that old state instead of from what the outgoing graph just produced.

Two kinds of state carry over:

1. **The tracking rectangle ("previous crop").** The app already hands this across a swap: the
   external ROI inputs added in this fork override it on every tracked frame.
2. **Temporal smoothing** (One Euro landmark filters and visibility low-pass filters). The app
   cannot share this today, and it is what makes a swap visible. Fixing it requires an AAR rebuild.

The app-side debounce (`AdaptiveDetectorFamilyStabilizer`: 2 confirmations to go heavier, 6 to go
lighter, 1 s minimum dwell) makes swaps rare. It does not change what happens on each swap.

## How swapping works today (app)

- Adaptive mode fixes the pose pool at 2 lanes. Each lane holds all three families, so there are
  6 independent VIDEO-mode graphs (`PoseDetectorManager`, `DetectorRoundRobinPolicy`).
- A swap is a flag flip: `PoseDetectorManager.activeDetectorFamily`. Each admitted frame runs on
  the active family's landmarker in whichever lane it leased.
- ROI tracking is shared by all 6 graphs through one `ExternalPoseRoiTracker`
  (`PoseDetectorManager.kt:163`, `MediaPipePoseDetectorAdapter.kt:34`). Tracked frames go through
  `detectForVideoWithExternalRoiCrop` (GPU/RGBA crop done by the app) or
  `detectForVideo(image, externalPoseRect, ts)` (MediaPipe crops). A frame with no tracked region
  uses plain `detectForVideo`.
- When a pose is lost or the video is discontinuous, the tracker bumps its generation. Every
  landmarker then calls `resetTracking()` once before its next frame
  (`MediaPipePoseDetectorAdapter.kt:262`, `:423`).

## State each graph keeps between runs

| State | Where (fork) | What is retained | Handed across a swap today? |
| --- | --- | --- | --- |
| Tracking rect loopback | `PreviousLoopbackCalculator`, `pose_landmarker_graph.cc:445`, back edge `:504` | `POSE_RECTS_NEXT_FRAME` from the last frame *this graph* ran. When present, it skips the pose detector (`:477-486`) and positions the landmark crop. | **Yes on tracked frames.** Java sets `EXTERNAL_POSE_RECT` on every external call (`PoseLandmarker.java:630-665`); the graph merges it ahead of the loopback (`:473`). Pre-cropped input uses `FULL_IMAGE_TRACKING_RECT`. |
| Loopback on untracked frames | same | Same stale rect, used whenever `USE_EXTERNAL_POSE_RECT` is false | **Only after a loss.** `RESET_TRACKING` drops it (`:454`). An untracked frame with no generation bump would crop at a stale spot; the current tracker design does not produce one. |
| Normalized landmark smoothing | `LandmarksSmoothingCalculator` via `SmoothLandmarks`, `pose_landmarks_detector_graph.cc:930-956` | Per landmark x/y/z: `OneEuroFilter` = last raw value, last filtered value, derivative EMA (`LowPassFilter` x2), last timestamp, frequency. Uses `IMAGE_SIZE` and `OBJECT_SCALE_ROI`. | **No.** |
| World landmark smoothing | same, `:957-970` | Same filters, world space | **No.** |
| Visibility smoothing | `VisibilitySmoothingCalculator` x2 (alpha 0.1), `:941`, `:957` | Last visibility per landmark | **No.** |
| Graph timeline | task runner | Last input timestamp (monotonic per graph) | Per adapter (`lastTimestampMs`); correct, but it sets the One Euro frequency |

Segmentation smoothing is not involved: the app does not request segmentation masks.

### Why the smoothing state makes a swap visible

- The incoming graph smooths its first frame against its **own** last output, which may be
  seconds old and from a different part of the rep. Its frequency comes from that long gap.
  One Euro's speed term usually lets most of the new value through, so the first frames are
  effectively unsmoothed.
- Its derivative estimate is also stale, so the next few frames smooth by a different amount than
  the outgoing graph was using.
- Lite, Full, and Heavy place landmarks slightly differently (model bias). Only a filter that runs
  continuously across the swap can smooth that step; each graph's filter only ever sees its own
  model's output.
- Lanes make it worse. With 2 lanes, consecutive frames alternate between two graphs of the same
  family, each filtering every other frame with separate histories. This happens even without a
  swap.

Your `FilterLandMarks` (EMA plus `ExtendedKalmanFilter3D`) is continuous across lanes and
families, but it receives input that has already been filtered by a different graph on each frame.

## Options for the AAR

### Option A - make in-graph smoothing optional and let the app own it (recommended first)

Smallest native change. It removes every piece of retained temporal state except the loopback,
which is already overridden. This also removes the lane-alternation problem.

Fork changes:

1. `pose_landmarker_graph.cc:402-411` currently forces
   `set_smooth_landmarks(use_stream_mode())`. Only apply that default when the option is unset:
   `if (!pose_landmarks_detector_graph_options.has_smooth_landmarks())`. The field is proto2
   `optional bool smooth_landmarks = 3` (`pose_landmarks_detector_graph_options.proto:44`), so
   `has_` is available. Keep the existing multi-pose guard.
2. `PoseLandmarker.java`, `PoseLandmarkerOptions`: add `abstract Optional<Boolean> smoothLandmarks()`
   with a builder setter. In `convertToCalculatorGraphConfigOptions` (`:886-896`), call
   `smoothLandmarks().ifPresent(poseLandmarksDetectorGraphOptionsBuilder::setSmoothLandmarks)`.
   The default stays absent, so upstream behavior is unchanged.
3. Tests: a graph test proving an explicit `false` in stream mode produces no
   `LandmarksSmoothingCalculator` or `VisibilitySmoothingCalculator` nodes, and a
   `PoseLandmarkerTest` case for the Java option.

App changes:

- Build live VIDEO-mode landmarkers with `smoothLandmarks(false)`. Leave preprocessing as it is,
  or decide separately.
- `FilterLandMarks` becomes the only temporal filter. Retune its EMA/EKF, because it previously
  received One Euro output. Keep the stable `projectionSourceWidth/Height`: projection still uses
  the aspect ratio.
- Profile at 30 and 60 FPS with and without adaptive detection. Removing One Euro's velocity-
  adaptive behavior can add jitter when still or lag in fast lifts until the app filter is tuned.

### Option B - hand the smoothing state across a swap

Keeps One Euro, but has the largest native surface.

1. `mediapipe/util/filtering`: add state accessors. `LowPassFilter` has `raw_value_`,
   `stored_value_`, `initialized_`; `OneEuroFilter` has `x_`, `dx_`, `last_time_`, `frequency_`.
   Add a plain `State` struct with `GetState()` / `SetState()`.
2. `LandmarksSmoothingCalculator` and `VisibilitySmoothingCalculator`: add an optional output
   `FILTER_STATE` (a new proto: per-landmark, per-axis filter state plus object scale) and an
   optional input `SEED_FILTER_STATE`. When the seed is present, apply it before filtering that
   timestamp's packet.
3. Route both through the smoothing block in `pose_landmarks_detector_graph.cc:930-970` and out to
   new `PoseLandmarkerGraph` input and output tags.
4. Java/JNI: expose the state on the result (for example `PoseLandmarkerResult.smoothingState()`
   as opaque bytes). Add `seedSmoothingState(bytes)` that queues a packet for the next frame, the
   same way `resetTracking()` queues `RESET_TRACKING`.
5. App: on a swap, seed the incoming landmarker with the outgoing one's last state before its first
   frame. Seeding every frame from whichever graph ran last would also unify the two lanes, at the
   cost of per-frame serialization: 33 landmarks x 3 axes x 2 spaces of a few floats each.

Timestamp caveat: the seeded `last_time_` must be earlier than the incoming frame's timestamp in
that graph, or `OneEuroFilter::Apply` returns the raw value. Decoder-derived timestamps satisfy
this if the app seeds with the previous frame's time.

### Option C - one graph per lane with all three landmark models (long term)

- Put the Lite, Full, and Heavy landmark inference behind a `SwitchContainer`
  (`mediapipe/framework/tool/switch_container`), selected by a per-frame `SELECT` input stream.
  The pose detector, ROI loopback, projection, smoothing, and timeline become shared.
- A swap is then a change of select index, and every retained state is continuous by
  construction. It also loads the BlazePose detector once per lane instead of three times.
- Cost: significant graph and task-options rework (three landmark model assets in one task, a GPU
  delegate per inference node). The two lanes are still separate graphs, so combine this with A or
  B for lane continuity.

## Recommendation

1. Ship **Option A**. It is a small fork diff, removes the per-graph state that causes swap jank,
   and fixes lane alternation as a side effect.
2. Keep the app debounce. Fewer swaps still means fewer model-bias steps.
3. Revisit **B** only if tuning `FilterLandMarks` cannot match One Euro's still/fast balance.
   Treat **C** as the eventual architecture if adaptive detection becomes a primary mode.

## Rebuild and release checklist

- Build with `infra/tools/mediapipe/build_external_roi_aar.sh` (app repo): Bazel 7.7.0, NDK
  28.2.13676358, `arm64-v8a`, `opt`.
- `infra/tools/mediapipe/custom_binary_manifest.properties` treats the manifest and both AARs as one
  release unit. In the same change:
  - bump `artifactRevision` (currently `formfocus-mediapipe-0.10.29-r7`);
  - update the source hashes of every modified fork file (for Option A at least
    `forkPoseLandmarkerSha256` and `forkPoseLandmarkerGraphSha256`, plus any new file);
  - update `visionSha256` / `visionArm64Sha256` (and `coreSha256` if tasks-core changes).
  Gradle verifies these before every `preBuild`.
- The manifest records `forkBranch=BUG-GPU-desynch-identified` with base
  `e1020198fd44211983d7c6695b8aff5a41a8ef4b`. Current work is on `FORM-FOCUS-MASTER`; reconcile
  which branch the next artifact is cut from before rebuilding.
- App tests to update: `MediaPipePoseDetectorAdapterTest` (fake landmarker; add the options
  wiring), plus on-device profiling at 30 and 60 FPS.

## How to verify on device

- Adaptive detection on, squat or side-view set. Watch the frames around each
  `Selected preloaded adaptive detector family: X -> Y` log line.
- Compare the landmark displacement on the swap frame with its neighbors. Before the fix it should
  spike; after it, it should look like any other frame.
- Also compare lane-alternation jitter with adaptive detection off and round robin on. Option A
  should reduce it as well.

## Implementation record (2026-09-26)

Option A is implemented on `FORM-FOCUS-MASTER`, based on
`c01cbd7a2b960661cb9663e4c1a0650bb305537c`. The Java option remains absent by default;
an explicit `setSmoothLandmarks(false)` now reaches the native graph without being overwritten
by VIDEO-mode defaults. The multi-pose smoothing guard is retained.

The consumer change is in `/home/michaeljohnson/StudioProjects/PoseDetectionApp` on
`HDR-CONVERSTIONS-ONLY`. Its shared options cache disables graph smoothing for every model
family and delegate, including CPU fallback and temporary families. The delegate benchmark
uses the same setting. Preprocessing shares that detector pool, so newly inferred preprocessing
data also inherits the option; existing saved data is unchanged. App EMA/EKF values, family
debounce, ROI tracking/reset, and projection dimensions were preserved for owner tuning.

The app's `infra/tools/mediapipe/build_external_roi_aar.sh` rebuilt the native library and both
AARs using Bazel 7.7.0, NDK 28.2.13676358, JBR 25.0.3, Python 3.12, ARM64, and `opt` with
stripping. Shared-GL exports and external OpenCV linkage passed the script's packaging gates.
The coordinated development revision is `formfocus-mediapipe-0.10.29-r8`:

- Vision AAR SHA-256: `ccbeb67f2450393a4e12ffb26a192bacd6aa39c3dda649bbde29fccc6825e582`.
- ARM64 JNI SHA-256: `0654ce9ed4c35a4cf9a3c107168974ad9ad3e4ce36d2b90b3cde26f4b643978c`.
- Core AAR SHA-256: `794ff7cd448933bb6753be1f51feed520391adf9e57541fd08d8c0c65effc859`
  (byte-identical to r7).
- App manifest SHA-256: `e3a1f70048f7e7c757bb1812626ac31253d6f5a5f2d86f018a80962ca829b017`.

The manifest pins the actual fork branch/base and changed source/test/build files. The app's
artifact inventory and modification notice match the new payload. Production approval remains
on r7 pending owner testing and review.

Consumer verification passed offline with `:app:verifyCustomMediaPipeArtifacts` and
`:app:compileDevDebugKotlin`. The packaged AAR's Java builder also exposes the new setter.

Regression cases were added to `PoseLandmarkerTest`, `MediaPipePoseDetectorAdapterTest`, and
the newly exposed `//mediapipe/tasks/cc/vision/pose_landmarker:pose_landmarker_graph_test`
target (`--test_filter=PoseLandmarkerGraphSmoothingTest.*`). The native case checks that explicit
false removes both landmark and visibility smoothing nodes, while default/true retain them.
Test suites and device profiling were not run, as requested. The remaining owner work is
30/60 FPS playback validation and any EMA/EKF retuning those results justify.
