# MediaPipe Android Boundary Optimization Opportunities

Date: 2026-11-22  
Status: opportunities 1 and 3 implemented in r24 source/AARs; app/device validation pending
App implementation base: `8a6c09097`; uncommitted on `perf/mp-roi-refactor-assessment`
Paired MediaPipe base: `c44e94698`; uncommitted in `/home/michaeljohnson/StudioProjects/mediapipe`

## Purpose

The tracked pose path has already moved the largest image-side operations out of MediaPipe. The
application owns ROI selection, OES/FBO crop, rotation, source-aspect correction, resize, tensor
packing, source projection, ordered smoothing, and the shared tracking state. MediaPipe still owns
model inference, presence gating, tensor decoding, heatmap refinement, world-landmark decoding,
and construction of the crop-local next-frame rectangle.

The next useful work is therefore at the Android/MediaPipe ownership boundary. The goal is to
remove packet, object, and graph-routing overhead without reproducing correctness-sensitive model
math in Kotlin.

## r24 implementation status

Opportunities 1 and 3 are now implemented behind the existing adapter and a source rollback. A
`PreparedPoseSession` accepts a selected tensor slot, timestamp, and reset state; immutable
metadata packets are created once and submitted natively. Each CPU or GPU session owns two
persistent slots with explicit free/writing/in-flight/poisoned lifetime. CPU capture packs directly
into native-aligned slot memory, and GPU capture packs directly into the native tensor SSBO.

This completes the narrow Android/JNI transport boundary, not the later stateless graph. The
general graph still contains `PoseTensorInputCalculator` and the tracked/acquisition merge nodes.
App-projection construction does remove the three native source-projection inputs and their
routing nodes. Removing all remaining route nodes requires the separate tracked graph described in
opportunity 5.

The r24 core/vision AARs and ARM64 JNI compiled and packaged successfully with one Bazel worker.
No Android app compile, Gradle task, test run, device run, or performance profile was performed.
The expected benefit therefore remains a hypothesis pending the evidence listed below.

## Ranked opportunities

### 1. Dedicated prepared-tensor session - implemented in r24

Create a steady-state entry point for tracked frames that accepts only the prepared tensor,
timestamp, and tracking generation/reset state. Android should choose between this entry point and
the existing full-frame acquisition path.

The current prepared-tensor call still enters the general Tasks video API. It constructs a packet
map containing a placeholder image, normalized rectangle, external pose rectangle, source
projection rectangle, source dimensions, route flags, reset state, and the tensor. The graph then
uses `PoseTensorInputCalculator` and merge nodes to select prepared input and emit identity matrix
and zero-letterbox metadata.

A narrow tracked entry point can remove those placeholder packets and routing calculators while
leaving inference and output decoding unchanged. The expected benefit is lower CPU work,
allocation pressure, and tail-latency jitter; average model inference time should not materially
change.

r24 delivers the narrow entry point as one slot-acquire JNI call and one native submit JNI call.
It removes per-frame Java packet/map construction and caches the placeholder metadata payloads,
while the native submit re-timestamps them for the existing graph. With app-owned projection, the
source-projection trio is absent from graph construction. The remaining image/rectangle/tensor
route nodes are intentionally retained until opportunity 5 so acquisition behavior is unchanged.

### 2. Flat leased result buffer

Return the fixed single-pose result into an app-owned pooled direct buffer instead of building the
current protobuf-to-Java object graph. The buffer should carry:

- pose-present state and score;
- 33 normalized image landmarks with confidence fields;
- 33 world landmarks;
- auxiliary tracking landmarks or the crop-local next rectangle;
- detector-ran/reset telemetry needed by recovery.

The lease must remain unique to one accepted frame until every downstream consumer finishes.
Reusing a singleton buffer would corrupt concurrently processed or retained results. A small ring
owned by each exclusively leased pose resource is the appropriate lifetime boundary.

Android can project the flat image/world values through the frozen `PoseImageRegion` in one pass.
This removes `PoseLandmarkerResult`, nested immutable lists, per-landmark Java value objects,
`Optional` access, and the second layer of app wrapper objects.

### 3. Persistent input tensor slots - implemented in r24

The CPU route currently allocates a `CpuPoseTensor` packet/view for every tracked frame. The GPU
route retains its RGB float SSBO but creates a native tensor wrapper and performs one device copy
per submission.

Use two or more tensor slots per landmarker so a completed frame can release one slot while the
next frame is prepared. Reuse is legal only after graph consumption and GPU dependencies are
complete. A failed or indeterminate native submission must poison its slot until the detector is
drained and rebuilt.

This can be delivered as part of the dedicated prepared-tensor session. Direct delegate buffer
binding may eventually remove the remaining GPU device copy, but that is native delegate work,
not a Kotlin-side optimization.

r24 uses two slots per session. CPU slots expose stable native-order direct buffers; GPU slots
expose graph-owned shared-context SSBO names, so the app shader writes the tensor storage directly
and r23's device copy is gone. A foreign-packet cleanup returns an in-flight slot to free; a
failure at or after submit poisons it. Sessions are destroyed only after graph teardown. The r23
per-frame route remains the bounded runtime fallback and `PREPARED_SESSION_ENABLED = false` source
rollback.

### 4. Construct the next ROI in Android

MediaPipe currently converts auxiliary landmarks to a detection, aligns a rectangle, then applies
the pose rectangle transformation. Android already owns `ExternalPoseRoiTracker` and composes the
returned crop-local rectangle into source space.

After flat result transport is available, return the auxiliary tracking landmarks and reproduce
the three-calculator rectangle operation in Android. Require exact parity for center, extent,
rotation, expansion, non-square sources, and off-image regions before removing the native result.
This is primarily an ownership cleanup and should be expected to save less time than eliminating
the packet and result-object boundaries.

### 5. Stateless tracked landmark graph

Once Android owns route selection, tracking generations, ROI publication, and reacquisition, the
steady-state landmark graph can be stateless. The existing detector graph remains responsible for
cold acquisition and pose recovery; the tracked graph performs only landmark inference and
decoding for the supplied tensor.

This removes duplicate video-loopback/reset machinery but changes missing-pose and reacquisition
semantics. It should follow the lower-risk transport and result-buffer work.

## Work that should remain native

Keep the following in MediaPipe/native code unless profiling identifies a specific replacement
with output parity:

- delegate setup and model invocation;
- pose-presence thresholding and early gating;
- landmark activation and tensor decoding;
- heatmap refinement;
- world-landmark decoding and visibility propagation;
- full-frame pose detector acquisition.

Moving these operations to Kotlin would expose large intermediate tensors across JNI and duplicate
model-specific numerical behavior. Full-frame acquisition also runs less frequently than tracked
landmark inference, so it is a lower-value extraction target.

## Reliability work at the same boundary

Two existing native recovery gaps remain relevant:

1. An Android coroutine timeout cannot interrupt a synchronous JNI call that is hung in the native
   graph. A cancellation/abort boundary must be implemented in the native session or isolated
   behind a process boundary.
2. A runtime failure after zero-copy CPU tensor binding needs an explicit copying fallback policy.
   Initialization already permits one copying retry; a live-session failure must not silently
   reuse memory whose ownership is uncertain.

These do not move compute to Android, but they let Android's detector supervisor make a real
recovery decision instead of merely timing out its waiter.

## Recommended delivery sequence

1. **Implemented in r24:** introduce a `PreparedPoseSession` behind the existing detector adapter
   and keep the r23 Tasks entry point as a source-level and runtime rollback.
2. **Implemented at the Java/JNI boundary:** remove per-frame placeholder packet-map construction
   while preserving the existing result representation. Native cached metadata remains until the
   stateless tracked graph removes the remaining route nodes.
3. **Implemented in r24 source:** add pooled input slots with explicit success, cancellation,
   abandonment, submission-failure, poison, and graph-teardown ownership. Execution evidence is
   still pending.
4. Add a leased flat result buffer and compare every landmark/world/confidence value against the
   current `PoseLandmarkerResult` route.
5. Move next-ROI construction only after the flat result and projection parity tests pass.
6. Consider a stateless tracked graph after sustained 30/60 FPS profiling shows the remaining
   graph control machinery is material.

## Required evidence

Correctness comparison must cover CPU and GPU delegates, portrait and landscape sources, rotated
and off-image ROIs, missing pose, five-miss draw hysteresis, seek/reset/source replacement,
reacquisition, cancellation, and complete tensor/EGL/session shutdown.

Performance evidence must include average and tail inference/transport latency, allocation and GC
activity, rendered/janky frames, completed pose jobs, queue/drop counters, thermals, and memory at
30 and 60 FPS. A faster isolated JNI call is not sufficient if rendered frames or completed jobs
regress.

Current evidence is build-only: the r24 Java targets, ARM64 JNI, core AAR, and vision AAR compiled;
the packaging script checked the new JNI exports and OpenCV dependency. Written JVM/device tests
cover routing, reset propagation, CPU/GPU parity, 12-frame slot reuse, close, and the out-of-order
full-frame coordinate-space reset, but none were run. The r24 memory delta is about 1.5 MiB per
storage type (two 768 KiB slots), not including the lazily allocated r23 GPU fallback SSBO.

## Source map

| Concern | Source |
| --- | --- |
| Android route and frozen ROI ownership | [`MediaPipePoseDetectorAdapter.kt`](../app/src/main/java/com/formfocus/app/pose_detection/detector_management/MediaPipePoseDetectorAdapter.kt) |
| Prepared-session routing and r23 fallback | [`PreparedPoseInput.kt`](../app/src/main/java/com/formfocus/app/pose_detection/detector_management/PreparedPoseInput.kt) |
| CPU prepared-tensor packing | [`PoseTensorInput.kt`](../app/src/main/java/com/formfocus/app/pose_detection/detector_management/PoseTensorInput.kt) |
| GPU RGB-float SSBO packing | [`PoseGpuTensorConverter.kt`](../app/src/main/java/com/formfocus/app/pose_detection/detector_management/PoseGpuTensorConverter.kt) |
| Current Java result wrapping | [`MediaPipePoseWrapper.kt`](../app/src/main/java/com/formfocus/app/pose_detection/pose_processing/inheritance/MediaPipePoseWrapper.kt) |
| App-owned projection geometry | [`PoseImageRegion.kt`](../app/src/main/java/com/formfocus/app/pose_detection/detector_management/PoseImageRegion.kt) |
| Ordered app smoothing | [`MediaPipePoseSmoothing.kt`](../app/src/main/java/com/formfocus/app/pose_detection/pose_processing/post_processing/MediaPipePoseSmoothing.kt) |
| Custom AAR build and provenance | [`infra/tools/mediapipe/README.md`](../infra/tools/mediapipe/README.md) |
| Current external-ROI/tensor implementation record | [`mediapipe_external_roi_and_preprocessing_refactor_analysis_2026_11_21.md`](mediapipe_external_roi_and_preprocessing_refactor_analysis_2026_11_21.md) |

Paired-fork implementation paths reviewed with the app:

- `mediapipe/tasks/java/com/google/mediapipe/tasks/vision/poselandmarker/PoseLandmarker.java`
- `mediapipe/tasks/java/com/google/mediapipe/tasks/vision/poselandmarker/PoseLandmarkerResult.java`
- `mediapipe/tasks/cc/vision/pose_landmarker/pose_tensor_input_calculator.cc`
- `mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarks_detector_graph.cc`
