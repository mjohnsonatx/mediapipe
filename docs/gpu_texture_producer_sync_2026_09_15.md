# GPU texture producer synchronization - MediaPipe handoff

## Scope and review

The timing report correctly identifies a missing GPU dependency: waiting for the
capture fence on detector context B does not order work on graph context C.
`nativeCreateGpuImage` previously selected `kNoSync`, so C could read before the
capture completed. This is a correctness defect; it does not by itself establish
that the reported playback hitch was caused by this defect.

This change is confined to MediaPipe. Frame admission, capture timestamps,
post-processing order, Android fallback policy, and application AARs are outside
this phase. The report's separate captured-time ordering concern remains separate.

## Implemented contract

1. The app queues its capture-fence wait on the current detector context B.
2. `PacketCreator.createImage(TextureFrame)` now requests required synchronization
   in `nativeCreateGpuImage`, without changing the Java/JNI signature.
3. `WrapExternalGlTexture(kSync)` creates and flushes a fence on B. It attaches
   that producer token before returning the GPU image.
4. The existing graph `GlTextureView` read path waits on the token on C.
5. The existing consumer tokens return to the capture owner for texture reuse.

The new strict helper checks both the graph context's fence support and the
calling context's GL version. It checks the actual fence handle, so a non-null
token containing a failed/null GL fence cannot masquerade as synchronization.
It never calls `glFinish`, `glClientWaitSync`, or a sleep. The legacy optional
`kMaybeSyncOrFinish` path retains its existing behavior; `kNoSync` remains explicit.

### Failure and ownership

| Failure | Native status |
| --- | --- |
| Missing graph delegate | `InvalidArgument` |
| No current calling context | `FailedPrecondition` |
| Unsupported fences/context version | `Unimplemented` |
| `glFenceSync` returns null | `Internal` |

GPU-image errors include the operation, texture name, dimensions, format, and
underlying reason. JNI converts the status to `MediaPipeException`. Existing JNI
lookup/allocation exceptions are preserved.

Ownership transfers only after synchronization succeeds. On failure the wrapper
does not invoke the texture callback, JNI frees its temporary global references,
and the existing `PoseLandmarker` creation catch releases the Java frame once.
This avoids the old `kSync` failure path's native callback plus Java catch release.
On success, the callback owns the JNI references and releases them as before.

## Native regression

`//mediapipe/gpu:gl_app_texture_support_test` is an EGL C++ test, runnable as an
arm64 executable on a device without installing or rebuilding the application.
It covers:

- Missing calling context and missing graph delegate.
- Unsupported caller version and injected null fence, with ownership retained.
- Cancellation before a graph read, releasing exactly once.
- Explicit no-sync compatibility.
- 64 identifiable captures over two persistent textures, two external EGL
  contexts, and a dedicated MediaPipe graph GL context. The graph copies each
  input into retained output before returning its consumer fence. Pixel readback
  occurs only after all captures/reuses have been queued.

Test-only linker wrappers count the fence, flush, and graph-side server wait, and
reject CPU waits/finishes in submission and graph sampling. These assertions catch
a missing dependency even on a driver that happens to complete rendering early.
Per-pixel dependent shader arithmetic adds GPU work; this is correctness coverage,
not a playback or latency benchmark. Every test destroys its textures, EGL surfaces and external
contexts, and drains the graph's jobs and dedicated GL thread.

## Build and validation

Completed on 2026-09-15:

- Optimized arm64 JNI library and native regression executable built successfully.
- All 7 native tests passed on a Pixel 9 Pro, Android 16, Mali-G715, OpenGL ES 3.2.
  The 64-frame test verified both halves of every retained graph output after
  repeated lane reuse, plus fence/flush/server-wait and no-CPU-wait assertions.
- JNI disassembly: entry `0x556268`, `mov w9, #0x1` at `0x556280`, and
  `str w9, [sp]` at `0x5562a4` confirm required sync reaches the native helper.
- All 13 shared-GL exports required by the app packaging script remain present,
  as does `nativeCreateGpuImage`; the OpenCV shared-library dependency remains.
- Both application AARs still match the r6 manifest at this handoff.

The native library SHA-256 is
`2b3a32c9afc92c78e5a7a408fd21e8dbf802a16fc6028f17310251be632a0438`.
Full source/artifact hashes are in
[the native provenance record](gpu_texture_producer_sync_2026_09_15.provenance.json).
Device log and JUnit XML are at `/tmp/mediapipe-texture-sync-tftvl283`.
The temporary device executable and its directory were removed after the run.

The test uses static linkage to instrument calls inside production objects. This
also avoids duplicate CPU-feature symbols in the NDK's shared-library test path.

Build with the existing fork toolchain (Bazel 7.7.0, NDK 28.2.13676358):

```bash
export ANDROID_HOME=/home/michaeljohnson/Android/Sdk
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/28.2.13676358"
export JAVA_HOME=/opt/android-studio/jbr
bazel build -c opt --config=android_arm64 --strip=always --jobs=2 \
  --repo_env=HERMETIC_PYTHON_VERSION=3.12 \
  //mediapipe/gpu:gl_app_texture_support_test \
  //mediapipe/tasks/java/com/google/mediapipe/tasks/vision:libmediapipe_tasks_jni.so_copy
```

## Android worktree follow-up

The user subsequently supplied Android branch `BUG-texture-MISSING-GPU-BUILD`.
The r7 AAR pair is now integrated there, with 44 targeted unit tests passing and
the instrumentation APK compiled. The user will perform device testing and
playback profiling. The original phase checklist below remains for reference;
application validation and production artifact approval are still pending.
See the Android implementation report at
`PoseDetectionApp/md_files/LIVE_GL_TEXTURE_PRODUCER_SYNC_IMPLEMENTATION_2026_09_15.md`.

Original follow-up checklist:

1. Rebuild/package both custom AARs through
   `tools/mediapipe/build_external_roi_aar.sh`, with the native build enabled.
   Do not reuse the old JNI library through its Java-only shortcut.
2. Review current fork Java source drift against the r6 manifest. Update the
   complete release unit (both AARs and provenance) together. Include the changed
   JNI/GL sources in recorded provenance; old Java/graph hashes alone cannot
   identify this synchronization repair.
3. Verify the packaged arm64 JNI entry point selects required sync and reaches
   the strict fence helper; verify the existing shared-GL JNI exports and OpenCV
   dependency through the packaging workflow.
4. Run the application's direct-texture orientation, lease reuse, landmarker,
   failure/fallback and cancellation tests. Confirm native synchronization errors
   disable texture input for the session, retain RGBA fallback, and preserve
   frame/resource context in Crashlytics.
5. Profile the same 30/60 FPS clips, one/multiple pose lanes, pose plus equipment,
   seek/fullscreen/shutdown, and in-flight fallback. These app-level checks are
   required before claiming the visible hitch is fixed or the app is validated.
