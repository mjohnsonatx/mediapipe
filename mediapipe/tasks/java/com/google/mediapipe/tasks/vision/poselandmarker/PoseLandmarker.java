// Copyright 2023 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.mediapipe.tasks.vision.poselandmarker;

import android.content.Context;
import android.os.ParcelFileDescriptor;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.formats.proto.LandmarkProto.LandmarkList;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;
import com.google.mediapipe.formats.proto.RectProto;
import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
import com.google.mediapipe.framework.AndroidPacketGetter;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.TextureFrame;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.ByteBufferImageBuilder;
import com.google.mediapipe.framework.image.ByteBufferExtractor;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.framework.image.MPImageProperties;
import com.google.mediapipe.tasks.components.containers.Connection;
import com.google.mediapipe.tasks.components.containers.NormalizedRect;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.ErrorListener;
import com.google.mediapipe.tasks.core.OutputHandler;
import com.google.mediapipe.tasks.core.OutputHandler.ResultListener;
import com.google.mediapipe.tasks.core.TaskInfo;
import com.google.mediapipe.tasks.core.TaskOptions;
import com.google.mediapipe.tasks.core.TaskRunner;
import com.google.mediapipe.tasks.core.proto.BaseOptionsProto;
import com.google.mediapipe.tasks.vision.core.BaseVisionTaskApi;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.posedetector.proto.PoseDetectorGraphOptionsProto;
import com.google.mediapipe.tasks.vision.poselandmarker.proto.PoseLandmarkerGraphOptionsProto;
import com.google.mediapipe.tasks.vision.poselandmarker.proto.PoseLandmarksDetectorGraphOptionsProto;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Performs pose landmarks detection on images.
 *
 * <p>This API expects a pre-trained pose landmarks model asset bundle. See <TODO link
 * to the DevSite documentation page>.
 *
 * <ul>
 *   <li>Input image {@link MPImage}
 *       <ul>
 *         <li>The image that pose landmarks detection runs on.
 *       </ul>
 *   <li>Output PoseLandmarkerResult {@link PoseLandmarkerResult}
 *       <ul>
 *         <li>A PoseLandmarkerResult containing pose landmarks.
 *       </ul>
 * </ul>
 */
public final class PoseLandmarker extends BaseVisionTaskApi {
  private static final String TAG = PoseLandmarker.class.getSimpleName();
  private static final String IMAGE_IN_STREAM_NAME = "image_in";
  private static final String NORM_RECT_IN_STREAM_NAME = "norm_rect_in";
  private static final String EXTERNAL_POSE_RECT_IN_STREAM_NAME = "external_pose_rect_in";
  private static final String EXTERNAL_TENSORS_IN_STREAM_NAME = "external_tensors_in";
  private static final String USE_EXTERNAL_POSE_RECT_IN_STREAM_NAME =
      "use_external_pose_rect_in";
  private static final String EXTERNAL_SOURCE_PROJECTION_RECT_IN_STREAM_NAME =
      "external_source_projection_rect_in";
  private static final String EXTERNAL_SOURCE_IMAGE_SIZE_IN_STREAM_NAME =
      "external_source_image_size_in";
  private static final String USE_EXTERNAL_SOURCE_PROJECTION_IN_STREAM_NAME =
      "use_external_source_projection_in";
  private static final String RESET_TRACKING_IN_STREAM_NAME = "reset_tracking_in";
  private static final NormalizedRect FULL_IMAGE_TRACKING_RECT =
      NormalizedRect.create(0.5f, 0.5f, 1.0f, 1.0f, 0.0f);

  private final AtomicBoolean resetTrackingPending = new AtomicBoolean(false);

  @SuppressWarnings("ConstantCaseForConstants")
  private static final List<String> BASE_INPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList("IMAGE:" + IMAGE_IN_STREAM_NAME, "NORM_RECT:" + NORM_RECT_IN_STREAM_NAME));

  private static final int LANDMARKS_OUT_STREAM_INDEX = 0;
  private static final int WORLD_LANDMARKS_OUT_STREAM_INDEX = 1;
  private static final int OMITTED_OUT_STREAM_INDEX = -1;

  /**
   * Layout of the flat single-pose result (r25), in float words. Written by native code into an
   * application-owned array armed with {@link #armFlatPoseResult(float[])}. Every slot without
   * data is NaN; counts and flags are small integers stored exactly as floats.
   *
   * <p>Image landmarks are normalized to the graph input image (the 256x256 crop for prepared
   * tensors). Auxiliary landmarks are the two tracking points MediaPipe turns into the next-frame
   * rectangle (LandmarksToDetection, AlignmentPointsRects, RectTransformation); the native result
   * of that operation is also carried so callers can verify their own construction.
   */
  public static final class FlatPoseResult {
    /** Must equal the native writer's version word at {@link #VERSION_INDEX}. */
    public static final float LAYOUT_VERSION = 1f;

    public static final int VERSION_INDEX = 0;
    /** Bit set of the FLAG_* values below. */
    public static final int FLAGS_INDEX = 1;
    public static final int LANDMARK_COUNT_INDEX = 2;
    public static final int WORLD_COUNT_INDEX = 3;
    public static final int AUXILIARY_COUNT_INDEX = 4;
    /** Poses in the native output; only the first is written. */
    public static final int POSE_COUNT_INDEX = 5;
    /** x_center, y_center, width, height, rotation of MediaPipe's next-frame rectangle. */
    public static final int NATIVE_RECT_INDEX = 6;
    /** Per landmark: x, y, z, visibility, presence (NaN when the model omitted the field). */
    public static final int LANDMARKS_INDEX = 11;
    public static final int LANDMARK_STRIDE = 5;
    public static final int MAX_LANDMARKS = 33;
    /** Per landmark: x, y, z in meters. */
    public static final int WORLD_INDEX = 176;
    public static final int WORLD_STRIDE = 3;
    /** Per auxiliary landmark: x, y normalized to the graph input image. */
    public static final int AUXILIARY_INDEX = 275;
    public static final int AUXILIARY_STRIDE = 2;
    public static final int MAX_AUXILIARY = 2;
    public static final int SIZE = 279;

    public static final int FLAG_POSE_PRESENT = 1;
    public static final int FLAG_POSE_DETECTOR_RAN = 2;
    public static final int FLAG_NATIVE_RECT = 4;
    /**
     * A supplied visibility or presence was NaN. Without it, NaN means the model omitted the
     * field; with it, callers must treat every NaN confidence as supplied (and invalid).
     */
    public static final int FLAG_SUPPLIED_NAN_CONFIDENCE = 8;

    private FlatPoseResult() {}
  }

  /**
   * Caller-confined flat output for one synchronous VIDEO call; see armFlatPoseResult. Created
   * per landmarker (never static) so no static initializer is needed on the result path.
   */
  private static final class FlatResultTarget {
    // Returned instead of a built result when the flat write succeeded. Empty and immutable; the
    // caller learns which happened from disarmFlatPoseResult(), never from this object.
    final PoseLandmarkerResult placeholder =
        PoseLandmarkerResult.create(
            Collections.emptyList(),
            Collections.emptyList(),
            Collections.emptyList(),
            false,
            Optional.empty(),
            -1L);
    // Written by the serialized caller before graph input and read on the graph output thread.
    // Volatile because the Java memory model does not see the native idle barrier as an edge.
    volatile float[] armed;
    // Written on the graph output thread; read by the caller after the synchronous call.
    volatile boolean written;
    volatile RuntimeException failure;
  }

  private static final String TASK_GRAPH_NAME =
      "mediapipe.tasks.vision.pose_landmarker.PoseLandmarkerGraph";

  static {
    System.loadLibrary("mediapipe_tasks_jni");
  }

  /**
   * Creates a {@link PoseLandmarker} instance from a model file and the default {@link
   * PoseLandmarkerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelPath path to the pose landmarks model with metadata in the assets.
   * @throws MediaPipeException if there is an error during {@link PoseLandmarker} creation.
   */
  public static PoseLandmarker createFromFile(Context context, String modelPath) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(modelPath).build();
    return createFromOptions(
        context, PoseLandmarkerOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates a {@link PoseLandmarker} instance from a model file and the default {@link
   * PoseLandmarkerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelFile the pose landmarks model {@link File} instance.
   * @throws IOException if an I/O error occurs when opening the tflite model file.
   * @throws MediaPipeException if there is an error during {@link PoseLandmarker} creation.
   */
  public static PoseLandmarker createFromFile(Context context, File modelFile) throws IOException {
    try (ParcelFileDescriptor descriptor =
        ParcelFileDescriptor.open(modelFile, ParcelFileDescriptor.MODE_READ_ONLY)) {
      BaseOptions baseOptions =
          BaseOptions.builder().setModelAssetFileDescriptor(descriptor.getFd()).build();
      return createFromOptions(
          context, PoseLandmarkerOptions.builder().setBaseOptions(baseOptions).build());
    }
  }

  /**
   * Creates a {@link PoseLandmarker} instance from a model buffer and the default {@link
   * PoseLandmarkerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelBuffer a direct {@link ByteBuffer} or a {@link MappedByteBuffer} of the detection
   *     model.
   * @throws MediaPipeException if there is an error during {@link PoseLandmarker} creation.
   */
  public static PoseLandmarker createFromBuffer(Context context, final ByteBuffer modelBuffer) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetBuffer(modelBuffer).build();
    return createFromOptions(
        context, PoseLandmarkerOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates a {@link PoseLandmarker} instance from a {@link PoseLandmarkerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param landmarkerOptions a {@link PoseLandmarkerOptions} instance.
   * @throws MediaPipeException if there is an error during {@link PoseLandmarker} creation.
   */
  public static PoseLandmarker createFromOptions(
      Context context, PoseLandmarkerOptions landmarkerOptions) {
    return createFromOptionsInternal(context, landmarkerOptions, /* parentGlContext= */ 0L);
  }

  /**
   * Creates a {@link PoseLandmarker} whose GPU resources share application-owned GL objects.
   *
   * <p>The caller must keep {@code parentGlContext} alive until this landmarker is closed. This
   * overload is intended for application-owned {@link TextureFrame} inputs.
   */
  public static PoseLandmarker createFromOptions(
      Context context, PoseLandmarkerOptions landmarkerOptions, long parentGlContext) {
    if (parentGlContext == 0L) {
      throw new IllegalArgumentException("The parent GL context handle must be non-zero.");
    }
    return createFromOptionsInternal(context, landmarkerOptions, parentGlContext);
  }

  private static PoseLandmarker createFromOptionsInternal(
      Context context, PoseLandmarkerOptions landmarkerOptions, long parentGlContext) {
    List<String> inputStreams = new ArrayList<>(BASE_INPUT_STREAMS);
    if (landmarkerOptions.runningMode() == RunningMode.VIDEO) {
      if (landmarkerOptions.appTensorInput().orElse(false)) {
        inputStreams.add("EXTERNAL_TENSORS:" + EXTERNAL_TENSORS_IN_STREAM_NAME);
      }
      inputStreams.add("EXTERNAL_POSE_RECT:" + EXTERNAL_POSE_RECT_IN_STREAM_NAME);
      inputStreams.add(
          "USE_EXTERNAL_POSE_RECT:" + USE_EXTERNAL_POSE_RECT_IN_STREAM_NAME);
      // App-projection callers never use native source projection. Omitting the all-or-nothing
      // trio removes three placeholder packets per frame and their AllowIf/Merge routing nodes.
      if (landmarkerOptions.nativeSourceProjection().orElse(true)) {
        inputStreams.add(
            "EXTERNAL_SOURCE_PROJECTION_RECT:"
                + EXTERNAL_SOURCE_PROJECTION_RECT_IN_STREAM_NAME);
        inputStreams.add(
            "EXTERNAL_SOURCE_IMAGE_SIZE:" + EXTERNAL_SOURCE_IMAGE_SIZE_IN_STREAM_NAME);
        inputStreams.add(
            "USE_EXTERNAL_SOURCE_PROJECTION:"
                + USE_EXTERNAL_SOURCE_PROJECTION_IN_STREAM_NAME);
      }
      if (!landmarkerOptions.statelessTracking().orElse(false)) {
        inputStreams.add("RESET_TRACKING:" + RESET_TRACKING_IN_STREAM_NAME);
      }
    }
    List<String> outputStreams = new ArrayList<>();
    outputStreams.add("NORM_LANDMARKS:pose_landmarks");
    outputStreams.add("WORLD_LANDMARKS:world_landmarks");
    // Synchronous VIDEO returns only the task result. A lightweight bool packet keeps the
    // multi-stream callback live even when every semantic result stream is empty, without retaining
    // the input Image/TextureFrame through output synchronization. IMAGE keeps its established
    // graph surface, and LIVE_STREAM requires image_out for its ResultListener input callback.
    final int imageOutStreamIndex;
    if (landmarkerOptions.runningMode() == RunningMode.VIDEO) {
      imageOutStreamIndex = OMITTED_OUT_STREAM_INDEX;
      outputStreams.add("VIDEO_SYNC:video_sync");
    } else {
      imageOutStreamIndex = outputStreams.size();
      outputStreams.add("IMAGE:image_out");
    }
    final int poseRectsNextFrameOutStreamIndex = outputStreams.size();
    outputStreams.add("POSE_RECTS_NEXT_FRAME:pose_rects_next_frame");
    final int poseDetectionsOutStreamIndex = outputStreams.size();
    outputStreams.add("DETECTIONS:pose_detections");
    final int segmentationMasksOutStreamIndex;
    if (landmarkerOptions.outputSegmentationMasks()) {
      segmentationMasksOutStreamIndex = outputStreams.size();
      outputStreams.add("SEGMENTATION_MASK:segmentation_masks");
    } else {
      segmentationMasksOutStreamIndex = OMITTED_OUT_STREAM_INDEX;
    }
    // Object results also need the auxiliary landmarks for Android-owned next ROI.
    // Append last to preserve the existing stream indices and flat-result ABI.
    final FlatResultTarget flatTarget = landmarkerOptions.flatPoseResult().orElse(false)
        ? new FlatResultTarget() : null;
    final int auxiliaryLandmarksOutStreamIndex = outputStreams.size();
    outputStreams.add("AUXILIARY_LANDMARKS:auxiliary_landmarks");

    // TODO: Consolidate OutputHandler and TaskRunner.
    OutputHandler<PoseLandmarkerResult, MPImage> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<PoseLandmarkerResult, MPImage>() {
          @Override
          public PoseLandmarkerResult convertToTaskResult(List<Packet> packets) {
            if (flatTarget != null) {
              float[] flat = flatTarget.armed;
              if (flat != null) {
                try {
                  com.google.mediapipe.framework.PacketCreator.writeFlatPoseResult(
                      packets.get(LANDMARKS_OUT_STREAM_INDEX),
                      packets.get(WORLD_LANDMARKS_OUT_STREAM_INDEX),
                      packets.get(auxiliaryLandmarksOutStreamIndex),
                      packets.get(poseRectsNextFrameOutStreamIndex),
                      !packets.get(poseDetectionsOutStreamIndex).isEmpty(),
                      flat);
                  if (flat[FlatPoseResult.VERSION_INDEX] != FlatPoseResult.LAYOUT_VERSION) {
                    throw new IllegalStateException(
                        "Flat pose result layout mismatch: native="
                            + flat[FlatPoseResult.VERSION_INDEX]
                            + " java="
                            + FlatPoseResult.LAYOUT_VERSION
                            + ". Rebuild the core and vision AARs together.");
                  }
                  flatTarget.written = true;
                  return flatTarget.placeholder;
                } catch (RuntimeException failure) {
                  // Backup plan: this frame is still delivered, as objects built below. The
                  // caller reads the cause from takeFlatPoseResultFailure() and decides whether
                  // to keep using the flat route. Never rethrown here: OutputHandler only caches
                  // MediaPipeException, and the graph thread must not see an app error.
                  flatTarget.failure = failure;
                }
              }
            }
            // If there is no poses detected in the image, just returns empty lists.
            if (packets.get(LANDMARKS_OUT_STREAM_INDEX).isEmpty()) {
              return PoseLandmarkerResult.create(
                  new ArrayList<>(),
                  new ArrayList<>(),
                  new ArrayList<>(),
                  !packets.get(poseDetectionsOutStreamIndex).isEmpty(),
                  Optional.empty(),
                  BaseVisionTaskApi.generateResultTimestampMs(
                      landmarkerOptions.runningMode(), packets.get(LANDMARKS_OUT_STREAM_INDEX)));
            }
            /** Get segmentation masks */
            Optional<List<MPImage>> segmentedMasks = Optional.empty();
            if (landmarkerOptions.outputSegmentationMasks()) {
              segmentedMasks = getSegmentationMasks(packets, segmentationMasksOutStreamIndex);
            }

            return PoseLandmarkerResult.create(
                PacketGetter.getProtoVector(
                    packets.get(LANDMARKS_OUT_STREAM_INDEX), NormalizedLandmarkList.parser()),
                PacketGetter.getProtoVector(
                    packets.get(WORLD_LANDMARKS_OUT_STREAM_INDEX), LandmarkList.parser()),
                packets.get(poseRectsNextFrameOutStreamIndex).isEmpty()
                    ? new ArrayList<>()
                    : PacketGetter.getProtoVector(
                        packets.get(poseRectsNextFrameOutStreamIndex),
                        RectProto.NormalizedRect.parser()),
                !packets.get(poseDetectionsOutStreamIndex).isEmpty(),
                segmentedMasks,
                packets.get(auxiliaryLandmarksOutStreamIndex).isEmpty()
                    ? Collections.emptyList()
                    : PacketGetter.getProtoVector(packets.get(auxiliaryLandmarksOutStreamIndex),
                        NormalizedLandmarkList.parser()),
                BaseVisionTaskApi.generateResultTimestampMs(
                    landmarkerOptions.runningMode(), packets.get(LANDMARKS_OUT_STREAM_INDEX)));
          }

          @Override
          public MPImage convertToTaskInput(List<Packet> packets) {
            if (imageOutStreamIndex == OMITTED_OUT_STREAM_INDEX) {
              throw new IllegalStateException(
                  "Synchronous VIDEO mode does not expose the pass-through input image.");
            }
            return new BitmapImageBuilder(
                    AndroidPacketGetter.getBitmap(packets.get(imageOutStreamIndex)))
                .build();
          }
        });
    landmarkerOptions.resultListener().ifPresent(handler::setResultListener);
    landmarkerOptions.errorListener().ifPresent(handler::setErrorListener);
    TaskRunner runner =
        TaskRunner.create(
            context,
            TaskInfo.<PoseLandmarkerOptions>builder()
                .setTaskName(PoseLandmarker.class.getSimpleName())
                .setTaskRunningModeName(landmarkerOptions.runningMode().name())
                .setTaskGraphName(landmarkerOptions.statelessTracking().orElse(false)
                    ? "mediapipe.tasks.vision.pose_landmarker.StatelessPoseLandmarkerGraph"
                    : TASK_GRAPH_NAME)
                .setInputStreams(inputStreams)
                .setOutputStreams(outputStreams)
                .setTaskOptions(landmarkerOptions)
                .setEnableFlowLimiting(landmarkerOptions.runningMode() == RunningMode.LIVE_STREAM)
                .build(),
            handler,
            parentGlContext);
    return new PoseLandmarker(runner, landmarkerOptions.runningMode(),
        landmarkerOptions.borrowedImageInput().orElse(false),
        landmarkerOptions.appTensorInput().orElse(false),
        landmarkerOptions.runningMode() == RunningMode.VIDEO
            && landmarkerOptions.nativeSourceProjection().orElse(true),
        flatTarget, landmarkerOptions.statelessTracking().orElse(false));
  }

  @SuppressWarnings("ConstantCaseForConstants")
  public static final Set<Connection> POSE_LANDMARKS = PoseLandmarksConnections.POSE_LANDMARKS;

  /**
   * Constructor to initialize a {@link PoseLandmarker} from a {@link TaskRunner} and a {@link
   * RunningMode}.
   *
   * @param taskRunner a {@link TaskRunner}.
   * @param runningMode a mediapipe vision task {@link RunningMode}.
   */
  private final boolean borrowedImageInput;
  private final boolean appTensorInput;
  // True when the VIDEO graph connects the external source projection input trio.
  private final boolean nativeSourceProjection;
  private final boolean statelessTracking;

  public boolean isStatelessTracking() { return statelessTracking; }
  // Confined to the synchronous VIDEO caller; close follows the caller's inference drain.
  private Packet tensorImage;
  private Packet emptyTensorVector;
  private final java.util.concurrent.atomic.AtomicInteger borrowedImages =
      new java.util.concurrent.atomic.AtomicInteger();
  private final Runnable borrowedImageReleased = () -> borrowedImages.set(0);
  // Sessions opened on this landmarker. Graph packets may reference their slots until graph
  // teardown, so close() frees them only after the native graph is released.
  private final List<PreparedPoseSession> preparedSessions = new ArrayList<>();
  // Non-null only when constructed with setFlatPoseResult(true).
  private final FlatResultTarget flatTarget;

  private PoseLandmarker(TaskRunner taskRunner, RunningMode runningMode, boolean borrowedImageInput,
      boolean appTensorInput, boolean nativeSourceProjection, FlatResultTarget flatTarget, boolean statelessTracking) {
    super(taskRunner, runningMode, IMAGE_IN_STREAM_NAME, NORM_RECT_IN_STREAM_NAME);
    this.borrowedImageInput = borrowedImageInput;
    this.appTensorInput = appTensorInput;
    this.nativeSourceProjection = nativeSourceProjection;
    this.flatTarget = flatTarget;
    this.statelessTracking = statelessTracking;
  }

  /**
   * Directs the result of the next synchronous VIDEO call on this landmarker into {@code target},
   * laid out as {@link FlatPoseResult}. Applies to every VIDEO entry point (image, texture, app
   * tensor and {@link PreparedPoseSession}), so one result representation serves tracking and
   * acquisition alike.
   *
   * <p>Pair every arm with {@link #disarmFlatPoseResult()} in a finally block. While armed, a
   * successful flat write makes the detect method return an empty placeholder that must be
   * ignored. If the native write fails, that call still returns an ordinary {@link
   * PoseLandmarkerResult}, and {@link #takeFlatPoseResultFailure()} reports why. Same serialized
   * owner contract as detectForVideo; the array must not be read or reused until disarmed.
   *
   * @throws IllegalStateException if flat results were not enabled at construction, or a
   *     previous arm was never disarmed (a missing finally would otherwise redirect a later
   *     caller's result into a stale array).
   */
  public void armFlatPoseResult(float[] target) {
    if (flatTarget == null) {
      throw new IllegalStateException("Enable flat pose results at construction.");
    }
    if (target == null || target.length < FlatPoseResult.SIZE) {
      throw new IllegalArgumentException(
          "Flat pose result needs " + FlatPoseResult.SIZE + " floats.");
    }
    if (flatTarget.armed != null) {
      throw new IllegalStateException("A flat pose result is already armed; disarm it first.");
    }
    flatTarget.written = false;
    flatTarget.failure = null;
    flatTarget.armed = target;
  }

  /**
   * Ends the armed call. Returns true when its result was written into the armed array; false
   * when the call returned an ordinary result (no synchronized output, or the flat write failed:
   * see {@link #takeFlatPoseResultFailure()}). Idempotent; never throws.
   */
  public boolean disarmFlatPoseResult() {
    if (flatTarget == null) return false;
    flatTarget.armed = null;
    boolean written = flatTarget.written;
    flatTarget.written = false;
    return written;
  }

  /** Returns and clears the native flat-write failure of the last armed call, or null. */
  public RuntimeException takeFlatPoseResultFailure() {
    if (flatTarget == null) return null;
    RuntimeException failure = flatTarget.failure;
    flatTarget.failure = null;
    return failure;
  }

  /**
   * One native-owned CPU input. Fill all 256x256x3 RGB floats in [0,1], then submit exactly once.
   * Do not write during/after submission or access the buffer after close. The caller closes this
   * handle in finally after the synchronous call, before closing the landmarker; native packets
   * retain their own ownership. Like detectForVideo, this API requires serialized owner access.
   */
  public static final class CpuPoseTensor implements AutoCloseable {
    private final PoseLandmarker owner;
    private final java.nio.ByteBuffer buffer;
    private Packet packet;
    private boolean submitted;

    private CpuPoseTensor(PoseLandmarker owner, Packet packet, java.nio.ByteBuffer buffer) {
      this.owner = owner;
      this.packet = packet;
      this.buffer = buffer;
    }

    public java.nio.ByteBuffer buffer() {
      if (packet == null || submitted) throw new IllegalStateException("Pose tensor is not writable.");
      return buffer;
    }

    @Override public void close() {
      if (packet != null) { packet.release(); packet = null; }
    }
  }

  /** Allocates native input storage; the application writes directly, with no RGB buffer copy. */
  public CpuPoseTensor createCpuPoseTensor() {
    requireAppTensorInput();
    Packet packet = runner.getPacketCreator().createCpuPoseTensor();
    try {
      return new CpuPoseTensor(this, packet, runner.getPacketCreator().getCpuPoseTensorBuffer(packet));
    } catch (RuntimeException | Error failure) {
      packet.release();
      throw failure;
    }
  }

  /** Skips landmark image preprocessing; returns crop-local results for the app's frozen ROI. */
  public PoseLandmarkerResult detectForVideoWithTensor(CpuPoseTensor tensor, long timestampMs) {
    requireAppTensorInput();
    if (tensor.owner != this || tensor.packet == null || tensor.submitted) {
      throw new IllegalArgumentException("Pose tensor must be fresh and owned by this landmarker.");
    }
    tensor.submitted = true;
    return processPreparedPoseTensor(tensor.packet.copy(), timestampMs);
  }

  /**
   * GPU counterpart: accepts an app-normalized FLOAT32 RGB SSBO. The native bridge performs one
   * device-to-device copy and GPU fences, never CPU readback. The caller keeps SSBO and texture
   * ownership and must retire the SSBO if this call fails before reuse ordering can be confirmed.
   */
  public PoseLandmarkerResult detectForVideoWithTensor(int rgbSsbo, long timestampMs) {
    requireAppTensorInput();
    return processPreparedPoseTensor(runner.getPacketCreator().createPoseTensorFromGlBuffer(rgbSsbo),
        timestampMs);
  }

  private void requireAppTensorInput() {
    if (!appTensorInput) throw new IllegalStateException("Enable app tensor input at construction.");
  }

  private void requireNativeSourceProjection() {
    if (!nativeSourceProjection) {
      throw new IllegalStateException(
          "Native source projection was disabled when this landmarker was created.");
    }
  }

  private PoseLandmarkerResult processPreparedPoseTensor(Packet tensors, long timestampMs) {
    try {
      if (tensorImage == null) tensorImage = runner.getPacketCreator().createPoseTensorImage();
      return processVideoPackets(tensorImage.copy(), FULL_IMAGE_TRACKING_RECT.toProto(),
          Optional.of(FULL_IMAGE_TRACKING_RECT), Optional.empty(), 256, 256, timestampMs, tensors);
    } finally {
      tensors.release();
    }
  }

  /**
   * Dedicated tracked-frame entry point: persistent prepared-tensor slots and a narrow
   * submission that takes only a slot, a timestamp and the tracking-reset state.
   *
   * <p>Per frame: {@link #acquireSlot()}; fill {@link #cpuBuffer(int)} with 256x256x3 RGB floats
   * in [0,1] (CPU storage) or write {@link #glBuffer(int)} on a context sharing this graph's
   * parent context (GPU storage); then {@link #detect(int, long, boolean)}. An acquired slot that
   * is not submitted must be returned with {@link #abandonSlot(int)}.
   *
   * <p>Native code caches the full-image metadata packets that the Tasks entry point rebuilds
   * every frame, so the graph receives the same inputs and returns the same crop-local {@link
   * PoseLandmarkerResult}. A slot returns to the pool only after the graph drops its last
   * reference; a failure at or after submission poisons it until the landmarker is rebuilt.
   *
   * <p>Not thread-safe: the owner serializes every call, exactly as for detectForVideo. Buffers
   * and SSBO names are valid until this landmarker closes and must not be touched afterwards.
   */
  public static final class PreparedPoseSession {
    /** {@link #acquireSlot()} result: every usable slot is still referenced by the graph. */
    public static final int SLOT_BUSY = -1;
    /** {@link #acquireSlot()} result: every slot is poisoned; rebuild the landmarker. */
    public static final int SLOTS_EXHAUSTED = -2;
    /** Two slots let a completed frame release one while the next frame is prepared. */
    public static final int SLOT_COUNT = 2;

    private final PoseLandmarker owner;
    private final boolean gpu;
    private final ByteBuffer[] cpuBuffers;
    private final int[] glBuffers;
    // Zeroed by the owner's close(); every native call goes through requireOpen().
    private volatile long handle;
    // Arguments for the native submitter, valid only inside detect() under the runner lock.
    private int pendingSlot;
    private boolean pendingReset;
    private boolean submitAttempted;
    private final TaskRunner.NativeInputSubmitter submitter = this::submitPendingFrame;

    private PreparedPoseSession(PoseLandmarker owner, long handle, boolean gpu,
        ByteBuffer[] cpuBuffers, int[] glBuffers) {
      this.owner = owner;
      this.handle = handle;
      this.gpu = gpu;
      this.cpuBuffers = cpuBuffers;
      this.glBuffers = glBuffers;
    }

    /** True for SSBO slots written on the GPU; false for CPU ByteBuffer slots. */
    public boolean isGpu() {
      return gpu;
    }

    /**
     * Returns a writable slot index, {@link #SLOT_BUSY} while every usable slot is in flight, or
     * {@link #SLOTS_EXHAUSTED} once every slot is poisoned.
     */
    public int acquireSlot() {
      return owner.packets().acquirePreparedPoseSlot(requireOpen());
    }

    /** Persistent native-order RGB float storage for one CPU slot. */
    public ByteBuffer cpuBuffer(int slot) {
      requireOpen();
      if (gpu) throw new IllegalStateException("GPU prepared pose session has no CPU slots.");
      return cpuBuffers[slot];
    }

    /** Persistent SSBO name for one GPU slot (std430 float RGB, 256x256x3). */
    public int glBuffer(int slot) {
      requireOpen();
      if (!gpu) throw new IllegalStateException("CPU prepared pose session has no GPU slots.");
      return glBuffers[slot];
    }

    /** Returns an acquired but unsubmitted slot to the pool; a submitted slot is unaffected. */
    public void abandonSlot(int slot) {
      owner.packets().releasePreparedPoseSlot(requireOpen(), slot, /* poison= */ false);
    }

    /**
     * Submits an acquired, fully written slot and waits for the synchronized result. A reset
     * armed through {@link PoseLandmarker#resetTracking()} is also consumed here, so both entry
     * points share one reset contract. GPU storage must be called with the writing context
     * current; native code fences that write without a CPU wait.
     */
    public PoseLandmarkerResult detect(int slot, long timestampMs) {
      return detect(slot, timestampMs, false);
    }

    public PoseLandmarkerResult detect(int slot, long timestampMs, boolean resetTracking) {
      if (owner.statelessTracking && resetTracking) {
        abandonSlot(slot);
        throw new IllegalStateException("Stateless pose session cannot reset tracking.");
      }
      long session = requireOpen();
      boolean reset = !owner.statelessTracking &&
          (owner.resetTrackingPending.getAndSet(false) || resetTracking);
      pendingSlot = slot;
      pendingReset = reset;
      submitAttempted = false;
      try {
        return owner.processPreparedSessionFrame(submitter, timestampMs);
      } catch (RuntimeException | Error failure) {
        // Before submission the slot is unreferenced and returns to the pool. At or after
        // submission the graph may still reference it or have a GPU read pending.
        try {
          owner.packets().releasePreparedPoseSlot(session, slot, submitAttempted);
        } catch (RuntimeException | Error cleanup) {
          failure.addSuppressed(cleanup);
        }
        throw failure;
      }
    }

    private void submitPendingFrame(long inputTimestamp) {
      submitAttempted = true;
      owner.packets().submitPreparedPoseFrame(
          requireOpen(), pendingSlot, inputTimestamp, pendingReset);
    }

    private long requireOpen() {
      long current = handle;
      if (current == 0) throw new IllegalStateException("Prepared pose session is closed.");
      return current;
    }

    /** Owner-only: closes the Java side and returns the native handle to free, or zero. */
    private long detach() {
      long current = handle;
      handle = 0;
      return current;
    }
  }

  private com.google.mediapipe.framework.AndroidPacketCreator packets() {
    return runner.getPacketCreator();
  }

  /** Same synchronous VIDEO contract as processVideoPackets(), with natively cached inputs. */
  private PoseLandmarkerResult processPreparedSessionFrame(
      TaskRunner.NativeInputSubmitter submitter, long timestampMs) {
    if (runningMode != RunningMode.VIDEO) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Prepared pose sessions require the video running mode.");
    }
    PoseLandmarkerResult result = (PoseLandmarkerResult) runner.processNativeInputs(
        submitter, timestampMs * MICROSECONDS_PER_MILLISECOND);
    if (result == null) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INTERNAL.ordinal(),
          "Pose landmarker graph produced no synchronized output for prepared timestampMs="
              + timestampMs);
    }
    return result;
  }

  /**
   * Opens a dedicated tracked-frame session with {@link PreparedPoseSession#SLOT_COUNT}
   * persistent input slots. GPU storage requires a landmarker created with a parent GL context,
   * and the caller's current context must share that parent. Sessions live until {@link
   * #close()}, which frees them only after graph teardown.
   */
  public synchronized PreparedPoseSession openPreparedPoseSession(boolean gpuStorage) {
    requireAppTensorInput();
    if (runner.isGraphReleased()) {
      throw new IllegalStateException("Cannot open a prepared pose session on a closed landmarker.");
    }
    com.google.mediapipe.framework.AndroidPacketCreator creator = packets();
    // Exactly the inputs processPreparedPoseTensor() sends, minus the tensor and reset packets
    // that change every frame. Order: [tensor, reset, metadata...] paired with metadata packets.
    List<String> streams = new ArrayList<>();
    List<Packet> metadata = new ArrayList<>();
    long handle = 0;
    try {
      streams.add(EXTERNAL_TENSORS_IN_STREAM_NAME);
      streams.add(statelessTracking ? "" : RESET_TRACKING_IN_STREAM_NAME);
      streams.add(IMAGE_IN_STREAM_NAME);
      metadata.add(creator.createPoseTensorImage());
      streams.add(NORM_RECT_IN_STREAM_NAME);
      metadata.add(creator.createProto(FULL_IMAGE_TRACKING_RECT.toProto()));
      streams.add(EXTERNAL_POSE_RECT_IN_STREAM_NAME);
      metadata.add(creator.createProto(FULL_IMAGE_TRACKING_RECT.toProto()));
      streams.add(USE_EXTERNAL_POSE_RECT_IN_STREAM_NAME);
      metadata.add(creator.createBool(true));
      if (nativeSourceProjection) {
        streams.add(EXTERNAL_SOURCE_PROJECTION_RECT_IN_STREAM_NAME);
        metadata.add(creator.createProto(FULL_IMAGE_TRACKING_RECT.toProto()));
        streams.add(EXTERNAL_SOURCE_IMAGE_SIZE_IN_STREAM_NAME);
        metadata.add(creator.createInt32Pair(256, 256));
        streams.add(USE_EXTERNAL_SOURCE_PROJECTION_IN_STREAM_NAME);
        metadata.add(creator.createBool(false));
      }
      handle = creator.createPreparedPoseSession(gpuStorage, PreparedPoseSession.SLOT_COUNT,
          streams.toArray(new String[0]), metadata.toArray(new Packet[0]));
      ByteBuffer[] cpuBuffers = gpuStorage ? null : new ByteBuffer[PreparedPoseSession.SLOT_COUNT];
      int[] glBuffers = gpuStorage ? new int[PreparedPoseSession.SLOT_COUNT] : null;
      for (int slot = 0; slot < PreparedPoseSession.SLOT_COUNT; ++slot) {
        if (gpuStorage) {
          glBuffers[slot] = creator.getPreparedPoseGlBuffer(handle, slot);
        } else {
          cpuBuffers[slot] = creator.getPreparedPoseCpuBuffer(handle, slot);
        }
      }
      PreparedPoseSession session =
          new PreparedPoseSession(this, handle, gpuStorage, cpuBuffers, glBuffers);
      preparedSessions.add(session);
      handle = 0; // Ownership moved to close().
      return session;
    } finally {
      // No slot was submitted yet, so a failed open can free its native session immediately.
      if (handle != 0) {
        creator.destroyPreparedPoseSession(handle);
      }
      for (Packet packet : metadata) {
        packet.release();
      }
    }
  }

  @Override public void close() {
    // Packet handles belong to Graph's registry and cannot be released after teardown. Any
    // in-flight graph packets hold independent references to the immutable payloads.
    if (tensorImage != null) { tensorImage.release(); tensorImage = null; }
    if (emptyTensorVector != null) { emptyTensorVector.release(); emptyTensorVector = null; }
    try {
      super.close();
    } finally {
      releasePreparedSessions();
    }
  }

  private synchronized void releasePreparedSessions() {
    if (preparedSessions.isEmpty()) return;
    // Slot packets can stay referenced until graph teardown releases them; freeing a session
    // earlier would leave dangling foreign packets. If teardown did not complete, leak the
    // sessions (a few MB) rather than risk a use-after-free.
    boolean graphReleased = runner.isGraphReleased();
    for (PreparedPoseSession session : preparedSessions) {
      long handle = session.detach();
      if (handle != 0 && graphReleased) {
        packets().destroyPreparedPoseSession(handle);
      }
    }
    if (!graphReleased) {
      android.util.Log.w(TAG, "Graph teardown incomplete; leaked " + preparedSessions.size()
          + " prepared pose session(s) to avoid freeing slots the graph may still reference.");
    }
    preparedSessions.clear();
  }

  /**
   * Performs pose landmarks detection on the provided single image with default image processing
   * options, i.e. without any rotation applied. Only use this method when the {@link
   * PoseLandmarker} is created with {@link RunningMode.IMAGE}. TODO update java doc
   * for input image format.
   *
   * <p>{@link PoseLandmarker} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @throws MediaPipeException if there is an internal error.
   */
  public PoseLandmarkerResult detect(MPImage image) {
    return detect(image, ImageProcessingOptions.builder().build());
  }

  /**
   * Performs pose landmarks detection on the provided single image. Only use this method when the
   * {@link PoseLandmarker} is created with {@link RunningMode.IMAGE}. TODO update java
   * doc for input image format.
   *
   * <p>{@link PoseLandmarker} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference. Note that region-of-interest is <b>not</b> supported
   *     by this task: specifying {@link ImageProcessingOptions#regionOfInterest()} will result in
   *     this method throwing an IllegalArgumentException.
   * @throws IllegalArgumentException if the {@link ImageProcessingOptions} specify a
   *     region-of-interest.
   * @throws MediaPipeException if there is an internal error.
   */
  public PoseLandmarkerResult detect(MPImage image, ImageProcessingOptions imageProcessingOptions) {
    validateImageProcessingOptions(imageProcessingOptions);
    return (PoseLandmarkerResult) processImageData(image, imageProcessingOptions);
  }

  /**
   * Performs pose landmarks detection on the provided video frame with default image processing
   * options, i.e. without any rotation applied. Only use this method when the {@link
   * PoseLandmarker} is created with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame's timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link PoseLandmarker} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public PoseLandmarkerResult detectForVideo(MPImage image, long timestampMs) {
    return detectForVideo(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  /**
   * Clears the retained pose rectangle on the next video frame while keeping this landmarker and
   * its loaded model alive.
   *
   * <p>Use this at an input discontinuity such as a seek or source replacement, before submitting
   * the first frame from the new sequence. The next frame bypasses the previous-frame rectangle and
   * runs pose detection again. Input timestamps must remain monotonically increasing; this method
   * does not restart the graph or its timestamp timeline.
   *
   * @throws MediaPipeException if this task is not in video mode.
   */
  public void resetTracking() {
    if (statelessTracking) {
      throw new IllegalStateException("Stateless pose landmarker cannot reset tracking.");
    }
    if (runningMode != RunningMode.VIDEO) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Pose tracking reset requires video mode. Current running mode:" + runningMode.name());
    }
    resetTrackingPending.set(true);
  }

  /**
   * Performs pose landmarks detection on the provided video frame. Only use this method when the
   * {@link PoseLandmarker} is created with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame's timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link PoseLandmarker} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference. Note that region-of-interest is <b>not</b> supported
   *     by this task: specifying {@link ImageProcessingOptions#regionOfInterest()} will result in
   *     this method throwing an IllegalArgumentException.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws IllegalArgumentException if the {@link ImageProcessingOptions} specify a
   *     region-of-interest.
   * @throws MediaPipeException if there is an internal error.
   */
  public PoseLandmarkerResult detectForVideo(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    validateImageProcessingOptions(imageProcessingOptions);
    return processVideoData(
        image,
        imageProcessingOptions,
        Optional.empty(),
        Optional.empty(),
        image.getWidth(),
        image.getHeight(),
        timestampMs);
  }

  /**
   * Performs pose landmarks detection on a video frame using an externally supplied pose region.
   *
   * <p>The external rectangle bypasses this landmarker instance's retained tracking rectangle and
   * is passed directly to pose landmark inference. A rectangle returned by {@link
   * PoseLandmarkerResult#poseRectsNextFrame()} can be passed without further expansion.
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param externalPoseRect the normalized pose rectangle to use for landmark inference.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws IllegalArgumentException if the external pose rectangle is invalid.
   * @throws MediaPipeException if there is an internal error.
   */
  public PoseLandmarkerResult detectForVideo(
      MPImage image, NormalizedRect externalPoseRect, long timestampMs) {
    return detectForVideo(
        image, ImageProcessingOptions.builder().build(), externalPoseRect, timestampMs);
  }

  /**
   * Performs pose landmarks detection on a video frame using image processing options and an
   * externally supplied pose region.
   *
   * <p>The external rectangle bypasses this landmarker instance's retained tracking rectangle and
   * is passed directly to pose landmark inference. A rectangle returned by {@link
   * PoseLandmarkerResult#poseRectsNextFrame()} can be passed without further expansion.
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param imageProcessingOptions options specifying image rotation. Region-of-interest remains
   *     unsupported because {@code externalPoseRect} is the landmark inference region.
   * @param externalPoseRect the normalized pose rectangle to use for landmark inference.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws IllegalArgumentException if the options or external pose rectangle are invalid.
   * @throws MediaPipeException if there is an internal error.
   */
  public PoseLandmarkerResult detectForVideo(
      MPImage image,
      ImageProcessingOptions imageProcessingOptions,
      NormalizedRect externalPoseRect,
      long timestampMs) {
    validateImageProcessingOptions(imageProcessingOptions);
    validateExternalPoseRect(externalPoseRect);
    return processVideoData(
        image,
        imageProcessingOptions,
        Optional.of(externalPoseRect),
        Optional.empty(),
        image.getWidth(),
        image.getHeight(),
        timestampMs);
  }

  /**
   * Performs tracked pose landmark detection on an image whose ROI was already cropped by the
   * caller.
   *
   * <p>A full-image tracking sentinel keeps the pose detector gate closed while making landmark
   * preprocessing consume the complete supplied image. Returned image landmarks and next-frame
   * rectangles therefore remain in this cropped image's coordinate system and must be remapped by
   * the caller.
   */
  public PoseLandmarkerResult detectForVideoWithExternalRoiCrop(MPImage image, long timestampMs) {
    return processVideoData(
        image,
        ImageProcessingOptions.builder().build(),
        Optional.of(FULL_IMAGE_TRACKING_RECT),
        Optional.empty(),
        image.getWidth(),
        image.getHeight(),
        timestampMs);
  }

  /**
   * Runs on an application-cropped image and returns crop-space pose landmarks, world landmarks,
   * and next-frame rectangles. The caller must project these outputs through the same source ROI
   * used to make the input image before tracking or smoothing them.
   */
  public PoseLandmarkerResult detectForVideoWithAppProjection(MPImage image, long timestampMs) {
    return detectForVideoWithExternalRoiCrop(image, timestampMs);
  }

  /**
   * Performs tracked pose landmark detection on an application-cropped image while preserving
   * MediaPipe's source-space landmark projection and temporal smoothing.
   *
   * <p>The full-image tracking sentinel prevents a second image crop. The source projection
   * rectangle and dimensions are used only after inference, for landmark/world-landmark
   * projection, next-frame ROI construction, and MediaPipe's existing smoothing calculators.
   */
  public PoseLandmarkerResult detectForVideoWithExternalRoiCrop(
      MPImage image,
      NormalizedRect sourceProjectionRect,
      int sourceImageWidth,
      int sourceImageHeight,
      long timestampMs) {
    requireNativeSourceProjection();
    validateExternalPoseRect(sourceProjectionRect);
    validateSourceImageDimensions(sourceImageWidth, sourceImageHeight);
    return processVideoData(
        image,
        ImageProcessingOptions.builder().build(),
        Optional.of(FULL_IMAGE_TRACKING_RECT),
        Optional.of(sourceProjectionRect),
        sourceImageWidth,
        sourceImageHeight,
        timestampMs);
  }

  /** Performs pose landmark detection directly from an application-owned OpenGL texture. */
  public PoseLandmarkerResult detectForVideo(TextureFrame image, long timestampMs) {
    return processVideoTextureData(
        image,
        Optional.empty(),
        Optional.empty(),
        /* sourceImageWidth= */ 0,
        /* sourceImageHeight= */ 0,
        timestampMs);
  }

  /** Performs pose landmark detection from a texture using an externally supplied tracking ROI. */
  public PoseLandmarkerResult detectForVideo(
      TextureFrame image, NormalizedRect externalPoseRect, long timestampMs) {
    validateExternalPoseRect(externalPoseRect);
    return processVideoTextureData(
        image,
        Optional.of(externalPoseRect),
        Optional.empty(),
        /* sourceImageWidth= */ 0,
        /* sourceImageHeight= */ 0,
        timestampMs);
  }

  /**
   * Texture counterpart to {@link #detectForVideoWithExternalRoiCrop(MPImage, long)}.
   *
   * <p>The texture must already contain the rotated ROI expanded to its full dimensions.
   */
  public PoseLandmarkerResult detectForVideoWithExternalRoiCrop(
      TextureFrame image, long timestampMs) {
    return processVideoTextureData(
        image,
        Optional.of(FULL_IMAGE_TRACKING_RECT),
        Optional.empty(),
        /* sourceImageWidth= */ 0,
        /* sourceImageHeight= */ 0,
        timestampMs);
  }

  /** Texture counterpart of {@link #detectForVideoWithAppProjection(MPImage, long)}. */
  public PoseLandmarkerResult detectForVideoWithAppProjection(TextureFrame image, long timestampMs) {
    return detectForVideoWithExternalRoiCrop(image, timestampMs);
  }

  /** Texture counterpart that preserves source-space projection and smoothing. */
  public PoseLandmarkerResult detectForVideoWithExternalRoiCrop(
      TextureFrame image,
      NormalizedRect sourceProjectionRect,
      int sourceImageWidth,
      int sourceImageHeight,
      long timestampMs) {
    requireNativeSourceProjection();
    validateExternalPoseRect(sourceProjectionRect);
    validateSourceImageDimensions(sourceImageWidth, sourceImageHeight);
    return processVideoTextureData(
        image,
        Optional.of(FULL_IMAGE_TRACKING_RECT),
        Optional.of(sourceProjectionRect),
        sourceImageWidth,
        sourceImageHeight,
        timestampMs);
  }

  private PoseLandmarkerResult processVideoData(
      MPImage image,
      ImageProcessingOptions imageProcessingOptions,
      Optional<NormalizedRect> externalPoseRect,
      Optional<NormalizedRect> externalSourceProjectionRect,
      int sourceImageWidth,
      int sourceImageHeight,
      long timestampMs) {
    RectProto.NormalizedRect imageRect = convertToNormalizedRect(imageProcessingOptions, image);
    // Bitmap inputs (offline/benchmark callers) retain their existing copying contract.
    MPImageProperties properties = borrowedImageInput ? image.getContainedImageProperties().get(0) : null;
    boolean borrow = borrowedImageInput
        && properties.getStorageType() == MPImage.STORAGE_TYPE_BYTEBUFFER;
    if (borrow && properties.getImageFormat() != MPImage.IMAGE_FORMAT_RGBA) {
      throw new IllegalArgumentException("Borrowed video input requires RGBA ByteBuffer MPImage.");
    }
    Packet imagePacket;
    if (borrow) {
      if (!borrowedImages.compareAndSet(0, 1)) {
        throw new IllegalStateException("A previous borrowed pose input is still owned by native code.");
      }
      try {
        imagePacket = runner.getPacketCreator().createBorrowedRgbaImage(
            ByteBufferExtractor.extract(image), image.getWidth(), image.getHeight(), borrowedImageReleased);
      } catch (RuntimeException | Error failure) {
        borrowedImages.set(0);
        throw failure;
      }
    } else {
      imagePacket = runner.getPacketCreator().createImage(image);
    }
    Throwable primaryFailure = null;
    try {
      return processVideoPackets(
        imagePacket,
        imageRect,
        externalPoseRect,
        externalSourceProjectionRect,
        sourceImageWidth,
        sourceImageHeight,
        timestampMs);
    } catch (RuntimeException | Error failure) {
      primaryFailure = failure;
      throw failure;
    } finally {
      // Never return a mutable app buffer while native packets still retain it. The graph
      // normally releases the image before its idle barrier. On errors/retention, drain and
      // close this graph before the app's existing stage recovery can recycle the lease.
      if (borrow && borrowedImages.get() != 0) {
        IllegalStateException retained = new IllegalStateException(
            "Borrowed pose input retained after graph idle; closing graph before lease recovery.");
        try {
          runner.close();
        } catch (RuntimeException | Error cleanupFailure) {
          retained.addSuppressed(cleanupFailure);
        }
        if (primaryFailure != null) {
          primaryFailure.addSuppressed(retained);
        } else {
          throw retained;
        }
      }
    }
  }

  private PoseLandmarkerResult processVideoTextureData(
      TextureFrame image,
      Optional<NormalizedRect> externalPoseRect,
      Optional<NormalizedRect> externalSourceProjectionRect,
      int sourceImageWidth,
      int sourceImageHeight,
      long timestampMs) {
    if (image == null) {
      throw new IllegalArgumentException("The GPU texture frame must not be null.");
    }
    if (image.getWidth() <= 0 || image.getHeight() <= 0) {
      image.release();
      throw new IllegalArgumentException("The GPU texture frame must have positive dimensions.");
    }
    if (!externalSourceProjectionRect.isPresent()) {
      sourceImageWidth = image.getWidth();
      sourceImageHeight = image.getHeight();
    }
    RectProto.NormalizedRect fullImageRect = FULL_IMAGE_TRACKING_RECT.toProto();
    Packet imagePacket;
    try {
      imagePacket = runner.getPacketCreator().createImage(image);
    } catch (RuntimeException creationFailure) {
      image.release();
      throw creationFailure;
    }
    return processVideoPackets(
        imagePacket,
        fullImageRect,
        externalPoseRect,
        externalSourceProjectionRect,
        sourceImageWidth,
        sourceImageHeight,
        timestampMs);
  }

  private PoseLandmarkerResult processVideoPackets(
      Packet imagePacket,
      RectProto.NormalizedRect imageRect,
      Optional<NormalizedRect> externalPoseRect,
      Optional<NormalizedRect> externalSourceProjectionRect,
      int sourceImageWidth,
      int sourceImageHeight,
      long timestampMs) {
    return processVideoPackets(imagePacket, imageRect, externalPoseRect,
        externalSourceProjectionRect, sourceImageWidth, sourceImageHeight, timestampMs, null);
  }

  private PoseLandmarkerResult processVideoPackets(
      Packet imagePacket,
      RectProto.NormalizedRect imageRect,
      Optional<NormalizedRect> externalPoseRect,
      Optional<NormalizedRect> externalSourceProjectionRect,
      int sourceImageWidth,
      int sourceImageHeight,
      long timestampMs,
      Packet externalTensors) {
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(IMAGE_IN_STREAM_NAME, imagePacket);
    try {
      if (appTensorInput) {
        if (externalTensors == null) {
          if (emptyTensorVector == null) {
            emptyTensorVector = runner.getPacketCreator().createEmptyPoseTensorVector();
          }
          externalTensors = emptyTensorVector.copy();
        }
        inputPackets.put(EXTERNAL_TENSORS_IN_STREAM_NAME, externalTensors);
      }
      inputPackets.put(
          NORM_RECT_IN_STREAM_NAME, runner.getPacketCreator().createProto(imageRect));
      inputPackets.put(
          EXTERNAL_POSE_RECT_IN_STREAM_NAME,
          runner
              .getPacketCreator()
              .createProto(externalPoseRect.orElse(FULL_IMAGE_TRACKING_RECT).toProto()));
      inputPackets.put(
          USE_EXTERNAL_POSE_RECT_IN_STREAM_NAME,
          runner.getPacketCreator().createBool(externalPoseRect.isPresent()));
      if (nativeSourceProjection) {
        inputPackets.put(
            EXTERNAL_SOURCE_PROJECTION_RECT_IN_STREAM_NAME,
            runner
                .getPacketCreator()
                .createProto(
                    externalSourceProjectionRect.orElse(FULL_IMAGE_TRACKING_RECT).toProto()));
        inputPackets.put(
            EXTERNAL_SOURCE_IMAGE_SIZE_IN_STREAM_NAME,
            runner.getPacketCreator().createInt32Pair(sourceImageWidth, sourceImageHeight));
        inputPackets.put(
            USE_EXTERNAL_SOURCE_PROJECTION_IN_STREAM_NAME,
            runner.getPacketCreator().createBool(externalSourceProjectionRect.isPresent()));
      } else if (externalSourceProjectionRect.isPresent()) {
        // Public overloads reject this first; keep the graph contract explicit for new callers.
        throw new IllegalStateException(
            "Native source projection was disabled when this landmarker was created.");
      }
      if (!statelessTracking) {
        inputPackets.put(RESET_TRACKING_IN_STREAM_NAME,
            runner.getPacketCreator().createBool(resetTrackingPending.getAndSet(false)));
      }
      PoseLandmarkerResult result =
          (PoseLandmarkerResult) super.processVideoData(inputPackets, timestampMs);
      if (result == null) {
        throw new MediaPipeException(
            MediaPipeException.StatusCode.INTERNAL.ordinal(),
            "Pose landmarker graph produced no synchronized output for timestampMs="
                + timestampMs
                + ", externalSourceProjection="
                + externalSourceProjectionRect.isPresent());
      }
      return result;
    } finally {
      // TaskRunner nulls packets transferred into the graph and releases packets it could not
      // transfer. Packet.release() is idempotent, so this also covers failures before addPackets.
      for (Packet packet : inputPackets.values()) {
        if (packet != null) {
          packet.release();
        }
      }
    }
  }

  private static void validateExternalPoseRect(NormalizedRect rect) {
    if (rect == null
        || !isFinite(rect.xCenter())
        || !isFinite(rect.yCenter())
        || !isFinite(rect.width())
        || !isFinite(rect.height())
        || !isFinite(rect.rotationRadians())
        || rect.width() <= 0.0f
        || rect.height() <= 0.0f) {
      throw new IllegalArgumentException(
          "External pose rectangle values must be finite and its width and height must be positive.");
    }
  }

  private static void validateSourceImageDimensions(int width, int height) {
    if (width <= 0 || height <= 0) {
      throw new IllegalArgumentException("Source image dimensions must be positive.");
    }
  }

  private static boolean isFinite(float value) {
    return !Float.isNaN(value) && !Float.isInfinite(value);
  }

  /**
   * Sends live image data to perform pose landmarks detection with default image processing
   * options, i.e. without any rotation applied, and the results will be available via the {@link
   * ResultListener} provided in the {@link PoseLandmarkerOptions}. Only use this method when the
   * {@link PoseLandmarker } is created with {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the pose landmarker. The input timestamps must be monotonically increasing.
   *
   * <p>{@link PoseLandmarker} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public void detectAsync(MPImage image, long timestampMs) {
    detectAsync(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  /**
   * Sends live image data to perform pose landmarks detection, and the results will be available
   * via the {@link ResultListener} provided in the {@link PoseLandmarkerOptions}. Only use this
   * method when the {@link PoseLandmarker} is created with {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the pose landmarker. The input timestamps must be monotonically increasing.
   *
   * <p>{@link PoseLandmarker} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference. Note that region-of-interest is <b>not</b> supported
   *     by this task: specifying {@link ImageProcessingOptions#regionOfInterest()} will result in
   *     this method throwing an IllegalArgumentException.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws IllegalArgumentException if the {@link ImageProcessingOptions} specify a
   *     region-of-interest.
   * @throws MediaPipeException if there is an internal error.
   */
  public void detectAsync(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    validateImageProcessingOptions(imageProcessingOptions);
    sendLiveStreamData(image, imageProcessingOptions, timestampMs);
  }

  /** Options for setting up an {@link PoseLandmarker}. */
  @AutoValue
  public abstract static class PoseLandmarkerOptions extends TaskOptions {

    /** Builder for {@link PoseLandmarkerOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {
      /** Sets the base options for the pose landmarker task. */
      public abstract Builder setBaseOptions(BaseOptions value);

      /**
       * Sets the running mode for the pose landmarker task. Default to the image mode. Pose
       * landmarker has three modes:
       *
       * <ul>
       *   <li>IMAGE: The mode for detecting pose landmarks on single image inputs.
       *   <li>VIDEO: The mode for detecting pose landmarks on the decoded frames of a video.
       *   <li>LIVE_STREAM: The mode for for detecting pose landmarks on a live stream of input
       *       data, such as from camera. In this mode, {@code setResultListener} must be called to
       *       set up a listener to receive the detection results asynchronously.
       * </ul>
       */
      public abstract Builder setRunningMode(RunningMode value);

      /** Sets the maximum number of poses can be detected by the PoseLandmarker. */
      public abstract Builder setNumPoses(Integer value);

      /** Sets minimum confidence score for the pose detection to be considered successful */
      public abstract Builder setMinPoseDetectionConfidence(Float value);

      /** Sets minimum confidence score of pose presence score in the pose landmark detection. */
      public abstract Builder setMinPosePresenceConfidence(Float value);

      /** Sets the minimum confidence score for the pose tracking to be considered successful. */
      public abstract Builder setMinTrackingConfidence(Float value);

      /**
       * Overrides temporal smoothing of pose landmarks and visibility. When unset, smoothing
       * defaults to enabled for a single pose in VIDEO and LIVE_STREAM modes and disabled in
       * IMAGE mode. Smoothing is only supported for a single pose.
       *
       * <p>Set to false when the caller owns temporal filtering across landmarker instances.
       */
      public abstract Builder setSmoothLandmarks(Boolean value);
      /** VIDEO-only opt-in for direct RGBA inputs; bitmap inputs keep the copying path. */
      public abstract Builder setBorrowedImageInput(Boolean value);
      /** Opt-in single-pose VIDEO input; requires the matching prepared-tensor native graph. */
      public abstract Builder setAppTensorInput(Boolean value);
      /**
       * VIDEO graphs connect native source projection by default. Set false when the caller
       * projects crop-local results itself; the overloads taking a source projection rectangle
       * then throw, and every frame omits three placeholder packets and their routing nodes.
       */
      public abstract Builder setNativeSourceProjection(Boolean value);
      /**
       * Single-pose VIDEO opt-in (r25): adds the auxiliary tracking-landmark output and lets
       * {@link PoseLandmarker#armFlatPoseResult(float[])} receive results as a flat float array
       * instead of protobuf-derived Java objects. Unarmed calls keep the object result.
       */
      public abstract Builder setFlatPoseResult(Boolean value);

      /** Input-driven single-pose VIDEO graph with no temporal smoothing or loopback. */
      public abstract Builder setStatelessTracking(Boolean value);

      public abstract Builder setOutputSegmentationMasks(boolean value);

      /**
       * Sets the result listener to receive the detection results asynchronously when the pose
       * landmarker is in the live stream mode.
       */
      public abstract Builder setResultListener(
          ResultListener<PoseLandmarkerResult, MPImage> value);

      /** Sets an optional error listener. */
      public abstract Builder setErrorListener(ErrorListener value);

      abstract PoseLandmarkerOptions autoBuild();

      /**
       * Validates and builds the {@link PoseLandmarkerOptions} instance.
       *
       * @throws IllegalArgumentException if the result listener and the running mode are not
       *     properly configured. The result listener should only be set when the pose landmarker is
       *     in the live stream mode.
       */
      public final PoseLandmarkerOptions build() {
        PoseLandmarkerOptions options = autoBuild();
        if (options.statelessTracking().orElse(false) &&
            (options.runningMode() != RunningMode.VIDEO || options.numPoses().orElse(1) != 1 ||
             options.outputSegmentationMasks())) {
          throw new IllegalArgumentException("Stateless tracking requires single-pose VIDEO without masks.");
        }
        if (options.appTensorInput().orElse(false) &&
            (options.runningMode() != RunningMode.VIDEO || options.numPoses().orElse(1) != 1 ||
             options.outputSegmentationMasks())) {
          throw new IllegalArgumentException(
              "App tensor input requires single-pose VIDEO mode without segmentation masks.");
        }
        if (options.borrowedImageInput().orElse(false) && options.runningMode() != RunningMode.VIDEO) {
          throw new IllegalArgumentException("Borrowed image input requires synchronous VIDEO mode.");
        }
        if (options.flatPoseResult().orElse(false) &&
            (options.runningMode() != RunningMode.VIDEO || options.numPoses().orElse(1) != 1 ||
             options.outputSegmentationMasks())) {
          // The layout holds exactly one pose and no masks; a silent truncation would be a bug.
          throw new IllegalArgumentException(
              "Flat pose results require single-pose VIDEO mode without segmentation masks.");
        }
        if (options.runningMode() == RunningMode.LIVE_STREAM) {
          if (!options.resultListener().isPresent()) {
            throw new IllegalArgumentException(
                "The pose landmarker is in the live stream mode, a user-defined result listener"
                    + " must be provided in PoseLandmarkerOptions.");
          }
        } else if (options.resultListener().isPresent()) {
          throw new IllegalArgumentException(
              "The pose landmarker is in the image or the video mode, a user-defined result"
                  + " listener shouldn't be provided in PoseLandmarkerOptions.");
        }
        return options;
      }
    }

    public abstract BaseOptions baseOptions();

    public abstract Builder toBuilder();

    abstract RunningMode runningMode();

    abstract Optional<Integer> numPoses();

    abstract Optional<Float> minPoseDetectionConfidence();

    abstract Optional<Float> minPosePresenceConfidence();

    abstract Optional<Float> minTrackingConfidence();

    abstract Optional<Boolean> smoothLandmarks();
    abstract Optional<Boolean> borrowedImageInput();
    abstract Optional<Boolean> appTensorInput();
    abstract Optional<Boolean> nativeSourceProjection();
    abstract Optional<Boolean> flatPoseResult();
    public abstract Optional<Boolean> statelessTracking();

    @Override
    public void configureGraph(com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.Builder graph) {
      configureCpuGraph(graph, baseOptions());
    }

    abstract boolean outputSegmentationMasks();

    abstract Optional<ResultListener<PoseLandmarkerResult, MPImage>> resultListener();

    abstract Optional<ErrorListener> errorListener();

    public static Builder builder() {
      return new AutoValue_PoseLandmarker_PoseLandmarkerOptions.Builder()
          .setRunningMode(RunningMode.IMAGE)
          .setNumPoses(1)
          .setMinPoseDetectionConfidence(0.5f)
          .setMinPosePresenceConfidence(0.5f)
          .setMinTrackingConfidence(0.5f)
          .setOutputSegmentationMasks(false);
    }

    /** Converts a {@link PoseLandmarkerOptions} to a {@link CalculatorOptions} protobuf message. */
    @Override
    public CalculatorOptions convertToCalculatorOptionsProto() {
      PoseLandmarkerGraphOptionsProto.PoseLandmarkerGraphOptions.Builder taskOptionsBuilder =
          PoseLandmarkerGraphOptionsProto.PoseLandmarkerGraphOptions.newBuilder()
              .setBaseOptions(
                  BaseOptionsProto.BaseOptions.newBuilder()
                      .setUseStreamMode(runningMode() != RunningMode.IMAGE)
                      .mergeFrom(convertBaseOptionsToProto(baseOptions()))
                      .build());

      // Setup PoseDetectorGraphOptions.
      PoseDetectorGraphOptionsProto.PoseDetectorGraphOptions.Builder
          poseDetectorGraphOptionsBuilder =
              PoseDetectorGraphOptionsProto.PoseDetectorGraphOptions.newBuilder();
      numPoses().ifPresent(poseDetectorGraphOptionsBuilder::setNumPoses);
      minPoseDetectionConfidence()
          .ifPresent(poseDetectorGraphOptionsBuilder::setMinDetectionConfidence);

      // Setup PoseLandmarkerGraphOptions.
      PoseLandmarksDetectorGraphOptionsProto.PoseLandmarksDetectorGraphOptions.Builder
          poseLandmarksDetectorGraphOptionsBuilder =
              PoseLandmarksDetectorGraphOptionsProto.PoseLandmarksDetectorGraphOptions.newBuilder();
      minPosePresenceConfidence()
          .ifPresent(poseLandmarksDetectorGraphOptionsBuilder::setMinDetectionConfidence);
      smoothLandmarks().ifPresent(poseLandmarksDetectorGraphOptionsBuilder::setSmoothLandmarks);
      minTrackingConfidence().ifPresent(taskOptionsBuilder::setMinTrackingConfidence);

      taskOptionsBuilder
          .setPoseDetectorGraphOptions(poseDetectorGraphOptionsBuilder.build())
          .setPoseLandmarksDetectorGraphOptions(poseLandmarksDetectorGraphOptionsBuilder.build());

      return CalculatorOptions.newBuilder()
          .setExtension(
              PoseLandmarkerGraphOptionsProto.PoseLandmarkerGraphOptions.ext,
              taskOptionsBuilder.build())
          .build();
    }
  }

  /**
   * Validates that the provided {@link ImageProcessingOptions} doesn't contain a
   * region-of-interest.
   */
  private static void validateImageProcessingOptions(
      ImageProcessingOptions imageProcessingOptions) {
    if (imageProcessingOptions.regionOfInterest().isPresent()) {
      throw new IllegalArgumentException("PoseLandmarker doesn't support region-of-interest.");
    }
  }

  private static Optional<List<MPImage>> getSegmentationMasks(
      List<Packet> packets, int segmentationMasksOutStreamIndex) {
    Optional<List<MPImage>> segmentedMasks = Optional.of(new ArrayList<>());
    int width =
        PacketGetter.getImageWidthFromImageList(packets.get(segmentationMasksOutStreamIndex));
    int height =
        PacketGetter.getImageHeightFromImageList(packets.get(segmentationMasksOutStreamIndex));
    int imageListSize = PacketGetter.getImageListSize(packets.get(segmentationMasksOutStreamIndex));
    ByteBuffer[] buffersArray = new ByteBuffer[imageListSize];

    // Segmentation mask is a float type image.
    int numBytes = 4;
    for (int i = 0; i < imageListSize; i++) {
      buffersArray[i] = ByteBuffer.allocateDirect(width * height * numBytes);
    }

    if (!PacketGetter.getImageList(
        packets.get(segmentationMasksOutStreamIndex),
        buffersArray,
        /** deepCopy= */
        true)) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INTERNAL.ordinal(),
          "There is an error getting segmented masks.");
    }
    for (ByteBuffer buffer : buffersArray) {
      ByteBufferImageBuilder builder =
          new ByteBufferImageBuilder(buffer, width, height, MPImage.IMAGE_FORMAT_VEC32F1);
      segmentedMasks.get().add(builder.build());
    }
    return segmentedMasks;
  }
}
