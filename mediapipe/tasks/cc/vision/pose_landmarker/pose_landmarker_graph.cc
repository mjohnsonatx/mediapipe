/* Copyright 2023 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <optional>
#include <utility>
#include <vector>
#include "absl/status/status.h"
#include "mediapipe/calculators/core/clip_vector_size_calculator.pb.h"
#include "mediapipe/calculators/core/gate_calculator.pb.h"
#include "mediapipe/calculators/util/association_calculator.pb.h"
#include "mediapipe/calculators/util/collection_has_min_size_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/api2/stream/merge.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/components/utils/gate.h"
#include "mediapipe/tasks/cc/core/model_asset_bundle_resources.h"
#include "mediapipe/tasks/cc/core/model_resources_cache.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/metadata/utils/zip_utils.h"
#include "mediapipe/tasks/cc/vision/pose_detector/proto/pose_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/proto/pose_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/proto/pose_landmarks_detector_graph_options.pb.h"
#include "mediapipe/util/graph_builder_utils.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace pose_landmarker {

namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Merge;
using ::mediapipe::api2::builder::SidePacket;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::components::utils::AllowIf;
using ::mediapipe::tasks::components::utils::DisallowIf;
using ::mediapipe::tasks::core::ModelAssetBundleResources;
using ::mediapipe::tasks::metadata::SetExternalFile;
using ::mediapipe::tasks::vision::pose_detector::proto::
    PoseDetectorGraphOptions;
using ::mediapipe::tasks::vision::pose_landmarker::proto::
    PoseLandmarkerGraphOptions;
using ::mediapipe::tasks::vision::pose_landmarker::proto::
    PoseLandmarksDetectorGraphOptions;

constexpr char kImageTag[] = "IMAGE";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kExternalPoseRectTag[] = "EXTERNAL_POSE_RECT";
constexpr char kUseExternalPoseRectTag[] = "USE_EXTERNAL_POSE_RECT";
constexpr char kExternalSourceProjectionRectTag[] =
    "EXTERNAL_SOURCE_PROJECTION_RECT";
constexpr char kExternalSourceImageSizeTag[] = "EXTERNAL_SOURCE_IMAGE_SIZE";
constexpr char kUseExternalSourceProjectionTag[] =
    "USE_EXTERNAL_SOURCE_PROJECTION";
constexpr char kResetTrackingTag[] = "RESET_TRACKING";
constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kWorldLandmarksTag[] = "WORLD_LANDMARKS";
constexpr char kAuxiliaryLandmarksTag[] = "AUXILIARY_LANDMARKS";
constexpr char kPoseRectsNextFrameTag[] = "POSE_RECTS_NEXT_FRAME";
constexpr char kExpandedPoseRectsTag[] = "EXPANDED_POSE_RECTS";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kLoopTag[] = "LOOP";
constexpr char kPrevLoopTag[] = "PREV_LOOP";
constexpr char kMainTag[] = "MAIN";
constexpr char kIterableTag[] = "ITERABLE";
constexpr char kSegmentationMaskTag[] = "SEGMENTATION_MASK";

constexpr char kPoseDetectorTFLiteName[] = "pose_detector.tflite";
constexpr char kPoseLandmarksDetectorTFLiteName[] =
    "pose_landmarks_detector.tflite";

struct PoseLandmarkerOutputs {
  Source<std::vector<NormalizedLandmarkList>> landmark_lists;
  Source<std::vector<LandmarkList>> world_landmark_lists;
  Source<std::vector<NormalizedLandmarkList>> auxiliary_landmark_lists;
  Source<std::vector<NormalizedRect>> pose_rects_next_frame;
  Source<std::vector<Detection>> pose_detections;
  std::optional<Source<std::vector<Image>>> segmentation_masks;
  Source<Image> image;
};

// Sets the base options in the sub tasks.
absl::Status SetSubTaskBaseOptions(const ModelAssetBundleResources& resources,
                                   PoseLandmarkerGraphOptions* options,
                                   bool is_copy) {
  auto* pose_detector_graph_options =
      options->mutable_pose_detector_graph_options();
  if (!pose_detector_graph_options->base_options().has_model_asset()) {
    ABSL_ASSIGN_OR_RETURN(const auto pose_detector_file,
                          resources.GetFile(kPoseDetectorTFLiteName));
    SetExternalFile(pose_detector_file,
                    pose_detector_graph_options->mutable_base_options()
                        ->mutable_model_asset(),
                    is_copy);
  }
  if (options->base_options().acceleration().has_gpu()) {
    core::proto::Acceleration gpu_accel;
    gpu_accel.mutable_gpu()->set_use_advanced_gpu_api(true);
    pose_detector_graph_options->mutable_base_options()
        ->mutable_acceleration()
        ->CopyFrom(gpu_accel);

  } else {
    pose_detector_graph_options->mutable_base_options()
        ->mutable_acceleration()
        ->CopyFrom(options->base_options().acceleration());
  }
  pose_detector_graph_options->mutable_base_options()->set_use_stream_mode(
      options->base_options().use_stream_mode());
  auto* pose_landmarks_detector_graph_options =
      options->mutable_pose_landmarks_detector_graph_options();
  if (!pose_landmarks_detector_graph_options->base_options()
           .has_model_asset()) {
    ABSL_ASSIGN_OR_RETURN(const auto pose_landmarks_detector_file,
                          resources.GetFile(kPoseLandmarksDetectorTFLiteName));
    SetExternalFile(
        pose_landmarks_detector_file,
        pose_landmarks_detector_graph_options->mutable_base_options()
            ->mutable_model_asset(),
        is_copy);
  }
  pose_landmarks_detector_graph_options->mutable_base_options()
      ->mutable_acceleration()
      ->CopyFrom(options->base_options().acceleration());
  pose_landmarks_detector_graph_options->mutable_base_options()
      ->set_use_stream_mode(options->base_options().use_stream_mode());

  pose_detector_graph_options->mutable_base_options()->set_gpu_origin(
      options->base_options().gpu_origin());
  pose_landmarks_detector_graph_options->mutable_base_options()->set_gpu_origin(
      options->base_options().gpu_origin());

  return absl::OkStatus();
}

}  // namespace

// A "mediapipe.tasks.vision.pose_landmarker.PoseLandmarkerGraph" performs pose
// landmarks detection. The PoseLandmarkerGraph consists of two subgraphs:
// PoseDetectorGraph, MultiplePoseLandmarksDetectorGraph
//
// MultiplePoseLandmarksDetectorGraph detects landmarks from bounding boxes
// produced by PoseDetectorGraph. PoseLandmarkerGraph tracks the landmarks over
// time, and skips the PoseDetectorGraph. If the tracking is lost or the
// detected poses are less than configured max number poses, PoseDetectorGraph
// would be triggered to detect poses.
//
//
// Inputs:
//   IMAGE - Image
//     Image to perform pose landmarks detection on.
//   NORM_RECT - NormalizedRect @Optional
//     Describes image rotation and region of image to perform landmarks
//     detection on. If not provided, whole image is used for pose landmarks
//     detection.
//   EXTERNAL_POSE_RECT - NormalizedRect @Optional
//     A pose region of interest to use directly for landmark detection instead
//     of the region retained by the internal stream-mode tracker. This input
//     must be accompanied by USE_EXTERNAL_POSE_RECT.
//   USE_EXTERNAL_POSE_RECT - bool @Optional
//     Whether EXTERNAL_POSE_RECT should override the internally tracked pose
//     rectangle for this timestamp.
//   EXTERNAL_SOURCE_PROJECTION_RECT - NormalizedRect @Optional
//     Source-image rectangle represented by an already-cropped IMAGE. This is
//     used for output projection and smoothing, not image preprocessing.
//   EXTERNAL_SOURCE_IMAGE_SIZE - std::pair<int, int> @Optional
//     Width and height of the source image described by
//     EXTERNAL_SOURCE_PROJECTION_RECT.
//   USE_EXTERNAL_SOURCE_PROJECTION - bool @Optional
//     Whether the external source projection rectangle and image size should
//     override the IMAGE coordinate system for this timestamp. All three
//     external source projection inputs must be connected together.
//   RESET_TRACKING - bool @Optional
//     When true, ignores the internally tracked pose rectangle for this
//     timestamp so pose detection runs again.
//
//
// Outputs:
//   NORM_LANDMARKS: - std::vector<NormalizedLandmarkList>
//     Vector of detected pose landmarks.
//   WORLD_LANDMARKS:  std::vector<LandmarkList>
//    Vector of detected world pose landmarks.
//   AUXILIARY_LANDMARKS: - std::vector<NormalizedLandmarkList>
//    Vector of detected auxiliary landmarks.
//   POSE_RECTS_NEXT_FRAME - std::vector<NormalizedRect>
//     Vector of the expanded rects enclosing the whole pose RoI for landmark
//     detection on the next frame.
//   POSE_RECTS - std::vector<NormalizedRect>
//     Detected pose bounding boxes in normalized coordinates from pose
//     detection.
//   SEGMENTATION_MASK -  std::vector<Image>
//     Segmentation masks.
//   IMAGE - Image
//     The input image that the pose landmarker runs on and has the pixel data
//     stored on the target storage (CPU vs GPU).
// All returned coordinates are in the unrotated and uncropped input image
// coordinates system.
//
// Example:
// node {
//   calculator: "mediapipe.tasks.vision.pose_landmarker.PoseLandmarkerGraph"
//   input_stream: "IMAGE:image_in"
//   input_stream: "NORM_RECT:norm_rect"
//   input_stream: "EXTERNAL_POSE_RECT:external_pose_rect"
//   input_stream: "USE_EXTERNAL_POSE_RECT:use_external_pose_rect"
//   input_stream:
//     "EXTERNAL_SOURCE_PROJECTION_RECT:external_source_projection_rect"
//   input_stream: "EXTERNAL_SOURCE_IMAGE_SIZE:external_source_image_size"
//   input_stream:
//     "USE_EXTERNAL_SOURCE_PROJECTION:use_external_source_projection"
//   input_stream: "RESET_TRACKING:reset_tracking"
//   output_stream: "NORM_LANDMARKS:pose_landmarks"
//   output_stream: "WORLD_LANDMARKS:world_landmarks"
//   output_stream: "AUXILIARY_LANDMARKS:auxiliary_landmarks"
//   output_stream: "POSE_RECTS_NEXT_FRAME:pose_rects_next_frame"
//   output_stream: "POSE_RECTS:pose_rects"
//   output_stream: "SEGMENTATION_MASK:segmentation_masks"
//   output_stream: "IMAGE:image_out"
//   options {
//     [mediapipe.tasks.vision.pose_landmarker.proto.PoseLandmarkerGraphOptions.ext]
//     {
//       base_options {
//          model_asset {
//            file_name: "pose_landmarker.task"
//          }
//       }
//       pose_detector_graph_options {
//         min_detection_confidence: 0.5
//         num_poses: 2
//       }
//       pose_landmarks_detector_graph_options {
//         min_detection_confidence: 0.5
//       }
//     }
//   }
// }
class PoseLandmarkerGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    Graph graph;
    bool output_segmentation_masks =
        HasOutput(sc->OriginalNode(), kSegmentationMaskTag);
    if (sc->Options<PoseLandmarkerGraphOptions>()
            .base_options()
            .has_model_asset()) {
      ABSL_ASSIGN_OR_RETURN(
          const auto* model_asset_bundle_resources,
          CreateModelAssetBundleResources<PoseLandmarkerGraphOptions>(sc));
      // Copies the file content instead of passing the pointer of file in
      // memory if the subgraph model resource service is not available.
      ABSL_RETURN_IF_ERROR(SetSubTaskBaseOptions(
          *model_asset_bundle_resources,
          sc->MutableOptions<PoseLandmarkerGraphOptions>(),
          !sc->Service(::mediapipe::tasks::core::kModelResourcesCacheService)
               .IsAvailable()));
    }
    const bool has_external_pose_rect =
        HasInput(sc->OriginalNode(), kExternalPoseRectTag);
    const bool has_use_external_pose_rect =
        HasInput(sc->OriginalNode(), kUseExternalPoseRectTag);
    if (has_external_pose_rect != has_use_external_pose_rect) {
      return absl::InvalidArgumentError(
          "EXTERNAL_POSE_RECT and USE_EXTERNAL_POSE_RECT must either both be "
          "connected or both be omitted.");
    }
    std::optional<Source<NormalizedRect>> external_pose_rect_in;
    std::optional<Source<bool>> use_external_pose_rect_in;
    if (has_external_pose_rect) {
      external_pose_rect_in =
          graph.In(kExternalPoseRectTag).Cast<NormalizedRect>();
      use_external_pose_rect_in =
          graph.In(kUseExternalPoseRectTag).Cast<bool>();
    }
    const bool has_external_source_projection_rect =
        HasInput(sc->OriginalNode(), kExternalSourceProjectionRectTag);
    const bool has_external_source_image_size =
        HasInput(sc->OriginalNode(), kExternalSourceImageSizeTag);
    const bool has_use_external_source_projection =
        HasInput(sc->OriginalNode(), kUseExternalSourceProjectionTag);
    if (has_external_source_projection_rect !=
            has_external_source_image_size ||
        has_external_source_projection_rect !=
            has_use_external_source_projection) {
      return absl::InvalidArgumentError(
          "EXTERNAL_SOURCE_PROJECTION_RECT, EXTERNAL_SOURCE_IMAGE_SIZE, and "
          "USE_EXTERNAL_SOURCE_PROJECTION must either all be connected or "
          "all be omitted.");
    }
    std::optional<Source<NormalizedRect>> external_source_projection_rect_in;
    std::optional<Source<std::pair<int, int>>> external_source_image_size_in;
    std::optional<Source<bool>> use_external_source_projection_in;
    if (has_external_source_projection_rect) {
      external_source_projection_rect_in =
          graph.In(kExternalSourceProjectionRectTag).Cast<NormalizedRect>();
      external_source_image_size_in =
          graph.In(kExternalSourceImageSizeTag).Cast<std::pair<int, int>>();
      use_external_source_projection_in =
          graph.In(kUseExternalSourceProjectionTag).Cast<bool>();
    }
    std::optional<Source<bool>> reset_tracking_in;
    if (HasInput(sc->OriginalNode(), kResetTrackingTag)) {
      reset_tracking_in = graph.In(kResetTrackingTag).Cast<bool>();
    }
    ABSL_ASSIGN_OR_RETURN(
        auto outs, BuildPoseLandmarkerGraph(
                       *sc->MutableOptions<PoseLandmarkerGraphOptions>(),
                       graph[Input<Image>(kImageTag)],
                       graph[Input<NormalizedRect>::Optional(kNormRectTag)],
                       external_pose_rect_in, use_external_pose_rect_in,
                       external_source_projection_rect_in,
                       external_source_image_size_in,
                       use_external_source_projection_in,
                       reset_tracking_in, graph, output_segmentation_masks));
    outs.landmark_lists >>
        graph[Output<std::vector<NormalizedLandmarkList>>(kNormLandmarksTag)];
    outs.world_landmark_lists >>
        graph[Output<std::vector<LandmarkList>>(kWorldLandmarksTag)];
    outs.auxiliary_landmark_lists >>
        graph[Output<std::vector<NormalizedLandmarkList>>(
            kAuxiliaryLandmarksTag)];
    outs.pose_rects_next_frame >>
        graph[Output<std::vector<NormalizedRect>>(kPoseRectsNextFrameTag)];
    outs.pose_detections >>
        graph[Output<std::vector<Detection>>(kDetectionsTag)];
    outs.image >> graph[Output<Image>(kImageTag)];
    if (outs.segmentation_masks) {
      *outs.segmentation_masks >>
          graph[Output<std::vector<Image>>(kSegmentationMaskTag)];
    }

    CalculatorGraphConfig config = graph.GetConfig();
    core::FixGraphBackEdges(config);
    return config;
  }

 private:
  // Adds a mediapipe pose landmarker graph into the provided builder::Graph
  // instance.
  //
  // tasks_options: the mediapipe tasks module PoseLandmarkerGraphOptions.
  // image_in: (mediapipe::Image) stream to run pose landmark detection on.
  // graph: the mediapipe graph instance to be updated.
  absl::StatusOr<PoseLandmarkerOutputs> BuildPoseLandmarkerGraph(
      PoseLandmarkerGraphOptions& tasks_options, Source<Image> image_in,
      Source<NormalizedRect> norm_rect_in,
      std::optional<Source<NormalizedRect>> external_pose_rect_in,
      std::optional<Source<bool>> use_external_pose_rect_in,
      std::optional<Source<NormalizedRect>>
          external_source_projection_rect_in,
      std::optional<Source<std::pair<int, int>>> external_source_image_size_in,
      std::optional<Source<bool>> use_external_source_projection_in,
      std::optional<Source<bool>> reset_tracking_in, Graph& graph,
      bool output_segmentation_masks) {
    const int max_num_poses =
        tasks_options.pose_detector_graph_options().num_poses();

    auto& pose_detector =
        graph.AddNode("mediapipe.tasks.vision.pose_detector.PoseDetectorGraph");
    auto& pose_detector_options =
        pose_detector.GetOptions<PoseDetectorGraphOptions>();
    pose_detector_options.Swap(
        tasks_options.mutable_pose_detector_graph_options());
    auto& clip_pose_rects =
        graph.AddNode("ClipNormalizedRectVectorSizeCalculator");
    clip_pose_rects.GetOptions<ClipVectorSizeCalculatorOptions>()
        .set_max_vec_size(max_num_poses);
    auto clipped_pose_rects = clip_pose_rects.Out("");

    auto& pose_landmarks_detector_graph = graph.AddNode(
        "mediapipe.tasks.vision.pose_landmarker."
        "MultiplePoseLandmarksDetectorGraph");
    auto& pose_landmarks_detector_graph_options =
        pose_landmarks_detector_graph
            .GetOptions<PoseLandmarksDetectorGraphOptions>();
    pose_landmarks_detector_graph_options.Swap(
        tasks_options.mutable_pose_landmarks_detector_graph_options());

    // Apply smoothing filter only on the single pose landmarks, because
    // landmarks smoothing calculator doesn't support multiple landmarks yet.
    if (pose_detector_options.num_poses() == 1) {
      pose_landmarks_detector_graph_options.set_smooth_landmarks(
          tasks_options.base_options().use_stream_mode());
    } else if (pose_detector_options.num_poses() > 1 &&
               pose_landmarks_detector_graph_options.smooth_landmarks()) {
      return absl::InvalidArgumentError(
          "Currently pose landmarks smoothing only supports a single pose.");
    }

    image_in >> pose_landmarks_detector_graph.In(kImageTag);
    clipped_pose_rects >> pose_landmarks_detector_graph.In(kNormRectTag);
    if (external_source_projection_rect_in && external_source_image_size_in &&
        use_external_source_projection_in) {
      *external_source_projection_rect_in >>
          pose_landmarks_detector_graph.In(kExternalSourceProjectionRectTag);
      *external_source_image_size_in >>
          pose_landmarks_detector_graph.In(kExternalSourceImageSizeTag);
      *use_external_source_projection_in >>
          pose_landmarks_detector_graph.In(kUseExternalSourceProjectionTag);
    }

    // TODO: Add landmarks smoothing calculators to
    // PoseLandmarkerGraph
    auto landmarks = pose_landmarks_detector_graph.Out("LANDMARKS")
                         .Cast<std::vector<NormalizedLandmarkList>>();
    auto world_landmarks = pose_landmarks_detector_graph.Out(kWorldLandmarksTag)
                               .Cast<std::vector<LandmarkList>>();
    auto aux_landmarks =
        pose_landmarks_detector_graph.Out(kAuxiliaryLandmarksTag)
            .Cast<std::vector<NormalizedLandmarkList>>();
    auto pose_rects_for_next_frame =
        pose_landmarks_detector_graph.Out(kPoseRectsNextFrameTag)
            .Cast<std::vector<NormalizedRect>>();
    std::optional<Source<std::vector<Image>>> segmentation_masks;
    if (output_segmentation_masks) {
      segmentation_masks =
          pose_landmarks_detector_graph.Out(kSegmentationMaskTag)
              .Cast<std::vector<Image>>();
    }

    if (tasks_options.base_options().use_stream_mode()) {
      auto& previous_loopback = graph.AddNode("PreviousLoopbackCalculator");
      image_in >> previous_loopback.In(kMainTag);
      auto prev_pose_rects_from_landmarks =
          previous_loopback[Output<std::vector<NormalizedRect>>(kPrevLoopTag)];

      // A reset is consumed with the next video frame. Dropping the loopback
      // packet makes has_enough_poses empty/false, which opens the detector
      // gates and prevents the stale rectangle from entering association.
      if (reset_tracking_in) {
        prev_pose_rects_from_landmarks = DisallowIf(
            prev_pose_rects_from_landmarks, *reset_tracking_in, graph);
      }

      // When supplied for this timestamp, an external pose rectangle takes
      // priority over the rectangle retained by this graph instance. The
      // singular external rectangle is vectorized to match the multi-pose
      // landmark detector interface.
      auto pose_rects_for_tracking = prev_pose_rects_from_landmarks;
      if (external_pose_rect_in && use_external_pose_rect_in) {
        auto& concatenate_external_pose_rect =
            graph.AddNode("ConcatenateNormalizedRectVectorCalculator");
        *external_pose_rect_in >> concatenate_external_pose_rect.In("");
        Source<std::vector<NormalizedRect>> external_pose_rects =
            concatenate_external_pose_rect.Out("")
                .Cast<std::vector<NormalizedRect>>();
        external_pose_rects = AllowIf(external_pose_rects,
                                      *use_external_pose_rect_in, graph);
        pose_rects_for_tracking =
            Merge(external_pose_rects, prev_pose_rects_from_landmarks, graph);
      }

      auto& min_size_node =
          graph.AddNode("NormalizedRectVectorHasMinSizeCalculator");
      pose_rects_for_tracking >> min_size_node.In(kIterableTag);
      min_size_node.GetOptions<CollectionHasMinSizeCalculatorOptions>()
          .set_min_size(max_num_poses);
      auto has_enough_poses = min_size_node.Out("").Cast<bool>();

      // While in stream mode, skip pose detector graph when we successfully
      // track the poses from the last frame.
      auto image_for_pose_detector =
          DisallowIf(image_in, has_enough_poses, graph);
      auto norm_rect_in_for_pose_detector =
          DisallowIf(norm_rect_in, has_enough_poses, graph);
      image_for_pose_detector >> pose_detector.In(kImageTag);
      norm_rect_in_for_pose_detector >> pose_detector.In(kNormRectTag);
      auto expanded_pose_rects_from_pose_detector =
          pose_detector.Out(kExpandedPoseRectsTag);
      auto& pose_association = graph.AddNode("AssociationNormRectCalculator");
      pose_association.GetOptions<mediapipe::AssociationCalculatorOptions>()
          .set_min_similarity_threshold(
              tasks_options.min_tracking_confidence());
      pose_rects_for_tracking >>
          pose_association[Input<std::vector<NormalizedRect>>::Multiple("")][0];
      expanded_pose_rects_from_pose_detector >>
          pose_association[Input<std::vector<NormalizedRect>>::Multiple("")][1];
      auto pose_rects = pose_association.Out("");
      pose_rects >> clip_pose_rects.In("");
      // Back edge.
      pose_rects_for_next_frame >> previous_loopback.In(kLoopTag);
    } else {
      if (external_pose_rect_in || use_external_pose_rect_in ||
          external_source_projection_rect_in ||
          external_source_image_size_in || use_external_source_projection_in ||
          reset_tracking_in) {
        return absl::InvalidArgumentError(
            "External pose rectangles, source projection, and tracking reset "
            "are only supported in stream mode.");
      }
      // While not in stream mode, the input images are not guaranteed to be in
      // series, and we don't want to enable the tracking and rect associations
      // between input images. Always use the pose detector graph.
      image_in >> pose_detector.In(kImageTag);
      norm_rect_in >> pose_detector.In(kNormRectTag);
      auto pose_rects = pose_detector.Out(kExpandedPoseRectsTag);
      pose_rects >> clip_pose_rects.In("");
    }

    // TODO: Replace PassThroughCalculator with a calculator that
    // converts the pixel data to be stored on the target storage (CPU vs GPU).
    auto& pass_through = graph.AddNode("PassThroughCalculator");
    image_in >> pass_through.In("");

    return {{
        /* landmark_lists= */ landmarks,
        /* world_landmarks= */ world_landmarks,
        /* aux_landmarks= */ aux_landmarks,
        /* pose_rects_next_frame= */ pose_rects_for_next_frame,
        /* pose_detections */
        pose_detector.Out(kDetectionsTag).Cast<std::vector<Detection>>(),
        /* segmentation_masks= */ segmentation_masks,
        /* image= */
        pass_through[Output<Image>("")],
    }};
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::pose_landmarker::PoseLandmarkerGraph);

}  // namespace pose_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
