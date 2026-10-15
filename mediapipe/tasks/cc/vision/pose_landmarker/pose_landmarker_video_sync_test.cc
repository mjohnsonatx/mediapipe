// Copyright 2026 The MediaPipe Authors.
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

// Run with POSE_LANDMARKER_TEST_MODEL pointing to a pose landmarker .task bundle.
// This also runs as an Android native executable against the app's model assets.
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/sink.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/proto/pose_landmarker_graph_options.pb.h"

namespace mediapipe::tasks::vision::pose_landmarker {
namespace {

CalculatorGraphConfig MakeConfig(bool stream_mode, bool video_sync,
                                 bool reset_tracking, bool segmentation) {
  api2::builder::Graph graph;
  auto& pose = graph.AddNode(
      "mediapipe.tasks.vision.pose_landmarker.PoseLandmarkerGraph");
  auto& options = pose.GetOptions<proto::PoseLandmarkerGraphOptions>();
  const char* model = std::getenv("POSE_LANDMARKER_TEST_MODEL");
  options.mutable_base_options()->mutable_model_asset()->set_file_name(
      model ? model : "mediapipe/tasks/testdata/vision/pose_landmarker.task");
  options.mutable_base_options()->set_use_stream_mode(stream_mode);
  options.mutable_pose_detector_graph_options()->set_num_poses(1);
  options.mutable_pose_landmarks_detector_graph_options()->set_smooth_landmarks(
      false);
  graph.In("IMAGE").SetName("image_in") >> pose.In("IMAGE");
  graph.In("NORM_RECT").SetName("norm_rect_in") >> pose.In("NORM_RECT");
  if (video_sync) {
    graph.In("EXTERNAL_POSE_RECT").SetName("external_pose_rect_in") >>
        pose.In("EXTERNAL_POSE_RECT");
    graph.In("USE_EXTERNAL_POSE_RECT").SetName("use_external_pose_rect_in") >>
        pose.In("USE_EXTERNAL_POSE_RECT");
    graph.In("EXTERNAL_SOURCE_PROJECTION_RECT").SetName("projection_in") >>
        pose.In("EXTERNAL_SOURCE_PROJECTION_RECT");
    graph.In("EXTERNAL_SOURCE_IMAGE_SIZE").SetName("source_size_in") >>
        pose.In("EXTERNAL_SOURCE_IMAGE_SIZE");
    graph.In("USE_EXTERNAL_SOURCE_PROJECTION").SetName("use_projection_in") >>
        pose.In("USE_EXTERNAL_SOURCE_PROJECTION");
    pose.Out("VIDEO_SYNC").SetName("video_sync") >> graph.Out("VIDEO_SYNC");
  } else {
    pose.Out("IMAGE").SetName("image_out") >> graph.Out("IMAGE");
  }
  if (reset_tracking) {
    // Different input/output names reproduce the Java TaskInfo graph boundary.
    graph.In("RESET_TRACKING").SetName("reset_tracking_in") >>
        pose.In("RESET_TRACKING");
  }
  pose.Out("NORM_LANDMARKS").SetName("pose_landmarks") >>
      graph.Out("NORM_LANDMARKS");
  pose.Out("WORLD_LANDMARKS").SetName("world_landmarks") >>
      graph.Out("WORLD_LANDMARKS");
  pose.Out("POSE_RECTS_NEXT_FRAME").SetName("pose_rects_next_frame") >>
      graph.Out("POSE_RECTS_NEXT_FRAME");
  pose.Out("DETECTIONS").SetName("pose_detections") >> graph.Out("DETECTIONS");
  if (segmentation) {
    pose.Out("SEGMENTATION_MASK").SetName("segmentation_masks") >>
        graph.Out("SEGMENTATION_MASK");
  }
  return graph.GetConfig();
}

Image MakeBlankImage() {
  auto frame = std::make_shared<ImageFrame>(ImageFormat::SRGB, 256, 256);
  frame->SetToZero();
  return Image(std::move(frame));
}

NormalizedRect FullImageRect() {
  NormalizedRect rect;
  rect.set_x_center(0.5f);
  rect.set_y_center(0.5f);
  rect.set_width(1.0f);
  rect.set_height(1.0f);
  rect.set_rotation(0.0f);
  return rect;
}

class PoseLandmarkerVideoSyncTest : public testing::TestWithParam<bool> {};

TEST_P(PoseLandmarkerVideoSyncTest, StartsAndReturnsEmptyFramesWithoutImage) {
  auto config = MakeConfig(/*stream_mode=*/true, /*video_sync=*/true,
                           /*reset_tracking=*/true, GetParam());
  std::vector<std::string> streams = {
      "pose_landmarks", "world_landmarks", "video_sync",
      "pose_rects_next_frame", "pose_detections"};
  if (GetParam()) streams.push_back("segmentation_masks");
  std::vector<std::vector<Packet>> results;
  std::map<std::string, Packet> side_packets;
  // Match Java TaskRunner: timestamp bounds alone do not invoke its callback.
  tool::AddMultiStreamCallback(
      streams,
      [&results](const std::vector<Packet>& packets) { results.push_back(packets); },
      &config, &side_packets, /*observe_timestamp_bounds=*/false);
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun(side_packets));
  // The synchronization producer must not retain an Image packet.
  for (const auto& node : graph.Config().node()) {
    if (node.calculator() == "PassThroughCalculator") {
      EXPECT_NE(node.input_stream(0), "image_in");
    }
  }
  for (int i = 0; i < 3; ++i) {
    const Timestamp timestamp(i * 33333);
    const bool reset = i == 1;
    const bool external_roi = i != 0;
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "image_in", MakePacket<Image>(MakeBlankImage()).At(timestamp)));
    for (const auto* stream :
         {"norm_rect_in", "external_pose_rect_in", "projection_in"}) {
      MP_ASSERT_OK(graph.AddPacketToInputStream(
          stream, MakePacket<NormalizedRect>(FullImageRect()).At(timestamp)));
    }
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "source_size_in",
        MakePacket<std::pair<int, int>>(512, 512).At(timestamp)));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "use_external_pose_rect_in", MakePacket<bool>(external_roi).At(timestamp)));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "use_projection_in", MakePacket<bool>(external_roi).At(timestamp)));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "reset_tracking_in", MakePacket<bool>(reset).At(timestamp)));
    MP_ASSERT_OK(graph.WaitUntilIdle());
    ASSERT_EQ(results.size(), i + 1);
    const auto& packets = results.back();
    ASSERT_EQ(packets.size(), streams.size());
    EXPECT_TRUE(packets[0].IsEmpty());
    EXPECT_TRUE(packets[1].IsEmpty());
    ASSERT_FALSE(packets[2].IsEmpty());
    EXPECT_EQ(packets[2].Timestamp(), timestamp);
    EXPECT_EQ(packets[2].Get<bool>(), reset);
    EXPECT_TRUE(packets[3].IsEmpty());
    if (i == 2) {
      // External ROI skips detection, so only VIDEO_SYNC is nonempty.
      EXPECT_TRUE(packets[4].IsEmpty());
    }
  }
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

INSTANTIATE_TEST_SUITE_P(Segmentation, PoseLandmarkerVideoSyncTest,
                         testing::Bool());

class PoseLandmarkerImageOutputTest : public testing::TestWithParam<bool> {};

TEST_P(PoseLandmarkerImageOutputTest, PreservesImageOutputWithoutVideoSync) {
  auto config = MakeConfig(GetParam(), /*video_sync=*/false,
                           /*reset_tracking=*/false, /*segmentation=*/false);
  std::vector<Packet> images;
  tool::AddVectorSink("image_out", &config, &images);
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "image_in", MakePacket<Image>(MakeBlankImage()).At(Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "norm_rect_in", MakePacket<NormalizedRect>(FullImageRect()).At(Timestamp(0))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(images.size(), 1);
  EXPECT_EQ(images[0].Get<Image>().width(), 256);
  EXPECT_EQ(images[0].Timestamp(), Timestamp(0));
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

INSTANTIATE_TEST_SUITE_P(StreamMode, PoseLandmarkerImageOutputTest,
                         testing::Bool());

TEST(PoseLandmarkerVideoSyncContractTest, RequiresPerFrameResetInput) {
  CalculatorGraph graph;
  auto status = graph.Initialize(
      MakeConfig(/*stream_mode=*/true, /*video_sync=*/true,
                 /*reset_tracking=*/false, /*segmentation=*/false));
  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("VIDEO_SYNC requires"), std::string::npos);
}

}  // namespace
}  // namespace mediapipe::tasks::vision::pose_landmarker
