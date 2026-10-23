// Copyright 2019 The MediaPipe Authors.
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

#include <atomic>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator.pb.h"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/memory_manager_service.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

namespace mediapipe {
namespace {

class TensorWriteWarningSink : public absl::LogSink {
 public:
  TensorWriteWarningSink() { absl::AddLogSink(this); }
  ~TensorWriteWarningSink() override { absl::RemoveLogSink(this); }

  void Send(const absl::LogEntry& entry) override {
    if (absl::StrContains(entry.text_message(),
                          "Tensors are designed for single writes")) {
      warnings.fetch_add(1, std::memory_order_relaxed);
    }
  }

  std::atomic<int> warnings{0};
};

class TensorsToDetectionsGlTest : public testing::Test {
 protected:
  void SetUp() override { helper_.InitializeForTest(gpu_resources_.get()); }

  GpuSharedData gpu_shared_;
  std::shared_ptr<GpuResources> gpu_resources_ = gpu_shared_.gpu_resources;
  GlCalculatorHelper helper_;
};

// Keep both service configurations in one test: Tensor logs this violation only
// once per process. The sink must cover the very first repeated GPU decode.
TEST_F(TensorsToDetectionsGlTest, RepeatedFramesUseSingleWriteTensors) {
  TensorWriteWarningSink warning_sink;
  for (bool use_pool : {true, false}) {
    const auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
      input_stream: "tensors"
      input_side_packet: "anchors"
      output_stream: "detections"
      node {
        calculator: "TensorsToDetectionsCalculator"
        input_stream: "TENSORS:tensors"
        input_side_packet: "ANCHORS:anchors"
        output_stream: "DETECTIONS:detections"
        options {
          [mediapipe.TensorsToDetectionsCalculatorOptions.ext] {
            num_classes: 1
            num_boxes: 1
            num_coords: 4
            box_format: XYWH
            x_scale: 1
            y_scale: 1
            w_scale: 1
            h_scale: 1
            min_score_thresh: 0.5
          }
        }
      }
    )pb");
    CalculatorGraph graph;
    MP_ASSERT_OK(graph.SetGpuResources(gpu_resources_));
    if (use_pool) {
      MP_ASSERT_OK(graph.SetServiceObject(kMemoryManagerService,
                                         std::make_shared<MemoryManager>()));
    }
    MP_ASSERT_OK(graph.Initialize(config));
    MP_ASSERT_OK_AND_ASSIGN(auto poller,
                           graph.AddOutputStreamPoller("detections"));
    std::vector<Anchor> anchors(1);
    anchors[0].set_x_center(0.5f);
    anchors[0].set_y_center(0.5f);
    anchors[0].set_w(1.f);
    anchors[0].set_h(1.f);
    MP_ASSERT_OK(graph.StartRun({
        {"anchors", MakePacket<std::vector<Anchor>>(std::move(anchors))}}));

    for (int frame = 0; frame < 20; ++frame) {
      std::vector<Tensor> tensors;
      tensors.emplace_back(Tensor::ElementType::kFloat32,
                           Tensor::Shape{1, 1, 4});
      tensors.emplace_back(Tensor::ElementType::kFloat32,
                           Tensor::Shape{1, 1, 1});
      {
        auto box_view = tensors[0].GetCpuWriteView();
        auto* boxes = box_view.buffer<float>();
        boxes[0] = frame * 0.01f;
        boxes[1] = 0.f;
        boxes[2] = 0.2f;
        boxes[3] = 0.4f;
        auto score_view = tensors[1].GetCpuWriteView();
        score_view.buffer<float>()[0] = frame % 2 == 0 ? 0.9f : 0.1f;
      }
      // Upload through read views so the input remains single-write too, and
      // force GPU postprocessing rather than silently exercising the CPU path.
      helper_.RunInGlContext([&tensors]() {
        for (auto& tensor : tensors) {
          auto view = tensor.GetOpenGlBufferReadView();
          ASSERT_NE(view.name(), 0);
        }
      });
      ASSERT_TRUE(tensors[0].ready_on_gpu());
      MP_ASSERT_OK(graph.AddPacketToInputStream(
          "tensors", MakePacket<std::vector<Tensor>>(std::move(tensors))
                         .At(Timestamp(frame))));
      MP_ASSERT_OK(graph.WaitUntilIdle());
      Packet output;
      ASSERT_TRUE(poller.Next(&output));
      const auto& detections = output.Get<std::vector<Detection>>();
      ASSERT_EQ(detections.size(), frame % 2 == 0 ? 1 : 0);
      if (!detections.empty()) {
        const auto& box = detections[0].location_data().relative_bounding_box();
        EXPECT_NEAR(box.xmin(), 0.4f + frame * 0.01f, 1e-5f);
        EXPECT_NEAR(box.ymin(), 0.3f, 1e-5f);
        EXPECT_NEAR(box.width(), 0.2f, 1e-5f);
        EXPECT_NEAR(box.height(), 0.4f, 1e-5f);
        EXPECT_NEAR(detections[0].score(0), 0.9f, 1e-5f);
      }
    }
    MP_ASSERT_OK(graph.CloseAllInputStreams());
    MP_ASSERT_OK(graph.WaitUntilDone());
  }
  EXPECT_EQ(warning_sink.warnings.load(std::memory_order_relaxed), 0);
}

}  // namespace
}  // namespace mediapipe
