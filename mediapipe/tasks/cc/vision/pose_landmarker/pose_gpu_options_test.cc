#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_gpu_options.h"

#include <set>
#include <string>
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe::tasks::vision::pose_landmarker {
namespace {
using Gpu = InferenceCalculatorOptions::Delegate::Gpu;

TEST(PoseGpuOptionsTest, PreservesBackendAndSeparatesBothModelsAndAllApis) {
  std::set<std::string> tokens;
  for (auto api : {Gpu::ANY, Gpu::OPENGL, Gpu::OPENCL}) {
    core::proto::Acceleration parent;
    auto* gpu = parent.mutable_gpu();
    gpu->set_api(api);
    gpu->set_cached_kernel_path("/private/kernels");
    gpu->set_serialized_model_dir("/private/models");
    gpu->set_model_token("full_model_hash");
    gpu->set_allow_precision_loss(false);
    gpu->set_usage(Gpu::SUSTAINED_SPEED);
    gpu->set_cache_writing_behavior(Gpu::TRY_WRITE);
    const auto original = parent.SerializeAsString();
    for (const auto* role : {"detector", "landmarks"}) {
      const auto child = PoseSubtaskAcceleration(parent, role);
      EXPECT_EQ(child.gpu().api(), api);
      EXPECT_TRUE(child.gpu().use_advanced_gpu_api());
      EXPECT_FALSE(child.gpu().allow_precision_loss());
      EXPECT_EQ(child.gpu().usage(), Gpu::SUSTAINED_SPEED);
      EXPECT_EQ(child.gpu().cache_writing_behavior(), Gpu::TRY_WRITE);
      EXPECT_EQ(child.gpu().cached_kernel_path(), "/private/kernels");
      EXPECT_EQ(child.gpu().serialized_model_dir(), "/private/models");
      EXPECT_TRUE(tokens.insert(child.gpu().model_token()).second);
      EXPECT_EQ(child.SerializeAsString(), PoseSubtaskAcceleration(parent, role).SerializeAsString());
    }
    EXPECT_EQ(parent.SerializeAsString(), original);
  }
  EXPECT_EQ(tokens.size(), 6);
}

TEST(PoseGpuOptionsTest, MissingTokenDisablesUnsafeEmbeddedModelCaching) {
  core::proto::Acceleration parent;
  parent.mutable_gpu()->set_api(Gpu::OPENCL);
  parent.mutable_gpu()->set_cached_kernel_path("/private/kernels");
  parent.mutable_gpu()->set_serialized_model_dir("/private/models");
  const auto child = PoseSubtaskAcceleration(parent, "detector");
  EXPECT_EQ(child.gpu().api(), Gpu::OPENCL);
  EXPECT_FALSE(child.gpu().has_cached_kernel_path());
  EXPECT_FALSE(child.gpu().has_serialized_model_dir());
}

TEST(PoseGpuOptionsTest, CpuAccelerationRemainsUnchanged) {
  core::proto::Acceleration parent;
  parent.mutable_xnnpack()->set_num_threads(4);
  EXPECT_EQ(PoseSubtaskAcceleration(parent, "detector").SerializeAsString(),
            parent.SerializeAsString());
}
}  // namespace
}  // namespace mediapipe::tasks::vision::pose_landmarker
