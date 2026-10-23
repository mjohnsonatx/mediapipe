#include "mediapipe/calculators/tensor/inference_on_disk_cache_helper.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::api2 {
TEST(InferenceOnDiskCacheHelperTest, ForcedOpenGlNeverEnablesOpenClSerialization) {
  InferenceCalculatorOptions options;
  InferenceCalculatorOptions::Delegate::Gpu gpu;
  gpu.set_model_token("model");
  gpu.set_cached_kernel_path("/private/kernels");
  gpu.set_serialized_model_dir("/private/models");
  InferenceOnDiskCacheHelper helper;
  for (auto api : {InferenceCalculatorOptions::Delegate::Gpu::ANY,
                   InferenceCalculatorOptions::Delegate::Gpu::OPENCL,
                   InferenceCalculatorOptions::Delegate::Gpu::OPENGL}) {
    gpu.set_api(api);
    MP_ASSERT_OK(helper.Init(options, gpu));
    EXPECT_EQ(helper.UseSerializedModel(),
              api != InferenceCalculatorOptions::Delegate::Gpu::OPENGL);
  }
}
}  // namespace mediapipe::api2
