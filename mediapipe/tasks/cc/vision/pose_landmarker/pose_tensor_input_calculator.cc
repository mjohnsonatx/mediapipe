// Copyright 2026 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

#include <array>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

// Empty tensor vectors select ordinary image preprocessing. Nonempty vectors are
// already packed by the application; no ImageClone/ImageToTensor work is emitted.
// Every timestamp must carry a vector, including acquisition, so BeginLoop CLONE
// inputs can advance even when the detector returns no rectangles.
class PoseTensorInputCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag("EXTERNAL_TENSORS").Set<std::vector<Tensor>>();
    cc->Inputs().Tag("IMAGE").Set<Image>();
    cc->Inputs().Tag("NORM_RECT").Set<NormalizedRect>();
    cc->Outputs().Tag("IMAGE").Set<Image>();
    cc->Outputs().Tag("NORM_RECT").Set<NormalizedRect>();
    cc->Outputs().Tag("TENSORS").Set<std::vector<Tensor>>();
    cc->Outputs().Tag("MATRIX").Set<std::array<float, 16>>();
    cc->Outputs().Tag("LETTERBOX_PADDING").Set<std::array<float, 4>>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    RET_CHECK(!cc->Inputs().Tag("EXTERNAL_TENSORS").IsEmpty());
    RET_CHECK(!cc->Inputs().Tag("IMAGE").IsEmpty());
    RET_CHECK(!cc->Inputs().Tag("NORM_RECT").IsEmpty());
    const auto& tensors = cc->Inputs().Tag("EXTERNAL_TENSORS").Get<std::vector<Tensor>>();
    if (tensors.empty()) {
      cc->Outputs().Tag("IMAGE").AddPacket(cc->Inputs().Tag("IMAGE").Value());
      cc->Outputs().Tag("NORM_RECT").AddPacket(cc->Inputs().Tag("NORM_RECT").Value());
      return absl::OkStatus();
    }
    RET_CHECK_EQ(tensors.size(), 1);
    RET_CHECK(tensors[0].element_type() == Tensor::ElementType::kFloat32);
    const auto& shape = tensors[0].shape().dims;
    RET_CHECK(shape.size() == 4 && shape[0] == 1 && shape[1] == 256 &&
              shape[2] == 256 && shape[3] == 3);
    const auto& image = cc->Inputs().Tag("IMAGE").Get<Image>();
    RET_CHECK(image.width() == 256 && image.height() == 256);
    const auto& rect = cc->Inputs().Tag("NORM_RECT").Get<NormalizedRect>();
    RET_CHECK(rect.x_center() == 0.5f && rect.y_center() == 0.5f &&
              rect.width() == 1.0f && rect.height() == 1.0f && rect.rotation() == 0.0f)
        << "Prepared tensors require the full-image tracking sentinel.";
    cc->Outputs().Tag("TENSORS").AddPacket(cc->Inputs().Tag("EXTERNAL_TENSORS").Value());
    cc->Outputs().Tag("MATRIX").AddPacket(
        MakePacket<std::array<float, 16>>(std::array<float, 16>{
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}).At(cc->InputTimestamp()));
    cc->Outputs().Tag("LETTERBOX_PADDING").AddPacket(
        MakePacket<std::array<float, 4>>(std::array<float, 4>{0, 0, 0, 0}).At(cc->InputTimestamp()));
    return absl::OkStatus();
  }
};

REGISTER_CALCULATOR(PoseTensorInputCalculator);
}  // namespace mediapipe
