// Copyright 2026 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"

namespace mediapipe {
namespace {

CalculatorGraphConfig::Node Config() {
  return ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "PoseTensorInputCalculator"
    input_stream: "IMAGE:image"
    input_stream: "NORM_RECT:rect"
    input_stream: "EXTERNAL_TENSORS:external"
    output_stream: "IMAGE:preprocess_image"
    output_stream: "NORM_RECT:preprocess_rect"
    output_stream: "TENSORS:tensors"
    output_stream: "MATRIX:matrix"
    output_stream: "LETTERBOX_PADDING:padding"
  )pb");
}

NormalizedRect FullRect() {
  NormalizedRect rect;
  rect.set_x_center(0.5f);
  rect.set_y_center(0.5f);
  rect.set_width(1.0f);
  rect.set_height(1.0f);
  rect.set_rotation(0.0f);
  return rect;
}

void AddFrame(CalculatorRunner& runner, int64_t time, int image_size,
              NormalizedRect rect, std::vector<Tensor> tensors) {
  auto frame = std::make_shared<ImageFrame>(ImageFormat::SRGB, image_size, image_size);
  frame->SetToZero();
  runner.MutableInputs()->Tag("IMAGE").packets.push_back(
      MakePacket<Image>(frame).At(Timestamp(time)));
  runner.MutableInputs()->Tag("NORM_RECT").packets.push_back(
      MakePacket<NormalizedRect>(std::move(rect)).At(Timestamp(time)));
  runner.MutableInputs()->Tag("EXTERNAL_TENSORS").packets.push_back(
      MakePacket<std::vector<Tensor>>(std::move(tensors)).At(Timestamp(time)));
}

std::vector<Tensor> Prepared(int channels = 3) {
  std::vector<Tensor> tensors;
  tensors.emplace_back(Tensor::ElementType::kFloat32, Tensor::Shape{1, 256, 256, channels});
  return tensors;
}

TEST(PoseTensorInputCalculatorTest, AlternatesAcquisitionTensorAndReacquisition) {
  CalculatorRunner runner(Config());
  AddFrame(runner, 1, 640, FullRect(), {});
  AddFrame(runner, 2, 256, FullRect(), Prepared());
  AddFrame(runner, 3, 640, FullRect(), {});
  ASSERT_TRUE(runner.Run().ok());
  const auto& images = runner.Outputs().Tag("IMAGE").packets;
  ASSERT_EQ(images.size(), 2);
  EXPECT_EQ(images[0].Timestamp(), Timestamp(1));
  EXPECT_EQ(images[1].Timestamp(), Timestamp(3));
  ASSERT_EQ(runner.Outputs().Tag("NORM_RECT").packets.size(), 2);
  const auto& tensors = runner.Outputs().Tag("TENSORS").packets;
  ASSERT_EQ(tensors.size(), 1);
  EXPECT_EQ(tensors[0].Timestamp(), Timestamp(2));
  const auto& matrices = runner.Outputs().Tag("MATRIX").packets;
  const auto& padding = runner.Outputs().Tag("LETTERBOX_PADDING").packets;
  ASSERT_EQ(matrices.size(), 1);
  ASSERT_EQ(padding.size(), 1);
  const std::array<float, 16> identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  const std::array<float, 4> zero{0, 0, 0, 0};
  EXPECT_EQ((matrices[0].Get<std::array<float, 16>>()), identity);
  EXPECT_EQ((padding[0].Get<std::array<float, 4>>()), zero);
}

TEST(PoseTensorInputCalculatorTest, RejectsWrongShape) {
  CalculatorRunner runner(Config());
  AddFrame(runner, 1, 256, FullRect(), Prepared(4));
  EXPECT_FALSE(runner.Run().ok());
}

TEST(PoseTensorInputCalculatorTest, RejectsNonIdentityRoi) {
  CalculatorRunner runner(Config());
  auto rect = FullRect();
  rect.set_rotation(0.1f);
  AddFrame(runner, 1, 256, rect, Prepared());
  EXPECT_FALSE(runner.Run().ok());
}

TEST(PoseTensorInputCalculatorTest, RejectsWrongMetadataImageSize) {
  CalculatorRunner runner(Config());
  AddFrame(runner, 1, 640, FullRect(), Prepared());
  EXPECT_FALSE(runner.Run().ok());
}
}  // namespace
}  // namespace mediapipe
