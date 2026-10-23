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

#include "mediapipe/calculators/tensor/image_to_tensor_converter_opencv.h"

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory>

#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_opencv.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

namespace mediapipe {
namespace {

uint8_t Pixel(int x, int y, int channel) {
  return static_cast<uint8_t>((x * 17 + y * 31 + channel * 67) % 256);
}

std::shared_ptr<ImageFrame> Pattern(int width, int height, int channels) {
  const auto format = channels == 1 ? ImageFormat::GRAY8
                      : channels == 3 ? ImageFormat::SRGB : ImageFormat::SRGBA;
  auto frame = std::make_shared<ImageFrame>(format, width, height, 64);
  for (int y = 0; y < height; ++y) {
    auto* row = frame->MutablePixelData() + y * frame->WidthStep();
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) row[x * channels + c] = Pixel(x, y, c);
    }
  }
  return frame;
}

TEST(ImageToTensorOpenCvConverterTest, AlignedPoseCropKeepsRgbAndNormalization) {
  const Image image(Pattern(256, 256, 4));
  auto converter = CreateOpenCvConverter(nullptr, BorderMode::kZero,
                                        Tensor::ElementType::kFloat32);
  ASSERT_TRUE(converter.ok()) << converter.status();
  Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape({1, 256, 256, 3}));
  ASSERT_TRUE((*converter)->Convert(image, {128.f, 128.f, 256.f, 256.f, 0.f},
                                   -1.f, 1.f, 0, tensor).ok());
  const auto view = tensor.GetCpuReadView();
  for (int y = 0; y < 256; ++y) {
    for (int x = 0; x < 256; ++x) {
      for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(view.buffer<float>()[(y * 256 + x) * 3 + c],
                    Pixel(x, y, c) * (2.f / 255.f) - 1.f, 1e-6f);
      }
    }
  }
}

TEST(ImageToTensorOpenCvConverterTest, AlignedInputHonorsRowStrideAndFloatOffset) {
  for (int channels : {1, 3, 4}) {
    const auto frame = Pattern(5, 3, channels);
    ASSERT_GT(frame->WidthStep(), 5 * channels);
    const Image image(frame);
    const int output_channels = channels == 1 ? 1 : 3;
    const int elements = 5 * 3 * output_channels;
    auto converter = CreateOpenCvConverter(nullptr, BorderMode::kReplicate,
                                          Tensor::ElementType::kFloat32);
    ASSERT_TRUE(converter.ok()) << converter.status();
    Tensor tensor(Tensor::ElementType::kFloat32,
                  Tensor::Shape({2, 3, 5, output_channels}));
    ASSERT_TRUE((*converter)->Convert(image, {2.5f, 1.5f, 5.f, 3.f, 0.f},
                                     0.f, 1.f, elements * sizeof(float), tensor).ok());
    const auto view = tensor.GetCpuReadView();
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 5; ++x) {
        for (int c = 0; c < output_channels; ++c) {
          EXPECT_NEAR(view.buffer<float>()[elements + (y * 5 + x) * output_channels + c],
                      Pixel(x, y, c) / 255.f, 1e-6f);
          EXPECT_EQ(frame->PixelData()[y * frame->WidthStep() + x * channels + c],
                    Pixel(x, y, c));
        }
      }
    }
  }
}

TEST(ImageToTensorOpenCvConverterTest, AlignedInputPreservesQuantizedRanges) {
  const Image image(Pattern(5, 3, 4));
  for (auto type : {Tensor::ElementType::kUInt8, Tensor::ElementType::kInt8}) {
    auto converter = CreateOpenCvConverter(nullptr, BorderMode::kZero, type);
    ASSERT_TRUE(converter.ok()) << converter.status();
    const bool signed_output = type == Tensor::ElementType::kInt8;
    Tensor tensor(type, Tensor::Shape({1, 3, 5, 3}));
    ASSERT_TRUE((*converter)->Convert(image, {2.5f, 1.5f, 5.f, 3.f, 0.f},
                                     signed_output ? -128.f : 0.f,
                                     signed_output ? 127.f : 255.f, 0, tensor).ok());
    const auto view = tensor.GetCpuReadView();
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 5; ++x) {
        for (int c = 0; c < 3; ++c) {
          const int index = (y * 5 + x) * 3 + c;
          const int actual = signed_output ? view.buffer<int8_t>()[index]
                                           : view.buffer<uint8_t>()[index];
          EXPECT_EQ(actual, static_cast<int>(Pixel(x, y, c)) - (signed_output ? 128 : 0));
        }
      }
    }
  }
}

TEST(ImageToTensorOpenCvConverterTest, NonIdentityRoisKeepTheExistingWarp) {
  const Image image(Pattern(9, 7, 4));
  const auto src = formats::MatView(&image);
  struct Case { RotatedRect roi; int width; int height; };
  const Case cases[] = {
      {{4.75f, 3.5f, 9.f, 7.f, 0.f}, 9, 7},  // Subpixel translation.
      {{4.5f, 3.5f, 9.f, 7.f, 0.3f}, 9, 7},  // Rotation.
      {{4.5f, 3.5f, 4.5f, 3.5f, 0.f}, 9, 7}, // Crop and upscale.
      {{4.5f, 3.5f, 9.f, 7.f, 0.f}, 5, 3},   // Resize only.
      {{0.f, 0.f, 9.f, 7.f, 0.f}, 9, 7},     // Out-of-bounds padding.
  };
  for (auto border : {BorderMode::kZero, BorderMode::kReplicate}) {
    auto converter = CreateOpenCvConverter(nullptr, border, Tensor::ElementType::kFloat32);
    ASSERT_TRUE(converter.ok()) << converter.status();
    for (const auto& item : cases) {
      // Reference the former unconditional OpenCV path, including border/interpolation semantics.
      cv::Mat source_points;
      cv::boxPoints(cv::RotatedRect({item.roi.center_x, item.roi.center_y},
                    {item.roi.width, item.roi.height}, item.roi.rotation * 180.f / M_PI),
                    source_points);
      const float w = item.width;
      const float h = item.height;
      float corners[] = {0.f, h, 0.f, 0.f, w, 0.f, w, h};
      const cv::Mat destination_points(4, 2, CV_32F, corners);
      cv::Mat warped, rgb, expected;
      cv::warpPerspective(*src, warped,
                          cv::getPerspectiveTransform(source_points, destination_points),
                          cv::Size(item.width, item.height), cv::INTER_LINEAR,
                          border == BorderMode::kZero ? cv::BORDER_CONSTANT : cv::BORDER_REPLICATE);
      cv::cvtColor(warped, rgb, cv::COLOR_RGBA2RGB);
      rgb.convertTo(expected, CV_32FC3, 1.f / 255.f);
      Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape({1, item.height, item.width, 3}));
      ASSERT_TRUE((*converter)->Convert(image, item.roi, 0.f, 1.f, 0, tensor).ok());
      const auto view = tensor.GetCpuReadView();
      for (int y = 0; y < item.height; ++y) {
        for (int x = 0; x < item.width * 3; ++x) {
          EXPECT_NEAR(view.buffer<float>()[y * item.width * 3 + x], expected.ptr<float>(y)[x], 1e-6f);
        }
      }
    }
  }
}

}  // namespace
}  // namespace mediapipe
