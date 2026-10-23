// Copyright 2026 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied.
#ifndef MEDIAPIPE_TASKS_CC_VISION_POSE_LANDMARKER_POSE_GPU_OPTIONS_H_
#define MEDIAPIPE_TASKS_CC_VISION_POSE_LANDMARKER_POSE_GPU_OPTIONS_H_

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/tasks/cc/core/proto/acceleration.pb.h"

namespace mediapipe::tasks::vision::pose_landmarker {

// Each embedded model owns a distinct compiled cache. Copy the entire parent
// acceleration so forced APIs, precision, usage, and cache-write policy survive
// graph expansion. Derive from the parent every time, never an already-suffixed child.
inline core::proto::Acceleration PoseSubtaskAcceleration(
    const core::proto::Acceleration& parent, absl::string_view model_role) {
  auto child = parent;
  if (!child.has_gpu()) return child;
  auto* gpu = child.mutable_gpu();
  gpu->set_use_advanced_gpu_api(true);
  if (!gpu->model_token().empty()) {
    gpu->set_model_token(absl::StrCat(
        gpu->model_token(), "_posecache_v2_", model_role, "_api", gpu->api(),
        "_precision", gpu->allow_precision_loss(), "_usage", gpu->usage()));
  } else {
    // Embedded models have no caller-provided unique identity. Avoid a shared
    // basename cache; inference still works without disk caching.
    gpu->clear_cached_kernel_path();
    gpu->clear_serialized_model_dir();
  }
  return child;
}

}  // namespace mediapipe::tasks::vision::pose_landmarker
#endif  // MEDIAPIPE_TASKS_CC_VISION_POSE_LANDMARKER_POSE_GPU_OPTIONS_H_
