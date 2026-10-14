// Copyright 2024 The MediaPipe Authors.
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
#ifndef MEDIAPIPE_FRAMEWORK_MEMORY_MANAGER_H_
#define MEDIAPIPE_FRAMEWORK_MEMORY_MANAGER_H_

#include <memory>

// Defines MEDIAPIPE_TENSOR_USE_AHWB
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/tensor_buffer_pool.h"
#include "mediapipe/gpu/multi_pool.h"

#ifdef MEDIAPIPE_TENSOR_USE_AHWB
#include "mediapipe/framework/formats/hardware_buffer_pool.h"
#endif

namespace mediapipe {

// Owns buffer pools to provide access to pooled buffer objects. Access is
// managed via shared_ptrs to allow clients of buffer objects to control their
// lifetime.
//
// Example usage:
// 1) Instantiate the MemoryManager and pass it to the kMemoryManagerService
//    before graph initialization:
//    CalculatorGraph graph;
//    graph.SetServiceObject(kMemoryManagerService,
//                           std::make_shared<MemoryManager>());
//    graph.Initialize(...);
// 2) Lookup MemoryManager instance in Calculator::Open():
//    absl::Status Calculator::Open(CalculatorContext* cc) {
//      auto memory_manager_service = cc->Service(kMemoryManagerService);
//      if (memory_manager_service.IsAvailable()) {
//        memory_manager_ = &memory_manager_service.GetObject();
//      } else {
//        ABSL_LOG(WARNING) << "MemoryManager not available";
//      }
//      ...
//    }
//    private:
//       MemoryManager* memory_manager_ = nullptr;
// 3) Pass Calculator::memory_manager_ to the Tensor class constructor:
//       Tensor tensor(Tensor::ElementType::kFloat32,
//                     Tensor::Shape{kTensorSize}, &memory_manager_);

class MemoryManager {
 public:
  // If prefer_ahwb is true, Tensors written on CPU and read as OpenGL buffers
  // or vice versa will use Android Hardware Buffers as storage. This prevents
  // unnecessary copies, but extra AHWB allocations may outweigh the benefits of
  // avoiding copies. Therefore, make sure to pick MultiPoolOptions carefully
  // to avoid unnecessary AHWB allocations.
  explicit MemoryManager(MultiPoolOptions options = kDefaultMultiPoolOptions,
                         bool prefer_ahwb = false) {
#ifdef MEDIAPIPE_TENSOR_USE_AHWB
    hardware_buffer_pool_ =
        std::make_shared<HardwareBufferPool>(options);
#endif
    tensor_buffer_pool_ = std::make_shared<TensorBufferPool>(options);
    prefer_ahwb_ = prefer_ahwb;
  }

#ifdef MEDIAPIPE_TENSOR_USE_AHWB
  std::shared_ptr<HardwareBufferPool> GetAndroidHardwareBufferPool() const {
    return hardware_buffer_pool_;
  }
#endif

  std::shared_ptr<TensorBufferPool> GetTensorBufferPool() const {
    return tensor_buffer_pool_;
  }

  bool PreferAhwb() const { return prefer_ahwb_; }

 private:
#ifdef MEDIAPIPE_TENSOR_USE_AHWB
  std::shared_ptr<HardwareBufferPool> hardware_buffer_pool_;
#endif
  std::shared_ptr<TensorBufferPool> tensor_buffer_pool_;
  bool prefer_ahwb_ = false;
};

}  //  namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_MEMORY_MANAGER_H_
