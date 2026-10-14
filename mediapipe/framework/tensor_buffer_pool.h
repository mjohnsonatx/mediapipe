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

#ifndef MEDIAPIPE_FRAMEWORK_TENSOR_BUFFER_POOL_H_
#define MEDIAPIPE_FRAMEWORK_TENSOR_BUFFER_POOL_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/gpu/multi_pool.h"

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gl_context.h"
#endif

namespace mediapipe {

// Identifies interchangeable Tensor SSBO backing. Dimensions are retained
// in the key even when two shapes have the same byte size so dynamic model
// topology changes cannot accidentally share a differently shaped allocation.
struct TensorBufferSpec {
  int element_type = 0;
  // Tensor shapes in MediaPipe are normally rank <= 4. Keeping the shape inline avoids replacing
  // the large SSBO allocation with a small heap allocation on every pool lease.
  absl::InlinedVector<int, 4> dimensions;
  int byte_size = 0;
  // Reuse only on the allocating context, never across unrelated GL share groups.
  uintptr_t context_id = 0;

  bool operator<(const TensorBufferSpec& other) const;
};

// A bounded, shape-keyed pool for Tensor OpenGL shader-storage buffers only.
// CPU backing and GPU-to-CPU readback retain their original allocation path.
// Defaults retain at most two buffers per shape and ten idle shape keys. In-use
// leases are never evicted; temporary overflow drains when those leases return.
class TensorBufferPool
    : public std::enable_shared_from_this<TensorBufferPool> {
 public:
  class Buffer {
   public:
    explicit Buffer(TensorBufferSpec spec);
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer();

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
    GLuint GetOpenGlBuffer(const std::shared_ptr<GlContext>& current_context);
    bool has_opengl_buffer() const {
      return opengl_buffer_ != GL_INVALID_INDEX;
    }
    // A reader on another context is not ordered by the allocating context's
    // command queue. Such a buffer takes the original deletion path, not reuse.
    void DisableReuse() { reusable_ = false; }
#endif
    bool reusable() const { return reusable_; }

   private:
    TensorBufferSpec spec_;
    bool reusable_ = true;

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
    GLuint opengl_buffer_ = GL_INVALID_INDEX;
    std::shared_ptr<GlContext> gl_context_;
#endif
  };

  explicit TensorBufferPool(
      MultiPoolOptions options = kDefaultMultiPoolOptions)
      : options_(options) {}

  // Returns an exclusive lease. Releasing the shared_ptr returns the backing
  // to this pool, up to keep_count entries for the exact shape key.
  std::shared_ptr<Buffer> GetBuffer(const TensorBufferSpec& spec);

 private:
  struct Entry {
    int in_use = 0;
    uint64_t last_use = 0;
    std::vector<std::unique_ptr<Buffer>> available;
  };

  void Return(const TensorBufferSpec& spec, std::unique_ptr<Buffer> buffer);
  void EvictUnusedEntries(std::vector<std::unique_ptr<Buffer>>* evicted)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  const MultiPoolOptions options_;
  absl::Mutex mutex_;
  uint64_t use_sequence_ ABSL_GUARDED_BY(mutex_) = 0;
  std::map<TensorBufferSpec, Entry> entries_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_TENSOR_BUFFER_POOL_H_
