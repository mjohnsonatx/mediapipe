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
// Each GL context exclusively owns its shape map and free lists. The only pool
// mutex registers new context owners; borrowing/returning on a registered owner
// does not acquire it. The key budget is shared, not multiplied per context.
class TensorBufferPool {
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
      MultiPoolOptions options = kDefaultMultiPoolOptions);
  ~TensorBufferPool();

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
  // Must be called with current_context current. The last lease may be released
  // on any thread: its deleter routes bookkeeping back to that context. A lease
  // outliving this pool deletes its backing instead of accessing retired state.
  std::shared_ptr<Buffer> GetBuffer(
      const TensorBufferSpec& spec,
      const std::shared_ptr<GlContext>& current_context);
#endif

 private:
  const MultiPoolOptions options_;
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
  struct KeyBudget;
  class ContextPool;
  const uint64_t pool_id_;
  std::shared_ptr<KeyBudget> key_budget_;
  // Cold registration only. Context caches hold non-owning pointers, so they
  // cannot form a GlContext -> pool -> GlContext ownership cycle.
  absl::Mutex registration_mutex_;
  std::vector<std::shared_ptr<ContextPool>> context_pools_
      ABSL_GUARDED_BY(registration_mutex_);
#endif
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_TENSOR_BUFFER_POOL_H_
