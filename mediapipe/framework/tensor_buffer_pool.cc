// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0

#include "mediapipe/framework/tensor_buffer_pool.h"

#include <algorithm>
#include <iterator>
#include <tuple>
#include <utility>

#include "absl/log/absl_check.h"

namespace mediapipe {

bool TensorBufferSpec::operator<(const TensorBufferSpec& other) const {
  return std::tie(element_type, dimensions, byte_size, context_id) <
         std::tie(other.element_type, other.dimensions, other.byte_size,
                  other.context_id);
}

TensorBufferPool::Buffer::Buffer(TensorBufferSpec spec)
    : spec_(std::move(spec)) {}

TensorBufferPool::Buffer::~Buffer() {
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
  GLuint buffer = std::exchange(opengl_buffer_, GL_INVALID_INDEX);
  if (buffer != GL_INVALID_INDEX && gl_context_ != nullptr) {
    if (gl_context_->IsCurrent()) {
      glDeleteBuffers(1, &buffer);
    } else {
      gl_context_->RunWithoutWaiting([buffer]() { glDeleteBuffers(1, &buffer); });
    }
  }
#endif
}

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
GLuint TensorBufferPool::Buffer::GetOpenGlBuffer(
    const std::shared_ptr<GlContext>& current_context) {
  ABSL_CHECK(current_context != nullptr);
  ABSL_CHECK(current_context->IsCurrent());

  ABSL_CHECK_EQ(spec_.context_id,
                reinterpret_cast<uintptr_t>(current_context.get()));

  if (opengl_buffer_ == GL_INVALID_INDEX) {
    gl_context_ = current_context;
    glGenBuffers(1, &opengl_buffer_);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, opengl_buffer_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, spec_.byte_size, nullptr,
                 GL_STREAM_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  }
  return opengl_buffer_;
}

#endif

std::shared_ptr<TensorBufferPool::Buffer> TensorBufferPool::GetBuffer(
    const TensorBufferSpec& spec) {
  std::unique_ptr<Buffer> buffer;
  {
    absl::MutexLock lock(&mutex_);
    Entry& entry = entries_[spec];
    entry.last_use = ++use_sequence_;
    ++entry.in_use;
    if (!entry.available.empty()) {
      buffer = std::move(entry.available.back());
      entry.available.pop_back();
    }
  }
  if (buffer == nullptr) {
    buffer = std::make_unique<Buffer>(spec);
  }

  std::weak_ptr<TensorBufferPool> weak_pool(shared_from_this());
  return std::shared_ptr<Buffer>(
      buffer.release(), [weak_pool, spec](Buffer* returned) {
        if (auto pool = weak_pool.lock()) {
          pool->Return(spec, std::unique_ptr<Buffer>(returned));
        } else {
          delete returned;
        }
      });
}

void TensorBufferPool::Return(const TensorBufferSpec& spec,
                              std::unique_ptr<Buffer> buffer) {
  std::vector<std::unique_ptr<Buffer>> evicted;
  {
    absl::MutexLock lock(&mutex_);
    auto entry_it = entries_.find(spec);
    if (entry_it == entries_.end()) {
      evicted.push_back(std::move(buffer));
    } else {
      Entry& entry = entry_it->second;
      --entry.in_use;
      entry.last_use = ++use_sequence_;
      const int retained_limit = std::max(0, options_.keep_count - entry.in_use);
      if (buffer->reusable() && entry.available.size() < retained_limit) {
        entry.available.push_back(std::move(buffer));
      } else {
        evicted.push_back(std::move(buffer));
      }
      EvictUnusedEntries(&evicted);
    }
  }
  // Release allocations without holding the pool lock. GL destruction may
  // enqueue work on the owning context.
}

void TensorBufferPool::EvictUnusedEntries(
    std::vector<std::unique_ptr<Buffer>>* evicted) {
  while (entries_.size() > std::max(1, options_.max_pool_count)) {
    auto oldest = entries_.end();
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->second.in_use != 0) continue;
      if (oldest == entries_.end() ||
          it->second.last_use < oldest->second.last_use) {
        oldest = it;
      }
    }
    if (oldest == entries_.end()) return;
    std::move(oldest->second.available.begin(),
              oldest->second.available.end(), std::back_inserter(*evicted));
    entries_.erase(oldest);
  }
}

}  // namespace mediapipe
