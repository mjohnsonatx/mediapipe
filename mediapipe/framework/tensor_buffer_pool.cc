// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0

#include "mediapipe/framework/tensor_buffer_pool.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <map>
#include <tuple>
#include <utility>

#include "absl/log/absl_check.h"
#include "mediapipe/framework/port/aligned_malloc_and_free.h"

namespace mediapipe {

bool TensorBufferSpec::operator<(const TensorBufferSpec& other) const {
  return std::tie(element_type, dimensions, byte_size, memory_alignment,
                  context_id) <
         std::tie(other.element_type, other.dimensions, other.byte_size,
                  other.memory_alignment, other.context_id);
}

TensorBufferPool::Buffer::Buffer(TensorBufferSpec spec)
    : spec_(std::move(spec)) {}

TensorBufferPool::Buffer::~Buffer() {
  if (spec_.memory_alignment > 0) {
    aligned_free(cpu_buffer_);
  } else {
    free(cpu_buffer_);
  }
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

void* TensorBufferPool::Buffer::GetCpuBuffer() {
  if (cpu_buffer_ == nullptr) {
    if (spec_.memory_alignment > 0) {
      cpu_buffer_ = aligned_malloc(
          std::max(spec_.memory_alignment, spec_.byte_size),
          spec_.memory_alignment);
    } else {
      cpu_buffer_ = malloc(spec_.byte_size);
    }
    ABSL_CHECK(cpu_buffer_ != nullptr)
        << "Failed to allocate Tensor CPU buffer.";
  }
  return cpu_buffer_;
}

bool TensorBufferPool::Buffer::AdoptCpuBuffer(void* cpu_buffer) {
  ABSL_CHECK(cpu_buffer != nullptr);
  if (cpu_buffer_ != nullptr) return false;
  cpu_buffer_ = cpu_buffer;
  return true;
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

// Counts shape entries across owners, including in-flight entries. Only shape
// creation/eviction changes the count; a steady-state return only loads it.
// Entries exceeding the shared cap may exist while live, just as before, but
// each returning owner evicts its own idle entries until the cap is satisfied.
struct TensorBufferPool::KeyBudget {
  explicit KeyBudget(int max_keys) : max_keys(std::max(1, max_keys)) {}
  const size_t max_keys;
  std::atomic<size_t> key_count{0};
};

class TensorBufferPool::ContextPool
    : public std::enable_shared_from_this<ContextPool> {
 public:
  using Cache = std::map<uint64_t, ContextPool*>;

  static Cache& GetCache(GlContext& context) {
    static constexpr GlContext::Attachment<Cache> kCache(
        [](GlContext&) { return GlContext::Attachment<Cache>::MakePtr(); });
    return context.GetCachedAttachment(kCache);
  }

  static std::shared_ptr<ContextPool> Create(
      uint64_t pool_id, int keep_count, std::shared_ptr<KeyBudget> budget,
      const std::shared_ptr<GlContext>& context) {
    return std::shared_ptr<ContextPool>(
        new ContextPool(pool_id, keep_count, std::move(budget), context),
        [](ContextPool* owner) {
          // Pool shutdown may occur on a Java/graph worker. Retire the cache
          // entry and free lists only on their owner, after queued work.
          auto context = owner->context_;
          if (context->IsCurrent()) {
            delete owner;
          } else {
            context->RunWithoutWaiting([owner, context] { delete owner; });
          }
        });
  }

  ~ContextPool() {
    ABSL_CHECK(context_->IsCurrent());
    GetCache(*context_).erase(pool_id_);
    budget_->key_count.fetch_sub(entries_.size());
    // Buffer destruction runs here with its allocating context current.
  }

  std::shared_ptr<Buffer> GetBuffer(const TensorBufferSpec& spec) {
    ABSL_DCHECK(context_->IsCurrent());
    auto [it, inserted] = entries_.try_emplace(spec);
    if (inserted) budget_->key_count.fetch_add(1);
    Entry& entry = it->second;
    entry.last_use = ++use_sequence_;
    ++entry.in_use;
    std::unique_ptr<Buffer> buffer;
    if (!entry.available.empty()) {
      buffer = std::move(entry.available.back());
      entry.available.pop_back();
    } else {
      buffer = std::make_unique<Buffer>(spec);
    }

    std::weak_ptr<ContextPool> weak_owner = shared_from_this();
    return std::shared_ptr<Buffer>(
        buffer.release(),
        [weak_owner, spec, context = context_](Buffer* returned) {
          auto release = [weak_owner, spec, returned] {
            if (auto owner = weak_owner.lock()) {
              owner->Return(spec, std::unique_ptr<Buffer>(returned));
            } else {
              delete returned;
            }
          };
          if (context->IsCurrent()) {
            release();
          } else {
            context->RunWithoutWaiting(std::move(release));
          }
        });
  }

 private:
  struct Entry {
    int in_use = 0;
    uint64_t last_use = 0;
    std::vector<std::unique_ptr<Buffer>> available;
  };

  ContextPool(uint64_t pool_id, int keep_count, std::shared_ptr<KeyBudget> budget,
              std::shared_ptr<GlContext> context)
      : pool_id_(pool_id),
        keep_count_(keep_count),
        budget_(std::move(budget)),
        context_(std::move(context)) {}

  void Return(const TensorBufferSpec& spec, std::unique_ptr<Buffer> buffer) {
    ABSL_DCHECK(context_->IsCurrent());
    auto it = entries_.find(spec);
    ABSL_CHECK(it != entries_.end());  // Live entries are never evicted.
    Entry& entry = it->second;
    --entry.in_use;
    entry.last_use = ++use_sequence_;
    const int retained_limit = std::max(0, keep_count_ - entry.in_use);
    if (buffer->reusable() && entry.available.size() < retained_limit) {
      entry.available.push_back(std::move(buffer));
    }
    EvictUnusedEntries();
  }

  void EvictUnusedEntries() {
    while (budget_->key_count.load() > budget_->max_keys) {
      auto oldest = entries_.end();
      for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->second.in_use != 0) continue;
        if (oldest == entries_.end() ||
            it->second.last_use < oldest->second.last_use) {
          oldest = it;
        }
      }
      if (oldest == entries_.end()) return;
      // Eviction is owner-local LRU. Never inspect or mutate another context's
      // free list, and never synchronously wait for a peer context to evict.
      entries_.erase(oldest);
      budget_->key_count.fetch_sub(1);
    }
  }

  const uint64_t pool_id_;
  const int keep_count_;
  const std::shared_ptr<KeyBudget> budget_;
  const std::shared_ptr<GlContext> context_;
  // Exclusively accessed with context_ current, even when that context migrates
  // between OS threads. No thread-local cache or cross-context mutable map.
  uint64_t use_sequence_ = 0;
  std::map<TensorBufferSpec, Entry> entries_;
};

std::shared_ptr<TensorBufferPool::Buffer> TensorBufferPool::GetBuffer(
    const TensorBufferSpec& spec,
    const std::shared_ptr<GlContext>& current_context) {
  ABSL_CHECK(current_context != nullptr && current_context->IsCurrent());
  ABSL_CHECK_EQ(spec.context_id,
                reinterpret_cast<uintptr_t>(current_context.get()));
  auto& cache = ContextPool::GetCache(*current_context);
  auto cached = cache.find(pool_id_);
  if (cached != cache.end()) {
    return cached->second->GetBuffer(spec);
  }

  auto owner = ContextPool::Create(pool_id_, options_.keep_count, key_budget_,
                                   current_context);
  bool registered = false;
  {
    absl::MutexLock lock(&registration_mutex_);
    // Also bound retained contexts. A graph with context churn must not retain
    // arbitrarily many GL threads just because their shape maps became empty.
    if (context_pools_.size() < key_budget_->max_keys) {
      context_pools_.push_back(owner);
      registered = true;
    }
  }
  if (!registered) return std::make_shared<Buffer>(spec);
  cache.emplace(pool_id_, owner.get());
  return owner->GetBuffer(spec);
}
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

TensorBufferPool::TensorBufferPool(MultiPoolOptions options)
    : options_(options)
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
      , pool_id_([] {
          static std::atomic<uint64_t> next_id{1};
          return next_id.fetch_add(1, std::memory_order_relaxed);
        }()),
      key_budget_(std::make_shared<KeyBudget>(options.max_pool_count))
#endif
{}

TensorBufferPool::~TensorBufferPool() = default;

}  // namespace mediapipe
