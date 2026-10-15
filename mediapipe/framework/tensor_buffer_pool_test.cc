// Copyright 2026 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0

#include "mediapipe/framework/tensor_buffer_pool.h"

#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

#include "absl/synchronization/barrier.h"
#include "mediapipe/framework/port/gtest.h"

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
namespace mediapipe {
namespace {

class TensorBufferPoolTest : public testing::Test {
 protected:
  void SetUp() override {
    auto context = GlContext::Create(nullptr, /*create_thread=*/true);
    ASSERT_TRUE(context.ok()) << context.status();
    context_ = *context;
    MultiPoolOptions options;
    options.keep_count = 2;
    options.max_pool_count = 2;
    pool_ = std::make_shared<TensorBufferPool>(options);
  }

  void TearDown() override {
    if (context_) context_->Run([this] { pool_.reset(); });
  }

  GLuint Acquire(const TensorBufferSpec& spec,
                 std::shared_ptr<TensorBufferPool::Buffer>* lease) {
    auto context_spec = spec;
    context_spec.context_id = reinterpret_cast<uintptr_t>(context_.get());
    *lease = pool_->GetBuffer(context_spec, context_);
    return (*lease)->GetOpenGlBuffer(context_);
  }

  void Return(std::shared_ptr<TensorBufferPool::Buffer>* lease) {
    lease->reset();
  }

  TensorBufferSpec spec_{/*element_type=*/1, /*dimensions=*/{1, 4},
                         /*byte_size=*/16};
  std::shared_ptr<GlContext> context_;
  std::shared_ptr<TensorBufferPool> pool_;
};

TEST_F(TensorBufferPoolTest, ReusesSsboWithoutAliasingActiveLease) {
  context_->Run([this] {
    std::shared_ptr<TensorBufferPool::Buffer> first, second, reused;
    const GLuint a = Acquire(spec_, &first);
    const GLuint b = Acquire(spec_, &second);
    EXPECT_NE(a, b);
    Return(&first);
    EXPECT_EQ(Acquire(spec_, &reused), a);
    EXPECT_TRUE(glIsBuffer(b));
    Return(&second);
    Return(&reused);
  });
}

TEST_F(TensorBufferPoolTest, ReusesCpuBackingWithItsSsboLease) {
  context_->Run([this] {
    std::shared_ptr<TensorBufferPool::Buffer> first, reused;
    const GLuint name = Acquire(spec_, &first);
    void* cpu_buffer = first->GetCpuBuffer();
    ASSERT_NE(cpu_buffer, nullptr);
    Return(&first);

    EXPECT_EQ(Acquire(spec_, &reused), name);
    EXPECT_EQ(reused->GetCpuBuffer(), cpu_buffer);
    Return(&reused);
  });
}

TEST_F(TensorBufferPoolTest, AdoptsCpuFirstAllocationIntoLease) {
  context_->Run([this] {
    std::shared_ptr<TensorBufferPool::Buffer> first, reused;
    Acquire(spec_, &first);
    void* cpu_buffer = malloc(spec_.byte_size);
    ASSERT_NE(cpu_buffer, nullptr);
    ASSERT_TRUE(first->AdoptCpuBuffer(cpu_buffer));
    Return(&first);

    Acquire(spec_, &reused);
    EXPECT_EQ(reused->GetCpuBuffer(), cpu_buffer);
    Return(&reused);
  });
}

TEST_F(TensorBufferPoolTest, ActiveLeasesDoNotAliasCpuBacking) {
  context_->Run([this] {
    std::shared_ptr<TensorBufferPool::Buffer> first, second;
    Acquire(spec_, &first);
    Acquire(spec_, &second);
    EXPECT_NE(first->GetCpuBuffer(), second->GetCpuBuffer());
    Return(&first);
    Return(&second);
  });
}

TEST_F(TensorBufferPoolTest, CpuAlignmentIsPartOfPoolKey) {
  context_->Run([this] {
    auto aligned_spec = spec_;
    aligned_spec.memory_alignment = 64;
    std::shared_ptr<TensorBufferPool::Buffer> aligned, unaligned;
    const GLuint aligned_name = Acquire(aligned_spec, &aligned);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(aligned->GetCpuBuffer()) % 64, 0);
    Return(&aligned);

    EXPECT_NE(Acquire(spec_, &unaligned), aligned_name);
    Return(&unaligned);
  });
}

TEST_F(TensorBufferPoolTest, EqualByteSizeDoesNotMixShapesOrElementTypes) {
  context_->Run([this] {
    std::shared_ptr<TensorBufferPool::Buffer> first, reshaped, retyped;
    const GLuint a = Acquire(spec_, &first);
    Return(&first);
    auto other = spec_;
    other.dimensions = {2, 2};
    const GLuint b = Acquire(other, &reshaped);
    EXPECT_NE(a, b);
    other.element_type = 2;
    const GLuint c = Acquire(other, &retyped);
    EXPECT_NE(b, c);
    Return(&reshaped);
    Return(&retyped);
  });
}

TEST_F(TensorBufferPoolTest, PerShapeLimitDeletesOverflow) {
  context_->Run([this] {
    std::shared_ptr<TensorBufferPool::Buffer> first, second, third;
    const GLuint a = Acquire(spec_, &first);
    const GLuint b = Acquire(spec_, &second);
    const GLuint c = Acquire(spec_, &third);
    Return(&first);
    EXPECT_FALSE(glIsBuffer(a));
    Return(&second);
    Return(&third);
    EXPECT_TRUE(glIsBuffer(b));
    EXPECT_TRUE(glIsBuffer(c));
  });
}

TEST_F(TensorBufferPoolTest, ShapeLimitEvictsLeastRecentlyUsedIdleBacking) {
  context_->Run([this] {
    GLuint names[3];
    for (int i = 0; i < 3; ++i) {
      auto spec = spec_;
      spec.dimensions = {i + 1, 4};
      spec.byte_size *= i + 1;
      std::shared_ptr<TensorBufferPool::Buffer> lease;
      names[i] = Acquire(spec, &lease);
      Return(&lease);
    }
    EXPECT_FALSE(glIsBuffer(names[0]));
    EXPECT_TRUE(glIsBuffer(names[1]));
    EXPECT_TRUE(glIsBuffer(names[2]));
  });
}

TEST_F(TensorBufferPoolTest, ShapeLimitDoesNotDeleteInFlightBuffers) {
  context_->Run([this] {
    std::vector<std::shared_ptr<TensorBufferPool::Buffer>> leases(3);
    GLuint names[3];
    for (int i = 0; i < 3; ++i) {
      auto spec = spec_;
      spec.dimensions = {i + 1, 4};
      spec.byte_size *= i + 1;
      names[i] = Acquire(spec, &leases[i]);
    }
    for (GLuint name : names) EXPECT_TRUE(glIsBuffer(name));
    Return(&leases[0]);
    EXPECT_FALSE(glIsBuffer(names[0]));
    EXPECT_TRUE(glIsBuffer(names[1]));
    EXPECT_TRUE(glIsBuffer(names[2]));
    Return(&leases[1]);
    Return(&leases[2]);
  });
}

TEST_F(TensorBufferPoolTest, OutstandingLeaseSurvivesPoolDestruction) {
  context_->Run([this] {
    std::shared_ptr<TensorBufferPool::Buffer> lease;
    const GLuint name = Acquire(spec_, &lease);
    pool_.reset();
    EXPECT_TRUE(glIsBuffer(name));
    Return(&lease);
    EXPECT_FALSE(glIsBuffer(name));
  });
}

TEST_F(TensorBufferPoolTest, CrossContextUseDisablesReuse) {
  context_->Run([this] {
    std::shared_ptr<TensorBufferPool::Buffer> lease;
    const GLuint name = Acquire(spec_, &lease);
    lease->DisableReuse();
    Return(&lease);
    EXPECT_FALSE(glIsBuffer(name));
  });
}

TEST_F(TensorBufferPoolTest, PoolKeysSeparateUnsharedContexts) {
  auto other_context = GlContext::Create(nullptr, /*create_thread=*/true);
  ASSERT_TRUE(other_context.ok()) << other_context.status();
  TensorBufferPool::Buffer* first_backing = nullptr;
  context_->Run([this, &first_backing] {
    std::shared_ptr<TensorBufferPool::Buffer> lease;
    Acquire(spec_, &lease);
    first_backing = lease.get();
    Return(&lease);
  });
  (*other_context)->Run([this, &other_context, first_backing] {
    auto other_spec = spec_;
    other_spec.context_id = reinterpret_cast<uintptr_t>(other_context->get());
    auto lease = pool_->GetBuffer(other_spec, *other_context);
    EXPECT_NE(lease.get(), first_backing);
    const GLuint name = lease->GetOpenGlBuffer(*other_context);
    EXPECT_TRUE(glIsBuffer(name));
  });
}

TEST_F(TensorBufferPoolTest, ForeignThreadReturnIsReusedOnItsOwner) {
  std::shared_ptr<TensorBufferPool::Buffer> lease;
  GLuint name = 0;
  context_->Run([&] { name = Acquire(spec_, &lease); });
  // Release on the test thread, not the GL owner. Run below is a queue barrier.
  lease.reset();
  context_->Run([&] {
    EXPECT_EQ(Acquire(spec_, &lease), name);
    Return(&lease);
  });
}

TEST_F(TensorBufferPoolTest, ShutdownBeforeForeignReturnDeletesBacking) {
  std::shared_ptr<TensorBufferPool::Buffer> lease;
  GLuint name = 0;
  context_->Run([&] { name = Acquire(spec_, &lease); });
  pool_.reset();
  lease.reset();
  context_->Run([&] { EXPECT_FALSE(glIsBuffer(name)); });
}

TEST_F(TensorBufferPoolTest, NewPoolOnSameContextDoesNotReuseRetiredCacheEntry) {
  std::shared_ptr<TensorBufferPool::Buffer> old_lease;
  GLuint old_name = 0;
  context_->Run([&] { old_name = Acquire(spec_, &old_lease); });
  pool_.reset();
  pool_ = std::make_shared<TensorBufferPool>();
  context_->Run([&] {
    std::shared_ptr<TensorBufferPool::Buffer> lease;
    EXPECT_NE(Acquire(spec_, &lease), old_name);
    EXPECT_TRUE(glIsBuffer(old_name));
    Return(&lease);
    old_lease.reset();
    EXPECT_FALSE(glIsBuffer(old_name));
  });
}

TEST_F(TensorBufferPoolTest, ConcurrentContextOwnersReuseWithoutAliasing) {
  auto other_context = GlContext::Create(nullptr, /*create_thread=*/true);
  ASSERT_TRUE(other_context.ok()) << other_context.status();
  absl::Barrier start(2);
  auto exercise = [&](const std::shared_ptr<GlContext>& context) {
    context->Run([&] {
      start.Block();
      auto spec = spec_;
      spec.context_id = reinterpret_cast<uintptr_t>(context.get());
      GLuint previous = 0;
      for (int i = 0; i < 200; ++i) {
        auto first = pool_->GetBuffer(spec, context);
        auto second = pool_->GetBuffer(spec, context);
        const GLuint a = first->GetOpenGlBuffer(context);
        const GLuint b = second->GetOpenGlBuffer(context);
        EXPECT_NE(a, b);
        if (previous != 0) EXPECT_EQ(a, previous);
        first.reset();
        second.reset();
        previous = b;  // Last returned backing is the next borrowed one.
      }
    });
  };
  std::thread first([&] { exercise(context_); });
  std::thread second([&] { exercise(*other_context); });
  first.join();
  second.join();
}

TEST_F(TensorBufferPoolTest, ShapeBudgetIsSharedAcrossContextOwners) {
  auto other_context = GlContext::Create(nullptr, /*create_thread=*/true);
  ASSERT_TRUE(other_context.ok()) << other_context.status();
  std::vector<std::shared_ptr<GlContext>> contexts{context_, *other_context};
  std::vector<std::shared_ptr<TensorBufferPool::Buffer>> leases(6);
  GLuint names[6];
  for (int c = 0; c < 2; ++c) {
    contexts[c]->Run([&, c] {
      for (int i = 0; i < 3; ++i) {
        auto spec = spec_;
        spec.dimensions = {i + 1, 4};
        spec.byte_size *= i + 1;
        spec.context_id = reinterpret_cast<uintptr_t>(contexts[c].get());
        leases[c * 3 + i] = pool_->GetBuffer(spec, contexts[c]);
        names[c * 3 + i] = leases[c * 3 + i]->GetOpenGlBuffer(contexts[c]);
      }
    });
  }
  // Releasing all six live keys must leave two retained keys TOTAL, not two
  // per context. No additional allocations can recycle the names being tested.
  leases.clear();
  int retained = 0;
  for (int c = 0; c < 2; ++c) {
    contexts[c]->Run([&, c] {
      for (int i = 0; i < 3; ++i) {
        if (glIsBuffer(names[c * 3 + i])) ++retained;
      }
    });
  }
  EXPECT_LE(retained, 2);
}

TEST_F(TensorBufferPoolTest, ContextOwnerCanMigrateBetweenThreads) {
  auto context = GlContext::Create(nullptr, /*create_thread=*/false);
  ASSERT_TRUE(context.ok()) << context.status();
  GLuint name = 0;
  auto spec = spec_;
  spec.context_id = reinterpret_cast<uintptr_t>(context->get());
  (*context)->Run([&] {
    auto lease = pool_->GetBuffer(spec, *context);
    name = lease->GetOpenGlBuffer(*context);
  });
  std::thread other_thread([&] {
    (*context)->Run([&] {
      auto lease = pool_->GetBuffer(spec, *context);
      EXPECT_EQ(lease->GetOpenGlBuffer(*context), name);
    });
  });
  other_thread.join();
  // Destruction outside any current context must not leave a context/cache
  // ownership cycle, including on a context without a dedicated GL thread.
  std::weak_ptr<GlContext> weak = *context;
  pool_.reset();
  context->reset();
  EXPECT_TRUE(weak.expired());
}

TEST_F(TensorBufferPoolTest, ContextRegistrationIsAlsoBounded) {
  MultiPoolOptions options;
  options.max_pool_count = 1;
  pool_ = std::make_shared<TensorBufferPool>(options);
  context_->Run([&] {
    std::shared_ptr<TensorBufferPool::Buffer> lease;
    Acquire(spec_, &lease);
  });
  auto other_context = GlContext::Create(nullptr, /*create_thread=*/true);
  ASSERT_TRUE(other_context.ok()) << other_context.status();
  (*other_context)->Run([&] {
    auto spec = spec_;
    spec.context_id = reinterpret_cast<uintptr_t>(other_context->get());
    auto lease = pool_->GetBuffer(spec, *other_context);
    const GLuint name = lease->GetOpenGlBuffer(*other_context);
    lease.reset();
    EXPECT_FALSE(glIsBuffer(name));  // Overflow uses unpooled driver lifetime.
  });
}

}  // namespace
}  // namespace mediapipe
#endif
