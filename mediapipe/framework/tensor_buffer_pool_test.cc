// Copyright 2026 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0

#include "mediapipe/framework/tensor_buffer_pool.h"

#include <memory>
#include <vector>

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
    *lease = pool_->GetBuffer(context_spec);
    return (*lease)->GetOpenGlBuffer(context_);
  }

  void Return(std::shared_ptr<TensorBufferPool::Buffer>* lease) {
    lease->reset();
  }

  TensorBufferSpec spec_{/*element_type=*/1, /*dimensions=*/{1, 4}, /*byte_size=*/16};
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
    auto lease = pool_->GetBuffer(other_spec);
    EXPECT_NE(lease.get(), first_backing);
    const GLuint name = lease->GetOpenGlBuffer(*other_context);
    EXPECT_TRUE(glIsBuffer(name));
  });
}

}  // namespace
}  // namespace mediapipe
#endif
