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

#include "mediapipe/gpu/gl_app_texture_support.h"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/gpu/gl_texture_util.h"
#include "mediapipe/gpu/gl_texture_view.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/gpu/shader_util.h"

namespace {

// Linker wrappers inject failures only in this test binary, on the calling
// thread. The graph GL thread continues using real driver calls.
struct GlAudit {
  bool enabled = false;
  bool fail_fence = false;
  const char* version = nullptr;
  int fences = 0;
  int flushes = 0;
  int finishes = 0;
  int cpu_waits = 0;
  int gpu_waits = 0;
};
thread_local GlAudit audit;

class ScopedGlAudit {
 public:
  ScopedGlAudit() { audit = GlAudit{.enabled = true}; }
  ~ScopedGlAudit() {
    EXPECT_EQ(audit.finishes, 0);
    EXPECT_EQ(audit.cpu_waits, 0);
    audit = {};
  }
};

}  // namespace

extern "C" {
GLsync __real_glFenceSync(GLenum condition, GLbitfield flags);
void __real_glFlush();
void __real_glFinish();
void __real_glWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout);
GLenum __real_glClientWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout);
const GLubyte* __real_glGetString(GLenum name);

GLsync __wrap_glFenceSync(GLenum condition, GLbitfield flags) {
  if (audit.enabled) {
    ++audit.fences;
    if (audit.fail_fence) return nullptr;
  }
  return __real_glFenceSync(condition, flags);
}
void __wrap_glFlush() {
  if (audit.enabled) ++audit.flushes;
  __real_glFlush();
}
void __wrap_glFinish() {
  if (audit.enabled) ++audit.finishes;
  __real_glFinish();
}
void __wrap_glWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout) {
  if (audit.enabled) ++audit.gpu_waits;
  __real_glWaitSync(sync, flags, timeout);
}
GLenum __wrap_glClientWaitSync(GLsync sync, GLbitfield flags,
                               GLuint64 timeout) {
  if (audit.enabled) ++audit.cpu_waits;
  return __real_glClientWaitSync(sync, flags, timeout);
}
const GLubyte* __wrap_glGetString(GLenum name) {
  if (audit.enabled && audit.version && name == GL_VERSION) {
    return reinterpret_cast<const GLubyte*>(audit.version);
  }
  return __real_glGetString(name);
}
}  // extern "C"

namespace mediapipe {
namespace {

class ExternalTextureSyncTest : public testing::Test {
 protected:
  static constexpr int kSize = 256;

  void SetUp() override {
    display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    ASSERT_NE(display_, EGL_NO_DISPLAY);
    ASSERT_TRUE(eglInitialize(display_, nullptr, nullptr));
    const EGLint config_attrs[] = {EGL_SURFACE_TYPE,
                                   EGL_PBUFFER_BIT,
                                   EGL_RENDERABLE_TYPE,
                                   EGL_OPENGL_ES2_BIT,
                                   EGL_RED_SIZE,
                                   8,
                                   EGL_GREEN_SIZE,
                                   8,
                                   EGL_BLUE_SIZE,
                                   8,
                                   EGL_ALPHA_SIZE,
                                   8,
                                   EGL_NONE};
    EGLConfig config;
    EGLint count = 0;
    ASSERT_TRUE(eglChooseConfig(display_, config_attrs, &config, 1, &count));
    ASSERT_EQ(count, 1);
    const EGLint context_attrs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    const EGLint surface_attrs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
    capture_ =
        eglCreateContext(display_, config, EGL_NO_CONTEXT, context_attrs);
    ASSERT_NE(capture_, EGL_NO_CONTEXT);
    detector_ = eglCreateContext(display_, config, capture_, context_attrs);
    ASSERT_NE(detector_, EGL_NO_CONTEXT);
    surface_ = eglCreatePbufferSurface(display_, config, surface_attrs);
    ASSERT_NE(surface_, EGL_NO_SURFACE);
    ASSERT_TRUE(MakeCurrent(capture_));
    glGenTextures(textures_.size(), textures_.data());
    for (GLuint texture : textures_) {
      glBindTexture(GL_TEXTURE_2D, texture);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, kSize, kSize, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, nullptr);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    glFlush();
    ASSERT_TRUE(MakeCurrent(EGL_NO_CONTEXT));
    MP_ASSERT_OK_AND_ASSIGN(resources_, GpuResources::Create(capture_));
    ASSERT_NE(resources_->gl_context()->native_context(), capture_);
    ASSERT_NE(resources_->gl_context()->native_context(), detector_);
  }

  void TearDown() override {
    audit = {};
    // Drain graph jobs/fence deletion and stop its dedicated GL thread before
    // destroying the two externally owned contexts.
    resources_.reset();
    if (capture_ != EGL_NO_CONTEXT && surface_ != EGL_NO_SURFACE) {
      EXPECT_TRUE(MakeCurrent(capture_));
      glDeleteTextures(textures_.size(), textures_.data());
      if (program_) glDeleteProgram(program_);
    }
    if (display_ != EGL_NO_DISPLAY) {
      EXPECT_TRUE(MakeCurrent(EGL_NO_CONTEXT));
      if (detector_ != EGL_NO_CONTEXT) {
        EXPECT_TRUE(eglDestroyContext(display_, detector_));
      }
      if (capture_ != EGL_NO_CONTEXT) {
        EXPECT_TRUE(eglDestroyContext(display_, capture_));
      }
      if (surface_ != EGL_NO_SURFACE) {
        EXPECT_TRUE(eglDestroySurface(display_, surface_));
      }
      // EGL displays are shared process-wide; do not terminate another user's
      // connection (the same policy as GlContext's EGL implementation).
      EXPECT_TRUE(eglReleaseThread());
    }
  }

  bool MakeCurrent(EGLContext context) {
    const auto surface = context == EGL_NO_CONTEXT ? EGL_NO_SURFACE : surface_;
    return eglMakeCurrent(display_, surface, surface, context);
  }

  absl::StatusOr<GpuBuffer> Wrap(
      GlTextureBuffer::DeletionCallback callback,
      WrapExternalGlTextureSyncMode mode = WrapExternalGlTextureSyncMode::kSync,
      int lane = 0) {
    return WrapExternalGlTexture(*resources_, GL_TEXTURE_2D, textures_[lane],
                                 kSize, kSize, GpuBufferFormat::kRGBA32,
                                 std::move(callback), mode);
  }

  EGLDisplay display_ = EGL_NO_DISPLAY;
  EGLContext capture_ = EGL_NO_CONTEXT;
  EGLContext detector_ = EGL_NO_CONTEXT;
  EGLSurface surface_ = EGL_NO_SURFACE;
  std::array<GLuint, 2> textures_{};
  GLuint program_ = 0;
  std::shared_ptr<GpuResources> resources_;
};

TEST_F(ExternalTextureSyncTest, MissingCallingContextKeepsOwnership) {
  int releases = 0;
  ScopedGlAudit scope;
  auto result = Wrap([&](GlSyncToken) { ++releases; });
  EXPECT_EQ(result.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(releases, 0);
  EXPECT_EQ(audit.fences, 0);
}

TEST_F(ExternalTextureSyncTest, MissingDelegateReturnsError) {
  ASSERT_TRUE(MakeCurrent(detector_));
  ScopedGlAudit scope;
  auto result =
      GlContext::CreateFenceSyncTokenForCurrentExternalContext(nullptr);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(audit.fences, 0);
}

TEST_F(ExternalTextureSyncTest, UnsupportedCallerDoesNotFinishOrTakeOwnership) {
  ASSERT_TRUE(MakeCurrent(detector_));
  int releases = 0;
  ScopedGlAudit scope;
  audit.version = "OpenGL ES 2.0";
  auto result = Wrap([&](GlSyncToken) { ++releases; });
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(releases, 0);
  EXPECT_EQ(audit.fences, 0);
}

TEST_F(ExternalTextureSyncTest, FailedFenceDoesNotFinishOrTakeOwnership) {
  ASSERT_TRUE(MakeCurrent(detector_));
  int releases = 0;
  ScopedGlAudit scope;
  audit.fail_fence = true;
  auto result = Wrap([&](GlSyncToken) { ++releases; });
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInternal);
  EXPECT_EQ(releases, 0);
  EXPECT_EQ(audit.fences, 1);
}

TEST_F(ExternalTextureSyncTest, CancelBeforeGraphReadReleasesExactlyOnce) {
  ASSERT_TRUE(MakeCurrent(detector_));
  int releases = 0;
  ScopedGlAudit scope;
  MP_ASSERT_OK_AND_ASSIGN(auto buffer, Wrap([&](GlSyncToken token) {
                            ASSERT_NE(token, nullptr);
                            ++releases;
                          }));
  EXPECT_EQ(audit.fences, 1);
  EXPECT_EQ(audit.flushes, 1);
  buffer = {};
  EXPECT_EQ(releases, 1);
}

TEST_F(ExternalTextureSyncTest, ExplicitNoSyncStillAllowsNoCallingContext) {
  int releases = 0;
  ScopedGlAudit scope;
  MP_ASSERT_OK_AND_ASSIGN(auto buffer,
                          Wrap([&](GlSyncToken) { ++releases; },
                               WrapExternalGlTextureSyncMode::kNoSync));
  EXPECT_EQ(audit.fences, 0);
  buffer = {};
  EXPECT_EQ(releases, 1);
}

TEST_F(ExternalTextureSyncTest, ThreeContextsPreservePixelsAcrossLaneReuse) {
  constexpr int kFrames = 64;
  ASSERT_TRUE(MakeCurrent(capture_));
  const char* vertex_shader = R"glsl(#version 300 es
    void main() {
      const vec2 vertices[3] = vec2[3](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
      gl_Position = vec4(vertices[gl_VertexID], 0, 1);
    })glsl";
  const char* fragment_shader = R"glsl(#version 300 es
    precision highp float;
    uniform vec3 frame_color;
    uniform float seed;
    out vec4 output_color;
    void main() {
      float v = fract(gl_FragCoord.x * 0.13 + gl_FragCoord.y * 0.31 + seed);
      for (int i = 0; i < 128; ++i) {
        v = fract(sin(v + float(i)) * 437.585);
      }
      vec3 color = gl_FragCoord.x < 128.0 ? frame_color : 1.0 - frame_color;
      output_color = vec4(color + v * 0.00001, 1);
    })glsl";
  ASSERT_EQ(GlhCreateProgram(vertex_shader, fragment_shader, 0, nullptr,
                             nullptr, &program_),
            GL_TRUE);
  const GLint color_location = glGetUniformLocation(program_, "frame_color");
  const GLint seed_location = glGetUniformLocation(program_, "seed");
  ASSERT_GE(color_location, 0);
  ASSERT_GE(seed_location, 0);
  std::array<GlSyncToken, 2> consumer_tokens;
  std::vector<GpuBuffer> sampled_frames;
  sampled_frames.reserve(kFrames);
  int releases = 0;
  for (int frame = 0; frame < kFrames; ++frame) {
    SCOPED_TRACE(frame);
    const int lane = frame % textures_.size();
    const std::array<uint8_t, 4> color = {
        static_cast<uint8_t>(frame * 37), static_cast<uint8_t>(frame * 73),
        static_cast<uint8_t>(frame * 113), 255};
    ASSERT_TRUE(MakeCurrent(capture_));
    GLsync producer_fence;
    {
      ScopedGlAudit scope;
      if (consumer_tokens[lane]) {
        consumer_tokens[lane]->WaitOnGpu();
        consumer_tokens[lane].reset();
      }
      TempGlFramebuffer framebuffer;
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, textures_[lane], 0);
      ASSERT_EQ(glCheckFramebufferStatus(GL_FRAMEBUFFER),
                GL_FRAMEBUFFER_COMPLETE);
      // Per-pixel dependent arithmetic adds GPU pressure. Its contribution is
      // below RGBA8 rounding precision, preserving identifiable half-colors.
      // No producer readback or CPU completion barrier before graph sampling.
      glDisable(GL_DITHER);
      glViewport(0, 0, kSize, kSize);
      glUseProgram(program_);
      glUniform3f(color_location, color[0] / 255.0f, color[1] / 255.0f,
                  color[2] / 255.0f);
      glUniform1f(seed_location, frame);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      glUseProgram(0);
      producer_fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
      ASSERT_NE(producer_fence, nullptr);
      glFlush();
    }
    ASSERT_TRUE(MakeCurrent(detector_));
    GpuBuffer input;
    {
      ScopedGlAudit scope;
      glWaitSync(producer_fence, 0, GL_TIMEOUT_IGNORED);
      glDeleteSync(producer_fence);
      MP_ASSERT_OK_AND_ASSIGN(input,
                              Wrap(
                                  [&, lane](GlSyncToken token) {
                                    ++releases;
                                    consumer_tokens[lane] = std::move(token);
                                  },
                                  WrapExternalGlTextureSyncMode::kSync, lane));
      EXPECT_EQ(audit.fences, 1);
      EXPECT_EQ(audit.flushes, 1);
    }
    // Match GPU-image packet storage. The graph's ordinary read view must
    // insert the B -> C server wait before copying pixels into retained output.
    Image image(std::move(input));
    GpuBuffer output(kSize, kSize, GpuBufferFormat::kRGBA32);
    MP_ASSERT_OK(resources_->gl_context()->Run([&] {
      ScopedGlAudit scope;
      TempGlFramebuffer framebuffer;
      auto source = image.GetGpuBuffer().GetReadView<GlTextureView>(0);
      EXPECT_EQ(audit.gpu_waits, 1);
      auto destination = output.GetWriteView<GlTextureView>(0);
      CopyGlTexture(source, destination);
      return absl::OkStatus();
    }));
    image = Image();
    ASSERT_EQ(releases, frame + 1);
    ASSERT_NE(consumer_tokens[lane], nullptr);
    sampled_frames.push_back(std::move(output));
  }
  // Read only retained graph outputs after ALL captures and reuses were queued.
  MP_ASSERT_OK(resources_->gl_context()->Run([&] {
    TempGlFramebuffer framebuffer;
    for (int frame = 0; frame < kFrames; ++frame) {
      SCOPED_TRACE(frame);
      auto view = sampled_frames[frame].GetReadView<GlTextureView>(0);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             view.target(), view.name(), 0);
      for (int half = 0; half < 2; ++half) {
        std::array<uint8_t, 4> pixel{};
        glReadPixels(kSize / 4 + half * kSize / 2, kSize / 2, 1, 1, GL_RGBA,
                     GL_UNSIGNED_BYTE, pixel.data());
        const std::array<int, 3> factors = {37, 73, 113};
        for (int channel = 0; channel < 3; ++channel) {
          const uint8_t value = frame * factors[channel];
          EXPECT_EQ(pixel[channel], half == 0 ? value : 255 - value);
        }
        EXPECT_EQ(pixel[3], 255);
      }
    }
    EXPECT_EQ(glGetError(), GL_NO_ERROR);
    return absl::OkStatus();
  }));
  ASSERT_TRUE(MakeCurrent(capture_));
  for (auto& token : consumer_tokens) {
    token->WaitOnGpu();
    token.reset();
  }
  sampled_frames.clear();
}

}  // namespace
}  // namespace mediapipe
