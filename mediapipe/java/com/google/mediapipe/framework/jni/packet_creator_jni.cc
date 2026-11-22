// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/java/com/google/mediapipe/framework/jni/packet_creator_jni.h"

#include <jni.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/camera_intrinsics.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/core_proto_inc.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/colorspace.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/graph.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/jni_util.h"
#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_app_texture_support.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/gpu/gl_texture_buffer.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace {

using ::mediapipe::WrapExternalGlTexture;
using ::mediapipe::WrapExternalGlTextureSyncMode;
using ::mediapipe::android::SerializedMessageIds;
using ::mediapipe::android::ThrowIfError;

template <class T>
int64_t CreatePacketScalar(jlong context, const T& value) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  mediapipe::Packet packet = mediapipe::Adopt(new T(value));
  return mediapipe_graph->WrapPacketIntoContext(packet);
}

// Creates a new internal::PacketWithContext object, and returns the native
// handle.
int64_t CreatePacketWithContext(jlong context,
                                const mediapipe::Packet& packet) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  return mediapipe_graph->WrapPacketIntoContext(packet);
}

absl::StatusOr<WrapExternalGlTextureSyncMode> ParseSyncMode(jint sync_mode) {
  switch (sync_mode) {
    case 0:
      return WrapExternalGlTextureSyncMode::kNoSync;
    case 1:
      return WrapExternalGlTextureSyncMode::kSync;
    case 2:
      return WrapExternalGlTextureSyncMode::kMaybeSyncOrFinish;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unexpected sync mode: ", sync_mode));
}

#if !MEDIAPIPE_DISABLE_GPU
absl::StatusOr<mediapipe::GpuBuffer> CreateGpuBuffer(
    JNIEnv* env, jobject thiz, jlong context, jint name, jint width,
    jint height, jint format, jobject texture_release_callback,
    jint sync_mode) {
  mediapipe::android::Graph* mediapipe_graph =
      reinterpret_cast<mediapipe::android::Graph*>(context);
  auto* gpu_resources = mediapipe_graph->GetGpuResources();
  RET_CHECK(gpu_resources)
      << "Cannot create a mediapipe::GpuBuffer packet on a "
         "graph without GPU support";

  const auto gpu_buffer_format = mediapipe::GpuBufferFormatForGlFormat(format);
  if (gpu_buffer_format == mediapipe::GpuBufferFormat::kUnknown) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported OpenGL texture format: ", format));
  }
  ABSL_ASSIGN_OR_RETURN(WrapExternalGlTextureSyncMode wrap_sync_mode,
                        ParseSyncMode(sync_mode));

  jobject java_callback = nullptr;
  jobject packet_creator = nullptr;
  absl::Cleanup cleanup_refs = [&] {
    if (java_callback) env->DeleteGlobalRef(java_callback);
    if (packet_creator) env->DeleteGlobalRef(packet_creator);
  };
  mediapipe::GlTextureBuffer::DeletionCallback cc_callback;
  if (texture_release_callback) {
    // TODO: see if this can be cached.
    // Note: we don't get this from the object because people may pass a
    // subclass of PacketCreator, and the method is private.
    jclass my_class =
        env->FindClass("com/google/mediapipe/framework/PacketCreator");
    RET_CHECK(my_class);
    jmethodID release_method =
        env->GetMethodID(my_class, "releaseWithSyncToken",
                         "(JL"
                         "com/google/mediapipe/framework/TextureReleaseCallback"
                         ";)V");
    env->DeleteLocalRef(my_class);
    RET_CHECK(release_method);

    java_callback = env->NewGlobalRef(texture_release_callback);
    RET_CHECK(java_callback);
    packet_creator = env->NewGlobalRef(thiz);
    RET_CHECK(packet_creator);
    cc_callback = [packet_creator, release_method,
                   java_callback](mediapipe::GlSyncToken release_token) {
      JNIEnv* env = mediapipe::java::GetJNIEnv();

      jlong raw_token = reinterpret_cast<jlong>(
          new mediapipe::GlSyncToken(std::move(release_token)));
      env->CallVoidMethod(packet_creator, release_method, raw_token,
                          java_callback);

      // Note that this callback is called only once, and is not saved
      // anywhere else, so we can and should delete it here.
      env->DeleteGlobalRef(java_callback);
      env->DeleteGlobalRef(packet_creator);
    };
  }

  ABSL_ASSIGN_OR_RETURN(
      auto buffer,
      WrapExternalGlTexture(*gpu_resources, GL_TEXTURE_2D, name, width, height,
                            gpu_buffer_format, std::move(cc_callback),
                            wrap_sync_mode));
  // Successful wrapping transfers the references to the release callback. On
  // failure only the references are cleaned up; Java still owns the frame.
  std::move(cleanup_refs).Cancel();
  return buffer;
}
#endif  // !MEDIAPIPE_DISABLE_GPU

// Create a 1, 3, or 4 channel 8-bit ImageFrame shared pointer from a Java
// ByteBuffer.
absl::StatusOr<std::unique_ptr<mediapipe::ImageFrame>>
CreateImageFrameFromByteBuffer(JNIEnv* env, jobject byte_buffer, jint width,
                               jint height, jint width_step,
                               mediapipe::ImageFormat::Format format) {
  const int64_t buffer_size = env->GetDirectBufferCapacity(byte_buffer);
  const void* buffer_data = env->GetDirectBufferAddress(byte_buffer);
  if (buffer_data == nullptr || buffer_size < 0) {
    return absl::InvalidArgumentError(
        "Cannot get direct access to the input buffer. It should be created "
        "using allocateDirect.");
  }

  const int expected_buffer_size = height * width_step;
  RET_CHECK_EQ(buffer_size, expected_buffer_size)
      << "Input buffer size should be " << expected_buffer_size
      << " but is: " << buffer_size;

  auto image_frame = std::make_unique<mediapipe::ImageFrame>();
  // TODO: we could retain the buffer with a special deleter and use
  // the data directly without a copy. May need a new Java API since existing
  // code might expect to be able to overwrite the buffer after creating an
  // ImageFrame from it.
  image_frame->CopyPixelData(
      format, width, height, width_step,
      static_cast<const uint8_t*>(buffer_data),
      mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);

  return image_frame;
}

}  // namespace

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreatePoseTensor)(
    JNIEnv* env, jobject, jlong context, jboolean allocate) {
  std::vector<mediapipe::Tensor> tensors;
  if (allocate) {
    tensors.emplace_back(mediapipe::Tensor::ElementType::kFloat32,
                         mediapipe::Tensor::Shape{1, 256, 256, 3},
                         reinterpret_cast<mediapipe::android::Graph*>(context)->GetMemoryManager(),
                         /*memory_alignment=*/64);
  }
  return CreatePacketWithContext(context,
      mediapipe::MakePacket<std::vector<mediapipe::Tensor>>(std::move(tensors)));
}

JNIEXPORT jobject JNICALL PACKET_CREATOR_METHOD(nativeGetCpuPoseTensorBuffer)(
    JNIEnv* env, jobject, jlong packet_handle) {
  const auto& packet = mediapipe::android::Graph::GetPacketFromHandle(packet_handle);
  auto status = packet.ValidateAsType<std::vector<mediapipe::Tensor>>();
  if (!status.ok()) { ThrowIfError(env, status); return nullptr; }
  const auto& tensors = packet.Get<std::vector<mediapipe::Tensor>>();
  if (tensors.size() != 1) {
    ThrowIfError(env, absl::InvalidArgumentError("Expected one pose tensor."));
    return nullptr;
  }
  const auto& dims = tensors[0].shape().dims;
  if (tensors[0].element_type() != mediapipe::Tensor::ElementType::kFloat32 ||
      dims.size() != 4 || dims[0] != 1 || dims[1] != 256 || dims[2] != 256 || dims[3] != 3 ||
      tensors[0].ready_on_cpu() || tensors[0].ready_on_gpu()) {
    ThrowIfError(env, absl::InvalidArgumentError("CPU pose tensor write view can be requested only once."));
    return nullptr;
  }
  // Exclusive pre-submission ownership permits Java to fill this allocation after the view
  // unlocks. The packet retains it through native inference; Java must not mutate after submit.
  auto view = tensors[0].GetCpuWriteView();
  return env->NewDirectByteBuffer(view.buffer<float>(), tensors[0].bytes());
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreatePoseTensorImage)(
    JNIEnv*, jobject, jlong context) {
  auto frame = std::make_shared<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, 256, 256);
  frame->SetToZero();
  return CreatePacketWithContext(context, mediapipe::MakePacket<mediapipe::Image>(frame));
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreatePoseTensorFromGlBuffer)(
    JNIEnv* env, jobject, jlong context, jint buffer) {
#if !MEDIAPIPE_DISABLE_GPU && MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
  auto* graph = reinterpret_cast<mediapipe::android::Graph*>(context);
  auto* resources = graph->GetGpuResources();
  if (!resources || !glIsBuffer(buffer)) {
    ThrowIfError(env, absl::InvalidArgumentError("Pose tensor needs a shared SSBO and GPU graph."));
    return 0;
  }
  constexpr int kBytes = 256 * 256 * 3 * sizeof(float);
  GLint previous_read = 0;
  glGetIntegerv(GL_COPY_READ_BUFFER_BINDING, &previous_read);
  glBindBuffer(GL_COPY_READ_BUFFER, buffer);
  GLint size = 0;
  glGetBufferParameteriv(GL_COPY_READ_BUFFER, GL_BUFFER_SIZE, &size);
  glBindBuffer(GL_COPY_READ_BUFFER, previous_read);
  if (size != kBytes || glGetError() != GL_NO_ERROR) {
    ThrowIfError(env, absl::InvalidArgumentError("Pose SSBO must contain exactly 256x256x3 floats."));
    return 0;
  }
  GLsync producer = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  if (!producer) {
    ThrowIfError(env, absl::InternalError("Could not fence app pose tensor producer."));
    return 0;
  }
  glFlush();
  GLsync consumed = nullptr;
  std::vector<mediapipe::Tensor> tensors;
  const auto status = resources->gl_context()->Run([&]() -> absl::Status {
    glWaitSync(producer, 0, GL_TIMEOUT_IGNORED);
    tensors.emplace_back(mediapipe::Tensor::ElementType::kFloat32,
                         mediapipe::Tensor::Shape{1, 256, 256, 3},
                         graph->GetMemoryManager());
    {
      auto view = tensors.front().GetOpenGlBufferWriteView();
      GLint old_read = 0, old_write = 0;
      glGetIntegerv(GL_COPY_READ_BUFFER_BINDING, &old_read);
      glGetIntegerv(GL_COPY_WRITE_BUFFER_BINDING, &old_write);
      glBindBuffer(GL_COPY_READ_BUFFER, buffer);
      glBindBuffer(GL_COPY_WRITE_BUFFER, view.name());
      glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, kBytes);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
      glBindBuffer(GL_COPY_READ_BUFFER, old_read);
      glBindBuffer(GL_COPY_WRITE_BUFFER, old_write);
    }
    consumed = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    glFlush();
    if (!consumed || glGetError() != GL_NO_ERROR) {
      return absl::InternalError("GPU pose tensor copy/fence failed; retire the app SSBO.");
    }
    return absl::OkStatus();
  });
  glDeleteSync(producer);
  if (consumed) {
    glWaitSync(consumed, 0, GL_TIMEOUT_IGNORED);
    glDeleteSync(consumed);
  }
  if (!status.ok()) { ThrowIfError(env, status); return 0; }
  if (glGetError() != GL_NO_ERROR) {
    ThrowIfError(env, absl::InternalError("Failed to order pose SSBO reuse; retire the app SSBO."));
    return 0;
  }
  return CreatePacketWithContext(context,
      mediapipe::MakePacket<std::vector<mediapipe::Tensor>>(std::move(tensors)));
#else
  ThrowIfError(env, absl::UnimplementedError("Pose SSBO tensors require OpenGL ES 3.1."));
  return 0;
#endif
}

// ---------------------------------------------------------------------------
// Prepared pose session (r24): persistent tracked-frame input for one PoseLandmarker.
//
// Ownership contract (enforced by PoseLandmarker.PreparedPoseSession in Java):
//   - One Java owner serializes acquire -> write -> submit under its exclusive lease.
//   - A slot is reused only after the graph dropped its last packet reference. Reads of a GPU
//     slot are complete by then: the synchronous VIDEO call read the model outputs on CPU, which
//     orders every earlier GPU command, including the input read, before it returned.
//   - Any failure at or after submission poisons the slot until the detector is rebuilt.
//   - The session is destroyed only after graph teardown released every slot packet.
// Per frame this replaces the Java packet map (8-9 JNI packet creations, 8-9 input calls and a
// fresh tensor) with one acquire call and one submit call. Metadata packets are immutable
// payloads created once and re-timestamped here.
// ---------------------------------------------------------------------------
namespace {

enum PreparedPoseSlotState : int {
  kPreparedPoseSlotFree = 0,      // Unreferenced by the graph; may be acquired.
  kPreparedPoseSlotWriting = 1,   // Owned by the Java writer; not yet submitted.
  kPreparedPoseSlotInFlight = 2,  // Referenced by graph packets until the foreign holder dies.
  kPreparedPoseSlotPoisoned = 3,  // Indeterminate after a failure; never reused.
};

constexpr int kPreparedPoseTensorSize = 256;
constexpr int kPreparedPoseMinSlots = 2;
constexpr int kPreparedPoseMaxSlots = 4;
constexpr jint kPreparedPoseSlotBusy = -1;
constexpr jint kPreparedPoseSlotsExhausted = -2;

struct PreparedPoseSlot {
  // Exactly one [1,256,256,3] FLOAT32 tensor; a vector because the graph input carries one.
  std::vector<mediapipe::Tensor> tensors;
  std::atomic<int> state{kPreparedPoseSlotFree};
  float* cpu = nullptr;  // CPU sessions: stable aligned storage behind a direct ByteBuffer.
  jint gl_buffer = 0;    // GPU sessions: SSBO name shared with the app's parent GL context.
};

struct PreparedPoseSession {
  bool gpu = false;
  // unique_ptr keeps each slot (and its atomic) at a stable address for packet cleanups.
  std::vector<std::unique_ptr<PreparedPoseSlot>> slots;
  std::string tensor_stream;
  std::string reset_stream;
  std::vector<std::pair<std::string, mediapipe::Packet>> metadata;
  mediapipe::Packet reset_true;
  mediapipe::Packet reset_false;
  int next_slot = 0;  // Writer-serialized round robin.
};

PreparedPoseSession* PreparedPoseSessionFromHandle(jlong handle) {
  return reinterpret_cast<PreparedPoseSession*>(handle);
}

absl::StatusOr<PreparedPoseSlot*> PreparedPoseSlotFromHandle(jlong handle,
                                                             jint slot) {
  PreparedPoseSession* session = PreparedPoseSessionFromHandle(handle);
  RET_CHECK(session != nullptr) << "Prepared pose session is closed.";
  RET_CHECK(slot >= 0 && slot < static_cast<jint>(session->slots.size()))
      << "Prepared pose slot " << slot << " is out of range.";
  return session->slots[slot].get();
}

absl::StatusOr<std::unique_ptr<PreparedPoseSession>> CreatePreparedPoseSession(
    JNIEnv* env, jlong context, jboolean gpu, jint slot_count,
    jobjectArray streams, jlongArray constants) {
  auto* graph = reinterpret_cast<mediapipe::android::Graph*>(context);
  RET_CHECK(graph != nullptr) << "Prepared pose session needs a live graph.";
  RET_CHECK(slot_count >= kPreparedPoseMinSlots &&
            slot_count <= kPreparedPoseMaxSlots)
      << "Prepared pose session needs " << kPreparedPoseMinSlots << " to "
      << kPreparedPoseMaxSlots << " slots, got " << slot_count;
  RET_CHECK(streams != nullptr && constants != nullptr)
      << "Prepared pose session needs stream names and metadata packets.";
  const jsize stream_count = env->GetArrayLength(streams);
  const jsize constant_count = env->GetArrayLength(constants);
  // [0] tensor stream, [1] reset stream, [2..] metadata streams paired with constants.
  RET_CHECK(stream_count >= 3 && constant_count == stream_count - 2)
      << "Prepared pose session got " << stream_count << " streams and "
      << constant_count << " metadata packets.";

  auto session = std::make_unique<PreparedPoseSession>();
  session->gpu = gpu == JNI_TRUE;
  std::vector<std::string> names;
  names.reserve(stream_count);
  for (jsize i = 0; i < stream_count; ++i) {
    auto* name = static_cast<jstring>(env->GetObjectArrayElement(streams, i));
    RET_CHECK(name != nullptr) << "Prepared pose stream " << i << " is null.";
    names.push_back(mediapipe::android::JStringToStdString(env, name));
    env->DeleteLocalRef(name);
    RET_CHECK(i == 1 || !names.back().empty()) << "Prepared pose stream " << i << " is empty.";
  }
  std::vector<jlong> handles(constant_count);
  env->GetLongArrayRegion(constants, 0, constant_count, handles.data());
  session->tensor_stream = names[0];
  session->reset_stream = names[1];
  for (jsize i = 0; i < constant_count; ++i) {
    RET_CHECK(handles[i] != 0) << "Prepared pose metadata packet " << i << " is missing.";
    // Copies share the immutable payload; the Java caller releases its own handles.
    session->metadata.emplace_back(
        names[i + 2], mediapipe::android::Graph::GetPacketFromHandle(handles[i]));
  }
  session->reset_true = mediapipe::MakePacket<bool>(true);
  session->reset_false = mediapipe::MakePacket<bool>(false);

  for (jint i = 0; i < slot_count; ++i) {
    auto slot = std::make_unique<PreparedPoseSlot>();
    // No memory manager: pooled storage is tied to the graph's lifetime, and a CPU slot must
    // keep one stable pointer for its Java ByteBuffer. 64-byte alignment matches the XNNPACK
    // zero-copy input binding used by the per-frame route.
    slot->tensors.emplace_back(
        mediapipe::Tensor::ElementType::kFloat32,
        mediapipe::Tensor::Shape{1, kPreparedPoseTensorSize,
                                 kPreparedPoseTensorSize, 3},
        /*memory_manager=*/nullptr, /*memory_alignment=*/64);
    session->slots.push_back(std::move(slot));
  }

  if (!session->gpu) {
    for (auto& slot : session->slots) {
      const mediapipe::Tensor& tensor = slot->tensors.front();
      auto view = tensor.GetCpuWriteView();
      slot->cpu = view.buffer<float>();
      RET_CHECK(slot->cpu != nullptr) << "Prepared pose CPU slot allocation failed.";
      // Defined contents before the first write; every frame overwrites all values.
      std::memset(slot->cpu, 0, tensor.bytes());
    }
    return session;
  }
#if !MEDIAPIPE_DISABLE_GPU && MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
  auto* resources = graph->GetGpuResources();
  RET_CHECK(resources != nullptr)
      << "GPU prepared pose slots need a graph created with a parent GL context.";
  PreparedPoseSession* raw = session.get();
  ABSL_RETURN_IF_ERROR(resources->gl_context()->Run([raw]() -> absl::Status {
    for (auto& slot : raw->slots) {
      // Allocates the SSBO on the graph context, whose share group includes the app context.
      auto view = slot->tensors.front().GetOpenGlBufferWriteView();
      slot->gl_buffer = static_cast<jint>(view.name());
      RET_CHECK(slot->gl_buffer != 0) << "Prepared pose GPU slot allocation failed.";
    }
    glFlush();
    const GLenum error = glGetError();
    RET_CHECK(error == GL_NO_ERROR)
        << "Prepared pose GPU slot allocation failed: GL error " << error;
    return absl::OkStatus();
  }));
  return session;
#else
  return absl::UnimplementedError("GPU prepared pose slots require OpenGL ES 3.1.");
#endif
}

absl::Status SubmitPreparedPoseFrame(mediapipe::android::Graph* graph,
                                     PreparedPoseSession* session, jint slot_index,
                                     int64_t timestamp_us, bool reset_tracking) {
  RET_CHECK(graph != nullptr) << "Prepared pose submission needs a live graph.";
  RET_CHECK(session != nullptr) << "Prepared pose session is closed.";
  ABSL_ASSIGN_OR_RETURN(PreparedPoseSlot * slot,
                        PreparedPoseSlotFromHandle(
                            reinterpret_cast<jlong>(session), slot_index));
  RET_CHECK(slot->state.load(std::memory_order_acquire) == kPreparedPoseSlotWriting)
      << "Prepared pose slot " << slot_index << " must be acquired and unsubmitted.";
  // Any failure from here on leaves the slot's graph references or GPU ordering uncertain.
  auto poison = absl::MakeCleanup([slot] {
    slot->state.store(kPreparedPoseSlotPoisoned, std::memory_order_release);
  });
  RET_CHECK(!session->reset_stream.empty() || !reset_tracking)
      << "Stateless prepared pose session cannot reset tracking.";
  const mediapipe::Tensor& tensor = slot->tensors.front();
  if (session->gpu) {
#if !MEDIAPIPE_DISABLE_GPU && MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
    auto* resources = graph->GetGpuResources();
    RET_CHECK(resources != nullptr) << "GPU prepared pose slots need a GPU graph.";
    // The app wrote the SSBO on the calling context; order graph reads after that write
    // without a CPU wait.
    GLsync producer = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    RET_CHECK(producer != nullptr) << "Could not fence the app pose tensor write.";
    glFlush();
    const absl::Status ordered =
        resources->gl_context()->Run([producer, &tensor]() -> absl::Status {
          glWaitSync(producer, 0, GL_TIMEOUT_IGNORED);
          glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
          // Re-arm: this frame's SSBO becomes the tensor's only valid representation, so no
          // reader reuses a CPU copy taken from an earlier frame.
          { auto view = tensor.GetOpenGlBufferWriteView(); }
          const GLenum error = glGetError();
          RET_CHECK(error == GL_NO_ERROR)
              << "Could not order the prepared pose GPU slot for inference: GL error "
              << error;
          return absl::OkStatus();
        });
    glDeleteSync(producer);
    ABSL_RETURN_IF_ERROR(ordered);
#else
    return absl::UnimplementedError("GPU prepared pose slots require OpenGL ES 3.1.");
#endif
  } else {
    // Java filled the stable pointer. Re-arm CPU as the only valid representation so a
    // GPU-delegate reader uploads this frame instead of reusing an earlier upload.
    auto view = tensor.GetCpuWriteView();
  }

  const mediapipe::Timestamp timestamp =
      mediapipe::Timestamp::CreateNoErrorChecking(timestamp_us);
  for (const auto& [stream, packet] : session->metadata) {
    ABSL_RETURN_IF_ERROR(graph->AddPacketToInputStream(stream, packet.At(timestamp)));
  }
  if (!session->reset_stream.empty()) {
    ABSL_RETURN_IF_ERROR(graph->AddPacketToInputStream(
        session->reset_stream,
        (reset_tracking ? session->reset_true : session->reset_false).At(timestamp)));
  }
  // The tensor goes last so a metadata failure never leaves the slot referenced.
  slot->state.store(kPreparedPoseSlotInFlight, std::memory_order_release);
  std::atomic<int>* state = &slot->state;
  mediapipe::Packet tensors =
      mediapipe::PointToForeign(&slot->tensors,
                                [state]() {
                                  // Last graph reference released. A poisoned slot stays
                                  // poisoned; only an in-flight slot returns to the pool.
                                  int in_flight = kPreparedPoseSlotInFlight;
                                  state->compare_exchange_strong(
                                      in_flight, kPreparedPoseSlotFree,
                                      std::memory_order_acq_rel);
                                })
          .At(timestamp);
  ABSL_RETURN_IF_ERROR(
      graph->AddPacketToInputStream(session->tensor_stream, std::move(tensors)));
  std::move(poison).Cancel();
  return absl::OkStatus();
}

// ---------------------------------------------------------------------------
// Flat single-pose result (r25). Replaces three proto-vector serializations, Java proto parsing
// and ~100 Java value objects per synchronous VIDEO frame with one bounded array copy.
//
// Layout, in float words. Must match PoseLandmarker.FlatPoseResult in the vision AAR; the leading
// version word lets Java reject a mismatched core/vision pair instead of reading shifted offsets.
//   [0]        layout version
//   [1]        flags: pose present 1, pose detector ran 2, native next-frame rect present 4,
//              a supplied visibility/presence was NaN 8 (so NaN-as-omitted stays unambiguous)
//   [2]        image landmark count          [3] world landmark count
//   [4]        auxiliary landmark count      [5] poses in the native output (the first is written)
//   [6..10]    native next-frame rect: x_center, y_center, width, height, rotation
//   [11..175]  33 image landmarks: x, y, z, visibility, presence (NaN when the proto omits a field)
//   [176..274] 33 world landmarks: x, y, z in meters
//   [275..278] 2 auxiliary tracking landmarks: x, y normalized to the graph input image
// Every slot without data is NaN, so a reader can never mistake stale storage for a coordinate.
// Counts and flags are small integers, exactly representable as float.
// ---------------------------------------------------------------------------
constexpr int kFlatPoseLayoutVersion = 1;
constexpr int kFlatPoseVersionIndex = 0;
constexpr int kFlatPoseFlagsIndex = 1;
constexpr int kFlatPoseLandmarkCountIndex = 2;
constexpr int kFlatPoseWorldCountIndex = 3;
constexpr int kFlatPoseAuxiliaryCountIndex = 4;
constexpr int kFlatPosePoseCountIndex = 5;
constexpr int kFlatPoseNativeRectIndex = 6;
constexpr int kFlatPoseLandmarksIndex = 11;
constexpr int kFlatPoseLandmarkStride = 5;
constexpr int kFlatPoseMaxLandmarks = 33;
constexpr int kFlatPoseWorldIndex =
    kFlatPoseLandmarksIndex + kFlatPoseLandmarkStride * kFlatPoseMaxLandmarks;
constexpr int kFlatPoseWorldStride = 3;
constexpr int kFlatPoseAuxiliaryIndex =
    kFlatPoseWorldIndex + kFlatPoseWorldStride * kFlatPoseMaxLandmarks;
constexpr int kFlatPoseAuxiliaryStride = 2;
constexpr int kFlatPoseMaxAuxiliary = 2;
constexpr int kFlatPoseSize =
    kFlatPoseAuxiliaryIndex + kFlatPoseAuxiliaryStride * kFlatPoseMaxAuxiliary;
constexpr int kFlatPoseFlagPosePresent = 1;
constexpr int kFlatPoseFlagDetectorRan = 2;
constexpr int kFlatPoseFlagNativeRect = 4;
constexpr int kFlatPoseFlagSuppliedNaNConfidence = 8;
static_assert(kFlatPoseWorldIndex == 176, "Update PoseLandmarker.FlatPoseResult.WORLD_INDEX");
static_assert(kFlatPoseAuxiliaryIndex == 275,
              "Update PoseLandmarker.FlatPoseResult.AUXILIARY_INDEX");
static_assert(kFlatPoseSize == 279, "Update PoseLandmarker.FlatPoseResult.SIZE");

// The Java packet objects own these references for the duration of the output callback; the
// copies taken here only keep the payloads alive while this function reads them.
mediapipe::Packet FlatPosePacket(jlong handle) {
  return handle == 0 ? mediapipe::Packet()
                     : mediapipe::android::Graph::GetPacketFromHandle(handle);
}

// Typed view of one vector stream, or nullptr when the stream is absent at this timestamp. A
// wrong type would abort the process inside Get(), so it is reported as a recoverable status.
template <typename T>
absl::StatusOr<const std::vector<T>*> FlatPoseVector(
    const mediapipe::Packet& packet, const char* stream) {
  if (packet.IsEmpty()) return static_cast<const std::vector<T>*>(nullptr);
  const absl::Status valid = packet.ValidateAsType<std::vector<T>>();
  if (!valid.ok()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Flat pose stream ", stream, " has an unexpected type: ", valid.message()));
  }
  return &packet.Get<std::vector<T>>();
}

// Fills `out` (kFlatPoseSize floats). Returns whether the graph produced a pose. Never touches
// JNI, so a failure leaves the caller's Java array unmodified.
absl::StatusOr<bool> WriteFlatPoseResult(jlong landmarks_handle, jlong world_handle,
                                         jlong auxiliary_handle, jlong rects_handle,
                                         bool pose_detector_ran, float* out) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::fill(out, out + kFlatPoseSize, nan);

  const mediapipe::Packet landmarks_packet = FlatPosePacket(landmarks_handle);
  const mediapipe::Packet world_packet = FlatPosePacket(world_handle);
  const mediapipe::Packet auxiliary_packet = FlatPosePacket(auxiliary_handle);
  const mediapipe::Packet rects_packet = FlatPosePacket(rects_handle);

  const auto landmark_lists = FlatPoseVector<mediapipe::NormalizedLandmarkList>(
      landmarks_packet, "landmarks");
  if (!landmark_lists.ok()) return landmark_lists.status();
  const auto world_lists =
      FlatPoseVector<mediapipe::LandmarkList>(world_packet, "world_landmarks");
  if (!world_lists.ok()) return world_lists.status();
  const auto auxiliary_lists = FlatPoseVector<mediapipe::NormalizedLandmarkList>(
      auxiliary_packet, "auxiliary_landmarks");
  if (!auxiliary_lists.ok()) return auxiliary_lists.status();
  const auto rects =
      FlatPoseVector<mediapipe::NormalizedRect>(rects_packet, "pose_rects_next_frame");
  if (!rects.ok()) return rects.status();

  int flags = pose_detector_ran ? kFlatPoseFlagDetectorRan : 0;
  int pose_count = 0;
  int landmark_count = 0;
  int world_count = 0;
  int auxiliary_count = 0;

  if (*landmark_lists != nullptr && !(*landmark_lists)->empty()) {
    pose_count = static_cast<int>((*landmark_lists)->size());
    flags |= kFlatPoseFlagPosePresent;
    const mediapipe::NormalizedLandmarkList& pose = (*landmark_lists)->front();
    landmark_count = pose.landmark_size();
    if (landmark_count > kFlatPoseMaxLandmarks) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Flat pose result holds ", kFlatPoseMaxLandmarks, " landmarks, got ",
          landmark_count));
    }
    for (int i = 0; i < landmark_count; ++i) {
      const mediapipe::NormalizedLandmark& landmark = pose.landmark(i);
      float* slot = out + kFlatPoseLandmarksIndex + i * kFlatPoseLandmarkStride;
      slot[0] = landmark.x();
      slot[1] = landmark.y();
      slot[2] = landmark.z();
      slot[3] = landmark.has_visibility() ? landmark.visibility() : nan;
      slot[4] = landmark.has_presence() ? landmark.presence() : nan;
      if ((landmark.has_visibility() && std::isnan(landmark.visibility())) ||
          (landmark.has_presence() && std::isnan(landmark.presence()))) {
        // NaN also encodes an omitted field; tell the reader a NaN here came from the model.
        flags |= kFlatPoseFlagSuppliedNaNConfidence;
      }
    }

    // World and auxiliary vectors are indexed like the image landmarks; pose 0 only.
    if (*world_lists != nullptr && !(*world_lists)->empty()) {
      const mediapipe::LandmarkList& world = (*world_lists)->front();
      world_count = world.landmark_size();
      if (world_count > kFlatPoseMaxLandmarks) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Flat pose result holds ", kFlatPoseMaxLandmarks, " world landmarks, got ",
            world_count));
      }
      for (int i = 0; i < world_count; ++i) {
        const mediapipe::Landmark& landmark = world.landmark(i);
        float* slot = out + kFlatPoseWorldIndex + i * kFlatPoseWorldStride;
        slot[0] = landmark.x();
        slot[1] = landmark.y();
        slot[2] = landmark.z();
      }
    }
    if (*auxiliary_lists != nullptr && !(*auxiliary_lists)->empty()) {
      const mediapipe::NormalizedLandmarkList& auxiliary = (*auxiliary_lists)->front();
      auxiliary_count = auxiliary.landmark_size();
      if (auxiliary_count > kFlatPoseMaxAuxiliary) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Flat pose result holds ", kFlatPoseMaxAuxiliary,
            " auxiliary landmarks, got ", auxiliary_count));
      }
      for (int i = 0; i < auxiliary_count; ++i) {
        const mediapipe::NormalizedLandmark& landmark = auxiliary.landmark(i);
        float* slot = out + kFlatPoseAuxiliaryIndex + i * kFlatPoseAuxiliaryStride;
        slot[0] = landmark.x();
        slot[1] = landmark.y();
      }
    }
    if (*rects != nullptr && !(*rects)->empty()) {
      const mediapipe::NormalizedRect& rect = (*rects)->front();
      float* slot = out + kFlatPoseNativeRectIndex;
      slot[0] = rect.x_center();
      slot[1] = rect.y_center();
      slot[2] = rect.width();
      slot[3] = rect.height();
      slot[4] = rect.rotation();
      flags |= kFlatPoseFlagNativeRect;
    }
  }

  out[kFlatPoseVersionIndex] = static_cast<float>(kFlatPoseLayoutVersion);
  out[kFlatPoseFlagsIndex] = static_cast<float>(flags);
  out[kFlatPoseLandmarkCountIndex] = static_cast<float>(landmark_count);
  out[kFlatPoseWorldCountIndex] = static_cast<float>(world_count);
  out[kFlatPoseAuxiliaryCountIndex] = static_cast<float>(auxiliary_count);
  out[kFlatPosePoseCountIndex] = static_cast<float>(pose_count);
  return pose_count > 0;
}

}  // namespace

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreatePreparedPoseSession)(
    JNIEnv* env, jobject, jlong context, jboolean gpu, jint slot_count,
    jobjectArray streams, jlongArray constants) {
  auto session =
      CreatePreparedPoseSession(env, context, gpu, slot_count, streams, constants);
  if (!session.ok()) {
    ThrowIfError(env, session.status());
    return 0;
  }
  std::unique_ptr<PreparedPoseSession> owned = std::move(session).value();
  return reinterpret_cast<jlong>(owned.release());
}

JNIEXPORT jobject JNICALL PACKET_CREATOR_METHOD(nativeGetPreparedPoseCpuBuffer)(
    JNIEnv* env, jobject, jlong handle, jint slot) {
  auto found = PreparedPoseSlotFromHandle(handle, slot);
  if (!found.ok()) {
    ThrowIfError(env, found.status());
    return nullptr;
  }
  if (PreparedPoseSessionFromHandle(handle)->gpu || (*found)->cpu == nullptr) {
    ThrowIfError(env, absl::InvalidArgumentError("Prepared pose slot is not CPU storage."));
    return nullptr;
  }
  // Valid until the owner destroys the session after graph teardown.
  return env->NewDirectByteBuffer((*found)->cpu, (*found)->tensors.front().bytes());
}

JNIEXPORT jint JNICALL PACKET_CREATOR_METHOD(nativeGetPreparedPoseGlBuffer)(
    JNIEnv* env, jobject, jlong handle, jint slot) {
  auto found = PreparedPoseSlotFromHandle(handle, slot);
  if (!found.ok()) {
    ThrowIfError(env, found.status());
    return 0;
  }
  if (!PreparedPoseSessionFromHandle(handle)->gpu || (*found)->gl_buffer == 0) {
    ThrowIfError(env, absl::InvalidArgumentError("Prepared pose slot is not GPU storage."));
    return 0;
  }
  return (*found)->gl_buffer;
}

JNIEXPORT jint JNICALL PACKET_CREATOR_METHOD(nativeAcquirePreparedPoseSlot)(
    JNIEnv* env, jobject, jlong handle) {
  PreparedPoseSession* session = PreparedPoseSessionFromHandle(handle);
  if (session == nullptr) {
    ThrowIfError(env, absl::FailedPreconditionError("Prepared pose session is closed."));
    return kPreparedPoseSlotsExhausted;
  }
  const int count = static_cast<int>(session->slots.size());
  bool usable = false;
  for (int offset = 0; offset < count; ++offset) {
    const int index = (session->next_slot + offset) % count;
    int expected = kPreparedPoseSlotFree;
    if (session->slots[index]->state.compare_exchange_strong(
            expected, kPreparedPoseSlotWriting, std::memory_order_acq_rel)) {
      session->next_slot = (index + 1) % count;
      return index;
    }
    if (expected != kPreparedPoseSlotPoisoned) usable = true;
  }
  // Busy: every usable slot is still referenced by the graph (transient). Exhausted: every slot
  // is poisoned and the detector needs a drained rebuild.
  return usable ? kPreparedPoseSlotBusy : kPreparedPoseSlotsExhausted;
}

JNIEXPORT void JNICALL PACKET_CREATOR_METHOD(nativeReleasePreparedPoseSlot)(
    JNIEnv* env, jobject, jlong handle, jint slot, jboolean poison) {
  auto found = PreparedPoseSlotFromHandle(handle, slot);
  if (!found.ok()) {
    ThrowIfError(env, found.status());
    return;
  }
  if (poison == JNI_TRUE) {
    (*found)->state.store(kPreparedPoseSlotPoisoned, std::memory_order_release);
    return;
  }
  // Only an unsubmitted slot returns here; a submitted slot is released by its packet cleanup.
  int writing = kPreparedPoseSlotWriting;
  (*found)->state.compare_exchange_strong(writing, kPreparedPoseSlotFree,
                                          std::memory_order_acq_rel);
}

JNIEXPORT void JNICALL PACKET_CREATOR_METHOD(nativeSubmitPreparedPoseFrame)(
    JNIEnv* env, jobject, jlong context, jlong handle, jint slot, jlong timestamp,
    jboolean reset_tracking) {
  const absl::Status status = SubmitPreparedPoseFrame(
      reinterpret_cast<mediapipe::android::Graph*>(context),
      PreparedPoseSessionFromHandle(handle), slot, timestamp,
      reset_tracking == JNI_TRUE);
  if (!status.ok()) ThrowIfError(env, status);
}

JNIEXPORT void JNICALL PACKET_CREATOR_METHOD(nativeDestroyPreparedPoseSession)(
    JNIEnv*, jobject, jlong handle) {
  // Called only after graph teardown released every packet that referenced a slot. GPU tensors
  // release their SSBOs through the GlContext they retain.
  delete PreparedPoseSessionFromHandle(handle);
}

JNIEXPORT jboolean JNICALL PACKET_CREATOR_METHOD(nativeWriteFlatPoseResult)(
    JNIEnv* env, jclass clazz, jlong landmarks, jlong world_landmarks,
    jlong auxiliary_landmarks, jlong pose_rects, jboolean pose_detector_ran,
    jfloatArray out) {
  if (out == nullptr || env->GetArrayLength(out) < kFlatPoseSize) {
    ThrowIfError(env, absl::InvalidArgumentError(absl::StrCat(
                          "Flat pose result needs ", kFlatPoseSize, " floats.")));
    return JNI_FALSE;
  }
  // Stage on the stack (~1.1 KB) so protos are read without a JNI critical section and the
  // Java array is written once, only after every stream validated.
  float values[kFlatPoseSize];
  const absl::StatusOr<bool> present = WriteFlatPoseResult(
      landmarks, world_landmarks, auxiliary_landmarks, pose_rects,
      pose_detector_ran == JNI_TRUE, values);
  if (!present.ok()) {
    ThrowIfError(env, present.status());
    return JNI_FALSE;
  }
  env->SetFloatArrayRegion(out, 0, kFlatPoseSize, values);
  return *present ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateReferencePacket)(
    JNIEnv* env, jobject thiz, jlong context, jlong packet) {
  auto mediapipe_graph = reinterpret_cast<mediapipe::android::Graph*>(context);
  mediapipe::Packet mediapipe_packet =
      mediapipe::android::Graph::GetPacketFromHandle(packet);
  auto reference_packet = mediapipe::AdoptAsUniquePtr(
      new mediapipe::SyncedPacket(mediapipe_packet));
  // assigned the initial value of the packet reference.
  return mediapipe_graph->WrapPacketIntoContext(reference_packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateRgbImage)(
    JNIEnv* env, jobject thiz, jlong context, jobject byte_buffer, jint width,
    jint height) {
  // We require 4-byte alignment. See Java method.
  constexpr int kAlignment = 4;
  int width_step = ((width * 3 - 1) | (kAlignment - 1)) + 1;
  auto image_frame_or =
      CreateImageFrameFromByteBuffer(env, byte_buffer, width, height,
                                     width_step, mediapipe::ImageFormat::SRGB);
  if (ThrowIfError(env, image_frame_or.status())) return 0L;

  mediapipe::Packet packet = mediapipe::Adopt(image_frame_or->release());
  return CreatePacketWithContext(context, packet);
}

absl::StatusOr<std::unique_ptr<mediapipe::ImageFrame>> CreateRgbImageFromRgba(
    JNIEnv* env, jobject byte_buffer, jint width, jint height) {
  const uint8_t* rgba_data =
      static_cast<uint8_t*>(env->GetDirectBufferAddress(byte_buffer));
  int64_t buffer_size = env->GetDirectBufferCapacity(byte_buffer);
  if (rgba_data == nullptr || buffer_size < 0) {
    return absl::InvalidArgumentError(
        "Cannot get direct access to the input buffer. It should be created "
        "using allocateDirect.");
  }

  const int expected_buffer_size = width * height * 4;
  RET_CHECK_EQ(buffer_size, expected_buffer_size)
      << "Input buffer size should be " << expected_buffer_size
      << " but is: " << buffer_size;

  auto image_frame = std::make_unique<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, width, height,
      mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
  mediapipe::android::RgbaToRgb(rgba_data, width * 4, width, height,
                                image_frame->MutablePixelData(),
                                image_frame->WidthStep());
  return image_frame;
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateRgbImageFromRgba)(
    JNIEnv* env, jobject thiz, jlong context, jobject byte_buffer, jint width,
    jint height) {
  auto image_frame_or = CreateRgbImageFromRgba(env, byte_buffer, width, height);
  if (ThrowIfError(env, image_frame_or.status())) return 0L;

  mediapipe::Packet packet = mediapipe::Adopt(image_frame_or->release());
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateGrayscaleImage)(
    JNIEnv* env, jobject thiz, jlong context, jobject byte_buffer, jint width,
    jint height) {
  auto image_frame_or = CreateImageFrameFromByteBuffer(
      env, byte_buffer, width, height, width, mediapipe::ImageFormat::GRAY8);
  if (ThrowIfError(env, image_frame_or.status())) return 0L;

  mediapipe::Packet packet = mediapipe::Adopt(image_frame_or->release());
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateFloatImageFrame)(
    JNIEnv* env, jobject thiz, jlong context, jobject byte_buffer, jint width,
    jint height) {
  auto image_frame_or =
      CreateImageFrameFromByteBuffer(env, byte_buffer, width, height, width * 4,
                                     mediapipe::ImageFormat::VEC32F1);
  if (ThrowIfError(env, image_frame_or.status())) return 0L;
  mediapipe::Packet packet = mediapipe::Adopt(image_frame_or->release());
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateRgbaImageFrame)(
    JNIEnv* env, jobject thiz, jlong context, jobject byte_buffer, jint width,
    jint height) {
  auto image_frame_or =
      CreateImageFrameFromByteBuffer(env, byte_buffer, width, height, width * 4,
                                     mediapipe::ImageFormat::SRGBA);
  if (ThrowIfError(env, image_frame_or.status())) return 0L;
  mediapipe::Packet packet = mediapipe::Adopt(image_frame_or->release());
  return CreatePacketWithContext(context, packet);
}

static mediapipe::Packet createAudioPacket(const uint8_t* audio_sample,
                                           int num_samples, int num_channels) {
  std::unique_ptr<mediapipe::Matrix> matrix(
      new mediapipe::Matrix(num_channels, num_samples));
  // Preparing and normalize the audio data.
  // kMultiplier is same as what used in av_sync_media_decoder.cc.
  static const float kMultiplier = 1.f / (1 << 15);
  for (int sample = 0; sample < num_samples; ++sample) {
    for (int channel = 0; channel < num_channels; ++channel) {
      // MediaPipe createAudioPacket can currently only handle
      // AudioFormat.ENCODING_PCM_16BIT data, so here we are reading 2 bytes per
      // sample, using ByteOrder.LITTLE_ENDIAN byte order, which is
      // ByteOrder.nativeOrder() on Android
      // (https://developer.android.com/ndk/guides/abis.html).
      int16_t value = (audio_sample[1] & 0xff) << 8 | audio_sample[0];
      (*matrix)(channel, sample) = kMultiplier * value;
      audio_sample += 2;
    }
  }
  return mediapipe::Adopt(matrix.release());
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateAudioPacket)(
    JNIEnv* env, jobject thiz, jlong context, jbyteArray data, jint offset,
    jint num_channels, jint num_samples) {
  // Note, audio_data_ref is really a const jbyte* but this clashes with the
  // the expectation of ReleaseByteArrayElements below.
  jbyte* audio_data_ref = env->GetByteArrayElements(data, nullptr);
  const uint8_t* audio_sample =
      reinterpret_cast<uint8_t*>(audio_data_ref) + offset;
  mediapipe::Packet packet =
      createAudioPacket(audio_sample, num_samples, num_channels);
  env->ReleaseByteArrayElements(data, audio_data_ref, JNI_ABORT);
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateAudioPacketDirect)(
    JNIEnv* env, jobject thiz, jlong context, jobject data, jint num_channels,
    jint num_samples) {
  const uint8_t* audio_sample =
      reinterpret_cast<uint8_t*>(env->GetDirectBufferAddress(data));
  if (!audio_sample) {
    ThrowIfError(env, absl::InvalidArgumentError(
                          "Cannot get direct access to the input buffer. It "
                          "should be created using allocateDirect."));
    return 0L;
  }
  mediapipe::Packet packet =
      createAudioPacket(audio_sample, num_samples, num_channels);
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateInt16)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong context,
                                                                 jshort value) {
  return CreatePacketScalar<int16_t>(context, value);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateInt32)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong context,
                                                                 jint value) {
  return CreatePacketScalar<int>(context, value);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateInt64)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong context,
                                                                 jlong value) {
  return CreatePacketScalar<int64_t>(context, value);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateFloat32)(
    JNIEnv* env, jobject thiz, jlong context, jfloat value) {
  return CreatePacketScalar<float>(context, value);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateFloat64)(
    JNIEnv* env, jobject thiz, jlong context, jdouble value) {
  return CreatePacketScalar<double>(context, value);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateBool)(
    JNIEnv* env, jobject thiz, jlong context, jboolean value) {
  return CreatePacketScalar<bool>(context, value);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateString)(
    JNIEnv* env, jobject thiz, jlong context, jstring value) {
  return CreatePacketScalar<std::string>(
      context, mediapipe::android::JStringToStdString(env, value));
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateVideoHeader)(
    JNIEnv* env, jobject thiz, jlong context, jint width, jint height) {
  mediapipe::VideoHeader header;
  header.format = mediapipe::ImageFormat::SRGB;
  header.width = width;
  header.height = height;
  return CreatePacketScalar<mediapipe::VideoHeader>(context, header);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateTimeSeriesHeader)(
    JNIEnv* env, jobject thiz, jlong context, jint num_channels,
    jdouble sample_rate) {
  mediapipe::TimeSeriesHeader header;
  header.set_num_channels(num_channels);
  header.set_sample_rate(sample_rate);
  return CreatePacketScalar<mediapipe::TimeSeriesHeader>(context, header);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateMatrix)(
    JNIEnv* env, jobject thiz, jlong context, jint rows, jint cols,
    jfloatArray data) {
  if (env->GetArrayLength(data) != rows * cols) {
    ThrowIfError(
        env, absl::InvalidArgumentError(absl::StrCat(
                 "Please check the matrix data size, has to be rows * cols = ",
                 rows * cols)));
    return 0L;
  }
  std::unique_ptr<mediapipe::Matrix> matrix(new mediapipe::Matrix(rows, cols));
  // Android is always
  // little-endian(https://developer.android.com/ndk/guides/abis.html), even
  // though Java's ByteBuffer defaults to
  // big-endian(https://docs.oracle.com/javase/7/docs/api/java/nio/ByteBuffer.html),
  // there is no Java ByteBuffer involved, JNI does not change the endianness(we
  // have PacketGetterTest.testEndianOrder() to cover this case), so we can
  // safely copy data directly here.
  env->GetFloatArrayRegion(data, 0, rows * cols, matrix->data());
  mediapipe::Packet packet = mediapipe::Adopt(matrix.release());
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateMatrixDirect)(
    JNIEnv* env, jobject thiz, jlong context, jint rows, jint cols,
    jobject data) {
  const float* float_data =
      reinterpret_cast<const float*>(env->GetDirectBufferAddress(data));
  int64_t capacity = env->GetDirectBufferCapacity(data);
  if (!float_data || capacity < 0) {
    ThrowIfError(env, absl::InvalidArgumentError(
                          "Cannot get direct access to the input buffer. It "
                          "should be created using allocateDirect."));
    return 0L;
  }
  int64_t required_size = static_cast<int64_t>(rows) * cols * sizeof(float);
  if (capacity < required_size) {
    ThrowIfError(env, absl::InvalidArgumentError(
                          "Buffer capacity is too small for the requested "
                          "matrix size."));
    return 0L;
  }
  auto matrix = std::make_unique<mediapipe::Matrix>(rows, cols);
  std::memcpy(matrix->data(), float_data, required_size);
  mediapipe::Packet packet = mediapipe::Adopt(matrix.release());
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateCpuImage)(
    JNIEnv* env, jobject thiz, jlong context, jobject byte_buffer, jint width,
    jint height, jint width_step, jint num_channels) {
  mediapipe::ImageFormat::Format format;
  switch (num_channels) {
    case 4:
      format = mediapipe::ImageFormat::SRGBA;
      break;
    case 3:
      format = mediapipe::ImageFormat::SRGB;
      break;
    case 1:
      format = mediapipe::ImageFormat::GRAY8;
      break;
    default:
      ThrowIfError(env, absl::InvalidArgumentError(absl::StrCat(
                            "Channels must be either 1, 3, or 4, but are ",
                            num_channels)));
      return 0L;
  }

  auto image_frame_or = CreateImageFrameFromByteBuffer(
      env, byte_buffer, width, height, width_step, format);
  if (ThrowIfError(env, image_frame_or.status())) return 0L;

  mediapipe::Packet packet =
      mediapipe::MakePacket<mediapipe::Image>(*std::move(image_frame_or));
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateBorrowedRgbaImage)(
    JNIEnv* env, jobject thiz, jlong context, jobject byte_buffer, jint width,
    jint height, jobject on_release) {
  if (width <= 0 || height <= 0 || width > std::numeric_limits<int>::max() / 4 ||
      height > std::numeric_limits<int>::max() / (width * 4)) {
    ThrowIfError(env, absl::InvalidArgumentError("Borrowed RGBA image dimensions overflow storage."));
    return 0L;
  }
  const int64_t expected = static_cast<int64_t>(width) * height * 4;
  auto* pixels = static_cast<uint8_t*>(env->GetDirectBufferAddress(byte_buffer));
  if (width <= 0 || height <= 0 || width > std::numeric_limits<int>::max() / 4 ||
      expected > std::numeric_limits<int>::max() || pixels == nullptr ||
      env->GetDirectBufferCapacity(byte_buffer) != expected || on_release == nullptr) {
    ThrowIfError(env, absl::InvalidArgumentError(
        "Borrowed RGBA image requires exact-capacity direct storage and positive dimensions."));
    return 0L;
  }
  jclass callback_class = env->GetObjectClass(on_release);
  if (callback_class == nullptr) return 0L;
  jmethodID run = env->GetMethodID(callback_class, "run", "()V");
  env->DeleteLocalRef(callback_class);
  if (run == nullptr) return 0L;
  jobject buffer_ref = env->NewGlobalRef(byte_buffer);
  if (buffer_ref == nullptr) return 0L;
  jobject release_ref = env->NewGlobalRef(on_release);
  if (release_ref == nullptr) {
    env->DeleteGlobalRef(buffer_ref);
    return 0L;
  }
  auto frame = std::make_unique<mediapipe::ImageFrame>();
  frame->AdoptPixelData(mediapipe::ImageFormat::SRGBA, width, height, width * 4, pixels,
      [buffer_ref, release_ref, run](uint8_t*) {
        // The app buffer owns the pixels. Release only our strong references, and acknowledge
        // after the last native ImageFrame owner is gone (possibly on a graph worker).
        JNIEnv* release_env = mediapipe::java::GetJNIEnv();
        release_env->DeleteGlobalRef(buffer_ref);
        release_env->CallVoidMethod(release_ref, run);
        release_env->DeleteGlobalRef(release_ref);
      });
  return CreatePacketWithContext(context,
      mediapipe::MakePacket<mediapipe::Image>(std::move(frame)));
}

#if !MEDIAPIPE_DISABLE_GPU

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateGpuImage)(
    JNIEnv* env, jobject thiz, jlong context, jint name, jint width,
    jint height, jint format, jobject texture_release_callback) {
  auto buffer_or =
      CreateGpuBuffer(env, thiz, context, name, width, height, format,
                      texture_release_callback, /*sync_mode=*/1);
  // Preserve pending JNI lookup/allocation exceptions after reference cleanup.
  if (env->ExceptionCheck()) return 0L;
  // The caller may already have queued a wait for a capture-context fence. The
  // new producer fence extends that dependency to every graph read context.
  if (!buffer_or.ok()) {
    ThrowIfError(
        env, absl::Status(buffer_or.status().code(),
                          absl::StrCat("nativeCreateGpuImage: texture=", name,
                                       ", size=", width, "x", height,
                                       ", format=", format,
                                       ", required producer fence: ",
                                       buffer_or.status().message())));
    return 0L;
  }
  mediapipe::Packet packet =
      mediapipe::MakePacket<mediapipe::Image>(std::move(buffer_or).value());
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateGpuBuffer)(
    JNIEnv* env, jobject thiz, jlong context, jint name, jint width,
    jint height, jint format, jobject texture_release_callback,
    jint sync_mode) {
  auto buffer_or = CreateGpuBuffer(env, thiz, context, name, width, height,
                                   format, texture_release_callback, sync_mode);
  if (env->ExceptionCheck()) return 0L;
  if (ThrowIfError(env, buffer_or.status())) return 0L;
  mediapipe::Packet packet =
      mediapipe::MakePacket<mediapipe::GpuBuffer>(std::move(buffer_or).value());
  return CreatePacketWithContext(context, packet);
}

#endif  // !MEDIAPIPE_DISABLE_GPU

// TODO: Add vector creators.

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateFloat32Array)(
    JNIEnv* env, jobject thiz, jlong context, jfloatArray data) {
  jsize count = env->GetArrayLength(data);
  jfloat* data_ref = env->GetFloatArrayElements(data, nullptr);
  float* floats = new float[count];
  // jfloat is a "machine-dependent native type" which represents a 32-bit
  // float. C++ makes no guarantees about the size of floating point types, and
  // some exotic architectures don't even have 32-bit floats (or even binary
  // floats), but on all architectures we care about this is a float.
  static_assert(std::is_same<float, jfloat>::value, "jfloat must be float");
  std::memcpy(floats, data_ref, count * sizeof(float));
  env->ReleaseFloatArrayElements(data, data_ref, JNI_ABORT);

  // The reinterpret_cast is needed to make the Adopt template recognize
  // that this is an array - this way Holder will call delete[].
  mediapipe::Packet packet =
      mediapipe::Adopt(reinterpret_cast<float (*)[]>(floats));
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateFloat32Vector)(
    JNIEnv* env, jobject thiz, jlong context, jfloatArray data) {
  jsize count = env->GetArrayLength(data);
  jfloat* data_ref = env->GetFloatArrayElements(data, nullptr);
  // jfloat is a "machine-dependent native type" which represents a 32-bit
  // float. C++ makes no guarantees about the size of floating point types, and
  // some exotic architectures don't even have 32-bit floats (or even binary
  // floats), but on all architectures we care about this is a float.
  static_assert(std::is_same<float, jfloat>::value, "jfloat must be float");
  std::unique_ptr<std::vector<float>> floats =
      std::make_unique<std::vector<float>>(data_ref, data_ref + count);

  env->ReleaseFloatArrayElements(data, data_ref, JNI_ABORT);
  mediapipe::Packet packet = mediapipe::Adopt(floats.release());
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateInt32Array)(
    JNIEnv* env, jobject thiz, jlong context, jintArray data) {
  jsize count = env->GetArrayLength(data);
  jint* data_ref = env->GetIntArrayElements(data, nullptr);
  int32_t* ints = new int32_t[count];
  static_assert(std::is_same<int32_t, jint>::value, "jint must be int32_t");
  std::memcpy(ints, data_ref, count * sizeof(int32_t));
  env->ReleaseIntArrayElements(data, data_ref, JNI_ABORT);

  // The reinterpret_cast is needed to make the Adopt template recognize
  // that this is an array - this way Holder will call delete[].
  mediapipe::Packet packet =
      mediapipe::Adopt(reinterpret_cast<int32_t (*)[]>(ints));
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateInt32Pair)(
    JNIEnv* env, jobject thiz, jlong context, jint first, jint second) {
  static_assert(std::is_same<int32_t, jint>::value, "jint must be int32_t");

  mediapipe::Packet packet = mediapipe::MakePacket<std::pair<int32_t, int32_t>>(
      std::make_pair(first, second));
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateStringFromByteArray)(
    JNIEnv* env, jobject thiz, jlong context, jbyteArray data) {
  jsize count = env->GetArrayLength(data);
  jbyte* data_ref = env->GetByteArrayElements(data, nullptr);
  mediapipe::Packet packet = mediapipe::Adopt(
      new std::string(reinterpret_cast<char*>(data_ref), count));
  env->ReleaseByteArrayElements(data, data_ref, JNI_ABORT);
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateCalculatorOptions)(
    JNIEnv* env, jobject thiz, jlong context, jbyteArray data) {
  jsize count = env->GetArrayLength(data);
  jbyte* data_ref = env->GetByteArrayElements(data, nullptr);
  auto options = std::make_unique<mediapipe::CalculatorOptions>();
  if (!options->ParseFromArray(data_ref, count)) {
    ThrowIfError(env, absl::InvalidArgumentError(
                          "Parsing binary-encoded CalculatorOptions failed."));
    return 0L;
  }
  mediapipe::Packet packet = mediapipe::Adopt(options.release());
  env->ReleaseByteArrayElements(data, data_ref, JNI_ABORT);
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateProto)(JNIEnv* env,
                                                                 jobject thiz,
                                                                 jlong context,
                                                                 jobject data) {
  // Convert type_name and value from Java data.
  static SerializedMessageIds ids(env, data);
  jstring j_type_name = (jstring)env->GetObjectField(data, ids.type_name_id);
  std::string type_name =
      mediapipe::android::JStringToStdString(env, j_type_name);
  jbyteArray value_array = (jbyteArray)env->GetObjectField(data, ids.value_id);
  jsize value_len = env->GetArrayLength(value_array);
  jbyte* value_ref = env->GetByteArrayElements(value_array, nullptr);

  // Create the C++ MessageLite and Packet.
  mediapipe::Packet packet;
  auto packet_or = mediapipe::packet_internal::PacketFromDynamicProto(
      type_name, std::string((char*)value_ref, value_len));
  if (!ThrowIfError(env, packet_or.status())) {
    packet = packet_or.value();
  }
  env->ReleaseByteArrayElements(value_array, value_ref, JNI_ABORT);
  return CreatePacketWithContext(context, packet);
}

JNIEXPORT jlong JNICALL PACKET_CREATOR_METHOD(nativeCreateCameraIntrinsics)(
    JNIEnv* env, jobject thiz, jlong context, jfloat fx, jfloat fy, jfloat cx,
    jfloat cy, jfloat width, jfloat height) {
  mediapipe::Packet packet =
      mediapipe::MakePacket<CameraIntrinsics>(fx, fy, cx, cy, width, height);
  return CreatePacketWithContext(context, packet);
}
