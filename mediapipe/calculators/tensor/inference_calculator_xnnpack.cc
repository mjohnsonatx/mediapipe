// Copyright 2022 The MediaPipe Authors.
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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "mediapipe/calculators/tensor/inference_calculator.h"
#include "mediapipe/calculators/tensor/inference_calculator_utils.h"
#include "mediapipe/calculators/tensor/inference_interpreter_delegate_runner.h"
#include "mediapipe/calculators/tensor/inference_runner.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#include "pthreadpool.h"

namespace mediapipe {
namespace api2 {

namespace {

// Preserve every task/custom op registration, but suppress resolver-provided lazy delegates.
// Otherwise a partially delegated model may allocate a second implicit XNNPACK pool after the
// explicitly configured pool has been verified. The retained packet owns the source resolver.
class ExplicitCpuOpResolver final : public tflite::OpResolver {
 public:
  explicit ExplicitCpuOpResolver(Packet<tflite::OpResolver> source)
      : source_(std::move(source)) {}

  const TfLiteRegistration* FindOp(tflite::BuiltinOperator op, int version) const override {
    return source_.Get().FindOp(op, version);
  }
  const TfLiteRegistration* FindOp(const char* op, int version) const override {
    return source_.Get().FindOp(op, version);
  }

 private:
  Packet<tflite::OpResolver> source_;
};

}  // namespace

class InferenceCalculatorXnnpackImpl
    : public InferenceCalculatorNodeImpl<InferenceCalculatorXnnpack,
                                         InferenceCalculatorXnnpackImpl> {
 public:
  static absl::Status UpdateContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::StatusOr<std::vector<Tensor>> Process(
      CalculatorContext* cc, const TensorSpan& tensor_span) override;
  absl::StatusOr<std::unique_ptr<InferenceRunner>> CreateInferenceRunner(
      CalculatorContext* cc);
  absl::StatusOr<TfLiteDelegatePtr> CreateDelegate(CalculatorContext* cc);

  std::unique_ptr<InferenceRunner> inference_runner_;
};

absl::Status InferenceCalculatorXnnpackImpl::UpdateContract(
    CalculatorContract* cc) {
  ABSL_RETURN_IF_ERROR(TensorContractCheck(cc));

  const auto& options = cc->Options<mediapipe::InferenceCalculatorOptions>();
  RET_CHECK(!options.model_path().empty() ^ kSideInModel(cc).IsConnected())
      << "Either model as side packet or model path in options is required.";

  return absl::OkStatus();
}

absl::Status InferenceCalculatorXnnpackImpl::Open(CalculatorContext* cc) {
  ABSL_ASSIGN_OR_RETURN(inference_runner_, CreateInferenceRunner(cc));
  return InferenceCalculatorNodeImpl::UpdateIoMapping(
      cc, inference_runner_->GetInputOutputTensorNames());
}

absl::StatusOr<std::vector<Tensor>> InferenceCalculatorXnnpackImpl::Process(
    CalculatorContext* cc, const TensorSpan& tensor_span) {
  ABSL_ASSIGN_OR_RETURN(std::vector<Tensor> output_tensors,
                        inference_runner_->Run(cc, tensor_span));
  return output_tensors;
}

absl::Status InferenceCalculatorXnnpackImpl::Close(CalculatorContext* cc) {
  inference_runner_ = nullptr;
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<InferenceRunner>>
InferenceCalculatorXnnpackImpl::CreateInferenceRunner(CalculatorContext* cc) {
  ABSL_ASSIGN_OR_RETURN(auto model_packet, GetModelAsPacket(cc));
  ABSL_ASSIGN_OR_RETURN(auto op_resolver_packet, GetOpResolverAsPacket(cc));
  auto explicit_resolver = PacketAdopting<tflite::OpResolver>(
      std::make_unique<ExplicitCpuOpResolver>(std::move(op_resolver_packet)));
  const auto& calculator_opts =
      cc->Options<mediapipe::InferenceCalculatorOptions>();
  const int interpreter_num_threads = calculator_opts.cpu_num_thread();
  ABSL_ASSIGN_OR_RETURN(TfLiteDelegatePtr delegate, CreateDelegate(cc));
  return CreateInferenceInterpreterDelegateRunner(
      std::move(model_packet), std::move(explicit_resolver),
      std::move(delegate), interpreter_num_threads,
      &calculator_opts.input_output_config(),
      calculator_opts.delegate().xnnpack().enable_zero_copy_tensor_io());
}

absl::StatusOr<TfLiteDelegatePtr>
InferenceCalculatorXnnpackImpl::CreateDelegate(CalculatorContext* cc) {
  const auto& calculator_opts =
      cc->Options<mediapipe::InferenceCalculatorOptions>();
  auto opts_delegate = calculator_opts.delegate();
  if (!kDelegate(cc).IsEmpty()) {
    const mediapipe::InferenceCalculatorOptions::Delegate&
        input_side_packet_delegate = kDelegate(cc).Get();
    RET_CHECK(
        input_side_packet_delegate.has_xnnpack() ||
        input_side_packet_delegate.delegate_case() ==
            mediapipe::InferenceCalculatorOptions::Delegate::DELEGATE_NOT_SET)
        << "inference_calculator_cpu only supports delegate input side packet "
        << "for TFLite, XNNPack";
    opts_delegate.MergeFrom(input_side_packet_delegate);
  }
  const bool opts_has_delegate =
      calculator_opts.has_delegate() || !kDelegate(cc).IsEmpty();

  auto xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
  xnnpack_opts.num_threads =
      GetXnnpackNumThreads(opts_has_delegate, opts_delegate);
  TfLiteDelegatePtr delegate(TfLiteXNNPackDelegateCreate(&xnnpack_opts),
                             &TfLiteXNNPackDelegateDelete);
  RET_CHECK(delegate != nullptr) << "CPU XNNPACK delegate creation failed: node="
                                << cc->NodeName();
  const auto pool = static_cast<pthreadpool_t>(
      TfLiteXNNPackDelegateGetThreadPool(delegate.get()));
  const size_t effective_threads = pool == nullptr ? 1 : pthreadpool_get_threads_count(pool);
  // Initialization only. This is measured native pool capacity, not a claim that every op uses
  // every worker. The invoking thread participates; Close destroys the runner/delegate and joins
  // its persistent helpers. Graph executors and other model instances have independent pools.
  ABSL_LOG(INFO) << "MediaPipeCpuThreads node=" << cc->NodeName()
                 << " requested=" << xnnpack_opts.num_threads
                 << " effective=" << effective_threads
                 << " helpers=" << (effective_threads - 1)
                 << " interpreter=" << calculator_opts.cpu_num_thread();
  if (opts_delegate.has_xnnpack() && opts_delegate.xnnpack().num_threads() > 0) {
    RET_CHECK_EQ(effective_threads, static_cast<size_t>(xnnpack_opts.num_threads))
        << "Explicit CPU thread count could not be honored: node=" << cc->NodeName();
  }
  return delegate;
}

}  // namespace api2
}  // namespace mediapipe
