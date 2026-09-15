#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
pose_app_root="${POSE_DETECTION_APP_DIR:-$(dirname "${repo_root}")/PoseDetectionApp}"
target="${1:-//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:libmediapipe_tasks_jni.so_copy}"
bazel_config="${BAZEL_CONFIG:-android_arm64}"
compilation_mode="${BAZEL_COMPILATION_MODE:-opt}"
strip_mode="${BAZEL_STRIP_MODE:-always}"
bazel_jobs="${BAZEL_JOBS:-2}"
python_version="${HERMETIC_PYTHON_VERSION:-3.12}"
ndk_version="${MEDIAPIPE_NDK_VERSION:-28.2.13676358}"
expected_bazel_version="$(tr -d '[:space:]' < "${repo_root}/.bazelversion")"

if [[ -n "${BAZEL_BIN:-}" ]]; then
  bazel_executable="${BAZEL_BIN}"
elif command -v bazelisk >/dev/null 2>&1; then
  bazel_executable="$(command -v bazelisk)"
elif command -v bazel >/dev/null 2>&1; then
  bazel_executable="$(command -v bazel)"
elif [[ -x "/tmp/bazel-${expected_bazel_version}" ]]; then
  bazel_executable="/tmp/bazel-${expected_bazel_version}"
else
  echo "Bazel ${expected_bazel_version} is required. Install Bazelisk or set BAZEL_BIN." >&2
  exit 1
fi

actual_bazel_version="$(cd "${repo_root}" && "${bazel_executable}" --version)"
if [[ "${actual_bazel_version}" != "bazel ${expected_bazel_version}" ]]; then
  echo "Expected Bazel ${expected_bazel_version}, but found ${actual_bazel_version}." >&2
  exit 1
fi

android_sdk="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"
if [[ -z "${android_sdk}" && -f "${pose_app_root}/local.properties" ]]; then
  android_sdk="$(awk -F= '$1 == "sdk.dir" { print substr($0, index($0, "=") + 1); exit }' \
    "${pose_app_root}/local.properties")"
fi
if [[ -z "${android_sdk}" ]]; then
  echo "Set ANDROID_HOME/ANDROID_SDK_ROOT or POSE_DETECTION_APP_DIR." >&2
  exit 1
fi

android_ndk="${ANDROID_NDK_HOME:-${android_sdk}/ndk/${ndk_version}}"
if [[ ! -d "${android_ndk}" ]]; then
  echo "Android NDK ${ndk_version} was not found at ${android_ndk}." >&2
  echo "Install the pinned NDK or set ANDROID_NDK_HOME explicitly." >&2
  exit 1
fi

bazel_environment=(
  env
  "ANDROID_HOME=${android_sdk}"
  "ANDROID_NDK_HOME=${android_ndk}"
  "HERMETIC_PYTHON_VERSION=${python_version}"
)

if [[ -n "${JAVA_HOME:-}" ]]; then
  bazel_environment+=("JAVA_HOME=${JAVA_HOME}")
elif [[ -d /opt/android-studio/jbr ]]; then
  bazel_environment+=("JAVA_HOME=/opt/android-studio/jbr")
fi

aquery_json="$(mktemp --tmpdir mediapipe-clion-aquery.XXXXXX.json)"
compile_commands_tmp="$(mktemp --tmpdir mediapipe-compile-commands.XXXXXX.json)"
trap 'rm -f "${aquery_json}" "${compile_commands_tmp}"' EXIT

cd "${repo_root}"

"${bazel_environment[@]}" "${bazel_executable}" aquery \
  "mnemonic(\"CppCompile\", deps(${target}))" \
  -c "${compilation_mode}" \
  "--config=${bazel_config}" \
  "--strip=${strip_mode}" \
  "--jobs=${bazel_jobs}" \
  "--repo_env=HERMETIC_PYTHON_VERSION=${python_version}" \
  --output=jsonproto \
  "--output_file=${aquery_json}"

execution_root="$("${bazel_environment[@]}" "${bazel_executable}" info execution_root)"

jq \
  --arg execution_root "${execution_root}" \
  --arg repo_root "${repo_root}" \
  '
    [
      .actions[]
      | select(.mnemonic == "CppCompile")
      | .arguments as $arguments
      | ($arguments | index("-c")) as $compile_index
      | select($compile_index != null)
      | $arguments[$compile_index + 1] as $source
      | select($source | startswith("mediapipe/"))
      | ($repo_root + "/" + $source) as $absolute_source
      | {
          directory: $execution_root,
          file: $absolute_source,
          arguments: (
            $arguments[0:$compile_index + 1]
            + [$absolute_source]
            + $arguments[$compile_index + 2:]
          )
        }
    ]
    | unique_by(.file)
  ' "${aquery_json}" > "${compile_commands_tmp}"

mv "${compile_commands_tmp}" "${repo_root}/compile_commands.json"

entry_count="$(jq length "${repo_root}/compile_commands.json")"
echo "Generated ${repo_root}/compile_commands.json with ${entry_count} MediaPipe C++ entries."
echo "Target: ${target} (-c ${compilation_mode} --config=${bazel_config} --strip=${strip_mode})"
