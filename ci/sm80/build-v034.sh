#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

jobs=$1
shift
recipe=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d "$RUNNER_TEMP/sm80-wheel.XXXXXX")
context="$work/context"
mkdir -p "$context/wheels"
wheel=vllm-0.34.0-cp38-abi3-manylinux_2_28_x86_64.whl
curl --fail --location --retry 3 --output "$work/$wheel" \
    "https://github.com/vllm-project/vllm/releases/download/v0.34.0/$wheel"
echo "79ca4b3cd74aec08c4bf476bdf0cb15735b605ae5368526a2faae1e30f044ca2  $work/$wheel" | sha256sum --check

python3 -m venv "$work/venv"
"$work/venv/bin/pip" --disable-pip-version-check install --quiet wheel==0.45.1
"$work/venv/bin/wheel" unpack "$work/$wheel" --dest "$work/unpacked"
unpacked="$work/unpacked/vllm-0.34.0"
while IFS= read -r -d '' status && IFS= read -r -d '' path; do
    case "$status" in
        A) test ! -e "$unpacked/$path" ;;
        M|D)
            if ! cmp -s <(git show "$TARGET:$path") "$unpacked/$path"; then
                echo "::error::Official wheel differs from the release source at $path"
                exit 1
            fi ;;
        *) echo "::error::Unsupported wheel entry $status $path"; exit 1 ;;
    esac
    if [[ $status == D ]]; then
        rm "$unpacked/$path"
    else
        install -D -m 0644 "$path" "$unpacked/$path"
    fi
done < <(git diff --no-renames --name-status -z "$TARGET" HEAD -- vllm)
"$work/venv/bin/wheel" pack "$unpacked" --dest-dir "$context/wheels"

# Pin both upstream image manifests for the entire build. Buildx resolves
# linux/amd64 from the pinned index when a tag contains multiple platforms.
pinned_image() {
    local ref=$1 manifest digest
    manifest=$(docker buildx imagetools inspect "$ref")
    digest=$(awk '$1=="Digest:" {print $2; exit}' <<<"$manifest")
    [[ $digest =~ ^sha256:[0-9a-f]{64}$ ]] || return 1
    printf '%s@%s\n' "$ref" "$digest"
}
runtime=$(pinned_image docker.io/vllm/vllm-openai:v0.34.0)
toolkit=$(pinned_image docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04)
echo "Building from $runtime with $toolkit"
docker buildx build --platform linux/amd64 --file "$recipe/Dockerfile.v034" \
    --build-arg "VLLM_IMAGE=$runtime" --build-arg "CUDA_IMAGE=$toolkit" \
    --build-arg "JOBS=$jobs" --label "org.opencontainers.image.base.name=$runtime" \
    "$@" "$context"
