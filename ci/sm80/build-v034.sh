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

install -m 0644 docker/lmcache-torch-cxx.patch "$context/lmcache-torch-cxx.patch"
runtime=$(sed -n 's/^ARG VLLM_IMAGE=//p' "$recipe/Dockerfile.v034")
docker buildx build --platform linux/amd64 --file "$recipe/Dockerfile.v034" \
    --build-arg "JOBS=$jobs" --label "org.opencontainers.image.base.name=$runtime" \
    "$@" "$context"
