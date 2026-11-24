#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

checkout=${1:?expected the prepared v0.34.0 worktree}
checks="$RUNNER_TEMP/report/runtime-tests"
venv="$RUNNER_TEMP/sm80-v034-venv"
git -C "$checkout" diff --cached --check
if ! command -v uv >/dev/null; then
    pipx install uv
    export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.12 "$venv"
uv pip install --python "$venv/bin/python" triton==3.8.0 ruff

mapfile -t paths < <(git -C "$checkout" diff --cached --name-only --diff-filter=ACMR -- '*.py')
if (( ${#paths[@]} )); then
    (
        cd "$checkout"
        "$venv/bin/python" -m compileall -q "${paths[@]}"
        "$venv/bin/ruff" check --select E9,F63,F7,F82 "${paths[@]}"
    )
fi
"$venv/bin/python" ci/sm80/compile-v034.py "$checkout"

cpu_wheel='https://github.com/vllm-project/vllm/releases/download/v0.34.0/vllm-0.34.0%2Bcpu-cp38-abi3-manylinux_2_39_x86_64.whl'
uv pip install --python "$venv/bin/python" --torch-backend=cpu "$cpu_wheel" pytest==9.2.0
TRITON_DEFAULT_BACKEND=cpu "$venv/bin/python" ci/sm80/compile-v034.py "$checkout" --check-metadata
site=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
for path in "${paths[@]}"; do
    if [[ $path == vllm/* ]]; then
        install -D -m 0644 "$checkout/$path" "$site/$path"
    fi
done

# Use the same tests and pytest version in the hosted check and the GPU image.
mkdir -p "$checks/wheels"
cp "$checkout/tests/models/test_glm5next_fp8_pp_load.py" "$checks/"
cp "$checkout/tests/v1/attention/test_triton_mla_sparse.py" "$checks/"
python3 -m pip download --disable-pip-version-check --only-binary=:all: \
    --dest "$checks/wheels" pytest==9.2.0
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
TRITON_DEFAULT_BACKEND=cpu "$venv/bin/python" ci/sm80/smoke.py --cpu-indexer
(cd "$RUNNER_TEMP" && "$venv/bin/python" -m pytest --noconftest --import-mode=importlib \
    -p no:cacheprovider -q -ra "$checks"/test_*.py)
