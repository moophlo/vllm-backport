#!/usr/bin/env bash
set -euo pipefail
base=62d4bc3c7e33b1b6c906e233e6af2d3506c482ca
git fetch -q origin refs/heads/ports/v0.34.0-sm80-mrv2
tip=$(git rev-parse FETCH_HEAD)
checkout="$RUNNER_TEMP/sm80-verify"
bash ci/sm80/prepare.sh "$checkout" "$base" "$base" "$tip"
test "$(git -C "$checkout" write-tree)" = "$(git rev-parse "$tip^{tree}")"
git -C "$checkout" diff --cached --check
git -C "$checkout" diff --cached --stat
if ! command -v uv >/dev/null; then
    pipx install uv
    export PATH="$HOME/.local/bin:$PATH"
fi
venv="$RUNNER_TEMP/port-venv"
uv venv --python 3.12 "$venv"
uv pip install --python "$venv/bin/python" triton==3.8.0 ruff
mapfile -t paths < <(git -C "$checkout" diff --cached --name-only --diff-filter=ACMR -- '*.py')
(
  cd "$checkout"
  "$venv/bin/python" -m compileall -q "${paths[@]}"
  "$venv/bin/ruff" check --select E9,F63,F7,F82 "${paths[@]}"
)
"$venv/bin/python" ci/sm80/compile-v034.py "$checkout"
cpu_wheel='https://github.com/vllm-project/vllm/releases/download/v0.34.0/vllm-0.34.0%2Bcpu-cp38-abi3-manylinux_2_39_x86_64.whl'
uv pip install --python "$venv/bin/python" --torch-backend=cpu "$cpu_wheel" pytest
site=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
for path in "${paths[@]}"; do
    if [[ $path == vllm/* ]]; then install -D -m 0644 "$checkout/$path" "$site/$path"; fi
done
cd "$RUNNER_TEMP"
HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 "$venv/bin/python" -m pytest --noconftest --import-mode=importlib -q -ra \
    "$checkout/tests/models/test_glm5next_fp8_pp_load.py" \
    "$checkout/tests/v1/attention/test_triton_mla_sparse.py"
