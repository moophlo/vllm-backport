# Automatic SM80 backports

GitHub Actions checks the official `vllm-project/vllm` stable release at minute 23 every six hours (UTC). Changing the workflow or the files in `ci/sm80` on `master` also starts an attempt. Run **vLLM SM80 backport** from Actions for an immediate check.

The controller lives on `master`. `port.json` pins the official release commit and the maintained SM80 source commit. The workflow applies their difference to each new stable release with a three-way Git merge. It can handle diverging release branches. Changes from wtdcode enter the maintained port when its pins are updated.

## v0.34.0 and Model Runner V2

The port for v0.34.0 covers the shared DeepSeek V3.2 sparse MLA path, the GLM-5.3-Flash pooled indexer and GLM pipeline-parallel loading. On Ampere the indexer uses software E4M3 conversion and Triton paged logits. During decode it uses the physical page-table row and causal context length that Model Runner V2 prepares for each query token, including mixed speculative verification lengths. The sparse MLA backend supports the main latent cache in BF16 or FP16. The indexer keeps its separate packed FP8 cache. The fused DSA query kernel guards the FP8 packing instructions at compile time so the BF16 specialization can compile for SM80.

The original wtdcode snapshot is available at [29e66da](https://github.com/wtdcode/vllm-backport/tree/29e66dad461b11caae6039f510eaa7913f9dfb19). Its DeepSeek V4/V4.1 and Qwen-specific optimizations have separate code paths and are outside this release port's coverage. [Source changes](https://github.com/moophlo/vllm-backport/compare/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca...b2bfa4f6ab834e99b694863aef136c89a4cdc33a) · [SM80 cross-compilation and CPU numerical checks](https://github.com/moophlo/vllm-backport/actions/runs/45372873372/job/138371344388)

## Runners

| Job | Default runner | Requirements |
| --- | --- | --- |
| Poll, merge and report | `ubuntu-24.04` | GitHub-hosted. Runs even if the builder is offline. |
| Compile and push candidate | `vllm-sm80-build` | Linux x86-64 with Docker, Buildx/BuildKit, Git, jq, and disk space for CUDA build stages. Compilation can run on a CPU host. |
| Test the candidate digest | `vllm-sm80-test` | Linux x86-64 with Docker, NVIDIA Container Toolkit and a dedicated SM80 device visible to containers. |
| Publish release aliases | `ubuntu-24.04` | GitHub-hosted. Runs after the GPU job succeeds. |

One runner can carry both custom labels. To use existing labels, set the repository Actions variables `SM80_BUILD_RUNNER` and `SM80_TEST_RUNNER`. `SM80_GPU_DEVICE` selects the Docker device ID (default `0`). `SM80_MAX_JOBS` optionally caps the CUDA build parallelism. By default the build allows one job per 4 GiB of host RAM, bounded by the CPU count. Size free disk for both Docker layers and BuildKit cache. The build uses the runner's existing default Buildx builder and its cache.

Use dedicated runners for this public repository. This workflow accepts the schedule, trusted pushes to `master`, and manual runs on the default branch. The repository token publishes to GHCR. Package visibility and access follow the package settings.

## Outputs

The build targets `docker/Dockerfile`, stage `vllm-openai`, platform `linux/amd64`, with `torch_cuda_arch_list=8.0`, `nvcc_threads=1`, and `INSTALL_LMCACHE=true`. That architecture setting applies to the source build. Bundled third-party wheels can contain kernels for other architectures.

For v0.34.0, preparation also cross-compiles the actual Triton kernels to SM80 cubins, checks Model Runner V2 metadata and compares GLM token selection against a CPU reference. It stages the sparse MLA and GLM test modules from the resolved source tree along with pytest wheels for the GPU job. That job runs the tests offline against the candidate digest.

A successful compile uploads `ghcr.io/moophlo/vllm-backport:candidate-<build-id>`. The test pulls that exact digest, verifies compute capability 8.0, exercises the FP8 lookup, checks GLM kpool prefill and decode against an independent CPU reference (including four-token verification, non-contiguous cache pages and mixed-length requests that select different tokens), and runs prefill and decode with a small local BF16 Llama using dummy weights. A full GLM checkpoint, tensor parallelism, Ray Serve and NIXL require workload qualification.

Passing validation publishes `:<upstream-tag>-sm80` and, if the release is still current, `:latest-sm80`. Use the digest in `build.json` for an immutable deployment. Each successful GitHub Release attaches `backport.patch` and `build.json`. Its Git tag points to the controller revision. To reproduce the image source, check out the official commit recorded in `build.json`, run `git apply --index backport.patch` (skip an empty patch), and compare `git write-tree` with `tree` in the manifest.

## Resolve a source conflict

The failed run lists conflicting paths and uploads `conflicts.txt` and `conflicts.diff`. The same release, pins and recipe are skipped on subsequent polls. The manual **Retry the same source conflict** input repeats that attempt. Transient fetch failures and failed builds or GPU tests stay eligible for retry.

From a checkout of this fork, use the exact upstream SHA from the failed run. The command creates a new disposable checkout with the actual release as `HEAD`, a populated Git index and normal conflict markers.

```bash
target=<official-commit-sha-from-the-run>
candidate=../vllm-sm80-port
bash ci/sm80/prepare.sh "$candidate" "$target" \
    "$(jq -r .upstream_base ci/sm80/port.json)" \
    "$(jq -r .patch_tip ci/sm80/port.json)"
```

Exit code 3 means source conflicts. Resolve them in the candidate checkout and inspect the staged changes, including CUDA paths and their tests. After staging the resolution, commit it and push the commit to a branch of this fork. For example, for release `v0.34.0` use the following commands.

```bash
set -e
git -C "$candidate" status --short
# After editing the conflicted files
git -C "$candidate" add -A
# GitHub needs the history of the official release on the first push from
# this shallow, disposable checkout. Keep the fork's own automation on
# the maintenance branch.
git -C "$candidate" fetch --no-tags --unshallow upstream "$target"
git -C "$candidate" fetch --no-tags --depth=1 port \
    refs/heads/master:refs/remotes/port/master
git -C "$candidate" restore --staged --worktree --source=refs/remotes/port/master -- .github
git -C "$candidate" commit -m "Port SM80 changes onto v0.34.0"
git -C "$candidate" push port HEAD:refs/heads/ports/v0.34.0
tip=$(git -C "$candidate" rev-parse HEAD)
jq --arg base "$target" --arg tip "$tip" \
    '.upstream_base=$base | .patch_tip=$tip' ci/sm80/port.json > ci/sm80/port.json.new
mv ci/sm80/port.json.new ci/sm80/port.json
```

Commit the updated `port.json` on `master` (or merge it through a PR). CI then builds the resolved port for the pinned release and uses that delta for future releases. The `port` remote defaults to this fork. Override `PORT_REMOTE` or `UPSTREAM_REMOTE` when reproducing against a different clone.
