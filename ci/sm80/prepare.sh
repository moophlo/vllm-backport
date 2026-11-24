#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# != 4 ]]; then
    echo "Usage: bash prepare.sh EMPTY_DIRECTORY UPSTREAM_COMMIT UPSTREAM_BASE PATCH_TIP" >&2
    exit 2
fi
dest=$1
target=$2
base=$3
tip=$4
for sha in "$target" "$base" "$tip"; do
    [[ $sha =~ ^[0-9a-f]{40}$ ]] || { echo "Expected a full Git commit SHA: $sha" >&2; exit 2; }
done
upstream_remote=${UPSTREAM_REMOTE:-https://github.com/vllm-project/vllm.git}
port_remote=${PORT_REMOTE:-https://github.com/${GITHUB_REPOSITORY:-moophlo/vllm-backport}.git}

# This is a disposable, standalone checkout. Refuse to reuse a worktree.
mkdir "$dest"
cd "$dest"
git init -q
builder_git() {
    git -c user.name='SM80 backport builder' \
        -c user.email='41898282+github-actions[bot]@users.noreply.github.com' "$@"
}
git remote add upstream "$upstream_remote"
git remote add port "$port_remote"
git fetch -q --no-tags --depth=1 upstream "$target" "$base"
git fetch -q --no-tags --depth=1 port "$tip"

# The fork moved upstream CI to workflows.disabled. Keep CI and the two
# repository landing pages out of the source delta. Preserve build files,
# model code, tests, benchmarks and the LMCache Docker stage.
patch_file=$(git rev-parse --git-path sm80-overlay.patch)
git diff --no-ext-diff --no-renames --binary --full-index "$base" "$tip" -- . \
    ':(top,exclude).github' ':(top,exclude)README.md' \
    ':(top,exclude)README.vllm.md' ':(top,exclude)ci/sm80' > "$patch_file"
git read-tree "$base"
if [[ -s $patch_file ]]; then
    git apply --cached --whitespace=nowarn "$patch_file"
fi
overlay_tree=$(git write-tree)

# Force the explicit three-tree merge base. Release branches can diverge,
# and the original fork contains history-only merges. Git's normal merge
# machinery still handles file renames and exposes real index conflict stages.
release_side=$(builder_git commit-tree "$target^{tree}" -p "$base" -m 'Official release snapshot')
port_side=$(builder_git commit-tree "$overlay_tree" -p "$base" -m 'Maintained backport delta')
git reset --hard -q "$release_side"
git checkout -q --detach "$release_side"
status=0
builder_git -c merge.conflictStyle=diff3 -c rerere.enabled=false merge \
    --no-ff --no-commit "$port_side" || status=$?
if [[ -f $(git rev-parse --git-path MERGE_HEAD) ]]; then
    git merge --quit
fi
# Keep the index and worktree. A maintainer's eventual resolution commit
# must directly parent the actual official release, not a synthetic snapshot.
git update-ref --no-deref HEAD "$target" "$release_side"
if [[ $status != 0 ]]; then
    if [[ $status == 1 && -n $(git ls-files --unmerged) ]]; then
        echo "Source conflicts. Resolve in $PWD, then stage and commit the port." >&2
        exit 3
    fi
    echo "Git could not complete the merge (status $status)." >&2
    exit 1
fi
