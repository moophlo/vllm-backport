#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

official=vllm-project/vllm
tag=$(gh api "repos/$official/releases/latest" --jq \
    'select(.draft == false and .prerelease == false) | .tag_name')
[[ $tag =~ ^v[0-9][0-9A-Za-z._-]{0,65}$ ]] || { echo "Unexpected release tag: $tag" >&2; exit 1; }
# The commits endpoint peels annotated tags. target_commitish is often "main".
target=$(gh api "repos/$official/commits/$tag" --jq .sha)
base=$(jq -er '.upstream_base' ci/sm80/port.json)
tip=$(jq -er '.patch_tip' ci/sm80/port.json)
for sha in "$target" "$base" "$tip"; do
    [[ $sha =~ ^[0-9a-f]{40}$ ]] || { echo "Invalid commit: $sha" >&2; exit 1; }
done
id=$({ printf '%s\n' "$tag" "$target" "$base" "$tip"
       git rev-parse HEAD:ci/sm80 HEAD:.github/workflows/docker-publish.yml
     } | sha256sum | cut -c1-24)
image="ghcr.io/$(tr '[:upper:]' '[:lower:]' <<<"$GITHUB_REPOSITORY")"
release_ref="sm80-$tag-$id"
conflict_ref="sm80-conflict/$tag-$id"
state=pending
refs=$(git ls-remote --refs origin "refs/tags/$release_ref" "refs/tags/$conflict_ref")
if awk -v ref="refs/tags/$release_ref" '$2 == ref { found=1 } END { exit !found }' <<<"$refs"; then
    # A partial release may already have a Git tag. It is complete only
    # after the source patch and manifest have both been published.
    if gh release view "$release_ref" -R "$GITHUB_REPOSITORY" --json isDraft,assets | \
       jq -e '(.isDraft == false) and ([.assets[].name] | contains(["backport.patch","build.json"]))' >/dev/null; then
        state=published
    fi
elif [[ -n $refs && ${FORCE_RETRY:-false} != true ]]; then
    state=known-conflict
    echo "::warning::This input previously produced source conflicts. Update the port pins or retry manually."
fi
printf '%s\n' "tag=$tag" "target=$target" "base=$base" "tip=$tip" "id=$id" \
    "release_ref=$release_ref" "conflict_ref=$conflict_ref" "image=$image" \
    "state=$state" >> "$GITHUB_OUTPUT"
printf 'Upstream **%s** at %s. Integration state: **%s**.\n' \
    "$tag" "$target" "$state" >> "$GITHUB_STEP_SUMMARY"
