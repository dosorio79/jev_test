#!/usr/bin/env bash
set -euo pipefail

base_sha=${BASE_SHA:-}
if [[ -z "$base_sha" || "$base_sha" =~ ^0+$ ]] || ! git cat-file -e "$base_sha^{commit}"; then
  if git rev-parse --verify HEAD^ >/dev/null 2>&1; then
    base_sha=$(git rev-parse HEAD^)
  else
    base_sha=$(git hash-object -t tree /dev/null)
  fi
fi

git diff --check "$base_sha" "$GITHUB_SHA"
