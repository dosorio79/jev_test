#!/usr/bin/env bash
# Pins and SHA-256 digests are checked against official GitHub release assets.
set -euo pipefail
case "${1:?tool name required}" in
  gitleaks)
    repository=gitleaks/gitleaks
    version=8.30.1
    archive=gitleaks_${version}_linux_x64.tar.gz
    checksum=551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb
    ;;
  actionlint)
    repository=rhysd/actionlint
    version=1.7.12
    archive=actionlint_${version}_linux_amd64.tar.gz
    checksum=8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8
    ;;
  *) echo 'Unsupported CI tool' >&2; exit 2 ;;
esac
destination=${2:?destination directory required}
mkdir -p "$destination"
working=$(mktemp -d)
trap 'rm -rf "$working"' EXIT
curl --fail --silent --show-error --location --retry 3 \
  "https://github.com/$repository/releases/download/v$version/$archive" -o "$working/$archive"
printf '%s  %s\n' "$checksum" "$working/$archive" | sha256sum --check --status
tar -xzf "$working/$archive" -C "$destination" "$1"
"$destination/$1" --version
