#!/usr/bin/env bash
set -euo pipefail
working=$(mktemp -d)
trap 'rm -rf "$working"' EXIT
uv build --out-dir "$working/dist"
uv export --frozen --no-dev --no-emit-project --output-file "$working/requirements.txt" > /dev/null
uv venv "$working/venv"
uv pip install --python "$working/venv/bin/python" --require-hashes -r "$working/requirements.txt"
uv pip install --python "$working/venv/bin/python" --no-deps "$working"/dist/*.whl
# Run away from the checkout to prove imports and templates come from the wheel.
cd "$working"
"$working/venv/bin/cx-sentiment" --help
"$working/venv/bin/python" - <<'PY'
from importlib.resources import files

reports = files('cx_sentiment').joinpath('reports')
for name in ('benchmark.qmd', 'selection.qmd', 'comparison.qmd'):
    assert reports.joinpath(name).is_file(), f'Missing bundled template: {name}'
PY
