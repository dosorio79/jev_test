"""Execute the actual workflow guard code with safe synthetic events/results."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"


def workflow_python(name: str, step_name: str) -> str:
    step = (WORKFLOWS / name).read_text().split(f"- name: {step_name}\n", 1)[1]
    body = step.split("python - <<'PY'\n", 1)[1]
    lines = body.splitlines()
    end = next(index for index, line in enumerate(lines) if line.strip() == "PY")
    return textwrap.dedent("\n".join(lines[:end]))


class WorkflowGuards(unittest.TestCase):
    def test_aggregators_fail_closed(self) -> None:
        for filename, step in (
            ("ci.yml", "Require every dependency to succeed"),
            ("security.yml", "Require every scanner to succeed"),
        ):
            code = workflow_python(filename, step)
            for result in ("success", "failure", "cancelled", "skipped", "unexpected"):
                with self.subTest(workflow=filename, result=result):
                    needs = {"one": {"result": "success"}, "two": {"result": result}}
                    completed = subprocess.run(
                        [sys.executable, "-c", code],
                        env={**os.environ, "NEEDS_JSON": json.dumps(needs)},
                        capture_output=True,
                        check=False,
                    )
                    self.assertEqual(completed.returncode == 0, result == "success")
            completed = subprocess.run(
                [sys.executable, "-c", code],
                env={**os.environ, "NEEDS_JSON": "{}"},
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)

    def test_release_pr_source(self) -> None:
        code = workflow_python("ci.yml", "Check release PR source")
        cases = [
            ("main", "dev", "owner/repo", True),
            ("main", "feature", "owner/repo", False),
            ("main", "dev", "fork/repo", False),
            ("dev", "feature", "fork/repo", True),
            ("dev", "main", "owner/repo", True),
        ]
        with tempfile.TemporaryDirectory() as directory:
            event = Path(directory) / "event.json"
            for base, head, repository, allowed in cases:
                with self.subTest(base=base, head=head, repository=repository):
                    event.write_text(
                        json.dumps(
                            {
                                "pull_request": {
                                    "base": {"ref": base},
                                    "head": {"ref": head, "repo": {"full_name": repository}},
                                }
                            }
                        )
                    )
                    completed = subprocess.run(
                        [sys.executable, "-c", code],
                        env={
                            **os.environ,
                            "EVENT_NAME": "pull_request",
                            "REPOSITORY": "owner/repo",
                            "GITHUB_EVENT_PATH": str(event),
                        },
                        capture_output=True,
                        check=False,
                    )
                    self.assertEqual(completed.returncode == 0, allowed)


if __name__ == "__main__":
    unittest.main()
