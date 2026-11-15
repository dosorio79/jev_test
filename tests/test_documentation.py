from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_performance_study_defines_measurement_contract() -> None:
    document = (ROOT / "docs" / "PERFORMANCE_STUDY.md").read_text()
    required_terms = (
        "provider_latency_ms",
        "stage_wall_ms",
        "operation_wall_ms",
        "scheduler_wait_ms",
        "attempt_number",
        "measurement_scope",
        "cache_status",
        "cold",
        "warm",
        "interleav",
        "retry",
        "failure",
        "validation",
        "locked test",
    )

    for term in required_terms:
        assert term.casefold() in document.casefold(), term
