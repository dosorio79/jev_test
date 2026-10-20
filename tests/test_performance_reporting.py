import csv
import json
from importlib.resources import files
from pathlib import Path

from cx_sentiment.reporting import write_study_report
from cx_sentiment.study import OfflineStudyResult, ThresholdPoint


def _result() -> OfflineStudyResult:
    point = ThresholdPoint(
        threshold=0.7,
        accuracy=0.9,
        macro_f1=0.89,
        macro_f1_delta_from_openai=-0.005,
        macro_f1_delta_ci95=(-0.01, 0.001),
        fallback_fraction=0.2,
        jev_coverage=0.8,
        accepted_error_rate=0.05,
        deferred_error_rate=0.3,
        cost_usd_per_1000=1.2,
        by_language=(),
        qualifies_quality=True,
        qualifies_languages=True,
        non_dominated=True,
    )
    return OfflineStudyResult(
        dataset_sha256="abc",
        support=10,
        openai_macro_f1=0.895,
        thresholds=(point,),
        candidate_thresholds=(0.7,),
        recommendation_status="candidate <review>",
    )


def test_study_report_writes_normalized_artifacts_and_template(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("cx_sentiment.reporting.render_quarto", lambda _source: None)

    write_study_report(tmp_path, _result())

    payload = json.loads((tmp_path / "study.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((tmp_path / "thresholds.csv").open(encoding="utf-8")))
    source = (tmp_path / "report.qmd").read_text(encoding="utf-8")
    assert payload["measurement_scope"] == "simulated"
    assert rows[0]["threshold"] == "0.7"
    assert "Measurement audit" in source
    assert "Historical exploratory values" in source
    assert "Offline threshold simulation" in source
    assert "Controlled live measurements" in source
    assert "Language slices" in source
    assert "Limitations" in source
    assert "Recommendation status" in source


def test_performance_template_is_packaged() -> None:
    template = files("cx_sentiment").joinpath("reports/performance_study.qmd")
    assert template.is_file()
    assert "html.escape" in template.read_text(encoding="utf-8")
