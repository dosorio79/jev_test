from __future__ import annotations

import csv
import html
import json
from collections.abc import Sequence
from pathlib import Path

from cx_sentiment.metrics import EvaluationMetrics
from cx_sentiment.models import Prediction
from cx_sentiment.routing import ThresholdSelection


def write_report(
    output_dir: Path,
    predictions: Sequence[Prediction],
    metrics: EvaluationMetrics,
    *,
    title: str,
    metadata: dict[str, object] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(prediction.model_dump_json() + "\n")
    (output_dir / "metrics.json").write_text(
        metrics.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scope",
                "support",
                "accuracy",
                "macro_f1",
                "coverage",
                "latency_p50_ms",
                "latency_p95_ms",
                "cost_usd_total",
                "cost_usd_per_1000",
                "cost_priced_fraction",
                "reported_credits_total",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scope": "overall",
                "support": metrics.support,
                "accuracy": metrics.accuracy,
                "macro_f1": metrics.macro_f1,
                "coverage": metrics.operational_coverage,
                "latency_p50_ms": metrics.latency_p50_ms,
                "latency_p95_ms": metrics.latency_p95_ms,
                "cost_usd_total": metrics.cost_usd_total,
                "cost_usd_per_1000": metrics.cost_usd_per_1000,
                "cost_priced_fraction": metrics.cost_priced_fraction,
                "reported_credits_total": metrics.reported_credits_total,
            }
        )
        for language, values in metrics.by_language.items():
            writer.writerow({"scope": language, **values, "coverage": ""})
    meta_json = html.escape(json.dumps(metadata or {}, ensure_ascii=False, indent=2))
    cost_summary = (
        f"${metrics.cost_usd_total:.6f} total · ${metrics.cost_usd_per_1000:.4f} per 1,000"
        if metrics.cost_usd_total is not None and metrics.cost_usd_per_1000 is not None
        else (
            f"partial (${metrics.cost_usd_known_total:.6f} known; "
            f"{metrics.cost_priced_fraction:.1%} priced)"
        )
    )
    rows = "".join(
        f"<tr><td>{html.escape(language)}</td><td>{values['support']}</td>"
        f"<td>{values['accuracy']:.3f}</td><td>{values['macro_f1']:.3f}</td></tr>"
        for language, values in metrics.by_language.items()
    )
    provider_rows = "".join(
        f"<tr><td>{html.escape(provider)}</td><td>{values['support']}</td>"
        f"<td>{values['p50_ms']:.1f}</td><td>{values['p95_ms']:.1f}</td></tr>"
        for provider, values in metrics.latency_by_provider.items()
    )
    class_rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{values['support']}</td>"
        f"<td>{values['precision']:.3f}</td><td>{values['recall']:.3f}</td>"
        f"<td>{values['f1']:.3f}</td></tr>"
        for label, values in metrics.per_class.items()
    )
    confusion_rows = "".join(
        "<tr><th>"
        + html.escape(label)
        + "</th>"
        + "".join(f"<td>{value}</td>" for value in row)
        + "</tr>"
        for label, row in zip(
            ("negative", "neutral", "positive"), metrics.confusion_matrix, strict=True
        )
    )
    document = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:16px system-ui;max-width:960px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse}}th,td{{border:1px solid #bbb;padding:.4rem .7rem;text-align:right}}
th:first-child,td:first-child{{text-align:left}}code,pre{{background:#f5f5f5;padding:.5rem}}</style>
<h1>{html.escape(title)}</h1>
<p>Support: {metrics.support} · Accuracy: {metrics.accuracy:.3f} · Macro F1:
{metrics.macro_f1:.3f} · Operational coverage: {metrics.operational_coverage:.3f}</p>
<p>Latency P50/P95: {metrics.latency_p50_ms:.1f}/{metrics.latency_p95_ms:.1f} ms ·
Cost: {cost_summary} · Credits: {metrics.reported_credits_total}</p>
<h2>Language breakdown</h2><table><thead><tr><th>Language</th><th>Support</th>
<th>Accuracy</th><th>Macro F1</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Provider latency</h2><table><thead><tr><th>Provider</th><th>Support</th>
<th>P50 ms</th><th>P95 ms</th></tr></thead><tbody>{provider_rows}</tbody></table>
<h2>Class performance</h2><table><thead><tr><th>Class</th><th>Support</th>
<th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{class_rows}</tbody></table>
<h2>Confusion matrix</h2><p>Rows are true labels; columns are predicted labels.</p>
<table><thead><tr><th></th><th>negative</th><th>neutral</th><th>positive</th></tr></thead>
<tbody>{confusion_rows}</tbody></table>
<h2>Run metadata</h2><pre>{meta_json}</pre>
<p>Calibration metrics are emitted only when every prediction has a genuine probability vector.</p>
</html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def write_selection_report(path: Path, selection: ThresholdSelection) -> None:
    points = selection.frontier
    costs = [float(point["expected_cost_usd"]) for point in points]
    scores = [float(point["macro_f1"]) for point in points]
    max_cost = max(costs, default=0) or 1
    min_score = min(scores, default=0)
    score_span = (max(scores, default=1) - min_score) or 1
    circles = "".join(
        f'<circle cx="{40 + 520 * cost / max_cost:.1f}" '
        f'cy="{260 - 220 * (score - min_score) / score_span:.1f}" r="4" '
        f'fill="{"#167d32" if point["qualifies"] else "#b23a2b"}">'
        f"<title>threshold {point['threshold']:.2f}; cost ${cost:.6f}; "
        f"macro F1 {score:.4f}; fallback {point['fallback_fraction']:.1%}</title></circle>"
        for point, cost, score in zip(points, costs, scores, strict=True)
    )
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Routing threshold frontier</title><style>body{{font:16px system-ui;max-width:800px;
margin:2rem auto}}svg{{border:1px solid #aaa}}code{{background:#eee;padding:.2rem}}</style>
<h1>Routing threshold frontier</h1><p>Selected threshold: <code>{selection.threshold}</code> ·
baseline macro F1: {selection.baseline_macro_f1:.4f} · maximum degradation:
{selection.max_f1_degradation:.4f}</p><svg viewBox="0 0 600 300" width="100%" role="img"
aria-label="Expected USD cost versus macro F1"><line x1="40" y1="260" x2="560" y2="260"
stroke="black"/><line x1="40" y1="40" x2="40" y2="260" stroke="black"/>{circles}
<text x="240" y="290">Expected cost (USD)</text><text x="8" y="25">Macro F1</text></svg>
<p>Green points meet the predefined macro-F1 degradation constraint. Hover a point for
threshold, cost, score, and fallback fraction.</p></html>"""
    path.with_suffix(".html").write_text(document, encoding="utf-8")
