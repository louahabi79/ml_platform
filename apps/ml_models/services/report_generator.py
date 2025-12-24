from __future__ import annotations

import io
from typing import Dict, Any, List, Optional, Tuple

from django.utils import timezone

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)
from reportlab.lib.utils import ImageReader

from apps.ml_models.models import Experiment, EvaluationResult
from apps.visualization.models import PlotArtifact


def _flatten_metrics(metrics: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    metrics stored as:
      {"test": {...}, "cv": {...}}
    Returns list of (metric_name, value_str).
    """
    rows = []
    test = metrics.get("test", {}) if isinstance(metrics, dict) else {}
    cv = metrics.get("cv", {}) if isinstance(metrics, dict) else {}

    for k, v in (test or {}).items():
        rows.append((f"test.{k}", str(v)))

    # cv is {metric: {mean, std}}
    if isinstance(cv, dict):
        for metric, stats in cv.items():
            if isinstance(stats, dict) and "mean" in stats and "std" in stats:
                rows.append((f"cv.{metric}.mean", str(stats["mean"])))
                rows.append((f"cv.{metric}.std", str(stats["std"])))
            else:
                rows.append((f"cv.{metric}", str(stats)))

    return rows


def _plot_file_path(plot: PlotArtifact) -> Optional[str]:
    # Check the correct field name first
    if hasattr(plot, "image_file") and plot.image_file:
        return plot.image_file.path
    # Fallbacks just in case
    if hasattr(plot, "image") and plot.image:
        return plot.image.path
    if hasattr(plot, "file") and plot.file:
        return plot.file.path
    return None


def generate_experiment_report_pdf(experiment: Experiment) -> bytes:
    """
    Produces a PDF with:
      - Project name, experiment name, algorithm, status
      - Metrics table
      - Embedded PNG plots from PlotArtifact linked to this experiment
    Returns PDF bytes.
    """
    evaluation = EvaluationResult.objects.filter(experiment=experiment).first()
    metrics = evaluation.metrics if evaluation else {}

    # Gather plots (order matters for presentation)
    wanted_order = [
        "exp_learning_curve",
        "exp_actual_vs_predicted",
        "exp_confusion_matrix",
    ]
    plots = list(PlotArtifact.objects.filter(experiment=experiment))
    plot_by_type = {p.plot_type: p for p in plots}
    ordered_plots = [plot_by_type[t] for t in wanted_order if t in plot_by_type]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=f"Experiment Report - {experiment.name}")

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Machine Learning Experiment Report", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))

    project_name = getattr(experiment.project, "name", "Project")
    story.append(Paragraph(f"<b>Project:</b> {project_name}", styles["Normal"]))
    story.append(Paragraph(f"<b>Experiment:</b> {experiment.name}", styles["Normal"]))
    story.append(Paragraph(f"<b>Algorithm:</b> {experiment.algorithm.display_name}", styles["Normal"]))
    story.append(Paragraph(f"<b>Status:</b> {experiment.status}", styles["Normal"]))
    story.append(Paragraph(f"<b>Generated:</b> {timezone.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 0.25 * inch))

    # Metrics Table
    story.append(Paragraph("Metrics", styles["Heading2"]))
    metric_rows = _flatten_metrics(metrics)
    if not metric_rows:
        story.append(Paragraph("No metrics available yet (experiment may still be running).", styles["Normal"]))
    else:
        data = [["Metric", "Value"]] + metric_rows
        table = Table(data, colWidths=[3.2 * inch, 2.8 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightyellow]),
        ]))
        story.append(table)

    story.append(Spacer(1, 0.25 * inch))

    # Plots
    story.append(Paragraph("Plots", styles["Heading2"]))
    if not ordered_plots:
        story.append(Paragraph("No plots available yet.", styles["Normal"]))
    else:
        for p in ordered_plots:
            path = _plot_file_path(p)
            if not path:
                continue
            story.append(Paragraph(f"<b>{p.title}</b>", styles["Normal"]))
            story.append(Spacer(1, 0.08 * inch))

            # Scale image to fit width
            img = Image(path)
            img._restrictSize(6.7 * inch, 4.5 * inch)
            story.append(img)
            story.append(Spacer(1, 0.2 * inch))

    doc.build(story)
    return buf.getvalue()
