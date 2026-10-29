"""Plotly figure builders. Every function copes with empty/missing data."""
from __future__ import annotations

from typing import Mapping, Sequence

import plotly.graph_objects as go

from config import settings

COLORS = {"positive": "#2E9E6B", "neutral": "#8A94A6", "negative": "#D64550"}
_LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), font=dict(family="Inter, Segoe UI, sans-serif"),
               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")


def _empty(message: str, title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color="#8A94A6"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title=title, height=300, **_LAYOUT)
    return fig


def sentiment_distribution_fig(counts: Mapping[str, int], title: str = "Sentiment distribution") -> go.Figure:
    labels = [l for l in settings.LABELS if counts.get(l, 0) > 0]
    if not labels:
        return _empty("No predictions yet", title)
    fig = go.Figure(go.Pie(
        labels=[l.title() for l in labels], values=[counts[l] for l in labels], hole=0.55,
        marker=dict(colors=[COLORS[l] for l in labels]), sort=False, textinfo="label+percent",
    ))
    fig.update_layout(title=title, height=320, showlegend=False, **_LAYOUT)
    return fig


def score_bar_fig(scores: Mapping[str, float], kind: str) -> go.Figure:
    title = "Model-estimated probabilities" if kind == "probability" else "Decision scores (not probabilities)"
    if not scores:
        return _empty("No score information for this model", title)
    labels = [l for l in settings.LABELS if l in scores] or list(scores)
    values = [scores[l] for l in labels]
    text = [f"{v:.1%}" if kind == "probability" else f"{v:+.2f}" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=[l.title() for l in labels], orientation="h", text=text, textposition="auto",
        marker_color=[COLORS.get(l, "#4F6BED") for l in labels],
    ))
    if kind == "probability":
        fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_layout(title=title, height=240, **_LAYOUT)
    return fig


def confusion_matrix_fig(matrix: Sequence[Sequence[int]] | None, labels: Sequence[str]) -> go.Figure:
    if not matrix or not len(matrix):
        return _empty("No confusion matrix available", "Confusion matrix (test set)")
    names = [l.title() for l in labels]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=names, y=names, colorscale="Blues", text=matrix, texttemplate="%{text}",
        hovertemplate="Actual %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>", showscale=False,
    ))
    fig.update_xaxes(title="Predicted", side="bottom")
    fig.update_yaxes(title="Actual", autorange="reversed")
    fig.update_layout(title="Confusion matrix (test set)", height=380, **_LAYOUT)
    return fig


def model_comparison_fig(rows: Sequence[Mapping] | None) -> go.Figure:
    title = "Validation comparison (cross-validated macro F1, training split)"
    if not rows:
        return _empty("No model comparison available", title)
    fig = go.Figure(go.Bar(
        x=[r["model"] for r in rows], y=[r["cv_f1_macro_mean"] for r in rows],
        error_y=dict(type="data", array=[r.get("cv_f1_macro_std", 0) for r in rows]),
        text=[f"{r['cv_f1_macro_mean']:.3f}" for r in rows], textposition="outside", marker_color="#4F6BED",
    ))
    fig.update_yaxes(range=[0, 1.1], title="Macro F1")
    fig.update_layout(title=title, height=360, **_LAYOUT)
    return fig


def per_class_fig(per_class: Mapping[str, Mapping[str, float]] | None) -> go.Figure:
    if not per_class:
        return _empty("No per-class metrics available", "Per-class performance (test set)")
    labels = list(per_class)
    fig = go.Figure()
    for metric, color in (("precision", "#4F6BED"), ("recall", "#2E9E6B"), ("f1", "#E8A33D")):
        fig.add_bar(name=metric.title() if metric != "f1" else "F1", x=[l.title() for l in labels],
                    y=[per_class[l][metric] for l in labels], marker_color=color)
    fig.update_yaxes(range=[0, 1.05])
    fig.update_layout(title="Per-class performance (test set)", barmode="group", height=340, **_LAYOUT)
    return fig
