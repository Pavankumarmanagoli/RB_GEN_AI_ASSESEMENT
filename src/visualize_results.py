"""Generate visualizations from the final evaluation results."""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from src.config import OUTPUT_DIR

FIGURES_DIR = OUTPUT_DIR / "figures"

SUMMARY_PATH = OUTPUT_DIR / "final_evaluation_summary.json"
PAIRWISE_PATH = OUTPUT_DIR / "final_pairwise_evaluation.json"

# Shared color system: one primary dark accent (Automated), one secondary
# muted accent (Human-reviewed), and neutral greys for supporting elements.
COLOR_AUTOMATED = "#1B2A4A"
COLOR_HUMAN = "#B08A4E"
COLOR_GREY_DARK = "#3A3A3A"
COLOR_GREY_MID = "#8C8C8C"
COLOR_GREY_LIGHT = "#D9D9D9"
COLOR_BG = "#FAFAF8"

# Subtle ordered tonal progression (weak -> strong), no traffic-light styling.
DISTRIBUTION_TONES = ["#C9CDD6", "#9AA3B5", "#6B7690", "#3E4A66", "#1B2A4A"]

FONT_FAMILY = "DejaVu Sans"

TITLE_KW = dict(fontsize=17, fontweight="bold", color="#1A1A1A", ha="left")
SUBTITLE_KW = dict(fontsize=11, color=COLOR_GREY_MID, ha="left")
LABEL_KW = dict(fontsize=10.5, color=COLOR_GREY_DARK)
DATA_LABEL_KW = dict(fontsize=10.5, fontweight="bold", color="#1A1A1A")
ANNOTATION_KW = dict(fontsize=9.5, color=COLOR_GREY_MID, style="italic")

DPI = 300


def load_summary() -> dict:
    with open(SUMMARY_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def load_pairwise() -> list[dict]:
    with open(PAIRWISE_PATH, encoding="utf-8") as handle:
        return json.load(handle)["results"]


def new_figure(figsize: tuple):
    plt.rcParams["font.family"] = FONT_FAMILY
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    return fig, ax


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_GREY_LIGHT)
    ax.spines["bottom"].set_color(COLOR_GREY_LIGHT)
    ax.tick_params(length=0)


def add_title_block(fig, title: str, subtitle: str | None = None, x: float = 0.06, y: float = 0.97):
    fig.text(x, y, title, **TITLE_KW)
    if subtitle:
        fig.text(x, y - 0.065, subtitle, **SUBTITLE_KW)


def add_footnote(fig, text: str, x: float = 0.06, y: float = 0.01):
    fig.text(x, y, text, **ANNOTATION_KW)


def save_figure(fig, stem: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def assert_close(actual, expected, label: str, tol: float = 1e-9):
    # Fail loudly rather than silently plotting a value that has drifted from the source data.
    if abs(actual - expected) > tol:
        raise ValueError(f"Frozen value mismatch for {label}: expected {expected}, got {actual}")


def figure_overall_match(summary: dict):
    """Save 01_overall_match_scores: Automated vs. Human-reviewed mean score."""
    automated = summary["automated_fidelity_mean"]
    human = summary["human_fidelity_mean"]
    assert_close(automated, 2.6333333333333333, "automated_fidelity_mean")
    assert_close(human, 2.533333333333333, "human_fidelity_mean")

    labels = ["Automated Match Score", "Human-Reviewed Match Score"]
    values = [automated, human]
    colors = [COLOR_AUTOMATED, COLOR_HUMAN]

    fig, ax = new_figure((7.5, 5.2))
    fig.subplots_adjust(top=0.83, bottom=0.12, left=0.1, right=0.95)

    x_positions = [0, 1]
    bars = ax.bar(x_positions, values, width=0.5, color=colors, zorder=3)

    ax.set_ylim(0, 5)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, **LABEL_KW)
    ax.set_yticks(range(0, 6))
    ax.set_yticklabels(range(0, 6), fontsize=9.5, color=COLOR_GREY_MID)
    ax.set_ylabel("Score (0-5 scale)", **LABEL_KW)
    ax.yaxis.grid(True, color=COLOR_GREY_LIGHT, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.12, f"{value:.2f} / 5",
                ha="center", va="bottom", **DATA_LABEL_KW)

    add_title_block(fig, "Overall Human-AI Match",
                     "Automated and Human review reach similar conclusions", x=0.1)

    save_figure(fig, "01_overall_match_scores")


def figure_match_distribution(pairwise_rows: list[dict]):
    """Save 02_match_distribution: Human-reviewed match score counts across all rows."""
    counts = Counter(row["human_fidelity_score"] for row in pairwise_rows)
    total = len(pairwise_rows)
    if total != 30:
        raise ValueError(f"Expected exactly 30 rows, got {total}")

    score_to_label = {
        5: "Very strong match",
        4: "Strong match",
        3: "Partial match",
        2: "Weak match",
        1: "Very weak match",
    }
    ordered_scores = [5, 4, 3, 2, 1]
    values = [counts.get(score, 0) for score in ordered_scores]
    labels = [score_to_label[score] for score in ordered_scores]

    expected = {5: 0, 4: 7, 3: 10, 2: 5, 1: 8}
    for score in ordered_scores:
        assert_close(counts.get(score, 0), expected[score], f"human_fidelity_score count for {score}")

    partial_or_below = counts.get(3, 0) + counts.get(2, 0) + counts.get(1, 0)

    fig, ax = new_figure((9, 5.2))
    fig.subplots_adjust(top=0.86, bottom=0.15, left=0.22, right=0.93)

    y_positions = list(range(len(labels)))[::-1]
    colors = list(reversed(DISTRIBUTION_TONES))
    bars = ax.barh(y_positions, values, height=0.6, color=colors, zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, **LABEL_KW)
    ax.set_xlim(0, max(values) + 3)
    ax.set_xlabel("Number of responses (of 30)", **LABEL_KW)
    ax.xaxis.grid(True, color=COLOR_GREY_LIGHT, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)

    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, str(value),
                ha="left", va="center", **DATA_LABEL_KW)

    add_title_block(fig, "Human-Reviewed Match Across 30 Responses", x=0.06)
    add_footnote(fig, f"{partial_or_below} of 30 responses were Partial Match or below")

    save_figure(fig, "02_match_distribution")


def figure_failure_modes(summary: dict):
    """Save 03_failure_modes: share of rows with each human-flagged failure mode."""
    evidence = summary["failure_evidence"]
    unsupported = evidence["human_unsupported_detail_rate"]
    omission = evidence["human_omission_rate"]
    contradiction = evidence["human_contradiction_rate"]

    assert_close(unsupported, 100.0, "human_unsupported_detail_rate")
    assert_close(omission, 73.33333333333333, "human_omission_rate")
    assert_close(contradiction, 36.666666666666664, "human_contradiction_rate")

    labels = ["Added unsupported detail", "Important omission", "Contradiction"]
    values = [unsupported, omission, contradiction]
    colors = ["#1B2A4A", "#4A5C82", "#8C97B5"]

    fig, ax = new_figure((9, 5.0))
    fig.subplots_adjust(top=0.86, bottom=0.24, left=0.24, right=0.93)

    y_positions = list(range(len(labels)))[::-1]
    bars = ax.barh(y_positions, values, height=0.55, color=colors, zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, **LABEL_KW)
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Share of 30 responses", **LABEL_KW)
    ax.xaxis.grid(True, color=COLOR_GREY_LIGHT, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)

    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%",
                ha="left", va="center", **DATA_LABEL_KW)

    add_title_block(fig, "Where the Simulations Diverge", x=0.06)
    add_footnote(
        fig,
        "An unsupported detail is information in the AI answer not supported by the Human reference;\n"
        "it is not necessarily false.",
        y=0.03,
    )

    save_figure(fig, "03_failure_modes")


def figure_person_comparison(summary: dict):
    """Save 04_person_comparison: Automated vs. Human-reviewed score per person."""
    person_summary = {row["person_id"]: row for row in summary["person_summary"]}
    consistency = summary["person_consistency"]

    expected = {
        "c_human": (2.90, 2.70, "Low"),
        "g_human": (2.30, 2.50, "Low"),
        "s_human": (2.70, 2.40, "Medium"),
    }
    person_ids = ["c_human", "g_human", "s_human"]
    if set(person_summary.keys()) != set(person_ids):
        raise ValueError(f"Unexpected person_ids: {set(person_summary.keys())}")

    automated_values = []
    human_values = []
    for person_id in person_ids:
        row = person_summary[person_id]
        automated = row["automated_fidelity_mean"]
        human = row["human_fidelity_mean"]
        exp_auto, exp_human, exp_consistency = expected[person_id]
        assert_close(automated, exp_auto, f"{person_id} automated_fidelity_mean")
        assert_close(human, exp_human, f"{person_id} human_fidelity_mean")
        if consistency[person_id] != exp_consistency:
            raise ValueError(f"Consistency mismatch for {person_id}: {consistency[person_id]}")
        automated_values.append(automated)
        human_values.append(human)

    fig, ax = new_figure((9.5, 5.4))
    fig.subplots_adjust(top=0.85, bottom=0.16, left=0.09, right=0.95)

    x = list(range(len(person_ids)))
    width = 0.32
    x_automated = [pos - width / 2 for pos in x]
    x_human = [pos + width / 2 for pos in x]

    bars_auto = ax.bar(x_automated, automated_values, width=width, color=COLOR_AUTOMATED,
                        label="Automated Match Score", zorder=3)
    bars_human = ax.bar(x_human, human_values, width=width, color=COLOR_HUMAN,
                         label="Human-Reviewed Match Score", zorder=3)

    consistency_suffix = {"c_human": "Low consistency", "g_human": "Low consistency", "s_human": "Medium consistency"}
    x_labels = [f"{pid}\n({consistency_suffix[pid]})" for pid in person_ids]

    ax.set_ylim(0, 5)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, **LABEL_KW)
    ax.set_yticks(range(0, 6))
    ax.set_yticklabels(range(0, 6), fontsize=9.5, color=COLOR_GREY_MID)
    ax.set_ylabel("Score (0-5 scale)", **LABEL_KW)
    ax.yaxis.grid(True, color=COLOR_GREY_LIGHT, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)

    for bars, bar_values in ((bars_auto, automated_values), (bars_human, human_values)):
        for bar, value in zip(bars, bar_values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.12, f"{value:.2f}",
                    ha="center", va="bottom", **DATA_LABEL_KW)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=10.5)

    add_title_block(fig, "Match Scores by Person", x=0.06)

    save_figure(fig, "04_person_comparison")


def generate_all_figures():
    summary = load_summary()
    pairwise_rows = load_pairwise()

    figure_overall_match(summary)
    figure_match_distribution(pairwise_rows)
    figure_failure_modes(summary)
    figure_person_comparison(summary)
