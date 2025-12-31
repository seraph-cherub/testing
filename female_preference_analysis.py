"""Utilities for analysing how different female demographics rate male demographics.

This version only relies on the Python standard library so it can run in
environments where installing third-party packages such as pandas is
impossible.  Beyond descriptive statistics it now surfaces higher-level
insights, including preference similarity across female groups, consensus
rankings, and polarisation markers for male demographics.

The pipeline performs the following steps:

* Cleans the raw ratings so that zero values or malformed entries are
  treated as missing and imputed with the column mean.
* Computes descriptive statistics for each (women, men) pair.
* Produces per-female preference tables and an overall male performance
  summary.
* Creates a heatmap-friendly structure that highlights the ratings matrix.
* Quantifies how similarly female groups behave relative to one another and
  where they diverge most strongly.
* Aggregates the per-female rankings into consensus and polarisation views
  that spotlight stable favourites versus contentious demographics.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List, Tuple
import itertools
import csv
import math


RATING_COLUMNS = [str(i) for i in range(20)]
PLOT_STYLE = "seaborn-v0_8"


@dataclass
class RatingRow:
    """A single row from the dataset after cleaning."""

    women: str
    men: str
    ratings: List[float]
    valid_ratings: List[float]
    @property
    def average_rating(self) -> float:
        values = self.valid_ratings or self.ratings
        return mean(values) if values else float("nan")


def _parse_rating(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number != 0 else None


def load_and_clean_data(csv_path: Path) -> List[RatingRow]:
    """Load the CSV file, clean the rating columns, and return structured rows."""

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        raw_rows: List[Dict[str, str]] = list(reader)

    # First pass: compute per-column means ignoring missing/zero values.
    column_sums: Dict[str, float] = defaultdict(float)
    column_counts: Dict[str, int] = defaultdict(int)

    for row in raw_rows:
        for column in RATING_COLUMNS:
            value = _parse_rating(row.get(column, ""))
            if value is not None:
                column_sums[column] += value
                column_counts[column] += 1

    column_means = {
        column: (column_sums[column] / column_counts[column]) if column_counts[column] else 0.0
        for column in RATING_COLUMNS
    }

    cleaned_rows: List[RatingRow] = []
    for row in raw_rows:
        ratings: List[float] = []
        valid_ratings: List[float] = []
        for column in RATING_COLUMNS:
            value = _parse_rating(row.get(column, ""))
            if value is None:
                value = column_means[column]
            else:
                valid_ratings.append(value)
            ratings.append(value)

        cleaned_rows.append(
            RatingRow(
                women=row.get("Women", "Unknown"),
                men=row.get("Men", "Unknown"),
                ratings=ratings,
                valid_ratings=valid_ratings,
            )
        )

    return cleaned_rows


def _quantile(sorted_values: List[float], fraction: float) -> float:
    if not sorted_values:
        return float("nan")
    if fraction <= 0:
        return sorted_values[0]
    if fraction >= 1:
        return sorted_values[-1]
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def compute_pair_statistics(rows: Iterable[RatingRow]) -> List[Dict[str, object]]:
    """Return descriptive statistics for each (women, men) pair."""

    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    observed_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    total_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    for row in rows:
        key = (row.women, row.men)
        values = row.valid_ratings or row.ratings
        grouped[key].extend(values)
        observed_counts[key] += len(row.valid_ratings)
        total_counts[key] += len(row.ratings)

    stats_tables: List[Dict[str, object]] = []
    for (women, men), values in grouped.items():
        sorted_values = sorted(values)
        q1 = _quantile(sorted_values, 0.25)
        q3 = _quantile(sorted_values, 0.75)
        total = total_counts[(women, men)]
        observed = observed_counts[(women, men)]
        stats_tables.append(
            {
                "Women": women,
                "Men": men,
                "mean_rating": mean(sorted_values),
                "median_rating": median(sorted_values),
                "std_rating": pstdev(sorted_values) if len(sorted_values) > 1 else 0.0,
                "min_rating": sorted_values[0],
                "max_rating": sorted_values[-1],
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "num_observed": observed,
                "num_imputed": total - observed,
                "imputed_ratio": (total - observed) / total if total else 0.0,
            }
        )

    stats_tables.sort(key=lambda item: (item["Women"], item["Men"]))
    return stats_tables


def summarise_female_preferences(
    pair_stats: List[Dict[str, object]]
) -> Tuple[Dict[str, List[Dict[str, object]]], Dict[str, float]]:
    """Produce rankings and deviation tables for each female demographic."""

    # Baseline: how each male group scores across all female groups on average.
    male_totals: Dict[str, List[float]] = defaultdict(list)
    for row in pair_stats:
        male_totals[row["Men"]].append(row["mean_rating"])  # type: ignore[index]

    male_baseline = {
        male: mean(values)
        for male, values in male_totals.items()
    }

    women_groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in pair_stats:
        women_groups[row["Women"]].append(row)

    summaries: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for women_group, rows in women_groups.items():
        group_average = mean(r["mean_rating"] for r in rows)  # type: ignore[index]
        sorted_rows = sorted(rows, key=lambda item: item["mean_rating"])  # type: ignore[index]
        for rank, row in enumerate(sorted_rows, start=1):
            men_group = row["Men"]
            summaries[women_group].append(
                {
                    "Men": men_group,
                    "rank": rank,
                    "mean_rating": row["mean_rating"],
                    "median_rating": row["median_rating"],
                    "std_rating": row["std_rating"],
                    "deviation_from_group": row["mean_rating"] - group_average,
                    "deviation_from_global": row["mean_rating"] - male_baseline[men_group],  # type: ignore[index]
                }
            )

    return summaries, male_baseline


def summarise_male_performance(pair_stats: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate how each male demographic performs across women."""

    male_groups: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for row in pair_stats:
        male_groups[row["Men"]].append((row["Women"], row["mean_rating"]))  # type: ignore[index]

    summary_rows: List[Dict[str, object]] = []
    for men_group, entries in male_groups.items():
        ratings = [rating for _, rating in entries]
        best = min(entries, key=lambda item: item[1])
        worst = max(entries, key=lambda item: item[1])
        summary_rows.append(
            {
                "Men": men_group,
                "mean_of_means": mean(ratings),
                "variability": pstdev(ratings) if len(ratings) > 1 else 0.0,
                "best_women_group": best[0],
                "best_group_score": best[1],
                "worst_women_group": worst[0],
                "worst_group_score": worst[1],
            }
        )

    summary_rows.sort(key=lambda item: item["mean_of_means"])  # type: ignore[index]
    for index, row in enumerate(summary_rows, start=1):
        row["rank"] = index
    return summary_rows


def summarise_female_groups(pair_stats: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """High-level overview for each female demographic."""

    overview_rows: List[Dict[str, object]] = []
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in pair_stats:
        grouped[row["Women"]].append(row)

    for women_group, rows in grouped.items():
        values = [r["mean_rating"] for r in rows]  # type: ignore[index]
        best = min(rows, key=lambda item: item["mean_rating"])  # type: ignore[index]
        worst = max(rows, key=lambda item: item["mean_rating"])  # type: ignore[index]
        overview_rows.append(
            {
                "Women": women_group,
                "overall_mean": mean(values),
                "spread": pstdev(values) if len(values) > 1 else 0.0,
                "favourite_men": best["Men"],
                "favourite_score": best["mean_rating"],
                "least_favourite_men": worst["Men"],
                "least_favourite_score": worst["mean_rating"],
            }
        )

    overview_rows.sort(key=lambda item: item["overall_mean"])  # type: ignore[index]
    return overview_rows


def preference_heatmap_data(pair_stats: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    """Pivot the mean ratings into a matrix for heatmap visualisation."""

    heatmap: Dict[str, Dict[str, float]] = defaultdict(dict)
    for row in pair_stats:
        heatmap[row["Women"]][row["Men"]] = row["mean_rating"]  # type: ignore[index]

    return {
        women: dict(sorted(men_items.items()))
        for women, men_items in sorted(heatmap.items())
    }


def _pearson_correlation(values_a: List[float], values_b: List[float]) -> float:
    """Compute the Pearson correlation coefficient for two vectors."""

    paired = [
        (a, b)
        for a, b in zip(values_a, values_b)
        if not math.isnan(a) and not math.isnan(b)
    ]
    n = len(paired)
    if n < 2:
        return float("nan")

    sum_x = sum(a for a, _ in paired)
    sum_y = sum(b for _, b in paired)
    sum_x2 = sum(a * a for a, _ in paired)
    sum_y2 = sum(b * b for _, b in paired)
    sum_xy = sum(a * b for a, b in paired)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt(
        (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)
    )
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def build_female_mean_matrix(
    pair_stats: List[Dict[str, object]]
) -> Dict[str, Dict[str, float]]:
    """Return female -> male -> mean rating mappings."""

    matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
    for row in pair_stats:
        matrix[row["Women"]][row["Men"]] = row["mean_rating"]  # type: ignore[index]

    ordered_men = sorted({row["Men"] for row in pair_stats})  # type: ignore[index]
    return {
        women: {men: matrix[women].get(men, float("nan")) for men in ordered_men}
        for women in sorted(matrix)
    }


def compute_female_similarity(
    pair_stats: List[Dict[str, object]]
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, object]]]:
    """Measure how closely aligned female demographics are with each other."""

    matrix = build_female_mean_matrix(pair_stats)
    female_groups = list(matrix.keys())

    correlation_table: Dict[str, Dict[str, float]] = {
        women: {other: float("nan") for other in female_groups}
        for women in female_groups
    }

    pairwise: List[Dict[str, object]] = []
    for women_a, women_b in itertools.combinations(female_groups, 2):
        values_a = list(matrix[women_a].values())
        values_b = list(matrix[women_b].values())
        correlation = _pearson_correlation(values_a, values_b)
        correlation_table[women_a][women_b] = correlation
        correlation_table[women_b][women_a] = correlation
        pairwise.append(
            {
                "Women A": women_a,
                "Women B": women_b,
                "correlation": correlation,
            }
        )

    for women in female_groups:
        correlation_table[women][women] = 1.0

    pairwise.sort(key=lambda item: item["correlation"], reverse=True)  # type: ignore[index]
    return correlation_table, pairwise


def compute_consensus_ranking(
    female_tables: Dict[str, List[Dict[str, object]]]
) -> List[Dict[str, object]]:
    """Aggregate individual female rankings into a consensus leaderboard."""

    if not female_tables:
        return []

    male_ranks: Dict[str, List[int]] = defaultdict(list)
    borda_scores: Dict[str, float] = defaultdict(float)
    top_finishes: Dict[str, int] = defaultdict(int)
    top_three_finishes: Dict[str, int] = defaultdict(int)
    bottom_three_finishes: Dict[str, int] = defaultdict(int)
    female_count = len(female_tables)
    all_men = sorted({row["Men"] for rows in female_tables.values() for row in rows})  # type: ignore[index]
    num_men = len(all_men)

    for rows in female_tables.values():
        for entry in rows:
            men_group = entry["Men"]  # type: ignore[index]
            rank = int(entry["rank"])  # type: ignore[index]
            male_ranks[men_group].append(rank)
            borda_scores[men_group] += num_men - rank
            if rank == 1:
                top_finishes[men_group] += 1
            if rank <= 3:
                top_three_finishes[men_group] += 1
            if rank >= num_men - 2:
                bottom_three_finishes[men_group] += 1

    summary: List[Dict[str, object]] = []
    for men_group in all_men:
        ranks = male_ranks[men_group]
        ranks_sorted = sorted(ranks)
        q1 = _quantile(ranks_sorted, 0.25)
        q3 = _quantile(ranks_sorted, 0.75)
        summary.append(
            {
                "Men": men_group,
                "average_rank": mean(ranks),
                "rank_std": pstdev(ranks) if len(ranks) > 1 else 0.0,
                "rank_range": max(ranks) - min(ranks),
                "iqr_rank": q3 - q1,
                "borda_score": borda_scores[men_group],
                "borda_share": borda_scores[men_group] / (female_count * (num_men - 1))
                if num_men > 1
                else float("nan"),
                "top_finish_rate": top_finishes[men_group] / female_count,
                "top_three_rate": top_three_finishes[men_group] / female_count,
                "bottom_three_rate": bottom_three_finishes[men_group] / female_count,
            }
        )

    summary.sort(key=lambda item: (item["average_rank"], -item["borda_score"]))  # type: ignore[index]
    for index, row in enumerate(summary, start=1):
        row["consensus_rank"] = index
    return summary


def compute_polarisation_report(
    pair_stats: List[Dict[str, object]]
) -> List[Dict[str, object]]:
    """Highlight which male groups are most divisive across women."""

    grouped: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for row in pair_stats:
        grouped[row["Men"]].append((row["Women"], row["mean_rating"]))  # type: ignore[index]

    report: List[Dict[str, object]] = []
    for men_group, entries in grouped.items():
        ratings = [rating for _, rating in entries]
        sorted_ratings = sorted(ratings)
        q10 = _quantile(sorted_ratings, 0.10)
        q90 = _quantile(sorted_ratings, 0.90)
        best = min(entries, key=lambda item: item[1])
        worst = max(entries, key=lambda item: item[1])
        report.append(
            {
                "Men": men_group,
                "polarisation": q90 - q10,
                "max_gap": worst[1] - best[1],
                "supporters": best[0],
                "supporter_score": best[1],
                "critics": worst[0],
                "critic_score": worst[1],
            }
        )

    report.sort(key=lambda item: item["polarisation"], reverse=True)  # type: ignore[index]
    return report


def compute_distinctive_preferences(
    female_tables: Dict[str, List[Dict[str, object]]]
) -> List[Dict[str, object]]:
    """Identify which men each female group uniquely supports or avoids."""

    highlights: List[Dict[str, object]] = []
    for women_group, rows in sorted(female_tables.items()):
        sorted_rows = sorted(
            rows,
            key=lambda item: item["deviation_from_global"],  # type: ignore[index]
            reverse=True,
        )
        favourites = sorted_rows[:3]
        aversions = sorted_rows[-3:]

        def format_entries(entries: List[Dict[str, object]]) -> str:
            return ", ".join(
                f"{entry['Men']} ({entry['deviation_from_global']:+.2f})"  # type: ignore[index]
                for entry in entries
            )

        highlights.append(
            {
                "Women": women_group,
                "signature_favourites": format_entries(favourites),
                "signature_avoids": format_entries(list(reversed(aversions))),
            }
        )

    return highlights


def _safe_import_matplotlib() -> Tuple[object | None, object | None]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib import cm  # type: ignore
    except ImportError:
        return None, None
    return plt, cm


def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_visualisations(
    output_dir: Path,
    heatmap_table: Dict[str, Dict[str, float]],
    male_summary: List[Dict[str, object]],
    female_overview: List[Dict[str, object]],
    correlation_matrix: Dict[str, Dict[str, float]],
    consensus_table: List[Dict[str, object]],
    polarisation_table: List[Dict[str, object]],
) -> List[Path]:
    """Generate visual summaries if matplotlib is available."""

    plt, cm = _safe_import_matplotlib()
    if plt is None or cm is None:
        print("Matplotlib not available; skipping visualisations.")
        return []

    output_dir = _ensure_output_dir(output_dir)
    plt.style.use(PLOT_STYLE)
    saved_paths: List[Path] = []

    women = list(heatmap_table.keys())
    men = sorted({men for row in heatmap_table.values() for men in row})
    heatmap_values = [
        [heatmap_table[women_group].get(men_group, float("nan")) for men_group in men]
        for women_group in women
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(men) * 0.7), max(6, len(women) * 0.6)))
    cax = ax.imshow(heatmap_values, cmap="viridis")
    ax.set_xticks(range(len(men)))
    ax.set_yticks(range(len(women)))
    ax.set_xticklabels(men, rotation=45, ha="right")
    ax.set_yticklabels(women)
    ax.set_title("Mean Ratings Heatmap (Women x Men)")
    fig.colorbar(cax, ax=ax, label="Mean rating")
    fig.tight_layout()
    heatmap_path = output_dir / "mean_ratings_heatmap.png"
    fig.savefig(heatmap_path, dpi=150)
    saved_paths.append(heatmap_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    men_labels = [row["Men"] for row in male_summary]
    men_means = [row["mean_of_means"] for row in male_summary]
    ax.barh(men_labels, men_means, color=cm.Blues(0.6))
    ax.set_xlabel("Average mean rating")
    ax.set_title("Overall Male Performance")
    ax.invert_yaxis()
    fig.tight_layout()
    male_path = output_dir / "male_performance_bar.png"
    fig.savefig(male_path, dpi=150)
    saved_paths.append(male_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    women_labels = [row["Women"] for row in female_overview]
    women_means = [row["overall_mean"] for row in female_overview]
    ax.barh(women_labels, women_means, color=cm.Purples(0.6))
    ax.set_xlabel("Average rating given")
    ax.set_title("Female Group Rating Baselines")
    ax.invert_yaxis()
    fig.tight_layout()
    female_path = output_dir / "female_baselines_bar.png"
    fig.savefig(female_path, dpi=150)
    saved_paths.append(female_path)
    plt.close(fig)

    corr_labels = sorted(correlation_matrix)
    corr_values = [
        [correlation_matrix[row].get(col, float("nan")) for col in corr_labels]
        for row in corr_labels
    ]
    fig, ax = plt.subplots(figsize=(max(7, len(corr_labels) * 0.7), max(6, len(corr_labels) * 0.6)))
    cax = ax.imshow(corr_values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_labels)))
    ax.set_yticks(range(len(corr_labels)))
    ax.set_xticklabels(corr_labels, rotation=45, ha="right")
    ax.set_yticklabels(corr_labels)
    ax.set_title("Female Similarity (Pearson Correlation)")
    fig.colorbar(cax, ax=ax, label="Correlation")
    fig.tight_layout()
    corr_path = output_dir / "female_similarity_heatmap.png"
    fig.savefig(corr_path, dpi=150)
    saved_paths.append(corr_path)
    plt.close(fig)

    polarisation_by_men = {row["Men"]: row["polarisation"] for row in polarisation_table}
    fig, ax = plt.subplots(figsize=(7, 5))
    x_vals = [row["average_rank"] for row in consensus_table]
    y_vals = [polarisation_by_men.get(row["Men"], 0.0) for row in consensus_table]
    labels = [row["Men"] for row in consensus_table]
    scatter = ax.scatter(x_vals, y_vals, c=y_vals, cmap="plasma", s=100, edgecolor="black")
    for x, y, label in zip(x_vals, y_vals, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Consensus average rank (lower = better)")
    ax.set_ylabel("Polarisation (90th-10th percentile)")
    ax.set_title("Consensus Rank vs Polarisation")
    fig.colorbar(scatter, ax=ax, label="Polarisation")
    fig.tight_layout()
    scatter_path = output_dir / "consensus_vs_polarisation.png"
    fig.savefig(scatter_path, dpi=150)
    saved_paths.append(scatter_path)
    plt.close(fig)

    return saved_paths


def run_analysis(csv_path: Path) -> Tuple[
    List[Dict[str, object]],
    Dict[str, List[Dict[str, object]]],
    List[Dict[str, object]],
    Dict[str, Dict[str, float]],
    List[Dict[str, object]],
    Dict[str, float],
]:
    """Execute the full female preference analysis pipeline."""

    rows = load_and_clean_data(csv_path)
    pair_stats = compute_pair_statistics(rows)
    female_preference_tables, male_baseline = summarise_female_preferences(pair_stats)
    male_summary = summarise_male_performance(pair_stats)
    female_overview = summarise_female_groups(pair_stats)
    heatmap_table = preference_heatmap_data(pair_stats)
    return (
        pair_stats,
        female_preference_tables,
        male_summary,
        heatmap_table,
        female_overview,
        male_baseline,
    )


def _stringify(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.3f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _format_table(rows: List[Dict[str, object]], headers: List[str]) -> str:
    """Return a simple aligned table representation."""

    if not rows:
        return "(no data)"

    is_numeric = {
        header: all(isinstance(row.get(header), (int, float)) for row in rows)
        for header in headers
    }

    formatted_rows = [
        {header: _stringify(row.get(header, "")) for header in headers}
        for row in rows
    ]

    column_widths = {
        header: max(len(header), *(len(row[header]) for row in formatted_rows))
        for header in headers
    }

    def render_row(row: Dict[str, str]) -> str:
        cells: List[str] = []
        for header in headers:
            alignment = ">" if is_numeric[header] else "<"
            cells.append(f"{row[header]:{alignment}{column_widths[header]}}")
        return " | ".join(cells)

    header_line = " | ".join(
        f"{header:>{column_widths[header]}}" if is_numeric[header] else f"{header:<{column_widths[header]}}"
        for header in headers
    )
    separator = "-+-".join("-" * column_widths[header] for header in headers)
    body = "\n".join(render_row(row) for row in formatted_rows)
    return f"{header_line}\n{separator}\n{body}"


def _format_nested_table(table: Dict[str, List[Dict[str, object]]], headers: List[str]) -> str:
    sections: List[str] = []
    for women_group, rows in sorted(table.items()):
        sections.append(f"-- {women_group.title()} --\n{_format_table(rows, headers)}")
    return "\n\n".join(sections)


def _format_heatmap(heatmap: Dict[str, Dict[str, float]]) -> str:
    if not heatmap:
        return "(no data)"

    men_headers = sorted({men for row in heatmap.values() for men in row})
    column_widths = {
        header: max(len(header), *(len(_stringify(row.get(header, float("nan")))) for row in heatmap.values()))
        for header in men_headers
    }
    women_width = max(len("Women"), *(len(women) for women in heatmap))

    lines = [
        "Women".ljust(women_width)
        + " | "
        + " | ".join(header.rjust(column_widths[header]) for header in men_headers)
    ]
    lines.append("-" * women_width + "-+-" + "-+-".join("-" * column_widths[h] for h in men_headers))

    for women, men_entries in heatmap.items():
        row_values = [
            _stringify(men_entries.get(header, float("nan"))) for header in men_headers
        ]
        row_values = [value.rjust(column_widths[header]) for value, header in zip(row_values, men_headers)]
        lines.append(women.ljust(women_width) + " | " + " | ".join(row_values))

    return "\n".join(lines)


def _format_square_matrix(matrix: Dict[str, Dict[str, float]]) -> str:
    """Render a symmetric matrix (e.g. correlation table)."""

    if not matrix:
        return "(no data)"

    headers = sorted(matrix)
    column_widths = {
        header: max(
            len(header),
            *(len(_stringify(matrix[row].get(header, float("nan")))) for row in headers)
        )
        for header in headers
    }
    row_width = max(len("Women"), *(len(header) for header in headers))

    header_line = (
        "Women".ljust(row_width)
        + " | "
        + " | ".join(header.rjust(column_widths[header]) for header in headers)
    )
    separator = "-" * row_width + "-+-" + "-+-".join("-" * column_widths[h] for h in headers)

    body_lines = []
    for row in headers:
        values = [
            _stringify(matrix[row].get(header, float("nan"))).rjust(column_widths[header])
            for header in headers
        ]
        body_lines.append(row.ljust(row_width) + " | " + " | ".join(values))

    return "\n".join([header_line, separator, *body_lines])


if __name__ == "__main__":
    dataset_path = Path(__file__).with_name("racial_smv_example_data.csv")
    visuals_dir = Path(__file__).with_name("analysis_outputs")

    (
        pair_stats,
        female_tables,
        male_summary,
        heatmap_table,
        female_overview,
        male_baseline,
    ) = run_analysis(dataset_path)

    correlation_matrix, similarity_pairs = compute_female_similarity(pair_stats)
    consensus_table = compute_consensus_ranking(female_tables)
    polarisation_table = compute_polarisation_report(pair_stats)
    distinctive_preferences = compute_distinctive_preferences(female_tables)
    global_baseline_rows = [
        {"Men": men, "global_mean": value}
        for men, value in sorted(male_baseline.items(), key=lambda item: item[1])
    ]
    saved_visuals = generate_visualisations(
        visuals_dir,
        heatmap_table,
        male_summary,
        female_overview,
        correlation_matrix,
        consensus_table,
        polarisation_table,
    )

    print("=== Pair Statistics (Women x Men) ===")
    print(
        _format_table(
            pair_stats,
            [
                "Women",
                "Men",
                "mean_rating",
                "median_rating",
                "std_rating",
                "q1",
                "q3",
                "iqr",
                "min_rating",
                "max_rating",
                "num_observed",
                "num_imputed",
                "imputed_ratio",
            ],
        )
    )
    print()

    print("=== Female Preference Tables ===")
    print(
        _format_nested_table(
            female_tables,
            [
                "rank",
                "Men",
                "mean_rating",
                "median_rating",
                "std_rating",
                "deviation_from_group",
                "deviation_from_global",
            ],
        )
    )

    print("\n=== Female Demographic Overview ===")
    print(
        _format_table(
            female_overview,
            [
                "Women",
                "overall_mean",
                "spread",
                "favourite_men",
                "favourite_score",
                "least_favourite_men",
                "least_favourite_score",
            ],
        )
    )

    print("\n=== Male Demographic Summary ===")
    print(
        _format_table(
            male_summary,
            [
                "rank",
                "Men",
                "mean_of_means",
                "variability",
                "best_women_group",
                "best_group_score",
                "worst_women_group",
                "worst_group_score",
            ],
        )
    )

    print("\n=== Heatmap Table (mean ratings) ===")
    print(_format_heatmap(heatmap_table))

    print("\n=== Global Male Baseline (all women combined) ===")
    print(
        _format_table(
            global_baseline_rows,
            ["Men", "global_mean"],
        )
    )

    print("\n=== Female Similarity Matrix (Pearson correlations) ===")
    print(_format_square_matrix(correlation_matrix))

    print("\nTop 5 strongest alignments")
    print(
        _format_table(
            similarity_pairs[:5],
            ["Women A", "Women B", "correlation"],
        )
    )

    print("\nTop 5 biggest disagreements")
    print(
        _format_table(
            list(reversed(similarity_pairs[-5:])),
            ["Women A", "Women B", "correlation"],
        )
    )

    print("\n=== Consensus Ranking Across Female Groups ===")
    print(
        _format_table(
            consensus_table,
            [
                "consensus_rank",
                "Men",
                "average_rank",
                "rank_std",
                "rank_range",
                "iqr_rank",
                "borda_score",
                "borda_share",
                "top_finish_rate",
                "top_three_rate",
                "bottom_three_rate",
            ],
        )
    )

    print("\n=== Polarisation Report (higher = more divisive) ===")
    print(
        _format_table(
            polarisation_table,
            [
                "Men",
                "polarisation",
                "max_gap",
                "supporters",
                "supporter_score",
                "critics",
                "critic_score",
            ],
        )
    )

    print("\n=== Distinctive Female Preferences ===")
    print(
        _format_table(
            distinctive_preferences,
            [
                "Women",
                "signature_favourites",
                "signature_avoids",
            ],
        )
    )

    if saved_visuals:
        print("\n=== Saved Visualisations ===")
        for path in saved_visuals:
            print(f"- {path}")
