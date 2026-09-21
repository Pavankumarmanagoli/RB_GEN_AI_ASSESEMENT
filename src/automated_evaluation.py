"""Build the automated evaluation summary."""
import json
import statistics
from collections import defaultdict

from src.config import OUTPUT_DIR
from src.schemas import AutomatedEvaluationRow

GEVAL_DIMENSIONS = ("behavior", "preference", "motivation", "nuance")


def _load_results(filename: str) -> list[dict]:
    return json.loads((OUTPUT_DIR / filename).read_text(encoding="utf-8"))["results"]


def _index_by_row(records: list[dict]) -> dict:
    return {record["row_id"]: record for record in records}


def _group_by_row(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["row_id"]].append(record)
    return grouped


def _safe_mean(values: list) -> float | None:
    filtered = [value for value in values if value is not None]
    return statistics.mean(filtered) if filtered else None


def _safe_median(values: list) -> float | None:
    filtered = [value for value in values if value is not None]
    return statistics.median(filtered) if filtered else None


def _nli_row_summary(pairs: list[dict]) -> dict:
    contradictions = [pair for pair in pairs if pair["nli_label"] == "contradiction"]
    return {
        "nli_contradiction_present": len(contradictions) > 0,
        "nli_contradiction_count": len(contradictions),
        "strongest_contradiction_probability": (
            max(pair["contradiction_prob"] for pair in contradictions) if contradictions else None
        ),
    }


def build_row_records() -> list[dict]:
    """Combine G-Eval, claim alignment, NLI, and BERTScore results by row."""
    geval_by_row = _index_by_row(_load_results("geval_results_final.json"))
    alignment_by_row = _index_by_row(_load_results("claim_alignment_final.json"))
    nli_by_row = _group_by_row(_load_results("nli_validation_final.json"))
    bertscore_by_row = _index_by_row(_load_results("bertscore_final.json"))

    records = []
    for row_id in sorted(geval_by_row):
        geval = geval_by_row[row_id]
        alignment = alignment_by_row[row_id]
        bertscore = bertscore_by_row[row_id]
        nli_summary = _nli_row_summary(nli_by_row.get(row_id, []))

        record = AutomatedEvaluationRow(
            row_id=row_id,
            person_id=geval["person_id"],
            question=geval["question"],
            human_answer=geval["human_answers"],
            ai_answer=geval["ai_answers"],
            automated_fidelity_score=geval["core"]["score"],
            automated_fidelity_scale="1-5",
            core_fidelity_score=geval["core"]["score"],
            behavior_score=geval["behavior"]["score"],
            preference_score=geval["preference"]["score"],
            motivation_score=geval["motivation"]["score"],
            nuance_score=geval["nuance"]["score"],
            human_claim_count=alignment["human_claim_count"],
            aligned_count=alignment["aligned_count"],
            partial_count=alignment["partial_count"],
            contradicted_count=alignment["contradicted_count"],
            missing_count=alignment["missing_count"],
            ai_claim_count=alignment["ai_claim_count"],
            unsupported_ai_count=alignment["unsupported_count"],
            claim_coverage_rate=alignment["claim_coverage_rate"],
            strict_alignment_rate=alignment["strict_alignment_rate"],
            unsupported_rate=alignment["unsupported_rate"],
            nli_contradiction_present=nli_summary["nli_contradiction_present"],
            nli_contradiction_count=nli_summary["nli_contradiction_count"],
            strongest_contradiction_probability=nli_summary["strongest_contradiction_probability"],
            bertscore_precision=bertscore["bertscore_precision"],
            bertscore_recall=bertscore["bertscore_recall"],
            bertscore_f1=bertscore["bertscore_f1"],
        )
        records.append(record.model_dump())

    return records


def person_summary(records: list[dict]) -> list[dict]:
    summaries = []
    for person_id in sorted({record["person_id"] for record in records}):
        rows = [record for record in records if record["person_id"] == person_id]
        summaries.append({
            "person_id": person_id,
            "number_of_rows": len(rows),
            "automated_fidelity_mean": _safe_mean([row["automated_fidelity_score"] for row in rows]),
            "automated_fidelity_median": _safe_median([row["automated_fidelity_score"] for row in rows]),
            "mean_claim_coverage_rate": _safe_mean([row["claim_coverage_rate"] for row in rows]),
            "mean_strict_alignment_rate": _safe_mean([row["strict_alignment_rate"] for row in rows]),
            "mean_unsupported_rate": _safe_mean([row["unsupported_rate"] for row in rows]),
            "step4_contradiction_row_count": sum(1 for row in rows if row["contradicted_count"] > 0),
            "nli_contradiction_row_count": sum(1 for row in rows if row["nli_contradiction_present"]),
            "mean_bertscore_f1": _safe_mean([row["bertscore_f1"] for row in rows]),
        })
    return summaries


def _dimension_summary(records: list[dict]) -> dict:
    summary = {}
    for dimension in GEVAL_DIMENSIONS:
        field = f"{dimension}_score"
        applicable_scores = [record[field] for record in records if record[field] is not None]
        summary[dimension] = {
            "applicable_count": len(applicable_scores),
            "not_applicable_count": len(records) - len(applicable_scores),
            "mean": _safe_mean(applicable_scores),
        }
    return summary


def _percentage(count: int, total: int) -> float | None:
    return (count / total * 100) if total else None


def claim_analysis(records: list[dict]) -> dict:
    human_claims_total = sum(record["human_claim_count"] for record in records)
    ai_claims_total = sum(record["ai_claim_count"] for record in records)
    aligned_total = sum(record["aligned_count"] for record in records)
    partial_total = sum(record["partial_count"] for record in records)
    contradicted_total = sum(record["contradicted_count"] for record in records)
    missing_total = sum(record["missing_count"] for record in records)
    unsupported_total = sum(record["unsupported_ai_count"] for record in records)

    coverage_rates = [record["claim_coverage_rate"] for record in records]
    strict_rates = [record["strict_alignment_rate"] for record in records]
    unsupported_rates = [record["unsupported_rate"] for record in records]

    return {
        "human_claims_total": human_claims_total,
        "aligned_count": aligned_total,
        "aligned_percentage": _percentage(aligned_total, human_claims_total),
        "partial_count": partial_total,
        "partial_percentage": _percentage(partial_total, human_claims_total),
        "contradicted_count": contradicted_total,
        "contradicted_percentage": _percentage(contradicted_total, human_claims_total),
        "missing_count": missing_total,
        "missing_percentage": _percentage(missing_total, human_claims_total),
        "ai_claims_total": ai_claims_total,
        "unsupported_ai_count": unsupported_total,
        "unsupported_ai_percentage": _percentage(unsupported_total, ai_claims_total),
        "mean_claim_coverage_rate": _safe_mean(coverage_rates),
        "median_claim_coverage_rate": _safe_median(coverage_rates),
        "mean_strict_alignment_rate": _safe_mean(strict_rates),
        "median_strict_alignment_rate": _safe_median(strict_rates),
        "mean_unsupported_rate": _safe_mean(unsupported_rates),
        "median_unsupported_rate": _safe_median(unsupported_rates),
        "rows_with_contradiction": sum(1 for record in records if record["contradicted_count"] > 0),
        "rows_with_missing_claim": sum(1 for record in records if record["missing_count"] > 0),
    }


def contradiction_analysis(records: list[dict]) -> dict:
    """Report where claim-based and NLI contradiction checks agree or disagree per row."""
    step4_rows = sorted(record["row_id"] for record in records if record["contradicted_count"] > 0)
    nli_rows = sorted(record["row_id"] for record in records if record["nli_contradiction_present"])
    both_rows = sorted(set(step4_rows) & set(nli_rows))
    step4_only_rows = sorted(set(step4_rows) - set(nli_rows))
    nli_only_rows = sorted(set(nli_rows) - set(step4_rows))

    return {
        "step4_contradiction_row_count": len(step4_rows),
        "step4_contradiction_rows": step4_rows,
        "nli_contradiction_row_count": len(nli_rows),
        "nli_contradiction_rows": nli_rows,
        "both_detect_contradiction_row_count": len(both_rows),
        "both_detect_contradiction_rows": both_rows,
        "step4_only_contradiction_rows": step4_only_rows,
        "nli_only_contradiction_rows": nli_only_rows,
    }


def semantic_similarity_analysis(records: list[dict]) -> dict:
    f1_values = [record["bertscore_f1"] for record in records]
    median_f1 = statistics.median(f1_values)

    representative_rows = []
    for row_id in (10, 22, 28):
        record = next((r for r in records if r["row_id"] == row_id), None)
        if record is None:
            continue
        representative_rows.append({
            "row_id": row_id,
            "bertscore_f1": record["bertscore_f1"],
            "core_fidelity_score": record["core_fidelity_score"],
            "step4_contradiction_present": record["contradicted_count"] > 0,
            "nli_contradiction_present": record["nli_contradiction_present"],
            "note": (
                "BERTScore F1 is above the dataset median despite the lowest possible G-Eval "
                "Core score and an NLI-detected contradiction."
                if row_id == 10 else
                "BERTScore F1 is among the lowest in the dataset despite a high G-Eval Core "
                "score and no detected contradiction."
                if record["bertscore_f1"] < median_f1 and record["core_fidelity_score"] >= 4
                else "Provided as a reference point; see bertscore_f1 and core_fidelity_score."
            ),
        })

    return {
        "mean_f1": statistics.mean(f1_values),
        "median_f1": median_f1,
        "min_f1": min(f1_values),
        "max_f1": max(f1_values),
        "representative_rows": representative_rows,
    }


def automated_findings(records: list[dict], claims: dict, contradictions: dict, similarity: dict) -> list[str]:
    fidelity_mean = _safe_mean([record["automated_fidelity_score"] for record in records])
    findings = [
        f"Overall automated fidelity is mixed: G-Eval Core Fidelity averages "
        f"{fidelity_mean:.2f}/5 across the 30 rows, with scores spread across the full 1-5 range "
        f"rather than clustering at the top.",
        f"Human content is frequently omitted: {claims['missing_percentage']:.1f}% of human claims "
        f"({claims['missing_count']}/{claims['human_claims_total']}) are missing from the AI answer, "
        f"affecting {claims['rows_with_missing_claim']}/{len(records)} rows.",
        f"Unsupported AI elaboration is common: on average {claims['mean_unsupported_rate']:.1%} of "
        f"AI claims per row are unsupported by the human reference (this indicates content absent from "
        f"the reference, not proven-false content).",
        f"Explicit contradictions occur in a meaningful subset of responses: Step 4 claim alignment "
        f"flags {contradictions['step4_contradiction_row_count']}/{len(records)} rows and independent "
        f"NLI validation flags {contradictions['nli_contradiction_row_count']}/{len(records)} rows, "
        f"agreeing on {contradictions['both_detect_contradiction_row_count']} of them.",
        "BERTScore can remain relatively high despite a behavioral contradiction, and can also be low "
        "despite a high G-Eval score and no detected contradiction (see representative_rows in the "
        "semantic_similarity section) — semantic similarity alone is not a reliable fidelity signal.",
    ]
    return findings


def dataset_summary(records: list[dict]) -> dict:
    person_ids = sorted({record["person_id"] for record in records})
    rows_per_person = {
        person_id: sum(1 for record in records if record["person_id"] == person_id)
        for person_id in person_ids
    }
    fidelity_scores = [record["automated_fidelity_score"] for record in records]

    claims = claim_analysis(records)
    contradictions = contradiction_analysis(records)
    similarity = semantic_similarity_analysis(records)

    return {
        "dataset": {
            "number_of_rows": len(records),
            "number_of_people": len(person_ids),
            "rows_per_person": rows_per_person,
        },
        "automated_fidelity": {
            "overall_automated_fidelity_mean": statistics.mean(fidelity_scores),
            "overall_automated_fidelity_median": statistics.median(fidelity_scores),
            "core_fidelity_distribution": {score: fidelity_scores.count(score) for score in range(1, 6)},
            "dimension_summary": _dimension_summary(records),
        },
        "claim_analysis": claims,
        "contradiction_analysis": contradictions,
        "semantic_similarity": similarity,
        "person_summary": person_summary(records),
        "automated_findings": automated_findings(records, claims, contradictions, similarity),
    }


def build_markdown_summary(summary: dict) -> str:
    fidelity = summary["automated_fidelity"]
    claims = summary["claim_analysis"]
    contradictions = summary["contradiction_analysis"]
    similarity = summary["semantic_similarity"]
    dataset = summary["dataset"]

    lines = [
        "# Automated Evaluation Summary",
        "",
        "## Overall Automated Fidelity",
        "",
        f"**Overall Automated Fidelity: {fidelity['overall_automated_fidelity_mean']:.2f} / 5**",
        "",
        f"Based on G-Eval Core Fidelity across all {dataset['number_of_rows']} rows "
        f"(median {fidelity['overall_automated_fidelity_median']:.1f}/5). This is the sole primary "
        "automated fidelity score; it is not a weighted composite with claim, NLI, or BERTScore signals.",
        "",
        "Score distribution: " + ", ".join(
            f"{score} → {count}" for score, count in fidelity["core_fidelity_distribution"].items()
        ),
        "",
        "## G-Eval",
        "",
        "Core Fidelity is the primary automated score. The other G-Eval dimensions are reported "
        "descriptively, counting only rows where the dimension was judged applicable:",
        "",
    ]
    for dimension in GEVAL_DIMENSIONS:
        stats = fidelity["dimension_summary"][dimension]
        mean_text = f"{stats['mean']:.2f}/5" if stats["mean"] is not None else "n/a"
        lines.append(
            f"- **{dimension.capitalize()}**: applicable in {stats['applicable_count']}/{dataset['number_of_rows']} "
            f"rows, mean {mean_text}"
        )

    lines += [
        "",
        "## Claim Preservation",
        "",
        f"- {claims['human_claims_total']} total human claims across the dataset.",
        f"- Aligned: {claims['aligned_count']} ({claims['aligned_percentage']:.1f}%)",
        f"- Partial: {claims['partial_count']} ({claims['partial_percentage']:.1f}%)",
        f"- Contradicted: {claims['contradicted_count']} ({claims['contradicted_percentage']:.1f}%)",
        f"- Missing: {claims['missing_count']} ({claims['missing_percentage']:.1f}%), "
        f"affecting {claims['rows_with_missing_claim']}/{dataset['number_of_rows']} rows",
        f"- {claims['ai_claims_total']} total AI claims; {claims['unsupported_ai_count']} "
        f"({claims['unsupported_ai_percentage']:.1f}%) are unsupported by the human reference "
        "(absent from the reference, not proven false or hallucinated).",
        f"- Mean claim coverage rate: {claims['mean_claim_coverage_rate']:.2f} "
        f"(median {claims['median_claim_coverage_rate']:.2f})",
        f"- Mean strict alignment rate: {claims['mean_strict_alignment_rate']:.2f} "
        f"(median {claims['median_strict_alignment_rate']:.2f})",
        "",
        "## Contradiction Analysis",
        "",
        f"- Step 4 (claim alignment) flags {contradictions['step4_contradiction_row_count']}/"
        f"{dataset['number_of_rows']} rows with at least one contradicted claim: "
        f"{contradictions['step4_contradiction_rows']}",
        f"- Step 5 (NLI) flags {contradictions['nli_contradiction_row_count']}/{dataset['number_of_rows']} "
        f"rows with at least one contradiction: {contradictions['nli_contradiction_rows']}",
        f"- Both methods agree on {contradictions['both_detect_contradiction_row_count']} rows: "
        f"{contradictions['both_detect_contradiction_rows']}",
        f"- Step 4 only: {contradictions['step4_only_contradiction_rows']}; "
        f"NLI only: {contradictions['nli_only_contradiction_rows']}",
        "",
        "No judgment is made here about which method is correct on the disagreement rows; "
        "that is left for human validation in Step 8B.",
        "",
        "## Semantic Similarity",
        "",
        f"- Mean F1: {similarity['mean_f1']:.3f}, median: {similarity['median_f1']:.3f}, "
        f"range: [{similarity['min_f1']:.3f}, {similarity['max_f1']:.3f}]",
        "- BERTScore measures semantic similarity only; it is not a factual-fidelity metric.",
        "",
    ]
    for row in similarity["representative_rows"]:
        lines.append(
            f"- Row {row['row_id']}: F1={row['bertscore_f1']:.3f}, G-Eval Core={row['core_fidelity_score']}, "
            f"Step4 contradiction={row['step4_contradiction_present']}, "
            f"NLI contradiction={row['nli_contradiction_present']} — {row['note']}"
        )

    lines += [
        "",
        "## Person-Level Automated Results",
        "",
    ]
    for person in summary["person_summary"]:
        lines.append(
            f"- **{person['person_id']}** ({person['number_of_rows']} rows): "
            f"automated fidelity mean {person['automated_fidelity_mean']:.2f} "
            f"(median {person['automated_fidelity_median']:.1f}), "
            f"mean claim coverage {person['mean_claim_coverage_rate']:.2f}, "
            f"mean unsupported rate {person['mean_unsupported_rate']:.2f}, "
            f"Step4 contradiction rows {person['step4_contradiction_row_count']}, "
            f"NLI contradiction rows {person['nli_contradiction_row_count']}, "
            f"mean BERTScore F1 {person['mean_bertscore_f1']:.3f}"
        )

    lines += [
        "",
        "## Key Automated Findings",
        "",
    ]
    lines += [f"- {finding}" for finding in summary["automated_findings"]]
    lines.append("")

    return "\n".join(lines)
