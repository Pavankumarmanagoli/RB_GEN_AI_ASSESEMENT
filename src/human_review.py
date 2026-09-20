import json
import re
import statistics
from collections import defaultdict

import pandas as pd
from scipy.stats import spearmanr

from src.config import OUTPUT_DIR, PROJECT_ROOT
from src.inspect_data import load_data
from src.schemas import HumanReviewRecord

REVIEW_PATH = PROJECT_ROOT / "data" / "human_fidelity_review.xlsx"
REQUIRED_COLUMNS = [
    "review_id", "question", "human_answers", "ai_answers",
    "fidelity_score", "contradiction", "omission", "unsupported_detail", "note",
]
YES_NO = {"YES", "NO"}


class HumanReviewError(ValueError):
    pass


def _normalize_text(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def load_review():
    if not REVIEW_PATH.exists():
        raise HumanReviewError(f"Human review file not found: {REVIEW_PATH}")
    review = pd.read_excel(REVIEW_PATH, sheet_name=0, keep_default_na=False)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in review.columns]
    if missing_columns:
        raise HumanReviewError(f"Human review file is missing required columns: {missing_columns}")
    # Ignore completely blank formatted rows (all required cells empty).
    is_blank = review[REQUIRED_COLUMNS].apply(
        lambda row: all(str(value).strip() == "" for value in row), axis=1,
    )
    return review.loc[~is_blank].reset_index(drop=True)


def validate_review(review) -> None:
    if len(review) != 30:
        raise HumanReviewError(f"Expected exactly 30 completed human review rows, found {len(review)}.")

    duplicate_ids = review["review_id"][review["review_id"].duplicated()].tolist()
    if duplicate_ids:
        raise HumanReviewError(f"Duplicate review_id values: {duplicate_ids}")

    def _is_valid_score(value) -> bool:
        try:
            return float(value).is_integer() and 1 <= int(value) <= 5
        except (TypeError, ValueError):
            return False

    bad_scores = review.loc[~review["fidelity_score"].apply(_is_valid_score), "review_id"].tolist()
    if bad_scores:
        raise HumanReviewError(f"fidelity_score must be an integer from 1 to 5 for review rows: {bad_scores}")

    for column in ("contradiction", "omission", "unsupported_detail"):
        normalized = review[column].astype(str).str.strip().str.upper()
        bad_values = review.loc[~normalized.isin(YES_NO), "review_id"].tolist()
        if bad_values:
            raise HumanReviewError(f"{column} must be YES or NO for review rows: {bad_values}")

    missing_text = review.loc[
        (review["human_answers"].astype(str).str.strip() == "")
        | (review["ai_answers"].astype(str).str.strip() == ""),
        "review_id",
    ].tolist()
    if missing_text:
        raise HumanReviewError(f"Missing Human or AI answer text for review rows: {missing_text}")


def map_reviews_to_dataset(review) -> list[dict]:
    """Recovers each blinded review row's original row_id/person_id by exact, whitespace-
    normalized matching of (question, human answer, AI answer) against the project's
    dataset loader. Fails loudly on zero matches, ambiguous matches, or incomplete
    coverage of the original 30 rows; never assumes review_id order."""
    dataset = load_data()
    candidates_by_key: dict[tuple, list[dict]] = defaultdict(list)
    for _, row in dataset.iterrows():
        key = (
            _normalize_text(row["question"]),
            _normalize_text(row["human_answers"]),
            _normalize_text(row["ai_answers"]),
        )
        candidates_by_key[key].append({
            "row_id": row["id"], "person_id": str(row["person_id"]),
            "question": row["question"], "human_answer": row["human_answers"], "ai_answer": row["ai_answers"],
        })

    mapped = []
    used_row_ids: set = set()
    unmatched, ambiguous, duplicate_mapping = [], [], []
    for _, review_row in review.iterrows():
        key = (
            _normalize_text(review_row["question"]),
            _normalize_text(review_row["human_answers"]),
            _normalize_text(review_row["ai_answers"]),
        )
        candidates = candidates_by_key.get(key, [])
        if len(candidates) == 0:
            unmatched.append(review_row["review_id"])
            continue
        if len(candidates) > 1:
            ambiguous.append((review_row["review_id"], [c["row_id"] for c in candidates]))
            continue
        match = candidates[0]
        if match["row_id"] in used_row_ids:
            duplicate_mapping.append((review_row["review_id"], match["row_id"]))
            continue
        used_row_ids.add(match["row_id"])
        mapped.append({
            "review_id": review_row["review_id"],
            "row_id": match["row_id"],
            "person_id": match["person_id"],
            "question": match["question"],
            "human_answer": match["human_answer"],
            "ai_answer": match["ai_answer"],
            "fidelity_score": int(review_row["fidelity_score"]),
            "contradiction": str(review_row["contradiction"]).strip().upper(),
            "omission": str(review_row["omission"]).strip().upper(),
            "unsupported_detail": str(review_row["unsupported_detail"]).strip().upper(),
            "note": review_row["note"],
        })

    if unmatched:
        raise HumanReviewError(f"No original dataset row found for review rows: {unmatched}")
    if ambiguous:
        raise HumanReviewError(f"Ambiguous mapping (multiple original rows match) for review rows: {ambiguous}")
    if duplicate_mapping:
        raise HumanReviewError(f"Duplicate mapping to an already-used original row: {duplicate_mapping}")

    all_dataset_ids = set(dataset["id"])
    missing_rows = all_dataset_ids - used_row_ids
    if missing_rows or len(used_row_ids) != len(all_dataset_ids):
        raise HumanReviewError(f"Not all original dataset rows were recovered; missing row_ids: {sorted(missing_rows)}")

    return mapped


def _load_by_row_id(filename: str) -> dict:
    data = json.loads((OUTPUT_DIR / filename).read_text(encoding="utf-8"))["results"]
    by_row = defaultdict(list)
    for record in data:
        by_row[record["row_id"]].append(record)
    return by_row


def build_joined_records(mapped: list[dict]) -> list[dict]:
    """Joins each mapped human review row with the frozen Step 3-6 outputs by row_id,
    without modifying or recomputing any of those frozen results."""
    geval_by_row = _load_by_row_id("geval_results_final.json")
    alignment_by_row = _load_by_row_id("claim_alignment_final.json")
    nli_by_row = _load_by_row_id("nli_validation_final.json")
    bertscore_by_row = _load_by_row_id("bertscore_final.json")

    records = []
    for entry in mapped:
        row_id = entry["row_id"]
        [geval] = geval_by_row[row_id]
        [alignment] = alignment_by_row[row_id]
        [bertscore] = bertscore_by_row[row_id]
        nli_pairs = nli_by_row.get(row_id, [])
        nli_contradictions = [pair for pair in nli_pairs if pair["nli_label"] == "contradiction"]

        record = HumanReviewRecord(
            review_id=entry["review_id"],
            row_id=row_id,
            person_id=entry["person_id"],
            question=entry["question"],
            human_answer=entry["human_answer"],
            ai_answer=entry["ai_answer"],
            human_fidelity_score=entry["fidelity_score"],
            human_contradiction=entry["contradiction"],
            human_omission=entry["omission"],
            human_unsupported_detail=entry["unsupported_detail"],
            reviewer_note=entry["note"],
            geval_core_fidelity=geval["core"]["score"],
            claim_coverage_rate=alignment["claim_coverage_rate"],
            strict_alignment_rate=alignment["strict_alignment_rate"],
            step4_contradiction_present=alignment["contradicted_count"] > 0,
            step4_missing_count=alignment["missing_count"],
            step4_unsupported_rate=alignment["unsupported_rate"],
            nli_contradiction_present=len(nli_contradictions) > 0,
            nli_contradiction_count=len(nli_contradictions),
            bertscore_precision=bertscore["bertscore_precision"],
            bertscore_recall=bertscore["bertscore_recall"],
            bertscore_f1=bertscore["bertscore_f1"],
        )
        records.append(record.model_dump())

    records.sort(key=lambda record: record["row_id"])
    return records


def fidelity_summary(records: list[dict]) -> dict:
    scores = [record["human_fidelity_score"] for record in records]
    return {
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
        "distribution": {value: scores.count(value) for value in range(1, 6)},
        "n": len(scores),
    }


def flag_rate(records: list[dict], field: str) -> dict:
    values = [record[field] for record in records]
    yes_count = sum(1 for value in values if value == "YES")
    return {
        "yes_count": yes_count,
        "no_count": len(values) - yes_count,
        "yes_percentage": (yes_count / len(values) * 100) if values else None,
        "n": len(values),
    }


def spearman(records: list[dict], x_field: str, y_field: str) -> dict:
    """Spearman correlation over rows where both fields are non-null. Reported cautiously:
    rho, p-value, and n are returned as-is with no strength label, given the small (<=30
    row) sample size."""
    pairs = [
        (record[x_field], record[y_field]) for record in records
        if record[x_field] is not None and record[y_field] is not None
    ]
    if len(pairs) < 2:
        return {"rho": None, "p_value": None, "n": len(pairs)}
    xs, ys = zip(*pairs)
    rho, p_value = spearmanr(xs, ys)
    return {"rho": float(rho), "p_value": float(p_value), "n": len(pairs)}


def contradiction_confusion(records: list[dict], predicted_field: str) -> dict:
    """Confusion matrix for an automated contradiction flag against the human
    `contradiction` label, treated here as the reference."""
    true_positives = true_negatives = false_positives = false_negatives = 0
    for record in records:
        human = record["human_contradiction"] == "YES"
        predicted = bool(record[predicted_field])
        if human and predicted:
            true_positives += 1
        elif not human and not predicted:
            true_negatives += 1
        elif not human and predicted:
            false_positives += 1
        else:
            false_negatives += 1

    n = len(records)
    agreement_count = true_positives + true_negatives
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else None
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "agreement_count": agreement_count,
        "agreement_percentage": (agreement_count / n * 100) if n else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def disagreement_rows(records: list[dict], predicted_field: str) -> list[dict]:
    return [
        {
            "row_id": record["row_id"],
            "review_id": record["review_id"],
            "human_contradiction": record["human_contradiction"],
            predicted_field: record[predicted_field],
            "reviewer_note": record["reviewer_note"],
        }
        for record in records
        if (record["human_contradiction"] == "YES") != bool(record[predicted_field])
    ]


def omission_comparison(records: list[dict]) -> dict:
    yes_missing = [record["step4_missing_count"] for record in records if record["human_omission"] == "YES"]
    no_missing = [record["step4_missing_count"] for record in records if record["human_omission"] == "NO"]
    return {
        "mean_step4_missing_count_when_human_omission_yes": statistics.mean(yes_missing) if yes_missing else None,
        "mean_step4_missing_count_when_human_omission_no": statistics.mean(no_missing) if no_missing else None,
        "human_omission_yes_but_step4_missing_zero_rows": [
            record["row_id"] for record in records
            if record["human_omission"] == "YES" and record["step4_missing_count"] == 0
        ],
        "human_omission_no_but_step4_missing_nonzero_rows": [
            record["row_id"] for record in records
            if record["human_omission"] == "NO" and record["step4_missing_count"] > 0
        ],
    }


def unsupported_detail_comparison(records: list[dict]) -> dict:
    yes_rates = [
        record["step4_unsupported_rate"] for record in records
        if record["human_unsupported_detail"] == "YES" and record["step4_unsupported_rate"] is not None
    ]
    no_rates = [
        record["step4_unsupported_rate"] for record in records
        if record["human_unsupported_detail"] == "NO" and record["step4_unsupported_rate"] is not None
    ]
    return {
        "all_rows_flagged_yes": all(record["human_unsupported_detail"] == "YES" for record in records),
        "mean_step4_unsupported_rate_when_human_yes": statistics.mean(yes_rates) if yes_rates else None,
        "mean_step4_unsupported_rate_when_human_no": statistics.mean(no_rates) if no_rates else None,
        "human_no_group_n": len(no_rates),
    }
