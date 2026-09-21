"""Run BERTScore evaluation over the dataset and export results."""
import pandas as pd

from src.bertscore_evaluator import (
    IDF,
    LANGUAGE,
    MODEL_NAME,
    RESCALE_WITH_BASELINE,
    BERTScoreError,
    find_length_overflows,
    load_scorer,
    score_batch,
)
from src.config import OUTPUT_DIR
from src.inspect_data import load_data
from src.run_geval import write_json
from src.schemas import BERTScoreResult


def _load_rows() -> list[dict]:
    data = load_data()
    return [
        {
            "row_id": row["id"],
            "person_id": str(row["person_id"]),
            "question": row["question"],
            "human_answer": row["human_answers"],
            "ai_answer": row["ai_answers"],
        }
        for _, row in data.iterrows()
    ]


def run_bertscore() -> int:
    """Score every row with BERTScore and write the frozen results."""
    rows = _load_rows()
    missing = [
        row["row_id"] for row in rows
        if not row["human_answer"].strip() or not row["ai_answer"].strip()
    ]
    if missing:
        raise BERTScoreError(f"Row IDs with empty/missing answer text cannot be scored: {missing}")

    overflows = find_length_overflows(rows)
    if overflows:
        print("Token-length pre-check found answers exceeding the model's max sequence length:")
        for overflow in overflows:
            print(f"  row {overflow['row_id']} ({overflow['side']}): "
                  f"{overflow['token_count']} tokens > {overflow['max_length']}")
    else:
        print(f"Token-length pre-check: all {len(rows) * 2} answers fit within the model's max sequence length.")

    scorer = load_scorer()
    precision, recall, f1 = score_batch(
        scorer, [row["human_answer"] for row in rows], [row["ai_answer"] for row in rows],
    )

    records = []
    for row, p, r, f in zip(rows, precision, recall, f1):
        result = BERTScoreResult(
            row_id=row["row_id"], person_id=row["person_id"], question=row["question"],
            human_answer=row["human_answer"], ai_answer=row["ai_answer"],
            bertscore_precision=p, bertscore_recall=r, bertscore_f1=f,
        )
        records.append(result.model_dump())

    metadata = {
        "model": MODEL_NAME,
        "language": LANGUAGE,
        "reference": "human_answer",
        "candidate": "ai_answer",
        "rescale_with_baseline": RESCALE_WITH_BASELINE,
        "idf": IDF,
        "bertscore_hash": scorer.hash,
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    write_json(OUTPUT_DIR / "bertscore_final.json", {**metadata, "results": records})
    pd.json_normalize(records, sep="_").to_csv(OUTPUT_DIR / "bertscore_final.csv", index=False)
    print(f"BERTScore model: {MODEL_NAME} ({metadata['bertscore_hash']})")
    print(f"Evaluated {len(records)} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_bertscore())
