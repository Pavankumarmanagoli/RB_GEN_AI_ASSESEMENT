"""Run claim alignment via the OpenAI API."""
import argparse
import json

import pandas as pd
from openai import APIConnectionError, APIError, APIStatusError, OpenAI
from pydantic import ValidationError

from src.claim_aligner import (
    ALIGNMENT_PROMPT,
    ALIGNMENT_PROMPT_VERSION,
    AlignmentError,
    align_claims,
    cache_key,
    validate_alignment,
)
from src.claim_extractor import CLAIM_PROMPT_VERSION
from src.config import OUTPUT_DIR, load_config
from src.run_geval import write_json
from src.schemas import AlignmentJudgment, ClaimAlignmentResult


def describe_failure(error: Exception) -> dict[str, str]:
    # SDK error bodies can contain credentials; never log or persist raw exceptions.
    if isinstance(error, AlignmentError):
        message = str(error)
    elif isinstance(error, APIStatusError):
        message = f"API request failed with HTTP {error.status_code}; check credentials, model access, or quota."
    elif isinstance(error, APIConnectionError):
        message = "Could not reach the OpenAI API; check network access."
    elif isinstance(error, ValidationError):
        message = "The alignment output failed schema validation."
    elif isinstance(error, APIError):
        message = "The API response could not be processed as a structured alignment."
    else:
        message = "Invalid row input or incomplete response."
    return {"error_type": type(error).__name__, "error_message": message}


def _load_extracted_claims() -> dict[int, dict]:
    """Load completed claim extraction records, keyed by row_id."""
    claims_by_row = {}
    # Files are read in sorted order, so a later-sorted file wins on row_id overlap.
    for path in sorted(OUTPUT_DIR.glob(f"claim_extraction_results_{CLAIM_PROMPT_VERSION}_*.json")):
        for record in json.loads(path.read_text(encoding="utf-8"))["results"]:
            if record.get("status") == "completed":
                claims_by_row[record["row_id"]] = record
    return claims_by_row


def _row_metrics(judgment: AlignmentJudgment, human_count: int, ai_count: int) -> dict:
    aligned = sum(1 for a in judgment.alignments if a.label == "aligned")
    partial = sum(1 for a in judgment.alignments if a.label == "partial")
    contradicted = sum(1 for a in judgment.alignments if a.label == "contradicted")
    missing = sum(1 for a in judgment.alignments if a.label == "missing")
    unsupported = len(judgment.unsupported_ai_claims)
    return {
        "human_claim_count": human_count,
        "ai_claim_count": ai_count,
        "aligned_count": aligned,
        "partial_count": partial,
        "contradicted_count": contradicted,
        "missing_count": missing,
        "unsupported_count": unsupported,
        "claim_coverage_rate": (aligned + partial) / human_count if human_count else None,
        "strict_alignment_rate": aligned / human_count if human_count else None,
        "unsupported_rate": unsupported / ai_count if ai_count else None,
    }


def _align_rows(client: OpenAI, model: str, prompt: str, row_ids: list[int],
                 claims_by_row: dict[int, dict], cache: dict, cache_path, result_stem: str) -> list[dict]:
    records = []
    for row_id in row_ids:
        extracted = claims_by_row[row_id]
        human_claims = extracted["human_claims"]
        ai_claims = extracted["ai_claims"]
        key = cache_key(human_claims, ai_claims, model, prompt)
        record = {
            "row_id": row_id,
            "person_id": extracted["person_id"],
            "question": extracted["question"],
            "aligner_model": model,
            "prompt_version": ALIGNMENT_PROMPT_VERSION,
            "cache_key": key,
        }
        try:
            cached = key in cache
            human_ids = {claim["claim_id"] for claim in human_claims}
            ai_ids = {claim["claim_id"] for claim in ai_claims}
            judgment = (
                validate_alignment(AlignmentJudgment.model_validate(cache[key]), human_ids, ai_ids) if cached
                else align_claims(client, model, prompt, human_claims, ai_claims)
            )
            result = ClaimAlignmentResult(row_id=row_id, **judgment.model_dump())
            if not cached:
                cache[key] = judgment.model_dump()
                write_json(cache_path, cache)
            record.update(result.model_dump())
            record.update(_row_metrics(judgment, len(human_claims), len(ai_claims)))
            record.update(status="completed", cached=cached)
            print(f"Row {row_id}: completed ({'cache' if cached else 'API'})", flush=True)
        except (APIError, ValidationError, ValueError) as error:
            record.update(status="failed", **describe_failure(error))
            print(f"Row {row_id}: {record['error_type']} — {record['error_message']}", flush=True)
        records.append(record)
        write_json(OUTPUT_DIR / f"{result_stem}.json", {"results": records})
        pd.json_normalize(records, sep="_").to_csv(OUTPUT_DIR / f"{result_stem}.csv", index=False)
    return records


def run_alignment(row_ids: list[int] | None) -> int:
    """Align claims for the selected rows and write checkpointed results."""
    try:
        api_key, model = load_config()
    except ValueError as error:
        print(str(error))
        return 1
    claims_by_row = _load_extracted_claims()
    selected_ids = sorted(claims_by_row) if row_ids is None else row_ids
    missing = [row_id for row_id in selected_ids if row_id not in claims_by_row]
    if missing:
        raise ValueError(
            f"No completed claim extraction found for row IDs {missing}. Run src.run_claim_extraction first."
        )
    OUTPUT_DIR.mkdir(exist_ok=True)
    cache_path = OUTPUT_DIR / ".alignment_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    print(f"Aligner model: {model}")
    suffix = "full" if row_ids is None else "rows_" + "-".join(map(str, selected_ids))
    result_stem = f"alignment_results_{ALIGNMENT_PROMPT_VERSION}_{suffix}"
    # The SDK bounds retries for transient connection, rate-limit, and server errors.
    with OpenAI(api_key=api_key, max_retries=2, timeout=60.0) as client:
        records = _align_rows(
            client, model, ALIGNMENT_PROMPT, selected_ids, claims_by_row, cache, cache_path, result_stem,
        )
    return int(any(record["status"] == "failed" for record in records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align atomic human/AI claims from extracted claim data.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--rows", type=int, nargs="+", metavar="ROW_ID",
        help="row IDs to align (must already have extracted claim results)",
    )
    selection.add_argument(
        "--full", action="store_true",
        help="align every row with completed extraction results",
    )
    args = parser.parse_args()
    try:
        raise SystemExit(run_alignment(None if args.full else args.rows))
    except ValueError:
        # Configuration messages are fixed strings, never API error bodies.
        print("Run could not start. Check OPENAI_API_KEY, EVALUATOR_MODEL, row IDs, and extracted claim results.")
        raise SystemExit(1)
