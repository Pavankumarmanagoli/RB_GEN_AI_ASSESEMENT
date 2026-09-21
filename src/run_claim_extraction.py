"""Run claim extraction via the OpenAI API."""
import argparse
import json
from pathlib import Path

import pandas as pd
from openai import APIConnectionError, APIError, APIStatusError, OpenAI
from pydantic import ValidationError

from src.claim_extractor import (
    CLAIM_PROMPT_VERSION,
    CLAIM_EXTRACTION_PROMPT,
    ExtractionError,
    cache_key,
    extract_claims,
)
from src.config import OUTPUT_DIR, load_config
from src.inspect_data import load_data
from src.run_geval import write_json
from src.schemas import AtomicClaim, ClaimExtractionResult, ClaimList


def describe_failure(error: Exception) -> dict[str, str]:
    # SDK error bodies can contain credentials; never log or persist raw exceptions.
    if isinstance(error, ExtractionError):
        message = str(error)
    elif isinstance(error, APIStatusError):
        message = f"API request failed with HTTP {error.status_code}; check credentials, model access, or quota."
    elif isinstance(error, APIConnectionError):
        message = "Could not reach the OpenAI API; check network access."
    elif isinstance(error, ValidationError):
        message = "The extraction output failed schema validation."
    elif isinstance(error, APIError):
        message = "The API response could not be processed as a structured extraction."
    else:
        message = "Invalid row input or incomplete response."
    return {"error_type": type(error).__name__, "error_message": message}


def _extract_side(
    client: OpenAI, model: str, prompt: str, question: str, text: str, cache: dict, cache_path: Path,
) -> tuple[list[str], bool]:
    key = cache_key(question, text, model, prompt)
    cached = key in cache
    claims = (
        ClaimList.model_validate(cache[key]).claims if cached
        else extract_claims(client, model, prompt, question, text).claims
    )
    if not cached:
        cache[key] = {"claims": claims}
        write_json(cache_path, cache)
    return claims, cached


def _extract_rows(
    client: OpenAI, model: str, prompt: str, data: pd.DataFrame, row_ids: list[int],
    cache: dict, cache_path: Path, result_stem: str,
) -> list[dict]:
    records = []
    for row_id in row_ids:
        row = data.loc[data["id"] == row_id].iloc[0]
        question, human_text, ai_text = row["question"], row["human_answers"], row["ai_answers"]
        record = {
            "row_id": row_id,
            "person_id": str(row["person_id"]),
            "question": question,
            "human_answers": human_text,
            "ai_answers": ai_text,
            "extractor_model": model,
            "prompt_version": CLAIM_PROMPT_VERSION,
        }
        try:
            human_claims, human_cached = _extract_side(client, model, prompt, question, human_text, cache, cache_path)
            ai_claims, ai_cached = _extract_side(client, model, prompt, question, ai_text, cache, cache_path)
            result = ClaimExtractionResult(
                row_id=row_id,
                human_claims=[AtomicClaim(claim_id=f"h{i}", claim=c) for i, c in enumerate(human_claims, start=1)],
                ai_claims=[AtomicClaim(claim_id=f"a{i}", claim=c) for i, c in enumerate(ai_claims, start=1)],
            )
            record.update(result.model_dump())
            record.update(status="completed", human_cached=human_cached, ai_cached=ai_cached)
            print(f"Row {row_id}: completed (human {'cache' if human_cached else 'API'}, "
                  f"ai {'cache' if ai_cached else 'API'})", flush=True)
        except (APIError, ValidationError, ValueError) as error:
            record.update(status="failed", **describe_failure(error))
            print(f"Row {row_id}: {record['error_type']} — {record['error_message']}", flush=True)
        records.append(record)
        write_json(OUTPUT_DIR / f"{result_stem}.json", {"results": records})
        pd.json_normalize(records, sep="_").to_csv(OUTPUT_DIR / f"{result_stem}.csv", index=False)
    return records


def run_extraction(row_ids: list[int] | None) -> int:
    """Extract claims for the selected rows and write checkpointed results."""
    try:
        api_key, model = load_config()
    except ValueError as error:
        print(str(error))
        return 1
    data = load_data()
    selected_ids = list(data["id"]) if row_ids is None else row_ids
    selected = data[data["id"].isin(selected_ids)]
    if len(selected) != len(selected_ids) or selected["id"].nunique() != len(selected_ids):
        raise ValueError("Each requested row ID must exist exactly once in the dataset.")
    OUTPUT_DIR.mkdir(exist_ok=True)
    cache_path = OUTPUT_DIR / ".claims_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    print(f"Extractor model: {model}")
    suffix = "full" if row_ids is None else "rows_" + "-".join(map(str, selected_ids))
    result_stem = f"claim_extraction_results_{CLAIM_PROMPT_VERSION}_{suffix}"
    # The SDK bounds retries for transient connection, rate-limit, and server errors.
    with OpenAI(api_key=api_key, max_retries=2, timeout=60.0) as client:
        records = _extract_rows(
            client, model, CLAIM_EXTRACTION_PROMPT, selected, selected_ids, cache, cache_path, result_stem,
        )
    return int(any(record["status"] == "failed" for record in records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract atomic claims from human/AI response pairs.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--rows", type=int, nargs="+", metavar="ROW_ID",
        help="row IDs to extract claims for",
    )
    selection.add_argument(
        "--full", action="store_true",
        help="extract claims for every row in the dataset",
    )
    args = parser.parse_args()
    try:
        raise SystemExit(run_extraction(None if args.full else args.rows))
    except ValueError:
        # Configuration messages are fixed strings, never API error bodies.
        print("Run could not start. Check OPENAI_API_KEY, EVALUATOR_MODEL, row IDs, and the JSON cache.")
        raise SystemExit(1)
