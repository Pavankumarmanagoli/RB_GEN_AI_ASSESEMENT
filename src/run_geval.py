import argparse
import json
from pathlib import Path

import pandas as pd
from openai import APIConnectionError, APIError, APIStatusError, OpenAI
from pydantic import ValidationError

from src.config import OUTPUT_DIR, load_config, model_options
from src.geval_evaluator import (
    PROMPT_VERSION,
    EVIDENCE_FORMAT_INSTRUCTIONS,
    EvaluationError,
    build_prompt,
    cache_key,
    evaluate_pair,
    validate_evidence,
)
from src.inspect_data import load_data
from src.schemas import FidelityEvaluation, FidelityJudgment


# Manually selected by reading the answers, before any automated evaluation.
PILOT_ROWS = {
    25: "Apparent strong match: efficacy, affordable price, and scent drive repurchase.",
    10: "Clear behavioral reversal: usually avoids bulk buying versus regularly buys in bulk.",
    20: "Preference mismatch: ingredient transparency, glass, texture, and oud versus different features.",
    22: "Motivation should be N/A: the human chooses option 2 without providing a reason.",
    24: "Nuanced answer: indifference to names, preference for visual design, and a conditional choice.",
}

# Manually selected from the fidelity-v2 full-dataset run: rows where Motivation
# scored 1 for a substituted, unrelated reason rather than a stated opposite.
MOTIVATION_CALIBRATION_ROWS = {
    12: "Unrelated reason (avoiding shopping trips) substituted for a discount-driven purchase.",
    13: "Unrelated reason (sustainability) substituted for a fame/attractiveness-driven celebrity choice.",
    14: "Unrelated reason (skincare expertise) substituted for a general-liking celebrity choice.",
    15: "Unrelated reason (grooming reliability) substituted where the human gave no celebrity choice.",
    16: "Unrelated reason (better ingredients) substituted for a family-habit brand history.",
    18: "Unrelated reason (friend recommendation) substituted for brand inattention.",
    24: "Unrelated reason (product texture) substituted for indifference to names/visual design.",
}

# Manually selected from the fidelity-v3 motivation-calibration run: rows where a
# hypothetical or forced-choice selection was inconsistently scored as Behavior.
BEHAVIOR_CALIBRATION_ROWS = {
    13: "Hypothetical celebrity choice previously scored Behavior = applicable.",
    14: "Hypothetical celebrity choice previously scored Behavior = applicable.",
    15: "Hypothetical inability to choose a celebrity previously scored Behavior = applicable, "
        "inconsistent with rows 13/14/24.",
    24: "Forced-choice product selection previously scored Behavior = applicable.",
}


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def describe_failure(error: Exception) -> dict[str, str]:
    # SDK error bodies can contain credentials; never log or persist raw exceptions.
    if isinstance(error, EvaluationError):
        message = str(error)
    elif isinstance(error, APIStatusError):
        message = f"API request failed with HTTP {error.status_code}; check credentials, model access, or quota."
    elif isinstance(error, APIConnectionError):
        message = "Could not reach the OpenAI API; check network access."
    elif isinstance(error, ValidationError):
        message = "The judge output failed schema or applicability validation."
    elif isinstance(error, APIError):
        message = "The API response could not be processed as a structured evaluation."
    else:
        message = "Invalid row input, incomplete response, or evidence validation failure."
    return {"error_type": type(error).__name__, "error_message": message}


def _evaluate_rows(
    client: OpenAI, model: str, prompt: str, original_prompt: str,
    data: pd.DataFrame, row_ids: list[int], cache: dict, cache_path: Path,
    result_stem: str, reasons: dict[int, str] | None = None,
) -> list[dict]:
    records = []
    for row_id in row_ids:
        row = data.loc[data["id"] == row_id].iloc[0]
        inputs = {field: row[field] for field in ("question", "human_answers", "ai_answers")}
        key = cache_key(inputs, model, prompt)
        original_key = cache_key(inputs, model, original_prompt)
        # Keep valid previous judgments; only new requests need stricter quote formatting.
        if key not in cache and original_key in cache:
            key = original_key
        record = {
            "row_id": row_id,
            "person_id": str(row["person_id"]),
            **inputs,
            **({"selection_reason": reasons[row_id]} if reasons is not None else {}),
            "evaluator_model": model,
            "model_options": model_options(model),
            "prompt_version": PROMPT_VERSION,
            "evidence_format": "original" if key == original_key else "single-excerpt-v1",
            "cache_key": key,
        }
        try:
            cached = key in cache
            judgment = (
                validate_evidence(FidelityJudgment.model_validate(cache[key]), inputs) if cached
                else evaluate_pair(client, model, prompt, inputs)
            )
            evaluation = FidelityEvaluation(
                row_id=row_id, person_id=str(row["person_id"]), **judgment.model_dump()
            )
            if not cached:
                cache[key] = judgment.model_dump()
                write_json(cache_path, cache)
            record.update(evaluation.model_dump())
            record.update(status="completed", cached=cached)
            print(f"Row {row_id}: completed ({'cache' if cached else 'API'})", flush=True)
        except (APIError, ValidationError, ValueError) as error:
            record.update(status="failed", **describe_failure(error))
            print(f"Row {row_id}: {record['error_type']} — {record['error_message']}", flush=True)
        records.append(record)
        write_json(OUTPUT_DIR / f"{result_stem}.json", {"results": records})
        pd.json_normalize(records, sep="_").to_csv(OUTPUT_DIR / f"{result_stem}.csv", index=False)
    return records


def run_pilot(row_ids: list[int] | None = None) -> int:
    try:
        api_key, model = load_config()
    except ValueError as error:
        print(str(error))
        return 1
    data = load_data()
    requested_ids = set(PILOT_ROWS) if row_ids is None else set(row_ids)
    unknown_ids = requested_ids - set(PILOT_ROWS)
    if unknown_ids:
        raise ValueError(f"Only manually selected pilot row IDs are allowed: {sorted(PILOT_ROWS)}.")
    selected_ids = [row_id for row_id in PILOT_ROWS if row_id in requested_ids]
    selected = data[data["id"].isin(selected_ids)]
    if len(selected) != len(selected_ids) or selected["id"].nunique() != len(selected_ids):
        raise ValueError("Each requested pilot row ID must occur exactly once.")
    original_prompt = build_prompt()
    prompt = original_prompt + EVIDENCE_FORMAT_INSTRUCTIONS
    OUTPUT_DIR.mkdir(exist_ok=True)
    cache_path = OUTPUT_DIR / ".geval_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    print(f"Evaluator model: {model}")
    selected_suffix = "all" if row_ids is None else "rows_" + "-".join(map(str, selected_ids))
    result_stem = f"geval_pilot_results_{PROMPT_VERSION}_{selected_suffix}"
    # The SDK bounds retries for transient connection, rate-limit, and server errors.
    with OpenAI(api_key=api_key, max_retries=2, timeout=60.0) as client:
        records = _evaluate_rows(
            client, model, prompt, original_prompt, selected, selected_ids,
            cache, cache_path, result_stem, reasons=PILOT_ROWS,
        )
    return int(any(record["status"] == "failed" for record in records))


def run_full() -> int:
    try:
        api_key, model = load_config()
    except ValueError as error:
        print(str(error))
        return 1
    data = load_data()
    row_ids = list(data["id"])
    original_prompt = build_prompt()
    prompt = original_prompt + EVIDENCE_FORMAT_INSTRUCTIONS
    OUTPUT_DIR.mkdir(exist_ok=True)
    cache_path = OUTPUT_DIR / ".geval_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    print(f"Evaluator model: {model}")
    result_stem = f"geval_results_{PROMPT_VERSION}_full"
    # The SDK bounds retries for transient connection, rate-limit, and server errors.
    with OpenAI(api_key=api_key, max_retries=2, timeout=60.0) as client:
        records = _evaluate_rows(
            client, model, prompt, original_prompt, data, row_ids, cache, cache_path, result_stem,
        )
    return int(any(record["status"] == "failed" for record in records))


def run_motivation_calibration() -> int:
    try:
        api_key, model = load_config()
    except ValueError as error:
        print(str(error))
        return 1
    data = load_data()
    row_ids = list(MOTIVATION_CALIBRATION_ROWS)
    selected = data[data["id"].isin(row_ids)]
    if len(selected) != len(row_ids) or selected["id"].nunique() != len(row_ids):
        raise ValueError("Each motivation calibration row ID must occur exactly once.")
    original_prompt = build_prompt()
    prompt = original_prompt + EVIDENCE_FORMAT_INSTRUCTIONS
    OUTPUT_DIR.mkdir(exist_ok=True)
    cache_path = OUTPUT_DIR / ".geval_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    print(f"Evaluator model: {model}")
    result_stem = f"geval_calibration_results_{PROMPT_VERSION}_motivation_rows_" + "-".join(map(str, row_ids))
    # The SDK bounds retries for transient connection, rate-limit, and server errors.
    with OpenAI(api_key=api_key, max_retries=2, timeout=60.0) as client:
        records = _evaluate_rows(
            client, model, prompt, original_prompt, selected, row_ids,
            cache, cache_path, result_stem, reasons=MOTIVATION_CALIBRATION_ROWS,
        )
    return int(any(record["status"] == "failed" for record in records))


def run_behavior_calibration() -> int:
    try:
        api_key, model = load_config()
    except ValueError as error:
        print(str(error))
        return 1
    data = load_data()
    row_ids = list(BEHAVIOR_CALIBRATION_ROWS)
    selected = data[data["id"].isin(row_ids)]
    if len(selected) != len(row_ids) or selected["id"].nunique() != len(row_ids):
        raise ValueError("Each behavior calibration row ID must occur exactly once.")
    original_prompt = build_prompt()
    prompt = original_prompt + EVIDENCE_FORMAT_INSTRUCTIONS
    OUTPUT_DIR.mkdir(exist_ok=True)
    cache_path = OUTPUT_DIR / ".geval_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    print(f"Evaluator model: {model}")
    result_stem = f"geval_calibration_results_{PROMPT_VERSION}_behavior_rows_" + "-".join(map(str, row_ids))
    # The SDK bounds retries for transient connection, rate-limit, and server errors.
    with OpenAI(api_key=api_key, max_retries=2, timeout=60.0) as client:
        records = _evaluate_rows(
            client, model, prompt, original_prompt, selected, row_ids,
            cache, cache_path, result_stem, reasons=BEHAVIOR_CALIBRATION_ROWS,
        )
    return int(any(record["status"] == "failed" for record in records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate human/AI response pairs.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--rows", type=int, nargs="+", metavar="ROW_ID",
        help="pilot row IDs to evaluate (defaults to all five selected pilot rows)",
    )
    selection.add_argument(
        "--full", action="store_true",
        help="evaluate every row in the dataset, not just the manually selected pilot rows",
    )
    selection.add_argument(
        "--motivation-calibration", action="store_true",
        help="evaluate the manually selected Motivation-calibration rows for the current prompt version",
    )
    selection.add_argument(
        "--behavior-calibration", action="store_true",
        help="evaluate the manually selected Behavior-calibration rows for the current prompt version",
    )
    args = parser.parse_args()
    try:
        if args.motivation_calibration:
            exit_code = run_motivation_calibration()
        elif args.behavior_calibration:
            exit_code = run_behavior_calibration()
        elif args.full:
            exit_code = run_full()
        else:
            exit_code = run_pilot(args.rows)
        raise SystemExit(exit_code)
    except ValueError:
        # Configuration messages are fixed strings, never API error bodies.
        print("Run could not start. Check OPENAI_API_KEY, EVALUATOR_MODEL, pilot IDs, and the JSON cache.")
        raise SystemExit(1)
