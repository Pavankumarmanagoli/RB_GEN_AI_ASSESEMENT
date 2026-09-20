import argparse
import json

import pandas as pd

from src.config import OUTPUT_DIR
from src.nli_validator import NLI_MODEL_NAME, load_nli_model, predict_nli, resolve_label_map
from src.run_geval import write_json
from src.schemas import NLIPrediction

# A small, manually curated set of frozen Step 4B claim pairs, selected by inspecting
# outputs/claim_extraction_final.json and outputs/claim_alignment_final.json — not
# guessed by row number. Identified as (row_id, human_claim_id); the matched
# ai_claim_id and Step 4B label are read from the frozen alignment output. Roughly
# 5 aligned, 5 contradicted (including one deliberately borderline case), and
# 5 partial pairs with useful scope/specificity differences.
NLI_CALIBRATION_PAIRS: list[tuple[int, str]] = [
    (5, "h4"),   # aligned: exact Beauty of Joseon sunscreen match
    (22, "h1"),  # aligned: near-exact Option 2 / Dove Body Love Light Hydration match
    (8, "h3"),   # aligned: glass bottle preference, paraphrased
    (25, "h1"),  # aligned: repurchase-if-it-works, paraphrased
    (27, "h1"),  # aligned: no single favorite brand, paraphrased
    (16, "h4"),  # contradicted: "never had a specific reason" vs a stated reason for switching
    (9, "h7"),   # contradicted: indifference to ingredient info vs preferring clear labeling
    (13, "h4"),  # contradicted: different named celebrity than the AI's forced choice
    (24, "h4"),  # contradicted: different named product than the AI's forced choice
    (12, "h1"),  # contradicted (borderline): "almost always" one item vs AI's "regularly buys in bulk"
    (1, "h1"),   # partial: "usually online" vs AI's broader "stores... or online platforms"
    (4, "h3"),   # partial: affordability as a stated reason vs AI asserting the product is affordable
    (6, "h3"),   # partial: "no-name discounter" vs AI's "sometimes buys no-name brands" (frequency)
    (9, "h1"),   # partial: ">300ml" preference vs AI's specific "500ml" value
    (2, "h1"),   # partial: orders from YesStyle specifically vs AI's broader "YesStyle and Soko Glam"
]


def _load_claim_extraction() -> dict[int, dict]:
    data = json.loads((OUTPUT_DIR / "claim_extraction_final.json").read_text(encoding="utf-8"))
    return {record["row_id"]: record for record in data["results"]}


def _load_claim_alignment() -> dict[int, dict]:
    data = json.loads((OUTPUT_DIR / "claim_alignment_final.json").read_text(encoding="utf-8"))
    return {record["row_id"]: record for record in data["results"]}


def eligible_pairs(extraction_by_row: dict[int, dict], alignment_by_row: dict[int, dict]) -> list[dict]:
    """Every Step 4B alignment with a non-null ai_claim_id — labels aligned, partial, or
    contradicted. Excludes `missing` human claims (no AI claim exists to compare) and
    unsupported AI claims (no human claim exists to compare), which stay Step 4-only
    diagnostics."""
    pairs = []
    for row_id, alignment in alignment_by_row.items():
        extraction = extraction_by_row[row_id]
        human_by_id = {claim["claim_id"]: claim["claim"] for claim in extraction["human_claims"]}
        ai_by_id = {claim["claim_id"]: claim["claim"] for claim in extraction["ai_claims"]}
        for entry in alignment["alignments"]:
            if entry["ai_claim_id"] is None:
                continue
            pairs.append({
                "row_id": row_id,
                "person_id": alignment["person_id"],
                "human_claim_id": entry["human_claim_id"],
                "ai_claim_id": entry["ai_claim_id"],
                "human_claim": human_by_id[entry["human_claim_id"]],
                "ai_claim": ai_by_id[entry["ai_claim_id"]],
                "step4b_label": entry["label"],
            })
    return pairs


def select_calibration_pairs(pairs: list[dict]) -> list[dict]:
    wanted_order = {key: index for index, key in enumerate(NLI_CALIBRATION_PAIRS)}
    selected = [pair for pair in pairs if (pair["row_id"], pair["human_claim_id"]) in wanted_order]
    found = {(pair["row_id"], pair["human_claim_id"]) for pair in selected}
    missing = set(wanted_order) - found
    if missing:
        raise ValueError(f"Calibration pairs not found in frozen Step 4 outputs: {sorted(missing)}")
    selected.sort(key=lambda pair: wanted_order[(pair["row_id"], pair["human_claim_id"])])
    return selected


def compare_with_step4b(step4b_label: str, nli_label: str) -> dict:
    """`aligned` expects entailment and `contradicted` expects contradiction; `partial` has
    no single expected NLI signal (a partial semantic match may reasonably read as
    entailment or neutral depending on specificity/scope), so it is never counted as
    disagreement."""
    expected = {"aligned": "entailment", "contradicted": "contradiction"}.get(step4b_label)
    if expected is None:
        return {"expected_nli_signal": None, "agrees_with_step4b": None}
    return {"expected_nli_signal": expected, "agrees_with_step4b": nli_label == expected}


def run_validation(calibration: bool) -> int:
    extraction_by_row = _load_claim_extraction()
    alignment_by_row = _load_claim_alignment()
    pairs = eligible_pairs(extraction_by_row, alignment_by_row)
    selected = select_calibration_pairs(pairs) if calibration else pairs

    tokenizer, model, device = load_nli_model()
    label_map = resolve_label_map(model)
    predictions = predict_nli(
        tokenizer, model, device, label_map,
        [(pair["human_claim"], pair["ai_claim"]) for pair in selected],
    )

    records = []
    for pair, prediction in zip(selected, predictions):
        validated = NLIPrediction.model_validate(prediction)
        record = {
            "row_id": pair["row_id"],
            "person_id": pair["person_id"],
            "human_claim_id": pair["human_claim_id"],
            "ai_claim_id": pair["ai_claim_id"],
            "premise": pair["human_claim"],
            "hypothesis": pair["ai_claim"],
            "step4b_label": pair["step4b_label"],
            "nli_model": NLI_MODEL_NAME,
            **validated.model_dump(),
        }
        record.update(compare_with_step4b(pair["step4b_label"], validated.nli_label))
        records.append(record)

    stem = "nli_calibration_results" if calibration else "nli_validation_final"
    OUTPUT_DIR.mkdir(exist_ok=True)
    write_json(OUTPUT_DIR / f"{stem}.json", {
        "nli_model": NLI_MODEL_NAME,
        "premise": "human_claim",
        "hypothesis": "ai_claim",
        "results": records,
    })
    pd.json_normalize(records, sep="_").to_csv(OUTPUT_DIR / f"{stem}.csv", index=False)
    print(f"NLI model: {NLI_MODEL_NAME}")
    print(f"Evaluated {len(records)} claim pairs ({'calibration' if calibration else 'full'}).")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-check frozen Step 4B claim alignment labels with a pretrained NLI model."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--calibration", action="store_true",
        help="run the fixed calibration subset of claim pairs (NLI_CALIBRATION_PAIRS)",
    )
    selection.add_argument(
        "--full", action="store_true",
        help="run every eligible (aligned/partial/contradicted) claim pair",
    )
    args = parser.parse_args()
    raise SystemExit(run_validation(calibration=args.calibration))
