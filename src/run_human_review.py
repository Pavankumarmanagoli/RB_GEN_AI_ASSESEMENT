"""Run the human review comparison and export reviewer outputs."""
import pandas as pd

from src.config import OUTPUT_DIR
from src.human_review import (
    build_joined_records,
    contradiction_confusion,
    disagreement_rows,
    fidelity_summary,
    flag_rate,
    load_review,
    map_reviews_to_dataset,
    omission_comparison,
    spearman,
    unsupported_detail_comparison,
    validate_review,
)
from src.run_geval import write_json


def run_human_review() -> int:
    """Map the human review to the dataset and write the joined comparison."""
    review = load_review()
    validate_review(review)
    mapped = map_reviews_to_dataset(review)
    records = build_joined_records(mapped)

    summary = {
        "human_fidelity_summary": fidelity_summary(records),
        "human_flag_rates": {
            "contradiction": flag_rate(records, "human_contradiction"),
            "omission": flag_rate(records, "human_omission"),
            "unsupported_detail": flag_rate(records, "human_unsupported_detail"),
        },
        "spearman_correlations": {
            "human_fidelity_vs_geval_core_fidelity": spearman(records, "human_fidelity_score", "geval_core_fidelity"),
            "human_fidelity_vs_claim_coverage_rate": spearman(records, "human_fidelity_score", "claim_coverage_rate"),
            "human_fidelity_vs_strict_alignment_rate": spearman(
                records, "human_fidelity_score", "strict_alignment_rate"
            ),
            "human_fidelity_vs_bertscore_f1": spearman(records, "human_fidelity_score", "bertscore_f1"),
        },
        "contradiction_agreement": {
            "step4_vs_human": contradiction_confusion(records, "step4_contradiction_present"),
            "nli_vs_human": contradiction_confusion(records, "nli_contradiction_present"),
        },
        "contradiction_disagreement_rows": {
            "step4_vs_human": disagreement_rows(records, "step4_contradiction_present"),
            "nli_vs_human": disagreement_rows(records, "nli_contradiction_present"),
        },
        "omission_comparison": omission_comparison(records),
        "unsupported_detail_comparison": unsupported_detail_comparison(records),
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    write_json(OUTPUT_DIR / "human_review_final.json", {"summary": summary, "results": records})
    pd.json_normalize(records, sep="_").to_csv(OUTPUT_DIR / "human_review_final.csv", index=False)
    print(f"Mapped and joined {len(records)} human review rows with the automated evaluation results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_human_review())
