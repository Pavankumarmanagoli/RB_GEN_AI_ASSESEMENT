"""Run the final evaluation and export reviewer-facing outputs."""
import pandas as pd

from src.config import OUTPUT_DIR
from src.final_evaluation import (
    build_excel_workbook,
    build_final_rows,
    build_markdown_summary,
    dataset_final_summary,
    load_automated_summary,
    load_human_review,
    load_person_manual_fields,
)
from src.run_geval import write_json


def run_final_evaluation() -> int:
    """Join automated and human review results and write the final outputs."""
    records = build_final_rows()
    _, human_summary = load_human_review()
    manual_by_person = load_person_manual_fields()
    automated_summary = load_automated_summary()

    summary = dataset_final_summary(records, human_summary, manual_by_person, automated_summary)
    markdown = build_markdown_summary(summary)
    workbook = build_excel_workbook(records, summary)

    OUTPUT_DIR.mkdir(exist_ok=True)
    write_json(OUTPUT_DIR / "final_pairwise_evaluation.json", {"results": records})
    pd.json_normalize(records, sep="_").to_csv(OUTPUT_DIR / "final_pairwise_evaluation.csv", index=False)
    write_json(OUTPUT_DIR / "final_evaluation_summary.json", summary)
    (OUTPUT_DIR / "final_evaluation_summary.md").write_text(markdown, encoding="utf-8")
    workbook.save(OUTPUT_DIR / "final_evaluation.xlsx")

    print(f"Joined {len(records)} rows into the final Human-vs-AI validation.")
    print(f"Automated fidelity mean: {summary['automated_fidelity_mean']:.2f}/5")
    print(f"Human fidelity mean: {summary['human_fidelity_mean']:.2f}/5")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_final_evaluation())
