import pandas as pd

from src.automated_evaluation import build_markdown_summary, build_row_records, dataset_summary
from src.config import OUTPUT_DIR
from src.run_geval import write_json


def run_automated_evaluation() -> int:
    records = build_row_records()
    summary = dataset_summary(records)
    markdown = build_markdown_summary(summary)

    OUTPUT_DIR.mkdir(exist_ok=True)
    write_json(OUTPUT_DIR / "automated_evaluation_by_row.json", {"results": records})
    pd.json_normalize(records, sep="_").to_csv(OUTPUT_DIR / "automated_evaluation_by_row.csv", index=False)
    write_json(OUTPUT_DIR / "automated_evaluation_summary.json", summary)
    (OUTPUT_DIR / "automated_evaluation_summary.md").write_text(markdown, encoding="utf-8")

    print(f"Joined {len(records)} rows from Steps 3-6 into the automated evaluation synthesis.")
    print(f"Overall Automated Fidelity: {summary['automated_fidelity']['overall_automated_fidelity_mean']:.2f} / 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_automated_evaluation())
