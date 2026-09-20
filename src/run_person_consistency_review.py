from src.config import OUTPUT_DIR
from src.person_consistency_review import (
    build_workbook,
    group_by_person,
    load_human_review_records,
    person_summary,
    validate_grouping,
)
from src.run_geval import write_json


def run_person_consistency_review() -> int:
    records = load_human_review_records()
    grouped = group_by_person(records)
    validate_grouping(records, grouped)

    workbook = build_workbook(grouped)
    OUTPUT_DIR.mkdir(exist_ok=True)
    workbook.save(OUTPUT_DIR / "person_consistency_review.xlsx")

    summary = person_summary(grouped)
    write_json(OUTPUT_DIR / "person_consistency_context.json", {"people": summary})

    print(f"Grouped {len(records)} rows into {len(grouped)} person sheets.")
    for entry in summary:
        print(
            f"  {entry['person_id']}: {entry['number_of_rows']} rows, "
            f"mean fidelity {entry['mean_human_fidelity_score']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_person_consistency_review())
