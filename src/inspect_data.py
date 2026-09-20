from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "RB_GenAI_Datatest.xlsx"
REQUIRED_COLUMNS = [
    "id",
    "question_category",
    "question",
    "person_id",
    "human_answers",
    "ai_answers",
]


def load_data() -> pd.DataFrame:
    return pd.read_excel(
        DATA_PATH, sheet_name="answer_pairs", engine="openpyxl", keep_default_na=False
    )


def inspect_data() -> None:
    data = load_data()
    print(f"File: {DATA_PATH.name}")
    print("Sheet: answer_pairs")
    print(f"Dataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"Columns: {', '.join(data.columns)}")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    print("Required columns: PASS (all six present)")
    if data.empty:
        raise ValueError("The dataset contains no rows.")

    # Treat empty and whitespace-only cells as missing without changing the source.
    missing_values = data.replace(r"^\s*$", pd.NA, regex=True).isna().sum()
    print("\nMissing values by column (including blank text):")
    print(missing_values.to_string())
    print(f"Total missing values: {missing_values.sum()}")
    print(f"\nDuplicate rows (beyond first occurrence): {data.duplicated().sum()}")

    people = data.replace(r"^\s*$", pd.NA, regex=True)
    print(f"Unique Person IDs (excluding missing): {people['person_id'].nunique()}")
    per_person = people.groupby("person_id", dropna=False).agg(
        rows=("question", "size"),
        questions=("question", "count"),
        unique_questions=("question", "nunique"),
    )
    print("\nRows/questions per person:")
    print(per_person.to_string())

    category_counts = people["question_category"].value_counts(dropna=False)
    distribution = category_counts.rename("rows").to_frame()
    distribution["percent"] = (category_counts / len(data) * 100).round(2)
    print("\nQuestion-category distribution:")
    print(distribution.to_string())


if __name__ == "__main__":
    inspect_data()
