import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.human_review import (
    HumanReviewError,
    build_joined_records,
    contradiction_confusion,
    load_review,
    map_reviews_to_dataset,
    spearman,
    validate_review,
)
from src.run_human_review import run_human_review


def valid_review_rows(n: int = 30) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "review_id": f"r{i}",
            "question": f"Question {i}?",
            "human_answers": f"Human answer {i}.",
            "ai_answers": f"AI answer {i}.",
            "fidelity_score": (i % 5) + 1,
            "contradiction": "YES" if i % 2 == 0 else "NO",
            "omission": "NO",
            "unsupported_detail": "YES",
            "note": f"Note {i}.",
        }
        for i in range(n)
    ])


class LoadReviewTests(unittest.TestCase):
    def test_ignores_completely_blank_rows(self):
        rows = valid_review_rows(2)
        blank = {column: "" for column in rows.columns}
        blank["fidelity_score"] = ""
        rows = pd.concat([rows, pd.DataFrame([blank])], ignore_index=True)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "review.xlsx"
            rows.to_excel(path, index=False)
            with patch("src.human_review.REVIEW_PATH", path):
                loaded = load_review()
        self.assertEqual(len(loaded), 2)


class ValidateReviewTests(unittest.TestCase):
    def test_requires_exactly_thirty_rows(self):
        with self.assertRaisesRegex(HumanReviewError, "exactly 30"):
            validate_review(valid_review_rows(5))

    def test_rejects_duplicate_review_ids(self):
        rows = valid_review_rows(30)
        rows.loc[1, "review_id"] = rows.loc[0, "review_id"]
        with self.assertRaisesRegex(HumanReviewError, "Duplicate review_id"):
            validate_review(rows)

    def test_rejects_out_of_range_or_non_integer_fidelity_scores(self):
        rows = valid_review_rows(30)
        rows.loc[0, "fidelity_score"] = 6
        with self.assertRaisesRegex(HumanReviewError, "fidelity_score"):
            validate_review(rows)
        rows = valid_review_rows(30)
        rows["fidelity_score"] = rows["fidelity_score"].astype(object)
        rows.loc[0, "fidelity_score"] = 2.5
        with self.assertRaisesRegex(HumanReviewError, "fidelity_score"):
            validate_review(rows)

    def test_normalizes_yes_no_case_and_whitespace_without_raising(self):
        rows = valid_review_rows(30)
        rows.loc[0, "contradiction"] = " yes "
        rows.loc[1, "omission"] = "no"
        validate_review(rows)  # should not raise

    def test_rejects_invalid_yes_no_values(self):
        rows = valid_review_rows(30)
        rows.loc[0, "contradiction"] = "MAYBE"
        with self.assertRaisesRegex(HumanReviewError, "contradiction"):
            validate_review(rows)

    def test_rejects_missing_answer_text(self):
        rows = valid_review_rows(30)
        rows.loc[0, "human_answers"] = ""
        with self.assertRaisesRegex(HumanReviewError, "Missing Human or AI"):
            validate_review(rows)


class MapReviewsToDatasetTests(unittest.TestCase):
    DATASET = pd.DataFrame([
        {"id": 1, "person_id": "p1", "question": "Q1?", "human_answers": "H1", "ai_answers": "A1"},
        {"id": 2, "person_id": "p2", "question": "Q2?", "human_answers": "H2", "ai_answers": "A2"},
        {"id": 3, "person_id": "p3", "question": "Q3?", "human_answers": "H3", "ai_answers": "A3"},
    ])

    @staticmethod
    def _review_row(review_id: str, row_id: int, **overrides) -> dict:
        base = {
            "review_id": review_id,
            "question": f"Q{row_id}?", "human_answers": f"H{row_id}", "ai_answers": f"A{row_id}",
            "fidelity_score": 3, "contradiction": "NO", "omission": "NO",
            "unsupported_detail": "YES", "note": "note",
        }
        return {**base, **overrides}

    def test_maps_shuffled_fake_ids_to_the_correct_original_row_by_content(self):
        # Fake IDs are deliberately out of order / non-corresponding to row_id.
        review = pd.DataFrame([
            self._review_row("r1", 3),
            self._review_row("r2", 1),
            self._review_row("r3", 2),
        ])
        with patch("src.human_review.load_data", return_value=self.DATASET):
            mapped = map_reviews_to_dataset(review)
        by_review_id = {entry["review_id"]: entry["row_id"] for entry in mapped}
        self.assertEqual(by_review_id, {"r1": 3, "r2": 1, "r3": 2})

    def test_uses_whitespace_normalized_exact_matching(self):
        review = pd.DataFrame([
            self._review_row("r1", 1, question=" Q1?  ", human_answers="H1\n", ai_answers="  A1"),
            self._review_row("r2", 2),
            self._review_row("r3", 3),
        ])
        with patch("src.human_review.load_data", return_value=self.DATASET):
            mapped = map_reviews_to_dataset(review)
        self.assertEqual({entry["row_id"] for entry in mapped}, {1, 2, 3})

    def test_raises_on_zero_matches(self):
        review = pd.DataFrame([
            self._review_row("r1", 1, human_answers="Something else entirely"),
            self._review_row("r2", 2),
            self._review_row("r3", 3),
        ])
        with patch("src.human_review.load_data", return_value=self.DATASET):
            with self.assertRaisesRegex(HumanReviewError, "No original dataset row"):
                map_reviews_to_dataset(review)

    def test_raises_on_duplicate_mapping_to_the_same_original_row(self):
        # Two review rows both matching dataset row 1's exact content.
        review = pd.DataFrame([
            self._review_row("r1", 1),
            self._review_row("r2", 1),
            self._review_row("r3", 3),
        ])
        with patch("src.human_review.load_data", return_value=self.DATASET):
            with self.assertRaisesRegex(HumanReviewError, "Duplicate mapping"):
                map_reviews_to_dataset(review)

    def test_raises_when_not_all_original_rows_are_recovered(self):
        # Only 2 review rows for a 3-row dataset: row_id 3 is never recovered.
        review = pd.DataFrame([
            self._review_row("r1", 1),
            self._review_row("r2", 2),
        ])
        with patch("src.human_review.load_data", return_value=self.DATASET):
            with self.assertRaisesRegex(HumanReviewError, "Not all original dataset rows"):
                map_reviews_to_dataset(review)


class SpearmanTests(unittest.TestCase):
    def test_uses_only_rows_where_both_fields_are_non_null(self):
        records = [
            {"x": 1, "y": 1},
            {"x": 2, "y": 2},
            {"x": 3, "y": None},
            {"x": None, "y": 4},
            {"x": 4, "y": 4},
        ]
        result = spearman(records, "x", "y")
        self.assertEqual(result["n"], 3)
        self.assertIsNotNone(result["rho"])
        self.assertIsNotNone(result["p_value"])

    def test_returns_none_when_fewer_than_two_matched_rows(self):
        records = [{"x": 1, "y": None}, {"x": None, "y": 2}]
        result = spearman(records, "x", "y")
        self.assertEqual(result, {"rho": None, "p_value": None, "n": 0})


class ContradictionConfusionTests(unittest.TestCase):
    def test_counts_confusion_matrix_correctly(self):
        records = [
            {"human_contradiction": "YES", "predicted": True},
            {"human_contradiction": "YES", "predicted": False},
            {"human_contradiction": "NO", "predicted": False},
            {"human_contradiction": "NO", "predicted": True},
        ]
        result = contradiction_confusion(records, "predicted")
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["false_negatives"], 1)
        self.assertEqual(result["true_negatives"], 1)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["agreement_count"], 2)
        self.assertAlmostEqual(result["agreement_percentage"], 50.0)


class BuildJoinedRecordsTests(unittest.TestCase):
    MAPPED = [{
        "review_id": "r1", "row_id": 1, "person_id": "p1",
        "question": "Q?", "human_answer": "H", "ai_answer": "A",
        "fidelity_score": 3, "contradiction": "NO", "omission": "NO",
        "unsupported_detail": "YES", "note": "note",
    }]

    def _write_frozen_outputs(self, directory: Path):
        (directory / "geval_results_final.json").write_text(json.dumps({
            "results": [{"row_id": 1, "core": {"score": 4}}],
        }))
        (directory / "claim_alignment_final.json").write_text(json.dumps({
            "results": [{
                "row_id": 1, "claim_coverage_rate": 0.5, "strict_alignment_rate": 0.25,
                "contradicted_count": 1, "missing_count": 2, "unsupported_rate": 0.3,
            }],
        }))
        (directory / "nli_validation_final.json").write_text(json.dumps({
            "results": [
                {"row_id": 1, "nli_label": "neutral"},
                {"row_id": 1, "nli_label": "contradiction"},
            ],
        }))
        (directory / "bertscore_final.json").write_text(json.dumps({
            "results": [{"row_id": 1, "bertscore_precision": 0.1, "bertscore_recall": 0.2, "bertscore_f1": 0.15}],
        }))

    def test_step4_contradiction_present_derived_from_contradicted_count(self):
        with TemporaryDirectory() as directory:
            self._write_frozen_outputs(Path(directory))
            with patch("src.human_review.OUTPUT_DIR", Path(directory)):
                [record] = build_joined_records(self.MAPPED)
        self.assertTrue(record["step4_contradiction_present"])
        self.assertEqual(record["step4_missing_count"], 2)

    def test_nli_contradiction_present_and_count_derived_from_matched_pairs(self):
        with TemporaryDirectory() as directory:
            self._write_frozen_outputs(Path(directory))
            with patch("src.human_review.OUTPUT_DIR", Path(directory)):
                [record] = build_joined_records(self.MAPPED)
        self.assertTrue(record["nli_contradiction_present"])
        self.assertEqual(record["nli_contradiction_count"], 1)


class RunHumanReviewTests(unittest.TestCase):
    def test_writes_exactly_the_expected_final_filenames(self):
        records = [{
            "review_id": "r1", "row_id": 1, "person_id": "p1", "question": "Q?",
            "human_answer": "H", "ai_answer": "A", "human_fidelity_score": 3,
            "human_contradiction": "NO", "human_omission": "NO", "human_unsupported_detail": "YES",
            "reviewer_note": "note", "geval_core_fidelity": 4, "claim_coverage_rate": 0.5,
            "strict_alignment_rate": 0.25, "step4_contradiction_present": False, "step4_missing_count": 1,
            "step4_unsupported_rate": 0.2, "nli_contradiction_present": False, "nli_contradiction_count": 0,
            "bertscore_precision": 0.1, "bertscore_recall": 0.2, "bertscore_f1": 0.15,
        }]
        with (
            TemporaryDirectory() as directory,
            patch("src.run_human_review.OUTPUT_DIR", Path(directory)),
            patch("src.run_human_review.load_review", return_value="review"),
            patch("src.run_human_review.validate_review"),
            patch("src.run_human_review.map_reviews_to_dataset", return_value="mapped"),
            patch("src.run_human_review.build_joined_records", return_value=records),
            patch("builtins.print"),
        ):
            self.assertEqual(run_human_review(), 0)
            output_dir = Path(directory)
            self.assertTrue((output_dir / "human_review_final.json").exists())
            self.assertTrue((output_dir / "human_review_final.csv").exists())
            self.assertFalse(list(output_dir.glob("*.tmp")))

    def test_does_not_modify_frozen_step3_to_6_outputs(self):
        records = [{
            "review_id": "r1", "row_id": 1, "person_id": "p1", "question": "Q?",
            "human_answer": "H", "ai_answer": "A", "human_fidelity_score": 3,
            "human_contradiction": "NO", "human_omission": "NO", "human_unsupported_detail": "YES",
            "reviewer_note": "note", "geval_core_fidelity": 4, "claim_coverage_rate": 0.5,
            "strict_alignment_rate": 0.25, "step4_contradiction_present": False, "step4_missing_count": 1,
            "step4_unsupported_rate": 0.2, "nli_contradiction_present": False, "nli_contradiction_count": 0,
            "bertscore_precision": 0.1, "bertscore_recall": 0.2, "bertscore_f1": 0.15,
        }]
        with (
            TemporaryDirectory() as directory,
            patch("src.run_human_review.OUTPUT_DIR", Path(directory)),
            patch("src.run_human_review.load_review", return_value="review"),
            patch("src.run_human_review.validate_review"),
            patch("src.run_human_review.map_reviews_to_dataset", return_value="mapped"),
            patch("src.run_human_review.build_joined_records", return_value=records),
            patch("builtins.print"),
        ):
            output_dir = Path(directory)
            frozen_files = [
                "geval_results_final.json", "claim_alignment_final.json",
                "nli_validation_final.json", "bertscore_final.json",
            ]
            before = {}
            for name in frozen_files:
                path = output_dir / name
                path.write_text(f'{{"marker": "{name}"}}')
                before[name] = path.read_text()
            run_human_review()
            for name in frozen_files:
                self.assertEqual((output_dir / name).read_text(), before[name])


if __name__ == "__main__":
    unittest.main()
