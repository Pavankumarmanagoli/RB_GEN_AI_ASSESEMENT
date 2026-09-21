import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pydantic import ValidationError

from src.automated_evaluation import (
    build_markdown_summary,
    build_row_records,
    claim_analysis,
    contradiction_analysis,
    dataset_summary,
    person_summary,
)
from src.run_automated_evaluation import run_automated_evaluation
from src.schemas import AutomatedEvaluationRow


def dim(score, applicable=True):
    return {"applicable": applicable, "score": score}


def geval_record(row_id, person_id="c_human", core=3, behavior=None, preference=None, motivation=None, nuance=None):
    return {
        "row_id": row_id, "person_id": person_id, "question": f"Q{row_id}?",
        "human_answers": f"H{row_id}", "ai_answers": f"A{row_id}",
        "core": dim(core),
        "behavior": dim(behavior) if behavior is not None else dim(None, applicable=False),
        "preference": dim(preference) if preference is not None else dim(None, applicable=False),
        "motivation": dim(motivation) if motivation is not None else dim(None, applicable=False),
        "nuance": dim(nuance) if nuance is not None else dim(None, applicable=False),
    }


def alignment_record(row_id, human=4, aligned=1, partial=1, contradicted=0, missing=2, ai=4, unsupported=1,
                      coverage=0.5, strict=0.25, unsupported_rate=0.25):
    return {
        "row_id": row_id, "human_claim_count": human, "aligned_count": aligned, "partial_count": partial,
        "contradicted_count": contradicted, "missing_count": missing, "ai_claim_count": ai,
        "unsupported_count": unsupported, "claim_coverage_rate": coverage, "strict_alignment_rate": strict,
        "unsupported_rate": unsupported_rate,
    }


def bertscore_record(row_id, precision=0.1, recall=0.2, f1=0.15):
    return {"row_id": row_id, "bertscore_precision": precision, "bertscore_recall": recall, "bertscore_f1": f1}


def nli_pair(row_id, label="neutral", contradiction_prob=0.1):
    return {"row_id": row_id, "nli_label": label, "contradiction_prob": contradiction_prob}


class FixtureOutputsMixin:
    def _write_frozen_outputs(self, directory: Path, geval, alignment, nli, bertscore):
        (directory / "geval_results_final.json").write_text(json.dumps({"results": geval}))
        (directory / "claim_alignment_final.json").write_text(json.dumps({"results": alignment}))
        (directory / "nli_validation_final.json").write_text(json.dumps({"results": nli}))
        (directory / "bertscore_final.json").write_text(json.dumps({"results": bertscore}))


class BuildRowRecordsTests(unittest.TestCase, FixtureOutputsMixin):
    def test_automated_fidelity_score_equals_geval_core(self):
        with TemporaryDirectory() as directory:
            self._write_frozen_outputs(
                Path(directory),
                [geval_record(1, core=4)],
                [alignment_record(1)],
                [],
                [bertscore_record(1)],
            )
            with patch("src.automated_evaluation.OUTPUT_DIR", Path(directory)):
                [record] = build_row_records()
        self.assertEqual(record["automated_fidelity_score"], 4)
        self.assertEqual(record["core_fidelity_score"], 4)
        self.assertEqual(record["automated_fidelity_scale"], "1-5")

    def test_not_applicable_dimensions_are_null(self):
        with TemporaryDirectory() as directory:
            self._write_frozen_outputs(
                Path(directory),
                [geval_record(1, core=3, behavior=5)],  # preference/motivation/nuance not applicable
                [alignment_record(1)], [], [bertscore_record(1)],
            )
            with patch("src.automated_evaluation.OUTPUT_DIR", Path(directory)):
                [record] = build_row_records()
        self.assertEqual(record["behavior_score"], 5)
        self.assertIsNone(record["preference_score"])
        self.assertIsNone(record["motivation_score"])
        self.assertIsNone(record["nuance_score"])

    def test_nli_contradiction_present_and_strongest_probability(self):
        with TemporaryDirectory() as directory:
            self._write_frozen_outputs(
                Path(directory),
                [geval_record(1)],
                [alignment_record(1)],
                [nli_pair(1, "neutral", 0.2), nli_pair(1, "contradiction", 0.6), nli_pair(1, "contradiction", 0.9)],
                [bertscore_record(1)],
            )
            with patch("src.automated_evaluation.OUTPUT_DIR", Path(directory)):
                [record] = build_row_records()
        self.assertTrue(record["nli_contradiction_present"])
        self.assertEqual(record["nli_contradiction_count"], 2)
        self.assertAlmostEqual(record["strongest_contradiction_probability"], 0.9)

    def test_no_nli_contradiction_gives_null_strongest_probability(self):
        with TemporaryDirectory() as directory:
            self._write_frozen_outputs(
                Path(directory), [geval_record(1)], [alignment_record(1)],
                [nli_pair(1, "neutral", 0.2)], [bertscore_record(1)],
            )
            with patch("src.automated_evaluation.OUTPUT_DIR", Path(directory)):
                [record] = build_row_records()
        self.assertFalse(record["nli_contradiction_present"])
        self.assertIsNone(record["strongest_contradiction_probability"])

    def test_joins_multiple_rows_correctly_and_preserves_text(self):
        with TemporaryDirectory() as directory:
            self._write_frozen_outputs(
                Path(directory),
                [geval_record(1, core=2), geval_record(2, core=5)],
                [alignment_record(1), alignment_record(2)],
                [],
                [bertscore_record(1, f1=0.1), bertscore_record(2, f1=0.9)],
            )
            with patch("src.automated_evaluation.OUTPUT_DIR", Path(directory)):
                records = build_row_records()
        by_row = {r["row_id"]: r for r in records}
        self.assertEqual(by_row[1]["automated_fidelity_score"], 2)
        self.assertEqual(by_row[2]["automated_fidelity_score"], 5)
        self.assertEqual(by_row[1]["bertscore_f1"], 0.1)
        self.assertEqual(by_row[2]["bertscore_f1"], 0.9)
        self.assertEqual(by_row[1]["human_answer"], "H1")
        self.assertEqual(by_row[1]["ai_answer"], "A1")


class SchemaTests(unittest.TestCase):
    BASE = dict(
        row_id=1, person_id="p", question="Q?", human_answer="H", ai_answer="A",
        automated_fidelity_score=3, automated_fidelity_scale="1-5", core_fidelity_score=3,
        behavior_score=None, preference_score=None, motivation_score=None, nuance_score=None,
        human_claim_count=4, aligned_count=1, partial_count=1, contradicted_count=0, missing_count=2,
        ai_claim_count=4, unsupported_ai_count=1, claim_coverage_rate=0.5, strict_alignment_rate=0.25,
        unsupported_rate=0.25, nli_contradiction_present=False, nli_contradiction_count=0,
        strongest_contradiction_probability=None, bertscore_precision=0.1, bertscore_recall=0.2,
        bertscore_f1=0.15,
    )

    def test_rejects_mismatch_between_automated_score_and_geval_core(self):
        with self.assertRaises(ValidationError):
            AutomatedEvaluationRow(**{**self.BASE, "automated_fidelity_score": 4, "core_fidelity_score": 3})

    def test_accepts_a_consistent_record(self):
        AutomatedEvaluationRow(**self.BASE)


class ClaimAnalysisTests(unittest.TestCase):
    def test_computes_totals_and_percentages(self):
        records = [
            {"human_claim_count": 4, "aligned_count": 1, "partial_count": 1, "contradicted_count": 1,
             "missing_count": 1, "ai_claim_count": 4, "unsupported_ai_count": 2,
             "claim_coverage_rate": 0.5, "strict_alignment_rate": 0.25, "unsupported_rate": 0.5},
            {"human_claim_count": 4, "aligned_count": 3, "partial_count": 0, "contradicted_count": 0,
             "missing_count": 1, "ai_claim_count": 4, "unsupported_ai_count": 0,
             "claim_coverage_rate": 0.75, "strict_alignment_rate": 0.75, "unsupported_rate": 0.0},
        ]
        result = claim_analysis(records)
        self.assertEqual(result["human_claims_total"], 8)
        self.assertEqual(result["aligned_count"], 4)
        self.assertAlmostEqual(result["aligned_percentage"], 50.0)
        self.assertEqual(result["rows_with_contradiction"], 1)
        self.assertEqual(result["rows_with_missing_claim"], 2)


class ContradictionAnalysisTests(unittest.TestCase):
    def test_identifies_agreement_and_disagreement_rows(self):
        records = [
            {"row_id": 1, "contradicted_count": 1, "nli_contradiction_present": True},
            {"row_id": 2, "contradicted_count": 0, "nli_contradiction_present": True},
            {"row_id": 3, "contradicted_count": 1, "nli_contradiction_present": False},
            {"row_id": 4, "contradicted_count": 0, "nli_contradiction_present": False},
        ]
        result = contradiction_analysis(records)
        self.assertEqual(result["both_detect_contradiction_rows"], [1])
        self.assertEqual(result["step4_only_contradiction_rows"], [3])
        self.assertEqual(result["nli_only_contradiction_rows"], [2])


class PersonSummaryTests(unittest.TestCase):
    def test_excludes_human_review_fields_and_computes_means(self):
        records = [
            {"person_id": "c_human", "automated_fidelity_score": 2, "claim_coverage_rate": 0.5,
             "strict_alignment_rate": 0.2, "unsupported_rate": 0.6, "contradicted_count": 1,
             "nli_contradiction_present": True, "bertscore_f1": 0.1},
            {"person_id": "c_human", "automated_fidelity_score": 4, "claim_coverage_rate": 0.7,
             "strict_alignment_rate": 0.4, "unsupported_rate": 0.4, "contradicted_count": 0,
             "nli_contradiction_present": False, "bertscore_f1": 0.3},
        ]
        [summary] = person_summary(records)
        self.assertEqual(summary["number_of_rows"], 2)
        self.assertAlmostEqual(summary["automated_fidelity_mean"], 3.0)
        self.assertEqual(summary["step4_contradiction_row_count"], 1)
        self.assertEqual(summary["nli_contradiction_row_count"], 1)
        for forbidden_field in ("human_fidelity_score", "overall_consistency", "human_contradiction"):
            self.assertNotIn(forbidden_field, summary)


class DatasetSummaryAndMarkdownTests(unittest.TestCase, FixtureOutputsMixin):
    def test_dataset_summary_has_no_weighted_composite_field(self):
        with TemporaryDirectory() as directory:
            self._write_frozen_outputs(
                Path(directory),
                [geval_record(i, person_id=p, core=c) for i, (p, c) in enumerate(
                    [("c_human", 3), ("s_human", 4), ("g_human", 2)], start=1)],
                [alignment_record(i) for i in (1, 2, 3)],
                [],
                [bertscore_record(i) for i in (1, 2, 3)],
            )
            with patch("src.automated_evaluation.OUTPUT_DIR", Path(directory)):
                records = build_row_records()
                summary = dataset_summary(records)
        dumped = json.dumps(summary)
        for forbidden_term in ("composite", "weighted_score", "overall_score"):
            self.assertNotIn(forbidden_term, dumped)
        self.assertAlmostEqual(summary["automated_fidelity"]["overall_automated_fidelity_mean"], 3.0)

    def test_markdown_contains_required_sections(self):
        with TemporaryDirectory() as directory:
            self._write_frozen_outputs(
                Path(directory),
                [geval_record(1, core=3)], [alignment_record(1)], [], [bertscore_record(1)],
            )
            with patch("src.automated_evaluation.OUTPUT_DIR", Path(directory)):
                records = build_row_records()
                summary = dataset_summary(records)
        markdown = build_markdown_summary(summary)
        for heading in (
            "# Automated Evaluation Summary", "## Overall Automated Fidelity", "## G-Eval",
            "## Claim Preservation", "## Contradiction Analysis", "## Semantic Similarity",
            "## Person-Level Automated Results", "## Key Automated Findings",
        ):
            self.assertIn(heading, markdown)


class RunAutomatedEvaluationTests(unittest.TestCase, FixtureOutputsMixin):
    def _fixture(self, directory: Path):
        persons = ["c_human", "s_human", "g_human"]
        geval = [geval_record(i, person_id=persons[(i - 1) % 3], core=((i - 1) % 5) + 1) for i in range(1, 31)]
        alignment = [alignment_record(i) for i in range(1, 31)]
        bertscore = [bertscore_record(i) for i in range(1, 31)]
        self._write_frozen_outputs(directory, geval, alignment, [], bertscore)

    def test_produces_exactly_thirty_rows_with_full_row_id_coverage_and_three_people(self):
        with (
            TemporaryDirectory() as directory,
            patch("src.automated_evaluation.OUTPUT_DIR", Path(directory)),
            patch("src.run_automated_evaluation.OUTPUT_DIR", Path(directory)),
            patch("builtins.print"),
        ):
            output_dir = Path(directory)
            self._fixture(output_dir)
            self.assertEqual(run_automated_evaluation(), 0)
            data = json.loads((output_dir / "automated_evaluation_by_row.json").read_text())

        results = data["results"]
        self.assertEqual(len(results), 30)
        self.assertEqual(sorted(r["row_id"] for r in results), list(range(1, 31)))
        self.assertEqual({r["person_id"] for r in results}, {"c_human", "s_human", "g_human"})
        for person_id in ("c_human", "s_human", "g_human"):
            self.assertEqual(sum(1 for r in results if r["person_id"] == person_id), 10)

    def test_writes_all_four_expected_files(self):
        with (
            TemporaryDirectory() as directory,
            patch("src.automated_evaluation.OUTPUT_DIR", Path(directory)),
            patch("src.run_automated_evaluation.OUTPUT_DIR", Path(directory)),
            patch("builtins.print"),
        ):
            output_dir = Path(directory)
            self._fixture(output_dir)
            run_automated_evaluation()
            for name in (
                "automated_evaluation_by_row.json", "automated_evaluation_by_row.csv",
                "automated_evaluation_summary.json", "automated_evaluation_summary.md",
            ):
                self.assertTrue((output_dir / name).exists(), name)

    def test_does_not_modify_frozen_step3_to_6_outputs(self):
        with (
            TemporaryDirectory() as directory,
            patch("src.automated_evaluation.OUTPUT_DIR", Path(directory)),
            patch("src.run_automated_evaluation.OUTPUT_DIR", Path(directory)),
            patch("builtins.print"),
        ):
            output_dir = Path(directory)
            self._fixture(output_dir)
            before = {
                name: (output_dir / name).read_text()
                for name in (
                    "geval_results_final.json", "claim_alignment_final.json",
                    "nli_validation_final.json", "bertscore_final.json",
                )
            }
            run_automated_evaluation()
            for name, content in before.items():
                self.assertEqual((output_dir / name).read_text(), content)

    def test_does_not_read_step7_human_review_files(self):
        with (
            TemporaryDirectory() as directory,
            patch("src.automated_evaluation.OUTPUT_DIR", Path(directory)),
            patch("src.run_automated_evaluation.OUTPUT_DIR", Path(directory)),
            patch("builtins.print"),
        ):
            output_dir = Path(directory)
            self._fixture(output_dir)
            # No human_review_final.json or person_consistency files exist in this
            # temp dir at all; a successful run proves Step 7 data is never required.
            self.assertEqual(run_automated_evaluation(), 0)
            self.assertFalse((output_dir / "human_review_final.json").exists())


if __name__ == "__main__":
    unittest.main()
