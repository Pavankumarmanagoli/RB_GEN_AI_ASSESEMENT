"""Tests for the BERTScore evaluator and its runner."""
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pandas as pd
import torch
from pydantic import ValidationError

from src.bertscore_evaluator import BERTScoreError, find_length_overflows, score_batch
from src.run_bertscore import _load_rows, run_bertscore
from src.schemas import BERTScoreResult


class ScoreBatchTests(unittest.TestCase):
    def test_ai_answer_is_passed_as_candidate_and_human_as_reference(self):
        scorer = MagicMock()
        scorer.score.return_value = (torch.tensor([0.5]), torch.tensor([0.6]), torch.tensor([0.7]))
        score_batch(scorer, ["human text"], ["ai text"])
        call_kwargs = scorer.score.call_args.kwargs
        self.assertEqual(call_kwargs["cands"], ["ai text"])
        self.assertEqual(call_kwargs["refs"], ["human text"])

    def test_batching_preserves_row_order(self):
        scorer = MagicMock()
        scorer.score.return_value = (
            torch.tensor([0.1, 0.2, 0.3]),
            torch.tensor([0.4, 0.5, 0.6]),
            torch.tensor([0.7, 0.8, 0.9]),
        )
        precision, recall, f1 = score_batch(scorer, ["h1", "h2", "h3"], ["a1", "a2", "a3"])
        self.assertEqual([round(v, 1) for v in precision], [0.1, 0.2, 0.3])
        self.assertEqual([round(v, 1) for v in recall], [0.4, 0.5, 0.6])
        self.assertEqual([round(v, 1) for v in f1], [0.7, 0.8, 0.9])

    def test_rejects_empty_answer_lists(self):
        with self.assertRaises(BERTScoreError):
            score_batch(MagicMock(), [], [])

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(BERTScoreError):
            score_batch(MagicMock(), ["h1", "h2"], ["a1"])

    def test_rejects_empty_or_missing_answer_text_explicitly(self):
        scorer = MagicMock()
        with self.assertRaisesRegex(BERTScoreError, "Empty or missing"):
            score_batch(scorer, ["human text", ""], ["ai text", "ai text"])
        with self.assertRaisesRegex(BERTScoreError, "Empty or missing"):
            score_batch(scorer, ["human text", "   "], ["ai text", "ai text"])
        scorer.score.assert_not_called()


class FindLengthOverflowsTests(unittest.TestCase):
    def test_flags_answers_exceeding_the_models_max_sequence_length(self):
        tokenizer = MagicMock()
        tokenizer.model_max_length = 10
        tokenizer.encode.side_effect = lambda text, add_special_tokens: [0] * (20 if text == "long" else 3)
        rows = [{"row_id": 1, "human_answer": "long", "ai_answer": "short"}]
        overflows = find_length_overflows(rows, tokenizer=tokenizer)
        self.assertEqual(len(overflows), 1)
        self.assertEqual(overflows[0], {"row_id": 1, "side": "human", "token_count": 20, "max_length": 10})

    def test_no_overflows_when_everything_fits(self):
        tokenizer = MagicMock()
        tokenizer.model_max_length = 512
        tokenizer.encode.return_value = [0] * 10
        rows = [{"row_id": 1, "human_answer": "short", "ai_answer": "also short"}]
        self.assertEqual(find_length_overflows(rows, tokenizer=tokenizer), [])


class BERTScoreResultSchemaTests(unittest.TestCase):
    BASE = dict(
        row_id=1, person_id="p1", question="Q?", human_answer="H", ai_answer="A",
        bertscore_precision=0.5, bertscore_recall=0.5, bertscore_f1=0.5,
    )

    def test_accepts_scores_outside_zero_to_one(self):
        # Baseline rescaling can push values below 0 or (rarely) above 1; these are
        # metric values, not probabilities, and must not be range-restricted.
        result = BERTScoreResult(**{**self.BASE, "bertscore_precision": -0.3, "bertscore_f1": 1.2})
        self.assertEqual(result.bertscore_precision, -0.3)
        self.assertEqual(result.bertscore_f1, 1.2)

    def test_rejects_non_finite_scores(self):
        for bad_value in (math.nan, math.inf, -math.inf):
            with self.subTest(bad_value=bad_value), self.assertRaises(ValidationError):
                BERTScoreResult(**{**self.BASE, "bertscore_f1": bad_value})

    def test_requires_nonempty_answers_and_question(self):
        with self.assertRaises(ValidationError):
            BERTScoreResult(**{**self.BASE, "human_answer": ""})
        with self.assertRaises(ValidationError):
            BERTScoreResult(**{**self.BASE, "ai_answer": ""})


class RunBertscoreTests(unittest.TestCase):
    ROWS = [
        {"id": 1, "person_id": "p1", "question": "Q1?", "human_answers": "H1", "ai_answers": "A1"},
        {"id": 2, "person_id": "p2", "question": "Q2?", "human_answers": "H2", "ai_answers": "A2"},
    ]

    def _fake_dataframe(self):
        return pd.DataFrame(self.ROWS)

    def test_load_rows_reuses_the_projects_dataset_loader(self):
        with patch("src.run_bertscore.load_data", return_value=self._fake_dataframe()):
            rows = _load_rows()
        self.assertEqual([row["row_id"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["human_answer"], "H1")
        self.assertEqual(rows[0]["ai_answer"], "A1")

    def test_full_run_selects_every_row_exactly_once_and_preserves_order(self):
        scorer = MagicMock()
        scorer.hash = "roberta-large_L17_no-idf_version=0.3.12(hug_trans=5.17.0)-rescaled"
        scorer.score.return_value = (
            torch.tensor([0.9, 0.8]), torch.tensor([0.7, 0.6]), torch.tensor([0.85, 0.75]),
        )
        with (
            TemporaryDirectory() as directory,
            patch("src.run_bertscore.OUTPUT_DIR", Path(directory)),
            patch("src.run_bertscore.load_data", return_value=self._fake_dataframe()),
            patch("src.run_bertscore.load_scorer", return_value=scorer),
            patch("src.run_bertscore.find_length_overflows", return_value=[]),
            patch("builtins.print"),
        ):
            self.assertEqual(run_bertscore(), 0)
            output_dir = Path(directory)
            data = json.loads((output_dir / "bertscore_final.json").read_text())

        self.assertEqual([r["row_id"] for r in data["results"]], [1, 2])
        self.assertEqual(len(data["results"]), len(self.ROWS))
        self.assertAlmostEqual(data["results"][0]["bertscore_precision"], 0.9)
        self.assertAlmostEqual(data["results"][1]["bertscore_f1"], 0.75)

    def test_output_schema_and_metadata_are_recorded(self):
        scorer = MagicMock()
        scorer.hash = "test-hash"
        scorer.score.return_value = (torch.tensor([0.9, 0.8]), torch.tensor([0.7, 0.6]), torch.tensor([0.85, 0.75]))
        with (
            TemporaryDirectory() as directory,
            patch("src.run_bertscore.OUTPUT_DIR", Path(directory)),
            patch("src.run_bertscore.load_data", return_value=self._fake_dataframe()),
            patch("src.run_bertscore.load_scorer", return_value=scorer),
            patch("src.run_bertscore.find_length_overflows", return_value=[]),
            patch("builtins.print"),
        ):
            run_bertscore()
            output_dir = Path(directory)
            data = json.loads((output_dir / "bertscore_final.json").read_text())

        for field in ("model", "language", "reference", "candidate", "rescale_with_baseline", "idf", "bertscore_hash"):
            self.assertIn(field, data)
        self.assertEqual(data["reference"], "human_answer")
        self.assertEqual(data["candidate"], "ai_answer")
        for record in data["results"]:
            for field in (
                "row_id", "person_id", "question", "human_answer", "ai_answer",
                "bertscore_precision", "bertscore_recall", "bertscore_f1",
            ):
                self.assertIn(field, record)

    def test_final_filenames_are_exact(self):
        scorer = MagicMock()
        scorer.hash = "test-hash"
        scorer.score.return_value = (torch.tensor([0.9, 0.8]), torch.tensor([0.7, 0.6]), torch.tensor([0.85, 0.75]))
        with (
            TemporaryDirectory() as directory,
            patch("src.run_bertscore.OUTPUT_DIR", Path(directory)),
            patch("src.run_bertscore.load_data", return_value=self._fake_dataframe()),
            patch("src.run_bertscore.load_scorer", return_value=scorer),
            patch("src.run_bertscore.find_length_overflows", return_value=[]),
            patch("builtins.print"),
        ):
            run_bertscore()
            output_dir = Path(directory)
            self.assertTrue((output_dir / "bertscore_final.json").exists())
            self.assertTrue((output_dir / "bertscore_final.csv").exists())

    def test_does_not_modify_other_step_output_files(self):
        scorer = MagicMock()
        scorer.hash = "test-hash"
        scorer.score.return_value = (torch.tensor([0.9, 0.8]), torch.tensor([0.7, 0.6]), torch.tensor([0.85, 0.75]))
        with (
            TemporaryDirectory() as directory,
            patch("src.run_bertscore.OUTPUT_DIR", Path(directory)),
            patch("src.run_bertscore.load_data", return_value=self._fake_dataframe()),
            patch("src.run_bertscore.load_scorer", return_value=scorer),
            patch("src.run_bertscore.find_length_overflows", return_value=[]),
            patch("builtins.print"),
        ):
            output_dir = Path(directory)
            untouched = output_dir / "geval_results_final.json"
            untouched.write_text('{"results": []}')
            before = untouched.read_text()
            run_bertscore()
            self.assertEqual(untouched.read_text(), before)

    def test_rejects_rows_with_empty_answer_text_before_scoring(self):
        rows = [{"id": 1, "person_id": "p1", "question": "Q?", "human_answers": "", "ai_answers": "A"}]
        with (
            patch("src.run_bertscore.load_data", return_value=pd.DataFrame(rows)),
            patch("src.run_bertscore.load_scorer") as load_scorer,
        ):
            with self.assertRaises(BERTScoreError):
                run_bertscore()
            load_scorer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
