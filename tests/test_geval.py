import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pandas as pd
from pydantic import ValidationError

from src.config import load_config
from src.geval_evaluator import (
    EvaluationError, PROMPT_VERSION, build_prompt,
    cache_key, evaluate_pair, validate_evidence,
)
from src.run_geval import (
    BEHAVIOR_CALIBRATION_ROWS, MOTIVATION_CALIBRATION_ROWS, PILOT_ROWS,
    describe_failure, run_behavior_calibration, run_full,
    run_motivation_calibration, run_pilot,
)
from src.schemas import DimensionEvaluation, FidelityJudgment


def dimension(**changes) -> DimensionEvaluation:
    values = {
        "applicable": True,
        "score": 5,
        "rationale": "The stated choice is preserved.",
        "human_evidence": "option 2",
        "ai_evidence": "option 2",
        "confidence": "high",
    }
    return DimensionEvaluation(**(values | changes))


class SchemaTests(unittest.TestCase):
    def test_applicable_score_must_be_integer_one_to_five(self):
        for score in (0, 6, None, True, 3.5, "5"):
            with self.subTest(score=score), self.assertRaises(ValidationError):
                dimension(score=score)
        for score in range(1, 6):
            self.assertEqual(dimension(score=score).score, score)

    def test_na_requires_null_score(self):
        for score in range(1, 6):
            with self.subTest(score=score), self.assertRaises(ValidationError):
                dimension(applicable=False, score=score)
        result = dimension(applicable=False, score=None, human_evidence=None, ai_evidence=None)
        self.assertIsNone(result.score)

    def test_applicable_evidence_and_rationale_are_required(self):
        for changes in ({"human_evidence": None}, {"ai_evidence": " "}, {"rationale": " "}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                dimension(**changes)

    def test_valid_pair_requires_applicable_core(self):
        values = {name: dimension() for name in FidelityJudgment.model_fields}
        values["core"] = dimension(applicable=False, score=None)
        with self.assertRaises(ValidationError):
            FidelityJudgment(**values)


class EvaluatorTests(unittest.TestCase):
    @staticmethod
    def calibrated_prompt() -> str:
        return " ".join(build_prompt().split())

    def test_calibration_uses_a_new_prompt_version(self):
        self.assertEqual(PROMPT_VERSION, "fidelity-v4")
        self.assertIn("fidelity-v4", build_prompt() + PROMPT_VERSION)

    def test_behavior_without_preference_and_preference_without_behavior(self):
        prompt = self.calibrated_prompt()
        self.assertIn("does not by itself establish a preference", prompt)
        self.assertIn("Behavior = 1", prompt)
        self.assertIn("Behavior = applicable and Preference = N/A", prompt)
        self.assertIn("Preference = applicable, but Behavior = N/A", prompt)
        self.assertIn("A preference such as “I prefer fragrance-free lotion” establishes Preference = applicable", prompt)
        self.assertIn("Behavior = N/A without evidence of actual purchase, use, intended action", prompt)
        self.assertIn("does not by itself establish a preference for buying as needed", prompt)

    def test_hypothetical_product_design_is_preference_not_behavior(self):
        prompt = self.calibrated_prompt()
        self.assertIn("hypothetical product design is", prompt)
        self.assertIn("normally preference evidence, not behavioral evidence", prompt)
        self.assertIn("Mark Behavior N/A unless the human also describes actual behavior", prompt)

    def test_hypothetical_or_forced_choice_selection_is_preference_not_behavior(self):
        prompt = self.calibrated_prompt()
        self.assertIn(
            "This principle extends beyond product design to any hypothetical or "
            "forced-choice selection, such as choosing a celebrity or an option among "
            "alternatives.",
            prompt,
        )
        self.assertIn("A hypothetical or forced-choice selection does not count as Behavior by itself", prompt)
        self.assertIn(
            "Behavior is applicable only when the human also provides evidence of an actual "
            "action, a repeated habit, an intended real-world action, a behavioral tendency, or "
            "a concrete decision pattern beyond the hypothetical selection itself.",
            prompt,
        )
        self.assertIn(
            "if I had to pick one, I would choose X,” “maybe X or Y,” or “I would prefer X” "
            "should normally be treated as Preference, not Behavior, unless the statement "
            "clearly describes an intended real-world action or behavioral tendency.",
            prompt,
        )

    def test_applicability_must_come_from_human_evidence(self):
        prompt = self.calibrated_prompt()
        self.assertIn("Applicability must be based on evidence in the HUMAN answer", prompt)
        self.assertIn("The question, the AI answer, or an indirect inference cannot", prompt)
        self.assertIn("An AI-only reason, or an indirect inference, cannot make it applicable", prompt)

    def test_severity_calibration_distinguishes_reversal_from_mismatch(self):
        prompt = self.calibrated_prompt()
        self.assertIn("Score 1 is reserved for a direct reversal", prompt)
        self.assertIn("Score 2 is a major mismatch that is not a clean opposite", prompt)
        self.assertIn("Core Fidelity, use 1 when the central meaning is directly opposite/reversed", prompt)
        self.assertIn("loss of uncertainty, conditions, qualifiers, or weak", prompt)
        self.assertIn("Nuance is normally 2", prompt)

    def test_motivation_reversal_requires_the_same_negated_driver(self):
        prompt = self.calibrated_prompt()
        self.assertIn("Different reasons are not automatically opposites", prompt)
        self.assertIn(
            "Score Motivation = 1 only when the AI explicitly reverses or negates the same decision",
            prompt,
        )
        self.assertIn(
            "I buy it because it is cheap” versus AI “I avoid cheap products because a low price is a "
            "negative for me”: Motivation = 1, because the AI reverses the same price driver.",
            prompt,
        )

    def test_motivation_substitution_is_a_mismatch_not_a_reversal(self):
        prompt = self.calibrated_prompt()
        self.assertIn(
            "Score Motivation = 2 when the AI instead substitutes a different, unrelated, invented, "
            "or mismatched reason",
            prompt,
        )
        self.assertIn(
            "someone for their sustainability values: Motivation = 2, not 1, because these are "
            "different decision drivers, not a stated opposite of the same one.",
            prompt,
        )
        self.assertIn(
            "avoiding frequent shopping trips: Motivation = 2, not 1, because the reasons are "
            "unrelated rather than contradictory.",
            prompt,
        )
        self.assertIn(
            "Use 2 when the AI substitutes a different, unrelated, or invented reason, even for the "
            "same action or choice; a different reason is a mismatch, not an opposite.",
            prompt,
        )

    def test_motivation_severity_does_not_leak_from_other_dimensions(self):
        prompt = self.calibrated_prompt()
        self.assertIn(
            "Motivation severity is judged independently from Core, Behavior, Preference, and Nuance",
            prompt,
        )
        self.assertIn(
            "a direct contradiction in one of those dimensions does not by itself make Motivation = 1",
            prompt,
        )
        self.assertIn(
            "When the human's reason and the AI's reason explain different actions or decisions, treat "
            "this as a major mismatch (2), not a direct reversal (1), unless the AI also explicitly "
            "states the opposite of that same underlying driver.",
            prompt,
        )

    def test_cache_key_changes_when_prompt_version_changes(self):
        inputs = {"question": "Which?", "human_answers": "option 2", "ai_answers": "option 2"}
        with patch("src.geval_evaluator.PROMPT_VERSION", "fidelity-v2"):
            v2_key = cache_key(inputs, "model-a", "identical prompt text")
        with patch("src.geval_evaluator.PROMPT_VERSION", "fidelity-v3"):
            v3_key = cache_key(inputs, "model-a", "identical prompt text")
        with patch("src.geval_evaluator.PROMPT_VERSION", "fidelity-v4"):
            v4_key = cache_key(inputs, "model-a", "identical prompt text")
        self.assertEqual(len({v2_key, v3_key, v4_key}), 3)
        self.assertEqual(cache_key(inputs, "model-a", "identical prompt text"), v4_key)
        self.assertEqual(PROMPT_VERSION, "fidelity-v4")

    @patch("src.config.load_dotenv")
    def test_missing_configuration_is_clear(self, dotenv):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY is missing"):
                load_config()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder"}, clear=True):
            with self.assertRaisesRegex(ValueError, "EVALUATOR_MODEL is missing"):
                load_config()

    def test_prompt_excludes_other_response_examples(self):
        prompt = self.calibrated_prompt()
        self.assertNotIn("## Illustrative examples", prompt)
        self.assertNotIn("Keira Knightley", prompt)
        self.assertIn("## 5. Nuance Preservation", prompt)
        self.assertIn("reference-unverifiable", prompt)

    def test_cache_changes_with_inputs_model_and_prompt(self):
        inputs = {"question": "Which?", "human_answers": "option 2", "ai_answers": "option 2"}
        original = cache_key(inputs, "model-a", "prompt")
        self.assertEqual(original, cache_key(inputs, "model-a", "prompt"))
        self.assertNotEqual(original, cache_key(inputs | {"ai_answers": "option 1"}, "model-a", "prompt"))
        self.assertNotEqual(original, cache_key(inputs, "model-b", "prompt"))
        self.assertNotEqual(original, cache_key(inputs, "model-a", "revised"))

    def test_only_pair_sent_and_fabricated_evidence_rejected(self):
        client = MagicMock()
        judgment = FidelityJudgment(**{name: dimension() for name in FidelityJudgment.model_fields})
        client.responses.parse.return_value.status = "completed"
        client.responses.parse.return_value.output_parsed = judgment
        inputs = {"question": "Which?", "human_answers": "option 2", "ai_answers": "option 2"}
        evaluate_pair(client, "gpt-4.1", "rubric", inputs)
        request = client.responses.parse.call_args.kwargs
        self.assertEqual(request["temperature"], 0)
        self.assertFalse(request["store"])
        with self.assertRaisesRegex(ValueError, "Only the question"):
            evaluate_pair(client, "gpt-4.1", "rubric", inputs | {"person_id": "private"})
        judgment.core.ai_evidence = "invented quote"
        with self.assertRaisesRegex(ValueError, "verbatim"):
            evaluate_pair(client, "gpt-4.1", "rubric", inputs)

    def test_failure_messages_do_not_echo_exception_contents(self):
        failure = describe_failure(ValueError("SECRET-MUST-NOT-APPEAR"))
        self.assertNotIn("SECRET-MUST-NOT-APPEAR", str(failure))
        self.assertEqual(failure["error_type"], "ValueError")

    def test_evidence_accepts_quotes_and_ordered_ellipsis_excerpts(self):
        inputs = {"human_answers": "option 2 is affordable and works well", "ai_answers": "option 2"}
        for evidence in (
            "“option 2”", '"option 2"', "“option 2...works well”", "option 2 … works well",
            "“option 2”; “works well”", '"option 2" and "works well"',
        ):
            with self.subTest(evidence=evidence):
                judgment = FidelityJudgment(**{
                    name: dimension(human_evidence=evidence) for name in FidelityJudgment.model_fields
                })
                checked = validate_evidence(judgment, inputs)
                self.assertFalse(checked.core.human_evidence.startswith("“"))

    def test_evidence_rejects_fabrication_reversal_and_empty_quotes(self):
        inputs = {"human_answers": "option 2 is affordable and works well", "ai_answers": "option 2"}
        for evidence in (
            "option 2...lasts forever", "works well...option 2", "…", "“”",
            "“option 2” therefore lasts forever", "“works well”; “option 2”",
        ):
            with self.subTest(evidence=evidence):
                judgment = FidelityJudgment(**{
                    name: dimension(human_evidence=evidence) for name in FidelityJudgment.model_fields
                })
                with self.assertRaises(EvaluationError) as raised:
                    validate_evidence(judgment, inputs)
                self.assertIn("core.human_evidence", describe_failure(raised.exception)["error_message"])

    def test_incomplete_response_and_refusal_are_distinct_failures(self):
        client = MagicMock()
        inputs = {"question": "Which?", "human_answers": "option 2", "ai_answers": "option 2"}
        response = client.responses.parse.return_value
        response.status = "incomplete"
        with self.assertRaisesRegex(EvaluationError, "incomplete"):
            evaluate_pair(client, "gpt-4.1", "rubric", inputs)
        response.status = "completed"
        response.output_parsed = None
        with self.assertRaisesRegex(EvaluationError, "no parsed judgment"):
            evaluate_pair(client, "gpt-4.1", "rubric", inputs)

    def test_pilot_checkpoints_successes_and_retries_only_failed_rows(self):
        judgment = FidelityJudgment(**{name: dimension() for name in FidelityJudgment.model_fields})
        rows = pd.DataFrame([
            {"id": row_id, "person_id": "person", "question": f"Question {row_id}",
             "human_answers": "option 2", "ai_answers": "option 2"}
            for row_id in PILOT_ROWS
        ])
        with (
            TemporaryDirectory() as directory,
            patch("src.run_geval.OUTPUT_DIR", Path(directory)),
            patch("src.run_geval.load_config", return_value=("test-placeholder", "gpt-4.1")),
            patch("src.run_geval.load_data", return_value=rows),
            patch("src.run_geval.OpenAI"),
            patch("builtins.print"),
            patch("src.run_geval.evaluate_pair") as evaluate,
        ):
            all_results_json = Path(directory) / f"geval_pilot_results_{PROMPT_VERSION}_all.json"
            all_results_csv = Path(directory) / f"geval_pilot_results_{PROMPT_VERSION}_all.csv"
            evaluate.side_effect = [judgment, ValueError("bad output"), judgment, judgment, judgment]
            self.assertEqual(run_pilot(), 1)
            saved = json.loads(all_results_json.read_text())
            failed = saved["results"][1]
            self.assertEqual(failed["row_id"], 10)
            self.assertEqual(failed["status"], "failed")
            self.assertNotIn("core", failed)
            evaluate.reset_mock(side_effect=True)
            evaluate.return_value = judgment
            self.assertEqual(run_pilot(), 0)
            evaluate.assert_called_once()
            saved = json.loads(all_results_json.read_text())
            self.assertEqual(len(saved["results"]), 5)
            self.assertEqual(sum(row["cached"] for row in saved["results"]), 4)
            self.assertEqual(len(pd.read_csv(all_results_csv)), 5)

    def test_targeted_rows_keep_legacy_outputs_and_cache_entries(self):
        judgment = FidelityJudgment(**{name: dimension() for name in FidelityJudgment.model_fields})
        rows = pd.DataFrame([
            {"id": row_id, "person_id": "person", "question": f"Question {row_id}",
             "human_answers": "option 2", "ai_answers": "option 2"}
            for row_id in PILOT_ROWS
        ])
        with (
            TemporaryDirectory() as directory,
            patch("src.run_geval.OUTPUT_DIR", Path(directory)),
            patch("src.run_geval.load_config", return_value=("test-placeholder", "gpt-4.1")),
            patch("src.run_geval.load_data", return_value=rows),
            patch("src.run_geval.OpenAI"),
            patch("builtins.print"),
            patch("src.run_geval.evaluate_pair", return_value=judgment) as evaluate,
        ):
            output_dir = Path(directory)
            legacy_json = output_dir / "geval_pilot_results.json"
            legacy_csv = output_dir / "geval_pilot_results.csv"
            legacy_json.write_text('{"results": "historical v1"}')
            legacy_csv.write_text("historical v1\n")
            cache_path = output_dir / ".geval_cache.json"
            cache_path.write_text('{"legacy-v1-key": {"score": 3}}')

            self.assertEqual(run_pilot([10, 20, 24]), 0)
            self.assertEqual(
                [call.args[3]["question"] for call in evaluate.call_args_list],
                ["Question 10", "Question 20", "Question 24"],
            )
            results_path = output_dir / f"geval_pilot_results_{PROMPT_VERSION}_rows_10-20-24.json"
            results = json.loads(results_path.read_text())["results"]
            self.assertEqual([record["row_id"] for record in results], [10, 20, 24])
            self.assertEqual([record["evidence_format"] for record in results], [
                "single-excerpt-v1", "single-excerpt-v1", "single-excerpt-v1"
            ])
            self.assertEqual(legacy_json.read_text(), '{"results": "historical v1"}')
            self.assertEqual(legacy_csv.read_text(), "historical v1\n")
            self.assertIn("legacy-v1-key", json.loads(cache_path.read_text()))

    def test_full_run_covers_every_row_and_uses_separate_output_files(self):
        judgment = FidelityJudgment(**{name: dimension() for name in FidelityJudgment.model_fields})
        rows = pd.DataFrame([
            {"id": row_id, "person_id": "person", "question": f"Question {row_id}",
             "human_answers": "option 2", "ai_answers": "option 2"}
            for row_id in range(1, 31)
        ])
        with (
            TemporaryDirectory() as directory,
            patch("src.run_geval.OUTPUT_DIR", Path(directory)),
            patch("src.run_geval.load_config", return_value=("test-placeholder", "gpt-4.1")),
            patch("src.run_geval.load_data", return_value=rows),
            patch("src.run_geval.OpenAI"),
            patch("builtins.print"),
            patch("src.run_geval.evaluate_pair", return_value=judgment) as evaluate,
        ):
            output_dir = Path(directory)
            pilot_json = output_dir / "geval_pilot_results.json"
            pilot_json.write_text('{"results": "historical v1"}')

            self.assertEqual(run_full(), 0)

            evaluate.assert_called()
            self.assertEqual(evaluate.call_count, 30)
            results_path = output_dir / f"geval_results_{PROMPT_VERSION}_full.json"
            results = json.loads(results_path.read_text())["results"]
            self.assertEqual([record["row_id"] for record in results], list(range(1, 31)))
            self.assertTrue(all(record["status"] == "completed" for record in results))
            self.assertTrue(all("selection_reason" not in record for record in results))
            self.assertEqual(len(pd.read_csv(output_dir / f"geval_results_{PROMPT_VERSION}_full.csv")), 30)
            self.assertEqual(pilot_json.read_text(), '{"results": "historical v1"}')

    def test_motivation_calibration_run_covers_only_the_selected_rows(self):
        judgment = FidelityJudgment(**{name: dimension() for name in FidelityJudgment.model_fields})
        rows = pd.DataFrame([
            {"id": row_id, "person_id": "person", "question": f"Question {row_id}",
             "human_answers": "option 2", "ai_answers": "option 2"}
            for row_id in range(1, 31)
        ])
        with (
            TemporaryDirectory() as directory,
            patch("src.run_geval.OUTPUT_DIR", Path(directory)),
            patch("src.run_geval.load_config", return_value=("test-placeholder", "gpt-4.1")),
            patch("src.run_geval.load_data", return_value=rows),
            patch("src.run_geval.OpenAI"),
            patch("builtins.print"),
            patch("src.run_geval.evaluate_pair", return_value=judgment) as evaluate,
        ):
            output_dir = Path(directory)
            full_json = output_dir / f"geval_results_{PROMPT_VERSION}_full.json"
            full_json.write_text('{"results": "historical fidelity-v2 full run"}')

            self.assertEqual(run_motivation_calibration(), 0)

            expected_ids = list(MOTIVATION_CALIBRATION_ROWS)
            self.assertEqual(evaluate.call_count, len(expected_ids))
            suffix = "-".join(map(str, expected_ids))
            results_path = output_dir / f"geval_calibration_results_{PROMPT_VERSION}_motivation_rows_{suffix}.json"
            results = json.loads(results_path.read_text())["results"]
            self.assertEqual([record["row_id"] for record in results], expected_ids)
            self.assertEqual(
                [record["selection_reason"] for record in results],
                list(MOTIVATION_CALIBRATION_ROWS.values()),
            )
            self.assertEqual(full_json.read_text(), '{"results": "historical fidelity-v2 full run"}')

    def test_behavior_calibration_run_covers_only_the_selected_rows(self):
        judgment = FidelityJudgment(**{name: dimension() for name in FidelityJudgment.model_fields})
        rows = pd.DataFrame([
            {"id": row_id, "person_id": "person", "question": f"Question {row_id}",
             "human_answers": "option 2", "ai_answers": "option 2"}
            for row_id in range(1, 31)
        ])
        with (
            TemporaryDirectory() as directory,
            patch("src.run_geval.OUTPUT_DIR", Path(directory)),
            patch("src.run_geval.load_config", return_value=("test-placeholder", "gpt-4.1")),
            patch("src.run_geval.load_data", return_value=rows),
            patch("src.run_geval.OpenAI"),
            patch("builtins.print"),
            patch("src.run_geval.evaluate_pair", return_value=judgment) as evaluate,
        ):
            output_dir = Path(directory)
            motivation_json = output_dir / f"geval_calibration_results_{PROMPT_VERSION}_motivation_rows_12-13-14-15-16-18-24.json"
            motivation_json.write_text('{"results": "historical fidelity-v3 motivation calibration"}')

            self.assertEqual(run_behavior_calibration(), 0)

            expected_ids = list(BEHAVIOR_CALIBRATION_ROWS)
            self.assertEqual(evaluate.call_count, len(expected_ids))
            suffix = "-".join(map(str, expected_ids))
            results_path = output_dir / f"geval_calibration_results_{PROMPT_VERSION}_behavior_rows_{suffix}.json"
            results = json.loads(results_path.read_text())["results"]
            self.assertEqual([record["row_id"] for record in results], expected_ids)
            self.assertEqual(
                [record["selection_reason"] for record in results],
                list(BEHAVIOR_CALIBRATION_ROWS.values()),
            )
            self.assertEqual(
                motivation_json.read_text(), '{"results": "historical fidelity-v3 motivation calibration"}'
            )


if __name__ == "__main__":
    unittest.main()
