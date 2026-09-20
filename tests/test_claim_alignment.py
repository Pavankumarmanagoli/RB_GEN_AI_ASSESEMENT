import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pandas as pd
from pydantic import ValidationError

from src.claim_aligner import (
    ALIGNMENT_PROMPT,
    ALIGNMENT_PROMPT_VERSION,
    AlignmentError,
    align_claims,
    cache_key,
    validate_alignment,
)
from src.claim_extractor import CLAIM_PROMPT_VERSION
from src.run_claim_alignment import (
    _load_extracted_claims,
    _row_metrics,
    describe_failure,
    run_alignment,
)
from src.schemas import AlignmentJudgment, ClaimAlignment, ClaimAlignmentResult, UnsupportedAIClaim


def alignment(**changes) -> ClaimAlignment:
    values = {
        "human_claim_id": "h1", "ai_claim_id": "a1", "label": "aligned", "rationale": "Matches.",
    }
    return ClaimAlignment(**(values | changes))


def unsupported(claim_id: str = "a2") -> UnsupportedAIClaim:
    return UnsupportedAIClaim(ai_claim_id=claim_id, rationale="No human counterpart.")


class SchemaTests(unittest.TestCase):
    def test_missing_label_requires_null_ai_claim_id(self):
        with self.assertRaises(ValidationError):
            alignment(label="missing", ai_claim_id="a1")
        result = alignment(label="missing", ai_claim_id=None)
        self.assertIsNone(result.ai_claim_id)

    def test_non_missing_labels_require_an_ai_claim_id(self):
        for label in ("aligned", "partial", "contradicted"):
            with self.subTest(label=label), self.assertRaises(ValidationError):
                alignment(label=label, ai_claim_id=None)

    def test_unsupported_ai_claim_requires_a_rationale(self):
        with self.assertRaises(ValidationError):
            UnsupportedAIClaim(ai_claim_id="a1", rationale="")

    def test_claim_alignment_result_attaches_row_id_locally(self):
        judgment = AlignmentJudgment(alignments=[alignment()], unsupported_ai_claims=[])
        result = ClaimAlignmentResult(row_id=7, **judgment.model_dump())
        self.assertEqual(result.row_id, 7)
        self.assertEqual(result.alignments[0].label, "aligned")


class ValidateAlignmentTests(unittest.TestCase):
    def test_accepts_a_fully_consistent_judgment(self):
        judgment = AlignmentJudgment(
            alignments=[alignment(human_claim_id="h1", ai_claim_id="a1")],
            unsupported_ai_claims=[unsupported("a2")],
        )
        validate_alignment(judgment, human_ids={"h1"}, ai_ids={"a1", "a2"})

    def test_allows_one_ai_claim_to_support_multiple_human_claims(self):
        judgment = AlignmentJudgment(
            alignments=[
                alignment(human_claim_id="h1", ai_claim_id="a1"),
                alignment(human_claim_id="h2", ai_claim_id="a1"),
            ],
            unsupported_ai_claims=[],
        )
        validate_alignment(judgment, human_ids={"h1", "h2"}, ai_ids={"a1"})

    def test_rejects_fabricated_human_claim_id(self):
        judgment = AlignmentJudgment(
            alignments=[alignment(human_claim_id="h9", ai_claim_id="a1")], unsupported_ai_claims=[],
        )
        with self.assertRaisesRegex(AlignmentError, "unknown human_claim_id"):
            validate_alignment(judgment, human_ids={"h1"}, ai_ids={"a1"})

    def test_rejects_fabricated_ai_claim_id(self):
        judgment = AlignmentJudgment(
            alignments=[alignment(human_claim_id="h1", ai_claim_id="a9")], unsupported_ai_claims=[],
        )
        with self.assertRaisesRegex(AlignmentError, "unknown ai_claim_id"):
            validate_alignment(judgment, human_ids={"h1"}, ai_ids={"a1"})

    def test_rejects_duplicate_human_claim_labels(self):
        judgment = AlignmentJudgment(
            alignments=[
                alignment(human_claim_id="h1", ai_claim_id="a1"),
                alignment(human_claim_id="h1", ai_claim_id="a2", label="partial"),
            ],
            unsupported_ai_claims=[],
        )
        with self.assertRaisesRegex(AlignmentError, "more than once"):
            validate_alignment(judgment, human_ids={"h1"}, ai_ids={"a1", "a2"})

    def test_rejects_incomplete_human_claim_coverage(self):
        judgment = AlignmentJudgment(
            alignments=[alignment(human_claim_id="h1", ai_claim_id="a1")], unsupported_ai_claims=[],
        )
        with self.assertRaisesRegex(AlignmentError, "missing labels for human claims"):
            validate_alignment(judgment, human_ids={"h1", "h2"}, ai_ids={"a1"})

    def test_rejects_ai_claim_both_referenced_and_unsupported(self):
        judgment = AlignmentJudgment(
            alignments=[alignment(human_claim_id="h1", ai_claim_id="a1")],
            unsupported_ai_claims=[unsupported("a1")],
        )
        with self.assertRaisesRegex(AlignmentError, "both referenced and marked unsupported"):
            validate_alignment(judgment, human_ids={"h1"}, ai_ids={"a1"})

    def test_rejects_ai_claim_neither_referenced_nor_unsupported(self):
        judgment = AlignmentJudgment(
            alignments=[alignment(human_claim_id="h1", ai_claim_id="a1")], unsupported_ai_claims=[],
        )
        with self.assertRaisesRegex(AlignmentError, "neither referenced nor marked unsupported"):
            validate_alignment(judgment, human_ids={"h1"}, ai_ids={"a1", "a2"})


class PromptTests(unittest.TestCase):
    @staticmethod
    def flattened_prompt() -> str:
        return " ".join(ALIGNMENT_PROMPT.split())

    def test_prompt_uses_a_dedicated_version_distinct_from_extraction(self):
        self.assertEqual(ALIGNMENT_PROMPT_VERSION, "alignment-v2")
        self.assertNotEqual(ALIGNMENT_PROMPT_VERSION, CLAIM_PROMPT_VERSION)

    def test_prompt_defines_all_labels(self):
        prompt = self.flattened_prompt()
        self.assertIn("aligned: the AI claim preserves the meaning of the human claim.", prompt)
        self.assertIn("partial: the claims overlap meaningfully, but an important detail", prompt)
        self.assertIn(
            "contradicted: the Human and AI claims cannot both reasonably be true at the same "
            "time, under the same scope, conditions, and context.",
            prompt,
        )
        self.assertIn("missing: no AI claim expresses this human claim; ai_claim_id must be null.", prompt)
        self.assertIn("identify every AI claim that was not used to support any human claim's label "
                      "and list it as unsupported", prompt)

    def test_prompt_requires_incompatibility_not_just_tension_for_contradiction(self):
        prompt = self.flattened_prompt()
        self.assertIn("Do not use contradicted merely because the claims differ in emphasis", prompt)
        self.assertIn("include different additional details", prompt)
        self.assertIn("are in some tension but could still both be true", prompt)
        self.assertIn("discuss different reasons for related behaviors", prompt)
        self.assertIn(
            "unless one claim explicitly negates or is logically incompatible with the other",
            prompt,
        )
        self.assertIn("Clear opposites and incompatible forced choices", prompt)

    def test_prompt_examples_distinguish_negated_reason_from_compatible_tension(self):
        prompt = self.flattened_prompt()
        self.assertIn(
            "because the AI asserts a specific reason exists where the human claim denies "
            "that any specific reason exists",
            prompt,
        )
        self.assertIn("do not automatically label this contradicted", prompt)
        self.assertIn(
            "rotating among a small set of brands can include long-term use of one of them",
            prompt,
        )

    def test_prompt_requires_exactly_one_label_per_human_claim(self):
        self.assertIn(
            "Every human claim must receive exactly one label. Do not skip a human claim and do "
            "not assign more than one label to the same human claim.",
            self.flattened_prompt(),
        )

    def test_prompt_requires_full_ai_claim_accounting(self):
        self.assertIn(
            "Every AI claim must end up either referenced by at least one alignment label, or "
            "listed as unsupported — never both, and never neither.",
            self.flattened_prompt(),
        )

    def test_prompt_preserves_negation_frequency_uncertainty_and_scope(self):
        self.assertIn(
            "Preserve sensitivity to negation, frequency, uncertainty, quantities, named entities, "
            "temporal information, conditionality, hypothetical framing, and scope.",
            self.flattened_prompt(),
        )

    def test_prompt_forbids_topic_only_matching_and_separates_action_from_reason(self):
        prompt = self.flattened_prompt()
        self.assertIn(
            "a shared action does not make its reason aligned, and a shared reason does not make "
            "its action aligned",
            prompt,
        )
        self.assertIn("Do not force a match based only on topic overlap or shared entities.", prompt)

    def test_prompt_allows_one_to_many_support_without_forcing_pairings(self):
        self.assertIn(
            "A single AI claim may justifiably support more than one human claim, but do not "
            "invent artificial one-to-one pairings that distort meaning.",
            self.flattened_prompt(),
        )


class AlignClaimsTests(unittest.TestCase):
    HUMAN = [{"claim_id": "h1", "claim": "The respondent usually buys online."}]
    AI = [{"claim_id": "a1", "claim": "The respondent typically buys online."}]

    def test_cache_key_changes_with_claims_model_and_prompt(self):
        original = cache_key(self.HUMAN, self.AI, "model-a", "prompt")
        self.assertEqual(original, cache_key(self.HUMAN, self.AI, "model-a", "prompt"))
        other_ai = [{"claim_id": "a1", "claim": "different"}]
        self.assertNotEqual(original, cache_key(self.HUMAN, other_ai, "model-a", "prompt"))
        self.assertNotEqual(original, cache_key(self.HUMAN, self.AI, "model-b", "prompt"))
        self.assertNotEqual(original, cache_key(self.HUMAN, self.AI, "model-a", "revised"))

    def test_align_claims_sends_claim_ids_and_texts(self):
        client = MagicMock()
        client.responses.parse.return_value.status = "completed"
        client.responses.parse.return_value.output_parsed = AlignmentJudgment(
            alignments=[alignment()], unsupported_ai_claims=[],
        )
        align_claims(client, "gpt-4.1", "prompt", self.HUMAN, self.AI)
        request = client.responses.parse.call_args.kwargs
        self.assertEqual(json.loads(request["input"]), {"human_claims": self.HUMAN, "ai_claims": self.AI})
        self.assertEqual(request["temperature"], 0)
        self.assertFalse(request["store"])
        self.assertIs(request["text_format"], AlignmentJudgment)

    def test_align_claims_rejects_empty_claim_lists(self):
        client = MagicMock()
        with self.assertRaisesRegex(AlignmentError, "must be nonempty"):
            align_claims(client, "gpt-4.1", "prompt", [], self.AI)
        with self.assertRaisesRegex(AlignmentError, "must be nonempty"):
            align_claims(client, "gpt-4.1", "prompt", self.HUMAN, [])

    def test_align_claims_rejects_fabricated_ids_from_the_model(self):
        client = MagicMock()
        client.responses.parse.return_value.status = "completed"
        client.responses.parse.return_value.output_parsed = AlignmentJudgment(
            alignments=[alignment(human_claim_id="h1", ai_claim_id="a99")], unsupported_ai_claims=[],
        )
        with self.assertRaisesRegex(AlignmentError, "unknown ai_claim_id"):
            align_claims(client, "gpt-4.1", "prompt", self.HUMAN, self.AI)

    def test_incomplete_response_and_refusal_are_distinct_failures(self):
        client = MagicMock()
        response = client.responses.parse.return_value
        response.status = "incomplete"
        with self.assertRaisesRegex(AlignmentError, "incomplete"):
            align_claims(client, "gpt-4.1", "prompt", self.HUMAN, self.AI)
        response.status = "completed"
        response.output_parsed = None
        with self.assertRaisesRegex(AlignmentError, "no parsed alignment"):
            align_claims(client, "gpt-4.1", "prompt", self.HUMAN, self.AI)

    def test_failure_messages_do_not_echo_exception_contents(self):
        failure = describe_failure(ValueError("SECRET-MUST-NOT-APPEAR"))
        self.assertNotIn("SECRET-MUST-NOT-APPEAR", str(failure))
        self.assertEqual(failure["error_type"], "ValueError")


class RowMetricsTests(unittest.TestCase):
    def test_calculates_coverage_strict_alignment_and_unsupported_rate(self):
        judgment = AlignmentJudgment(
            alignments=[
                alignment(human_claim_id="h1", ai_claim_id="a1", label="aligned"),
                alignment(human_claim_id="h2", ai_claim_id="a2", label="partial"),
                alignment(human_claim_id="h3", ai_claim_id="a3", label="contradicted"),
                alignment(human_claim_id="h4", ai_claim_id=None, label="missing"),
            ],
            unsupported_ai_claims=[unsupported("a4")],
        )
        metrics = _row_metrics(judgment, human_count=4, ai_count=4)
        self.assertEqual(metrics["aligned_count"], 1)
        self.assertEqual(metrics["partial_count"], 1)
        self.assertEqual(metrics["contradicted_count"], 1)
        self.assertEqual(metrics["missing_count"], 1)
        self.assertEqual(metrics["unsupported_count"], 1)
        self.assertAlmostEqual(metrics["claim_coverage_rate"], 0.5)
        self.assertAlmostEqual(metrics["strict_alignment_rate"], 0.25)
        self.assertAlmostEqual(metrics["unsupported_rate"], 0.25)

    def test_handles_zero_denominators_safely(self):
        judgment = AlignmentJudgment(alignments=[], unsupported_ai_claims=[])
        metrics = _row_metrics(judgment, human_count=0, ai_count=0)
        self.assertIsNone(metrics["claim_coverage_rate"])
        self.assertIsNone(metrics["strict_alignment_rate"])
        self.assertIsNone(metrics["unsupported_rate"])


class RunnerTests(unittest.TestCase):
    def test_load_extracted_claims_skips_failed_rows(self):
        extraction = {"results": [
            {"row_id": 1, "person_id": "p", "question": "Q", "human_claims": [], "ai_claims": [], "status": "completed"},
            {"row_id": 2, "status": "failed"},
        ]}
        with TemporaryDirectory() as directory:
            path = Path(directory) / f"claim_extraction_results_{CLAIM_PROMPT_VERSION}_rows_1-2.json"
            path.write_text(json.dumps(extraction))
            with patch("src.run_claim_alignment.OUTPUT_DIR", Path(directory)):
                claims_by_row = _load_extracted_claims()
        self.assertEqual(set(claims_by_row), {1})

    def test_run_alignment_requires_existing_extraction_results(self):
        with (
            TemporaryDirectory() as directory,
            patch("src.run_claim_alignment.OUTPUT_DIR", Path(directory)),
            patch("src.run_claim_alignment.load_config", return_value=("key", "gpt-4.1")),
        ):
            with self.assertRaises(ValueError):
                run_alignment([1])

    def test_run_alignment_checkpoints_and_reuses_cache(self):
        extraction = {"results": [
            {
                "row_id": 1, "person_id": "p1", "question": "Q1", "status": "completed",
                "human_claims": [{"claim_id": "h1", "claim": "Human claim."}],
                "ai_claims": [{"claim_id": "a1", "claim": "AI claim."}],
            },
            {
                "row_id": 2, "person_id": "p2", "question": "Q2", "status": "completed",
                "human_claims": [{"claim_id": "h1", "claim": "Human claim 2."}],
                "ai_claims": [{"claim_id": "a1", "claim": "AI claim 2."}],
            },
        ]}
        judgment = AlignmentJudgment(alignments=[alignment()], unsupported_ai_claims=[])
        with (
            TemporaryDirectory() as directory,
            patch("src.run_claim_alignment.OUTPUT_DIR", Path(directory)),
            patch("src.run_claim_alignment.load_config", return_value=("test-placeholder", "gpt-4.1")),
            patch("src.run_claim_alignment.OpenAI"),
            patch("builtins.print"),
            patch("src.run_claim_alignment.align_claims", return_value=judgment) as align,
        ):
            output_dir = Path(directory)
            (output_dir / f"claim_extraction_results_{CLAIM_PROMPT_VERSION}_rows_1-2.json").write_text(
                json.dumps(extraction)
            )
            self.assertEqual(run_alignment([1, 2]), 0)
            align.assert_called()
            self.assertEqual(align.call_count, 2)
            results_path = output_dir / f"alignment_results_{ALIGNMENT_PROMPT_VERSION}_rows_1-2.json"
            results = json.loads(results_path.read_text())["results"]
            self.assertEqual([r["row_id"] for r in results], [1, 2])
            self.assertFalse(results[0]["cached"])
            self.assertEqual(results[0]["human_claim_count"], 1)
            self.assertEqual(results[0]["aligned_count"], 1)

            align.reset_mock()
            self.assertEqual(run_alignment([1, 2]), 0)
            align.assert_not_called()
            results = json.loads(results_path.read_text())["results"]
            self.assertTrue(results[0]["cached"])
            self.assertTrue(results[1]["cached"])

    def test_full_run_uses_a_distinct_filename(self):
        extraction = {"results": [
            {
                "row_id": row_id, "person_id": "p", "question": "Q", "status": "completed",
                "human_claims": [{"claim_id": "h1", "claim": f"Human claim {row_id}."}],
                "ai_claims": [{"claim_id": "a1", "claim": f"AI claim {row_id}."}],
            }
            for row_id in (1, 2)
        ]}
        judgment = AlignmentJudgment(alignments=[alignment()], unsupported_ai_claims=[])
        with (
            TemporaryDirectory() as directory,
            patch("src.run_claim_alignment.OUTPUT_DIR", Path(directory)),
            patch("src.run_claim_alignment.load_config", return_value=("test-placeholder", "gpt-4.1")),
            patch("src.run_claim_alignment.OpenAI"),
            patch("builtins.print"),
            patch("src.run_claim_alignment.align_claims", return_value=judgment),
        ):
            output_dir = Path(directory)
            (output_dir / f"claim_extraction_results_{CLAIM_PROMPT_VERSION}_full.json").write_text(
                json.dumps(extraction)
            )
            self.assertEqual(run_alignment(None), 0)
            results_path = output_dir / f"alignment_results_{ALIGNMENT_PROMPT_VERSION}_full.json"
            results = json.loads(results_path.read_text())["results"]
            self.assertEqual([r["row_id"] for r in results], [1, 2])


if __name__ == "__main__":
    unittest.main()
