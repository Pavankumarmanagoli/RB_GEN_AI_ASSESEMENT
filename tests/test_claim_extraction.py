"""Tests for claim extraction."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pandas as pd
from pydantic import ValidationError

from src.claim_extractor import (
    CLAIM_EXTRACTION_PROMPT, CLAIM_PROMPT_VERSION, ExtractionError,
    cache_key, extract_claims,
)
from src.run_claim_extraction import describe_failure, run_extraction
from src.schemas import AtomicClaim, ClaimExtractionResult, ClaimList


def claim_list(*claims: str) -> ClaimList:
    return ClaimList(claims=list(claims))


class SchemaTests(unittest.TestCase):
    def test_claim_list_requires_at_least_one_claim(self):
        with self.assertRaises(ValidationError):
            ClaimList(claims=[])
        self.assertEqual(ClaimList(claims=["a claim"]).claims, ["a claim"])

    def test_claim_list_forbids_extra_fields(self):
        with self.assertRaises(ValidationError):
            ClaimList(claims=["a claim"], extra="not allowed")

    def test_atomic_claim_requires_nonempty_claim(self):
        with self.assertRaises(ValidationError):
            AtomicClaim(claim_id="h1", claim="")
        claim = AtomicClaim(claim_id="h1", claim="a claim")
        self.assertEqual(claim.claim_id, "h1")

    def test_claim_extraction_result_holds_both_sides(self):
        result = ClaimExtractionResult(
            row_id=1,
            human_claims=[AtomicClaim(claim_id="h1", claim="human claim")],
            ai_claims=[AtomicClaim(claim_id="a1", claim="ai claim")],
        )
        self.assertEqual(result.human_claims[0].claim_id, "h1")
        self.assertEqual(result.ai_claims[0].claim_id, "a1")


class PromptTests(unittest.TestCase):
    @staticmethod
    def flattened_prompt() -> str:
        return " ".join(CLAIM_EXTRACTION_PROMPT.split())

    def test_prompt_uses_a_dedicated_version(self):
        self.assertEqual(CLAIM_PROMPT_VERSION, "claims-v2")

    def test_prompt_requires_atomic_decomposition(self):
        prompt = self.flattened_prompt()
        self.assertIn("exactly one independently comparable proposition", prompt)
        self.assertIn("Split genuinely independent propositions into separate claims", prompt)
        self.assertIn("Do not split a phrase into pieces that only make sense together", prompt)

    def test_prompt_separates_behavior_from_reason_when_meaningful(self):
        self.assertIn(
            "including a stated behavior from its stated reason when both are present and "
            "independently meaningful",
            self.flattened_prompt(),
        )

    def test_prompt_splits_action_from_its_reason_when_independently_true_or_false(self):
        prompt = self.flattened_prompt()
        self.assertIn(
            "extract them as two separate atomic claims: one stating that the action "
            "or event happened, and one stating what the reason for it was",
            prompt,
        )
        self.assertIn(
            "\"I switched brands because my friend recommended one\" should become "
            "approximately: \"The respondent switched brands.\" and \"A friend's "
            "recommendation was the reason for switching brands.\"",
            prompt,
        )
        self.assertIn("Do not merge an action and its reason into a single claim.", prompt)
        self.assertIn(
            "Do not over-split expressions where the reason cannot meaningfully stand "
            "on its own apart from the action it explains.",
            prompt,
        )

    def test_prompt_preserves_negation_frequency_and_uncertainty(self):
        prompt = self.flattened_prompt()
        self.assertIn("negation (for example \"does not usually buy in bulk\"", prompt)
        self.assertIn("frequency and degree (for example \"usually\", \"sometimes\"", prompt)
        self.assertIn("uncertainty and tentativeness (for example \"maybe\", \"I'm not sure\"", prompt)

    def test_prompt_preserves_hypothetical_conditional_framing(self):
        self.assertIn(
            "\"if I had to pick one, I would choose X\" is a hypothetical or forced choice, "
            "not a stated actual behavior",
            self.flattened_prompt(),
        )

    def test_prompt_forbids_unsupported_inference_and_world_knowledge(self):
        prompt = self.flattened_prompt()
        self.assertIn(
            "Do not infer information that is not explicitly stated, and do not add outside "
            "or world knowledge.",
            prompt,
        )
        self.assertIn("never invent a claim to fill a gap", prompt)

    def test_prompt_isolates_the_answer_being_extracted(self):
        self.assertIn(
            "do not use the paired response from the other party", self.flattened_prompt()
        )


class ExtractorTests(unittest.TestCase):
    def test_cache_key_changes_with_question_text_model_and_prompt(self):
        original = cache_key("Which?", "option 2", "model-a", "prompt")
        self.assertEqual(original, cache_key("Which?", "option 2", "model-a", "prompt"))
        self.assertNotEqual(original, cache_key("Which?", "option 1", "model-a", "prompt"))
        self.assertNotEqual(original, cache_key("Different question?", "option 2", "model-a", "prompt"))
        self.assertNotEqual(original, cache_key("Which?", "option 2", "model-b", "prompt"))
        self.assertNotEqual(original, cache_key("Which?", "option 2", "model-a", "revised"))

    def test_extract_claims_sends_only_question_and_answer(self):
        client = MagicMock()
        client.responses.parse.return_value.status = "completed"
        client.responses.parse.return_value.output_parsed = claim_list("a claim")
        extract_claims(client, "gpt-4.1", "prompt", "Which?", "option 2")
        request = client.responses.parse.call_args.kwargs
        self.assertEqual(json.loads(request["input"]), {"question": "Which?", "answer": "option 2"})
        self.assertEqual(request["temperature"], 0)
        self.assertFalse(request["store"])
        self.assertIs(request["text_format"], ClaimList)

    def test_extract_claims_rejects_empty_question_or_text(self):
        client = MagicMock()
        with self.assertRaisesRegex(ExtractionError, "question must be nonempty"):
            extract_claims(client, "gpt-4.1", "prompt", "  ", "option 2")
        with self.assertRaisesRegex(ExtractionError, "answer must be nonempty"):
            extract_claims(client, "gpt-4.1", "prompt", "Which?", "")

    def test_incomplete_response_and_refusal_are_distinct_failures(self):
        client = MagicMock()
        response = client.responses.parse.return_value
        response.status = "incomplete"
        with self.assertRaisesRegex(ExtractionError, "incomplete"):
            extract_claims(client, "gpt-4.1", "prompt", "Which?", "option 2")
        response.status = "completed"
        response.output_parsed = None
        with self.assertRaisesRegex(ExtractionError, "no parsed extraction"):
            extract_claims(client, "gpt-4.1", "prompt", "Which?", "option 2")

    def test_failure_messages_do_not_echo_exception_contents(self):
        failure = describe_failure(ValueError("SECRET-MUST-NOT-APPEAR"))
        self.assertNotIn("SECRET-MUST-NOT-APPEAR", str(failure))
        self.assertEqual(failure["error_type"], "ValueError")


class RunnerTests(unittest.TestCase):
    def test_duplicate_or_unknown_row_ids_are_rejected(self):
        rows = pd.DataFrame([
            {"id": 1, "person_id": "p", "question": "Q", "human_answers": "H", "ai_answers": "A"},
        ])
        with (
            patch("src.run_claim_extraction.load_config", return_value=("key", "gpt-4.1")),
            patch("src.run_claim_extraction.load_data", return_value=rows),
        ):
            with self.assertRaises(ValueError):
                run_extraction([1, 1])
            with self.assertRaises(ValueError):
                run_extraction([999])

    def test_extraction_checkpoints_assigns_stable_ids_and_retries_only_failed_rows(self):
        rows = pd.DataFrame([
            {"id": 1, "person_id": "p1", "question": "Q1", "human_answers": "H1", "ai_answers": "A1"},
            {"id": 2, "person_id": "p2", "question": "Q2", "human_answers": "H2", "ai_answers": "A2"},
        ])
        with (
            TemporaryDirectory() as directory,
            patch("src.run_claim_extraction.OUTPUT_DIR", Path(directory)),
            patch("src.run_claim_extraction.load_config", return_value=("test-placeholder", "gpt-4.1")),
            patch("src.run_claim_extraction.load_data", return_value=rows),
            patch("src.run_claim_extraction.OpenAI"),
            patch("builtins.print"),
            patch("src.run_claim_extraction.extract_claims") as extract,
        ):
            results_path = Path(directory) / f"claim_extraction_results_{CLAIM_PROMPT_VERSION}_rows_1-2.json"

            extract.side_effect = [
                claim_list("human claim 1a", "human claim 1b"),  # row 1 human: succeeds and caches
                ValueError("bad output"),  # row 1 ai: fails, row 1 marked failed
                claim_list("human claim 2"),  # row 2 human
                claim_list("ai claim 2"),  # row 2 ai
            ]
            self.assertEqual(run_extraction([1, 2]), 1)
            results = json.loads(results_path.read_text())["results"]
            self.assertEqual(results[0]["status"], "failed")
            self.assertNotIn("human_claims", results[0])
            self.assertEqual(results[1]["status"], "completed")
            self.assertEqual(
                [c["claim_id"] for c in results[1]["human_claims"]], ["h1"]
            )
            self.assertEqual(
                [c["claim_id"] for c in results[1]["ai_claims"]], ["a1"]
            )
            self.assertFalse(results[1]["human_cached"])
            self.assertFalse(results[1]["ai_cached"])

            # Row 1's human side was cached before its ai side failed; only the
            # missing ai side needs a fresh call on retry.
            extract.reset_mock(side_effect=True)
            extract.side_effect = [claim_list("ai claim 1")]
            self.assertEqual(run_extraction([1, 2]), 0)
            self.assertEqual(extract.call_count, 1)
            results = json.loads(results_path.read_text())["results"]
            self.assertEqual(results[0]["status"], "completed")
            self.assertEqual(
                [(c["claim_id"], c["claim"]) for c in results[0]["human_claims"]],
                [("h1", "human claim 1a"), ("h2", "human claim 1b")],
            )
            self.assertEqual(
                [(c["claim_id"], c["claim"]) for c in results[0]["ai_claims"]],
                [("a1", "ai claim 1")],
            )
            self.assertTrue(results[0]["human_cached"])
            self.assertFalse(results[0]["ai_cached"])
            self.assertTrue(results[1]["human_cached"])
            self.assertTrue(results[1]["ai_cached"])

    def test_full_run_covers_every_row_with_a_distinct_filename(self):
        rows = pd.DataFrame([
            {"id": i, "person_id": "p", "question": f"Q{i}", "human_answers": f"H{i}", "ai_answers": f"A{i}"}
            for i in range(1, 4)
        ])
        with (
            TemporaryDirectory() as directory,
            patch("src.run_claim_extraction.OUTPUT_DIR", Path(directory)),
            patch("src.run_claim_extraction.load_config", return_value=("test-placeholder", "gpt-4.1")),
            patch("src.run_claim_extraction.load_data", return_value=rows),
            patch("src.run_claim_extraction.OpenAI"),
            patch("builtins.print"),
            patch("src.run_claim_extraction.extract_claims", return_value=claim_list("c")),
        ):
            self.assertEqual(run_extraction(None), 0)
            results_path = Path(directory) / f"claim_extraction_results_{CLAIM_PROMPT_VERSION}_full.json"
            results = json.loads(results_path.read_text())["results"]
            self.assertEqual([r["row_id"] for r in results], [1, 2, 3])
            self.assertEqual(
                len(pd.read_csv(Path(directory) / f"claim_extraction_results_{CLAIM_PROMPT_VERSION}_full.csv")), 3
            )


if __name__ == "__main__":
    unittest.main()
