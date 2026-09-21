"""Tests for the NLI cross-check of claim alignment."""
import unittest
from unittest.mock import MagicMock, patch

import torch
from pydantic import ValidationError

from src.nli_validator import NLIError, load_nli_model, predict_nli, resolve_label_map
from src.run_nli_validation import (
    NLI_CALIBRATION_PAIRS,
    compare_with_step4b,
    eligible_pairs,
    select_calibration_pairs,
)
from src.schemas import NLIPrediction


def fake_model(id2label: dict[int, str]) -> MagicMock:
    model = MagicMock()
    model.config.id2label = id2label
    return model


class LabelMapTests(unittest.TestCase):
    def test_normalizes_standard_roberta_mnli_order(self):
        model = fake_model({0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"})
        self.assertEqual(resolve_label_map(model), {0: "contradiction", 1: "neutral", 2: "entailment"})

    def test_does_not_assume_a_fixed_numeric_ordering(self):
        # A checkpoint could plausibly order its classes differently; the map must be
        # read from id2label, never assumed as index 0 = contradiction.
        model = fake_model({0: "ENTAILMENT", 1: "CONTRADICTION", 2: "NEUTRAL"})
        self.assertEqual(resolve_label_map(model), {0: "entailment", 1: "contradiction", 2: "neutral"})

    def test_rejects_unrecognized_labels(self):
        model = fake_model({0: "CONTRADICTION", 1: "NEUTRAL", 2: "SOMETHING_ELSE"})
        with self.assertRaises(NLIError):
            resolve_label_map(model)

    def test_rejects_a_map_missing_one_of_the_three_labels(self):
        model = fake_model({0: "CONTRADICTION", 1: "NEUTRAL", 2: "NEUTRAL"})
        with self.assertRaises(NLIError):
            resolve_label_map(model)


class PredictNliTests(unittest.TestCase):
    LABEL_MAP = {0: "contradiction", 1: "neutral", 2: "entailment"}

    @staticmethod
    def _tokenizer_and_model(logits: torch.Tensor):
        encoded = MagicMock()
        encoded.to.return_value = encoded
        tokenizer = MagicMock(return_value=encoded)
        model_output = MagicMock(logits=logits)
        model = MagicMock(return_value=model_output)
        return tokenizer, model

    def test_premise_is_human_claim_hypothesis_is_ai_claim(self):
        tokenizer, model = self._tokenizer_and_model(torch.tensor([[3.0, 0.0, 0.0]]))
        pairs = [("human claim text", "ai claim text")]
        predict_nli(tokenizer, model, torch.device("cpu"), self.LABEL_MAP, pairs)
        call_args = tokenizer.call_args
        self.assertEqual(call_args.args[0], ["human claim text"])
        self.assertEqual(call_args.args[1], ["ai claim text"])

    def test_probabilities_sum_to_approximately_one(self):
        tokenizer, model = self._tokenizer_and_model(torch.tensor([[2.0, -1.0, 0.5]]))
        [result] = predict_nli(tokenizer, model, torch.device("cpu"), self.LABEL_MAP, [("p", "h")])
        total = result["entailment_prob"] + result["neutral_prob"] + result["contradiction_prob"]
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_confidence_matches_the_predicted_labels_probability(self):
        tokenizer, model = self._tokenizer_and_model(torch.tensor([[0.0, 0.0, 5.0]]))
        [result] = predict_nli(tokenizer, model, torch.device("cpu"), self.LABEL_MAP, [("p", "h")])
        self.assertEqual(result["nli_label"], "entailment")
        self.assertAlmostEqual(result["nli_confidence"], result["entailment_prob"])
        self.assertGreater(result["entailment_prob"], result["neutral_prob"])
        self.assertGreater(result["entailment_prob"], result["contradiction_prob"])

    def test_empty_pairs_returns_empty_without_calling_the_model(self):
        tokenizer, model = self._tokenizer_and_model(torch.tensor([[0.0, 0.0, 0.0]]))
        self.assertEqual(predict_nli(tokenizer, model, torch.device("cpu"), self.LABEL_MAP, []), [])
        tokenizer.assert_not_called()
        model.assert_not_called()

    def test_batching_preserves_input_order(self):
        # Three distinct logit rows, each pointing to a different predicted label.
        logits = torch.tensor([
            [5.0, 0.0, 0.0],   # contradiction
            [0.0, 0.0, 5.0],   # entailment
            [0.0, 5.0, 0.0],   # neutral
        ])
        tokenizer, model = self._tokenizer_and_model(logits)
        pairs = [("p1", "h1"), ("p2", "h2"), ("p3", "h3")]
        results = predict_nli(tokenizer, model, torch.device("cpu"), self.LABEL_MAP, pairs)
        self.assertEqual([r["nli_label"] for r in results], ["contradiction", "entailment", "neutral"])


class NLIPredictionSchemaTests(unittest.TestCase):
    def test_accepts_a_consistent_record(self):
        NLIPrediction(
            nli_label="entailment", entailment_prob=0.8, neutral_prob=0.15,
            contradiction_prob=0.05, nli_confidence=0.8,
        )

    def test_rejects_confidence_not_matching_the_predicted_label(self):
        with self.assertRaises(ValidationError):
            NLIPrediction(
                nli_label="entailment", entailment_prob=0.8, neutral_prob=0.15,
                contradiction_prob=0.05, nli_confidence=0.15,
            )

    def test_rejects_probabilities_not_summing_to_one(self):
        with self.assertRaises(ValidationError):
            NLIPrediction(
                nli_label="entailment", entailment_prob=0.8, neutral_prob=0.8,
                contradiction_prob=0.05, nli_confidence=0.8,
            )


class EligiblePairsTests(unittest.TestCase):
    EXTRACTION = {
        1: {
            "human_claims": [{"claim_id": "h1", "claim": "Human one."}, {"claim_id": "h2", "claim": "Human two."}],
            "ai_claims": [{"claim_id": "a1", "claim": "AI one."}, {"claim_id": "a2", "claim": "AI two."}],
        },
    }

    def test_excludes_missing_human_claims(self):
        alignment = {1: {
            "person_id": "p1",
            "alignments": [
                {"human_claim_id": "h1", "ai_claim_id": "a1", "label": "aligned"},
                {"human_claim_id": "h2", "ai_claim_id": None, "label": "missing"},
            ],
            "unsupported_ai_claims": [],
        }}
        pairs = eligible_pairs(self.EXTRACTION, alignment)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["human_claim_id"], "h1")

    def test_excludes_unsupported_ai_claims(self):
        alignment = {1: {
            "person_id": "p1",
            "alignments": [{"human_claim_id": "h1", "ai_claim_id": "a1", "label": "aligned"}],
            "unsupported_ai_claims": [{"ai_claim_id": "a2", "rationale": "No human counterpart."}],
        }}
        pairs = eligible_pairs(self.EXTRACTION, alignment)
        self.assertEqual(len(pairs), 1)
        self.assertEqual({pair["ai_claim_id"] for pair in pairs}, {"a1"})

    def test_resolves_claim_text_from_frozen_step4_ids(self):
        alignment = {1: {
            "person_id": "p1",
            "alignments": [{"human_claim_id": "h2", "ai_claim_id": "a2", "label": "partial"}],
            "unsupported_ai_claims": [],
        }}
        [pair] = eligible_pairs(self.EXTRACTION, alignment)
        self.assertEqual(pair["human_claim"], "Human two.")
        self.assertEqual(pair["ai_claim"], "AI two.")
        self.assertEqual(pair["step4b_label"], "partial")


class SelectCalibrationPairsTests(unittest.TestCase):
    def test_selects_exactly_the_curated_pairs_in_curated_order(self):
        all_pairs = [
            {"row_id": row_id, "human_claim_id": claim_id, "marker": index}
            for index, (row_id, claim_id) in enumerate(reversed(NLI_CALIBRATION_PAIRS))
        ]
        selected = select_calibration_pairs(all_pairs)
        self.assertEqual(
            [(pair["row_id"], pair["human_claim_id"]) for pair in selected],
            NLI_CALIBRATION_PAIRS,
        )

    def test_raises_if_a_curated_pair_is_absent_from_frozen_step4_output(self):
        incomplete = [
            {"row_id": row_id, "human_claim_id": claim_id}
            for row_id, claim_id in NLI_CALIBRATION_PAIRS[1:]
        ]
        with self.assertRaises(ValueError):
            select_calibration_pairs(incomplete)


class CompareWithStep4BTests(unittest.TestCase):
    def test_aligned_expects_entailment(self):
        self.assertEqual(
            compare_with_step4b("aligned", "entailment"),
            {"expected_nli_signal": "entailment", "agrees_with_step4b": True},
        )
        self.assertEqual(compare_with_step4b("aligned", "neutral")["agrees_with_step4b"], False)

    def test_contradicted_expects_contradiction(self):
        self.assertEqual(
            compare_with_step4b("contradicted", "contradiction"),
            {"expected_nli_signal": "contradiction", "agrees_with_step4b": True},
        )
        self.assertEqual(compare_with_step4b("contradicted", "entailment")["agrees_with_step4b"], False)

    def test_partial_has_no_forced_expectation_and_is_never_disagreement(self):
        for nli_label in ("entailment", "neutral", "contradiction"):
            with self.subTest(nli_label=nli_label):
                result = compare_with_step4b("partial", nli_label)
                self.assertIsNone(result["expected_nli_signal"])
                self.assertIsNone(result["agrees_with_step4b"])


class LoadNliModelTests(unittest.TestCase):
    def test_loads_the_documented_checkpoint_in_eval_mode(self):
        with (
            patch("src.nli_validator.AutoTokenizer") as tokenizer_cls,
            patch("src.nli_validator.AutoModelForSequenceClassification") as model_cls,
        ):
            model = MagicMock()
            model_cls.from_pretrained.return_value = model
            load_nli_model()
            model_cls.from_pretrained.assert_called_once_with("roberta-large-mnli")
            tokenizer_cls.from_pretrained.assert_called_once_with("roberta-large-mnli")
            model.eval.assert_called_once()


if __name__ == "__main__":
    unittest.main()
