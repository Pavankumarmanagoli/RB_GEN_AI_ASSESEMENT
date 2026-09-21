import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DimensionEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    applicable: bool
    score: int | None = Field(ge=1, le=5)
    rationale: str = Field(min_length=1)
    human_evidence: str | None
    ai_evidence: str | None
    confidence: Literal["low", "medium", "high"]

    @model_validator(mode="after")
    def validate_applicability(self) -> Self:
        if self.applicable and self.score is None:
            raise ValueError("Applicable dimensions require a score from 1 to 5.")
        if not self.applicable and self.score is not None:
            raise ValueError("Non-applicable dimensions require score=None.")
        if not self.rationale.strip():
            raise ValueError("A concise rationale is required.")
        if self.applicable and not all(
            evidence and evidence.strip()
            for evidence in (self.human_evidence, self.ai_evidence)
        ):
            raise ValueError("Applicable dimensions require evidence from both answers.")
        return self


class FidelityJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    core: DimensionEvaluation
    behavior: DimensionEvaluation
    preference: DimensionEvaluation
    motivation: DimensionEvaluation
    nuance: DimensionEvaluation

    @model_validator(mode="after")
    def validate_core(self) -> Self:
        if not self.core.applicable:
            raise ValueError("Core Fidelity must apply to a valid response pair.")
        return self


class FidelityEvaluation(FidelityJudgment):
    # Identifiers are attached locally and are never sent to the judge.
    row_id: int | str
    person_id: str


class ClaimList(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    claims: list[str] = Field(min_length=1)


class AtomicClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    # claim_id is assigned locally from extraction order and is never sent to the model.
    claim_id: str
    claim: str = Field(min_length=1)


class ClaimExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    row_id: int | str
    human_claims: list[AtomicClaim]
    ai_claims: list[AtomicClaim]


class ClaimAlignment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    human_claim_id: str
    ai_claim_id: str | None
    label: Literal["aligned", "partial", "contradicted", "missing"]
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ai_claim_id(self) -> Self:
        if self.label == "missing" and self.ai_claim_id is not None:
            raise ValueError("A missing label must have ai_claim_id=null.")
        if self.label != "missing" and self.ai_claim_id is None:
            raise ValueError(f"A {self.label} label requires a non-null ai_claim_id.")
        return self


class UnsupportedAIClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    ai_claim_id: str
    rationale: str = Field(min_length=1)


class AlignmentJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    alignments: list[ClaimAlignment]
    unsupported_ai_claims: list[UnsupportedAIClaim]


class ClaimAlignmentResult(AlignmentJudgment):
    # row_id is attached locally and is never sent to the model.
    row_id: int | str


class NLIPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    nli_label: Literal["entailment", "neutral", "contradiction"]
    entailment_prob: float = Field(ge=0.0, le=1.0)
    neutral_prob: float = Field(ge=0.0, le=1.0)
    contradiction_prob: float = Field(ge=0.0, le=1.0)
    nli_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_confidence_matches_predicted(self) -> Self:
        probs = {
            "entailment": self.entailment_prob,
            "neutral": self.neutral_prob,
            "contradiction": self.contradiction_prob,
        }
        if abs(sum(probs.values()) - 1.0) > 1e-3:
            raise ValueError("NLI probabilities must sum to approximately 1.")
        if abs(probs[self.nli_label] - self.nli_confidence) > 1e-6:
            raise ValueError("nli_confidence must equal the probability of the predicted label.")
        return self


class BERTScoreResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    row_id: int | str
    person_id: str
    question: str = Field(min_length=1)
    human_answer: str = Field(min_length=1)
    ai_answer: str = Field(min_length=1)
    # Baseline-rescaled BERTScore values are metric values, not probabilities; they are
    # not restricted to [0, 1] and must only be checked for being finite numbers.
    bertscore_precision: float
    bertscore_recall: float
    bertscore_f1: float

    @model_validator(mode="after")
    def validate_scores_are_finite(self) -> Self:
        scores = {
            "bertscore_precision": self.bertscore_precision,
            "bertscore_recall": self.bertscore_recall,
            "bertscore_f1": self.bertscore_f1,
        }
        for name, value in scores.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value}.")
        return self


class HumanReviewRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    review_id: str
    row_id: int | str
    person_id: str
    question: str = Field(min_length=1)
    human_answer: str = Field(min_length=1)
    ai_answer: str = Field(min_length=1)
    human_fidelity_score: int = Field(ge=1, le=5)
    human_contradiction: Literal["YES", "NO"]
    human_omission: Literal["YES", "NO"]
    human_unsupported_detail: Literal["YES", "NO"]
    reviewer_note: str
    geval_core_fidelity: int | None = Field(ge=1, le=5)
    claim_coverage_rate: float | None
    strict_alignment_rate: float | None
    step4_contradiction_present: bool
    step4_missing_count: int = Field(ge=0)
    step4_unsupported_rate: float | None
    nli_contradiction_present: bool
    nli_contradiction_count: int = Field(ge=0)
    bertscore_precision: float
    bertscore_recall: float
    bertscore_f1: float


class AutomatedEvaluationRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    row_id: int | str
    person_id: str
    question: str = Field(min_length=1)
    human_answer: str = Field(min_length=1)
    ai_answer: str = Field(min_length=1)

    # automated_fidelity_score is G-Eval Core Fidelity, and only that; never a
    # weighted composite with claim, NLI, or BERTScore signals.
    automated_fidelity_score: int = Field(ge=1, le=5)
    automated_fidelity_scale: Literal["1-5"]

    core_fidelity_score: int = Field(ge=1, le=5)
    behavior_score: int | None = Field(ge=1, le=5)
    preference_score: int | None = Field(ge=1, le=5)
    motivation_score: int | None = Field(ge=1, le=5)
    nuance_score: int | None = Field(ge=1, le=5)

    human_claim_count: int = Field(ge=0)
    aligned_count: int = Field(ge=0)
    partial_count: int = Field(ge=0)
    contradicted_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    ai_claim_count: int = Field(ge=0)
    unsupported_ai_count: int = Field(ge=0)
    claim_coverage_rate: float | None
    strict_alignment_rate: float | None
    unsupported_rate: float | None

    nli_contradiction_present: bool
    nli_contradiction_count: int = Field(ge=0)
    strongest_contradiction_probability: float | None = Field(ge=0.0, le=1.0)

    bertscore_precision: float
    bertscore_recall: float
    bertscore_f1: float

    @model_validator(mode="after")
    def validate_primary_score_is_geval_core(self) -> Self:
        if self.automated_fidelity_score != self.core_fidelity_score:
            raise ValueError("automated_fidelity_score must equal G-Eval Core Fidelity exactly.")
        return self


# The Human fidelity score IS the final validated score; this only maps it to a
# readable label. No new number is calculated anywhere from this mapping.
FINAL_FIDELITY_LABELS = {
    5: "High fidelity",
    4: "Mostly faithful",
    3: "Mixed fidelity",
    2: "Low fidelity",
    1: "Very low fidelity",
}


class FinalPairwiseEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    row_id: int | str
    person_id: str
    question: str = Field(min_length=1)
    human_answer: str = Field(min_length=1)
    ai_answer: str = Field(min_length=1)

    automated_fidelity_score: int = Field(ge=1, le=5)
    automated_fidelity_scale: Literal["1-5"]
    behavior_score: int | None = Field(ge=1, le=5)
    preference_score: int | None = Field(ge=1, le=5)
    motivation_score: int | None = Field(ge=1, le=5)
    nuance_score: int | None = Field(ge=1, le=5)
    claim_coverage_rate: float | None
    strict_alignment_rate: float | None
    contradicted_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    unsupported_rate: float | None
    nli_contradiction_present: bool
    bertscore_f1: float

    human_fidelity_score: int = Field(ge=1, le=5)
    human_contradiction: Literal["YES", "NO"]
    human_omission: Literal["YES", "NO"]
    human_unsupported_detail: Literal["YES", "NO"]
    reviewer_note: str = Field(min_length=1)

    # score_difference is diagnostic only; it is never averaged with either score or
    # turned into a new combined number.
    score_difference: int
    exact_score_agreement: bool
    within_one_point: bool
    final_fidelity_label: str
    main_issue: str = Field(min_length=1)
    final_assessment_note: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_derived_fields(self) -> Self:
        expected_difference = self.automated_fidelity_score - self.human_fidelity_score
        if self.score_difference != expected_difference:
            raise ValueError("score_difference must equal automated_fidelity_score - human_fidelity_score.")
        if self.exact_score_agreement != (self.score_difference == 0):
            raise ValueError("exact_score_agreement must reflect score_difference == 0.")
        if self.within_one_point != (abs(self.score_difference) <= 1):
            raise ValueError("within_one_point must reflect abs(score_difference) <= 1.")
        expected_label = FINAL_FIDELITY_LABELS[self.human_fidelity_score]
        if self.final_fidelity_label != expected_label:
            raise ValueError(
                f"final_fidelity_label must be {expected_label!r} for human_fidelity_score="
                f"{self.human_fidelity_score}."
            )
        return self
