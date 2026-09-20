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
