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
