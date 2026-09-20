from bert_score import BERTScorer
from transformers import AutoTokenizer

# One fixed, official English BERTScore configuration. Reference = Human answer,
# candidate = AI answer: precision loosely reflects how much of the candidate overlaps
# the reference, recall loosely reflects how much of the reference is covered by the
# candidate, and F1 summarizes contextual token-level similarity. These are semantic
# similarity signals, not factual-verification metrics — precision is not a
# hallucination rate and recall is not an omission rate.
MODEL_NAME = "roberta-large"
LANGUAGE = "en"
IDF = False
RESCALE_WITH_BASELINE = True


class BERTScoreError(ValueError):
    pass


def load_scorer() -> BERTScorer:
    return BERTScorer(
        model_type=MODEL_NAME,
        lang=LANGUAGE,
        idf=IDF,
        rescale_with_baseline=RESCALE_WITH_BASELINE,
    )


def find_length_overflows(rows: list[dict], tokenizer=None) -> list[dict]:
    """Reports any Human/AI answer whose token count exceeds the model's max sequence
    length, using the same tokenizer BERTScore will score with. Does not truncate or
    chunk; overflow rows must be handled explicitly, not silently."""
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(MODEL_NAME)
    max_length = tokenizer.model_max_length
    overflows = []
    for row in rows:
        for side, text in (("human", row["human_answer"]), ("ai", row["ai_answer"])):
            token_count = len(tokenizer.encode(text, add_special_tokens=True))
            if token_count > max_length:
                overflows.append({
                    "row_id": row["row_id"], "side": side,
                    "token_count": token_count, "max_length": max_length,
                })
    return overflows


def score_batch(
    scorer: BERTScorer, human_answers: list[str], ai_answers: list[str],
) -> tuple[list[float], list[float], list[float]]:
    """Human answer = reference, AI answer = candidate. Scores the whole batch at once
    and returns (precision, recall, f1) lists in input order."""
    if len(human_answers) != len(ai_answers):
        raise BERTScoreError("human_answers and ai_answers must be the same length.")
    if not human_answers:
        raise BERTScoreError("At least one answer pair is required.")
    if any(not text or not text.strip() for text in (*human_answers, *ai_answers)):
        raise BERTScoreError("Empty or missing answer text cannot be scored.")
    precision, recall, f1 = scorer.score(cands=ai_answers, refs=human_answers)
    return precision.tolist(), recall.tolist(), f1.tolist()
