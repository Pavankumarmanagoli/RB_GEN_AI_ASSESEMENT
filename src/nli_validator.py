import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# A single pretrained MNLI checkpoint, used only to cross-check Step 4B's semantic
# labels. This is not an LLM judge: it sees only the two claim texts, never the
# Step 4B label, and produces no fidelity score.
NLI_MODEL_NAME = "roberta-large-mnli"
MAX_LENGTH = 256

NLI_LABELS = ("entailment", "neutral", "contradiction")


class NLIError(ValueError):
    pass


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_nli_model() -> tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
    device = resolve_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device


def resolve_label_map(model: AutoModelForSequenceClassification) -> dict[int, str]:
    """Normalize the checkpoint's configured id2label into entailment/neutral/contradiction.
    Never assumes a fixed numeric ordering such as 0=contradiction; reads the checkpoint's
    own mapping and verifies it covers exactly the three expected labels."""
    label_map = {}
    for class_id, raw_label in model.config.id2label.items():
        normalized = str(raw_label).strip().lower()
        if normalized not in NLI_LABELS:
            raise NLIError(
                f"Unrecognized NLI label {raw_label!r} at index {class_id} for {NLI_MODEL_NAME}; "
                f"expected one of {NLI_LABELS}."
            )
        label_map[int(class_id)] = normalized
    if set(label_map.values()) != set(NLI_LABELS):
        raise NLIError(
            f"{NLI_MODEL_NAME} label map {label_map} does not cover exactly {NLI_LABELS}."
        )
    return label_map


def predict_nli(
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    label_map: dict[int, str],
    pairs: list[tuple[str, str]],
) -> list[dict]:
    """Deterministic batch NLI inference. Each pair is (premise, hypothesis); by this
    module's convention premise=human claim, hypothesis=AI claim (see run_nli_validation.py).
    Returns one record per pair, in input order."""
    if not pairs:
        return []
    premises = [premise for premise, _ in pairs]
    hypotheses = [hypothesis for _, hypothesis in pairs]
    encoded = tokenizer(
        premises, hypotheses,
        return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH,
    ).to(device)
    with torch.no_grad():
        logits = model(**encoded).logits
    probabilities = torch.softmax(logits, dim=-1)
    results = []
    for row in probabilities:
        per_label = {label_map[class_id]: row[class_id].item() for class_id in range(row.shape[0])}
        predicted_label = max(per_label, key=per_label.get)
        results.append({
            "nli_label": predicted_label,
            "entailment_prob": per_label["entailment"],
            "neutral_prob": per_label["neutral"],
            "contradiction_prob": per_label["contradiction"],
            "nli_confidence": per_label[predicted_label],
        })
    return results
