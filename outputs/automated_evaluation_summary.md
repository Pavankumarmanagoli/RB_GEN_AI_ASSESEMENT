# Automated Evaluation Summary

## Overall Automated Fidelity

**Overall Automated Fidelity: 2.63 / 5**

Based on G-Eval Core Fidelity across all 30 rows (median 3.0/5). This is the sole primary automated fidelity score; it is not a weighted composite with claim, NLI, or BERTScore signals.

Score distribution: 1 → 5, 2 → 8, 3 → 12, 4 → 3, 5 → 2

## G-Eval

Core Fidelity is the primary automated score. The other G-Eval dimensions are reported descriptively, counting only rows where the dimension was judged applicable:

- **Behavior**: applicable in 18/30 rows, mean 3.06/5
- **Preference**: applicable in 22/30 rows, mean 2.77/5
- **Motivation**: applicable in 22/30 rows, mean 2.91/5
- **Nuance**: applicable in 29/30 rows, mean 2.38/5

## Claim Preservation

- 121 total human claims across the dataset.
- Aligned: 15 (12.4%)
- Partial: 35 (28.9%)
- Contradicted: 10 (8.3%)
- Missing: 61 (50.4%), affecting 26/30 rows
- 254 total AI claims; 198 (78.0%) are unsupported by the human reference (absent from the reference, not proven false or hallucinated).
- Mean claim coverage rate: 0.47 (median 0.50)
- Mean strict alignment rate: 0.15 (median 0.00)

## Contradiction Analysis

- Step 4 (claim alignment) flags 8/30 rows with at least one contradicted claim: [9, 12, 13, 14, 15, 16, 21, 24]
- Step 5 (NLI) flags 10/30 rows with at least one contradiction: [7, 9, 10, 11, 13, 14, 15, 16, 18, 24]
- Both methods agree on 6 rows: [9, 13, 14, 15, 16, 24]
- Step 4 only: [12, 21]; NLI only: [7, 10, 11, 18]

No judgment is made here about which method is correct on the disagreement rows; that is left for human validation in Step 8B.

## Semantic Similarity

- Mean F1: 0.144, median: 0.145, range: [-0.061, 0.315]
- BERTScore measures semantic similarity only; it is not a factual-fidelity metric.

- Row 10: F1=0.257, G-Eval Core=1, Step4 contradiction=False, NLI contradiction=True — BERTScore F1 is above the dataset median despite the lowest possible G-Eval Core score and an NLI-detected contradiction.
- Row 22: F1=0.038, G-Eval Core=4, Step4 contradiction=False, NLI contradiction=False — BERTScore F1 is among the lowest in the dataset despite a high G-Eval Core score and no detected contradiction.
- Row 28: F1=-0.061, G-Eval Core=4, Step4 contradiction=False, NLI contradiction=False — BERTScore F1 is among the lowest in the dataset despite a high G-Eval Core score and no detected contradiction.

## Person-Level Automated Results

- **c_human** (10 rows): automated fidelity mean 2.90 (median 3.0), mean claim coverage 0.67, mean unsupported rate 0.74, Step4 contradiction rows 2, NLI contradiction rows 4, mean BERTScore F1 0.142
- **g_human** (10 rows): automated fidelity mean 2.30 (median 2.5), mean claim coverage 0.33, mean unsupported rate 0.73, Step4 contradiction rows 5, NLI contradiction rows 4, mean BERTScore F1 0.151
- **s_human** (10 rows): automated fidelity mean 2.70 (median 3.0), mean claim coverage 0.41, mean unsupported rate 0.80, Step4 contradiction rows 1, NLI contradiction rows 2, mean BERTScore F1 0.139

## Key Automated Findings

- Overall automated fidelity is mixed: G-Eval Core Fidelity averages 2.63/5 across the 30 rows, with scores spread across the full 1-5 range rather than clustering at the top.
- Human content is frequently omitted: 50.4% of human claims (61/121) are missing from the AI answer, affecting 26/30 rows.
- Unsupported AI elaboration is common: on average 75.6% of AI claims per row are unsupported by the human reference (this indicates content absent from the reference, not proven-false content).
- Explicit contradictions occur in a meaningful subset of responses: Step 4 claim alignment flags 8/30 rows and independent NLI validation flags 10/30 rows, agreeing on 6 of them.
- BERTScore can remain relatively high despite a behavioral contradiction, and can also be low despite a high G-Eval score and no detected contradiction (see representative_rows in the semantic_similarity section) — semantic similarity alone is not a reliable fidelity signal.
