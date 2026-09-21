# Final Evaluation Summary

## Automated vs Human-Reviewed Match

- Automated Match Score: 2.63/5 (median 3.0)
- Human-Reviewed Match Score: 2.53/5 (median 3.0)
- Mean difference (Automated Match Score - Human-Reviewed Match Score): +0.10
- Mean absolute score difference: 0.50
- Exact score agreement: 15/30 (50.0%)
- Within one point: 30/30 (100.0%)
- Rows with score difference greater than one: []
- Spearman (human fidelity vs G-Eval Core, reused from Step 7B): rho=0.820, p=2.95e-08, n=30

## Contradiction Validation

- Correctly detected: Claim-based check 8, NLI check 9
- Missed by claim-based check: 3 (rows [10, 11, 18])
- Missed by NLI check: 2 (rows [12, 21])
- NLI check false positives: 1 (rows [7])

## Person-Level Results

- **c_human** (10 rows, consistency: Low): Automated Match Score 2.90, Human-Reviewed Match Score 2.70, contradictions 3, omissions 6
  - c_human's simulated answers show an Automated Match Score of 2.90/5 and a Human-Reviewed Match Score of 2.70/5, with overall person-level consistency rated Low. Some important preferences are kept correctly, but there are a few major contradictions and many extra details added across the answers.
- **g_human** (10 rows, consistency: Low): Automated Match Score 2.30, Human-Reviewed Match Score 2.50, contradictions 6, omissions 7
  - g_human's simulated answers show an Automated Match Score of 2.30/5 and a Human-Reviewed Match Score of 2.50/5, with overall person-level consistency rated Low. The AI keeps a few general preferences, but there are several major contradictions and many unsupported personal details across the person's profile.
- **s_human** (10 rows, consistency: Medium): Automated Match Score 2.70, Human-Reviewed Match Score 2.40, contradictions 2, omissions 9
  - s_human's simulated answers show an Automated Match Score of 2.70/5 and a Human-Reviewed Match Score of 2.40/5, with overall person-level consistency rated Medium. The AI keeps several real skincare preferences, but it also adds a lot of extra profile details and has a few clear inconsistencies.

## Representative Examples

- Row 10 (c_human): Automated Match Score 1/5, Human-Reviewed Match Score 1/5 (Human–AI Match: Very weak match), main issue: multiple issues. Behavioral reversal: The AI gives the opposite buying behavior; the Human says they do not usually buy in bulk, while the AI says they regularly do and also adds specific products and reasons that were not mentioned.
- Row 16 (c_human): Automated Match Score 2/5, Human-Reviewed Match Score 1/5 (Human–AI Match: Very weak match), main issue: multiple issues. Unsupported switching reason: The Human answer says the toothpaste has been used for around 10 years and there was never a specific reason to switch, while the AI shortens this to a few years and invents a switch because of better ingredients.
- Row 22 (c_human): Automated Match Score 4/5, Human-Reviewed Match Score 4/5 (Human–AI Match: Strong match), main issue: unsupported elaboration. Correct choice, heavy unsupported elaboration: The AI correctly keeps Option 2 as the preferred choice, but it adds several reasons about hydration, heaviness, greasiness, and daily use that were not mentioned.
- Row 7 (c_human): Automated Match Score 3/5, Human-Reviewed Match Score 3/5 (Human–AI Match: Partial match), main issue: omission + unsupported elaboration. NLI false positive: The AI keeps the approximate size and natural-material idea, but it misses the smell preference and adds several packaging, labeling, sustainability, and ease-of-use preferences that were not mentioned.
- Row 14 (s_human): Automated Match Score 2/5, Human-Reviewed Match Score 1/5 (Human–AI Match: Very weak match), main issue: multiple issues. Celebrity preference mismatch: The Human answer is uncertain and, if forced to choose, picks Keira Knightley, while the AI chooses Hyram Yarbro instead and adds several unsupported reasons and preferences.
- Row 28 (c_human): Automated Match Score 4/5, Human-Reviewed Match Score 3/5 (Human–AI Match: Partial match), main issue: unsupported elaboration. Broad theme preserved, heavy elaboration: The AI keeps friends and social media as the main discovery sources, but it adds many extra influences such as reviews, ads, influencers, product reputation, ingredients, and price that were not mentioned.

## Final Business Findings

- Automated and Human-reviewed match scores broadly agree: the Automated Match Score averages 2.63/5 versus a Human-Reviewed Match Score of 2.53/5, matching exactly on 50% of rows and within one point on 100%.
- Broad themes (what a person buys, cares about, or discovers) are usually preserved, but person-specific details, quantities, and named preferences are frequently lost or replaced.
- Omissions and unsupported elaboration are common: the human reviewer flagged omission in 73% of rows and unsupported detail in 100% of rows.
- Human review identified contradictions in 36.7% of rows, including several clear behavioral or preference reversals.
- Person-level consistency remains weak for c_human, g_human (Low) and is only moderate for s_human (Medium), based on the manual cross-row consistency review.

## Overall Conclusion

Across the 30 reviewed answers, the AI simulations show only a partial match to the real Human responses, with an Automated Match Score of 2.63/5 and a Human-Reviewed Match Score of 2.53/5. The simulations often preserve parts of the broad theme of an answer, but frequently omit person-specific details and introduce unsupported personal elaboration; Human review identified contradictions in 36.7% of rows, including several clear behavioral or preference reversals. Automated and Human judgments show strong rank agreement (Spearman ρ = 0.82) and differ by no more than one point across all 30 rows, indicating that automated evaluation can serve as a useful screening signal. Human review remains important for identifying meaningful reversals, determining which omissions matter, and resolving borderline cases where automated methods disagree.
