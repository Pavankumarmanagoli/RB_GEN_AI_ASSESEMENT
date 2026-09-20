# Human-simulation evaluation rubric

## Purpose and reference

Evaluate how faithfully an AI response represents the specific individual in the human answer. Use the original question to interpret both responses. The human answer is the held-out reference; a separate first interview used to create the simulation is unavailable.

This rubric defines five independent dimensions on a 1–5 scale, where 5 is the highest fidelity. It does not define automated scoring, weights, or an overall score.

## 1. Core Fidelity

Measures whether the AI response preserves the central meaning and important substantive claims of the human response. Focus on what the answer says overall, rather than wording or length.

| Score | Anchor |
| --- | --- |
| 5 | Central meaning and key claims are fully aligned. |
| 4 | Core meaning aligns; only minor omissions or differences. |
| 3 | Broadly aligned, but one or more meaningful differences exist. |
| 2 | Major portions of the human response are missing, changed, or misrepresented. |
| 1 | Central meaning is substantially incorrect or opposite. |

**Applicability:** Normally applicable to every valid response pair, including a human answer expressing uncertainty or indifference. A sparse answer is not automatically N/A.

## 2. Behavioral Fidelity

Measures whether the AI expresses the same action, decision, intended action, habit, frequency, or behavioral tendency as the human. Focus on what the person does or would do. A stated choice can be both a decision here and a preference below.

| Score | Anchor |
| --- | --- |
| 5 | Same behavior or decision. |
| 4 | Same core behavior with a minor difference. |
| 3 | Partially aligned behavior, with a meaningful difference. |
| 2 | Significant behavioral mismatch. |
| 1 | Opposite or directly contradictory behavior. |

**Applicability:** Mark N/A when the human response contains no meaningful behavior, action, decision, or behavioral tendency. A denial of a behavior, such as not buying in bulk, is meaningful behavior evidence.

## 3. Preference Fidelity

Measures whether the AI preserves the human's likes, dislikes, choices, priorities, indifference, and relative preferences. Focus on what the person favors or considers important, rather than assuming that every reported purchase proves a preference.

| Score | Anchor |
| --- | --- |
| 5 | Preferences and priorities fully align. |
| 4 | Main preference aligns with only a minor difference. |
| 3 | Partially aligned preference. |
| 2 | Major preference mismatch. |
| 1 | Opposite preference or priority. |

**Applicability:** Mark N/A when no meaningful preference is expressed. Explicit indifference, such as “I don't really care about the brand,” is a meaningful preference state and must not be treated as missing information. Not knowing a brand does not by itself establish indifference to it.

## 4. Motivation Fidelity

Measures whether the AI preserves the reasons or decision drivers expressed by the human. Focus on why the person acts or chooses, rather than the action or preference alone.

Motivations can include price, convenience, quality, habit, availability, travel, recommendations, prior experience, and values. These count as motivations when the human presents them as reasons or drivers; mentioning a topic alone is not enough.

| Score | Anchor |
| --- | --- |
| 5 | Same motivation/reasoning. |
| 4 | Main motivation aligns with a minor difference. |
| 3 | Partially aligned motivation. |
| 2 | Major reasoning difference. |
| 1 | Opposite or fundamentally different motivation. |

**Applicability:** Mark N/A if the human response does not provide a meaningful reason or motivation, even when the question asks “why” or the AI supplies a reason. Do not infer a reason solely from a choice.

Do not penalize the AI simply because it gives more explanation unless that explanation changes, contradicts, or misrepresents the human's stated motivation. Additional compatible reasons are reference-unverifiable; they do not earn extra credit or automatically reduce fidelity.

## 5. Nuance Preservation

Measures whether the AI preserves qualifiers, uncertainty, conditions, exceptions, frequency, and strength of statements. Focus on how strongly, how often, and under what circumstances the human's claim holds.

Pay particular attention to “sometimes,” “usually,” “often,” “rarely,” “never,” “always,” “maybe,” “depends,” “only when,” “unless,” “not really,” and “if I had to choose.” Equivalent wording can preserve nuance; verbatim matching is unnecessary.

| Score | Anchor |
| --- | --- |
| 5 | Nuance, strength, frequency, and conditions are fully preserved. |
| 4 | Minor nuance difference without changing the overall interpretation. |
| 3 | Some meaningful nuance is lost or exaggerated. |
| 2 | Major over-generalization, understatement, or loss of conditions. |
| 1 | Important nuance is reversed or materially distorted. |

**Applicability:** Mark N/A only when there is genuinely no meaningful nuance to compare. Uncertainty, indifference, conditional choices, and explicit absolutes can all carry meaningful nuance. Judge meaning in context, not just the presence of a keyword.

## General evaluation and applicability rules

- Evaluate the five dimensions independently and in parallel. Do not derive one dimension's score from another.
- Compare the AI response against the human answer as the held-out reference, while considering the original question. Do not infer an answer from the question alone.
- Evaluate fidelity to this specific individual, not whether the AI sounds generally plausible for a consumer.
- Do not reward verbosity. Do not penalize brevity if the important human meaning is preserved.
- Do not assume an additional AI detail is false merely because it is absent from the human answer. The separate first interview is unavailable. Call such details **reference-unverifiable**, not hallucinations.
- Compatible reference-unverifiable additions neither earn extra credit nor automatically incur a penalty. An addition that changes, contradicts, or misrepresents an explicit human claim can reduce the relevant fidelity score. Distinguish that conflict from absence of supporting evidence.
- Explicit human uncertainty or indifference is meaningful and must be preserved. Negations and frequency terms are critical: “rarely” versus “often” may represent a major behavioral difference.
- Decide applicability from the human answer in its question context. If the human supplies relevant evidence but the AI omits it, assess the omission using the score anchors; do not mark N/A. AI-only content does not make an otherwise inapplicable dimension applicable.
- N/A means the criterion genuinely does not apply. It is neither score 0 nor score 1, and is not a substitute for an uncertain judgment. A future N/A decision should include a short reason.
- Core Fidelity normally applies to every valid pair. Missing or unreadable responses are an input-validity issue, not evidence that all dimensions are N/A; handling such inputs must be agreed before implementation.
- Do not combine the dimensions into arbitrary weighted scores at this stage.
- Every score should later have a concise rationale and relevant evidence from the responses. Quote only what is needed to show alignment, omission, or conflict.

### Distinguishing neighboring scores and overlapping dimensions

A minor difference leaves the individual's main answer intact. A meaningful difference changes a substantive part of it. A major difference changes a central claim, behavior, preference, reason, or condition. Judge importance relative to the question and answer, rather than counting matching words or sentences.

Use 5 for full semantic alignment and 4 for a small but real difference. Use 3 for a meaningful mismatch with substantial alignment remaining. Use 2 for a major mismatch with limited alignment remaining. Use 1 when the dimension's lowest anchor applies: an opposite or direct contradiction, a substantially incorrect central meaning, fundamentally different motivation, or material distortion of important nuance. Not every omission warrants 1.

Overlap is sometimes necessary. Core Fidelity addresses the central message; the other dimensions identify specific aspects. A frequency reversal may affect both Behavioral Fidelity (the habit changes) and Nuance Preservation (the frequency changes). A choice may inform both behavior and preference. Assess each independently and explain its specific effect, without imposing an automatic cross-dimension penalty.

## Illustrative examples

These short examples are adapted from response types in `data/RB_GenAI_Datatest.xlsx`, sheet `answer_pairs`. Wording is simplified and AI contrasts may be constructed. They are rubric illustrations, not scored dataset rows or evaluation results.

| Case | Question context | Human | AI | Illustration |
| --- | --- | --- | --- | --- |
| High fidelity | Where do you shop, and why? | “I almost always buy personal care products during my grocery trip at the nearest supermarket because it is convenient and saves time.” | “I almost always get them at my nearest supermarket while buying groceries, for convenience and to save time.” | Preserves the central meaning, behavior, stated reasons, and frequency. Does not add an unstated brand preference. |
| Direct behavioral reversal | Do you buy in bulk? | “I don't usually buy in bulk; I buy things when I need them.” | “I regularly buy toothpaste and deodorant in bulk.” | Reverses the reported habit. This is a conflict with the reference, not merely an unverifiable addition. |
| Preference mismatch | What matters when choosing toothpaste? | “I don't care much about the brand; freshness and texture matter.” | “Brand is my top priority; freshness and texture don't matter.” | Reverses both brand indifference and the stated priorities. Preference Fidelity applies to indifference. |
| Motivation mismatch | Why do you keep buying this toothpaste? | “My mum used to buy it, and I simply continued out of habit.” | “I keep buying it solely because a dentist recommended it, not out of habit.” | Keeps the purchase behavior but explicitly replaces its reason. A compatible added reason alone would not establish a mismatch. |
| Nuance loss | Which celebrity would you choose? | “It depends on the day, but if I had to pick one, maybe Keira Knightley.” | “Keira Knightley is always my definite choice.” | Keeps the name but turns a tentative, conditional choice into an absolute. |
| N/A criterion | Which lotion would you choose, and why? | “I would choose option 2.” | “I would choose option 2 because it feels light.” | Motivation Fidelity is N/A: the human supplies no reason. The AI's explanation is reference-unverifiable. Core Fidelity still applies. |

## Open questions before evaluator implementation

1. **Borderline severity:** The anchors distinguish minor, meaningful, and major differences qualitatively. Agree on a few additional calibration examples before implementation, especially the boundary between 2 and 1 for Motivation Fidelity and Nuance Preservation. No numerical mismatch-count thresholds are defined.
2. **Added details and emphasis:** Compatible additions are reference-unverifiable, while contradictory additions affect fidelity. Agree on borderline cases where numerous added priorities appear to change emphasis without explicitly denying a human claim.
3. **Ambiguous reference wording:** A phrase such as “my first choice would be option 1 and option 2” does not clearly establish a ranking. Preserve the expressed alternatives and avoid inventing a priority; agree how unresolved interpretations will be handled.
4. **Scope of context:** This rubric uses the original question and its paired human answer. Decide explicitly whether later evaluation may also consult other answers from the same person; do not silently introduce that context or assume access to the first interview.
5. **Invalid inputs:** Agree how missing, unreadable, or unpaired responses will be handled. These are distinct from criterion-level N/A and are not assigned scores here.

Step 2 ends with this rubric. No automated scoring or dataset evaluation results are defined or produced.
