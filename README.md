# Human-simulation fidelity evaluation

## Step 3: G-Eval evaluation

G-Eval is the primary automated fidelity evaluator in this project. A rubric-based
LLM judge compares each AI answer with its paired human answer and original question.
It independently judges Core Fidelity, Behavioral Fidelity, Preference Fidelity,
Motivation Fidelity, and Nuance Preservation on the frozen **1–5 + N/A** scale in
[the rubric](docs/evaluation_rubric.md). There is no combined score.

The implementation uses the OpenAI SDK with Pydantic structured output rather than
DeepEval. This is G-Eval-style rubric judging, not a reproduction of token-probability
weighted G-Eval research scores. Each judgment includes applicability, a concise
rationale, verbatim evidence, and confidence. It does not request or save chain-of-thought.
Evidence can be one verbatim excerpt, ellipsis-separated excerpts, or a list of
quoted excerpts. The validator removes quotation formatting and checks each excerpt against
the original answer in order; it does not accept paraphrases or invented evidence.
New requests explicitly ask for a single continuous excerpt per evidence field to
reduce formatting errors. Existing valid judgments are retained under their original
prompt hash, revalidated, and labeled `evidence_format=original` in the saved results.
Only the current question and answer pair are sent as case evidence; identifiers,
categories, other responses, and previous results are excluded. Prompts live in
`src/geval_evaluator.py`. The judge uses calibration prompt version `fidelity-v4`;
it loads the rubric's definitions and general rules verbatim and excludes its
dataset-derived examples. Cache entries include the prompt version, so the v1
judgments stay available under their original hashes.

### Setup and run

Use the existing uv project and `.venv`:

```bash
uv sync
```

Set these variables locally in the existing `.env`; do not overwrite that file or
commit credentials. `.env.example` contains empty placeholders only.

```dotenv
OPENAI_API_KEY=<your local API key>
EVALUATOR_MODEL=<a model available to your account with structured-output support>
```

Both values are required before an API call. Configuration lives in `src/config.py`.
GPT-4.1 and GPT-4o families use temperature 0. Other models omit temperature to avoid
unsupported reasoning-model parameters; this does not guarantee deterministic output.
No model is silently substituted. The configured model and request options are saved.

From the project root:

```bash
.venv/bin/python -m src.run_geval
```

To rerun only selected pilot rows under `fidelity-v4`, for example:

```bash
.venv/bin/python -m src.run_geval --rows 10 20 24
```

The command evaluates **only these five manually selected rows**, in this order:

| Row ID | Selection reason |
| --- | --- |
| 25 | Apparent strong match on effectiveness, affordable price, and scent as repurchase drivers. |
| 10 | Clear reversal between usually avoiding bulk buying and regularly buying in bulk. |
| 20 | Meaningful mismatch in packaging, scent, texture, and ingredient-information preferences. |
| 22 | Human selects option 2 without giving a reason; Motivation should be N/A. |
| 24 | Indifference to names, preference for visual design, and a conditional product choice. |

Selection is based on reading the responses, not automated scores. A full-dataset
mode is also available (`--full`); it was used to produce the frozen final results
described below.

Results are checkpointed after each row to a versioned output file named after the
row selection and current prompt version. Successful identical judgments are reused
from `outputs/.geval_cache.json`, keyed by input, model/options, rubric, prompt
version, prompt text, output schema, and evidence-validation version. Changing any
of these invalidates the cache entry and causes a fresh paid call. Cached evidence
is validated again when it is read.

### Final frozen results

Step 3 is complete. The evaluator is frozen at calibration prompt version
`fidelity-v4`, after two rounds of calibration (Motivation reversal-vs-substitution,
then Behavior applicability for hypothetical/forced-choice selections; see git
history on `src/geval_evaluator.py`). The full 30-row dataset was evaluated under
this configuration with:

```bash
.venv/bin/python -m src.run_geval --full
```

The final, reviewer-facing results are:

- `outputs/geval_results_final.json`
- `outputs/geval_results_final.csv`

Intermediate pilot and calibration outputs (the v1 pilot, the fidelity-v2 full run,
and the fidelity-v3/fidelity-v4 calibration-only runs) have been removed from
`outputs/`; git history preserves the source and prompt changes that produced them.

Processing failures are recorded separately with row ID, error type, and a safe
message. They are never scores or N/A. The SDK retries transient errors at most twice;
parsing/validation failures are not automatically retried. Rerunning retries failed
rows while reusing successful cached rows. A failed row makes the command exit nonzero.
Local output failures identify whether the response is incomplete, lacks a parsed
judgment, or contains an invalid evidence excerpt (including the dimension and field).

Run the non-API tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Regenerate the dependency export after dependency changes:

```bash
uv export --format requirements-txt --no-hashes --no-emit-project --output-file requirements.txt
```

API implementation reference: [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

## Step 4: Atomic claim extraction and alignment

Step 4 decomposes each human and AI answer into atomic claims (Step 4A), then
independently aligns the human claims against the AI claims (Step 4B), labeling
each human claim `aligned`, `partial`, `contradicted`, or `missing`, and each
leftover AI claim `unsupported`. This is diagnostic evidence alongside G-Eval, not
a replacement for it; no combined fidelity score is produced.

Both stages are frozen:

- Claim extraction: `src/claim_extractor.py`, prompt version `claims-v2`.
- Claim alignment: `src/claim_aligner.py`, prompt version `alignment-v2`.

Each stage caches successful results separately from G-Eval and from each other,
in `outputs/.claims_cache.json` and `outputs/.alignment_cache.json`.

Run the full 30-row dataset:

```bash
.venv/bin/python -m src.run_claim_extraction --full
.venv/bin/python -m src.run_claim_alignment --full
```

Both runners also accept `--rows ID [ID ...]` to target specific rows. Alignment
consumes Step 4A's saved extraction output rather than re-extracting claims, so
extraction must be run first.

The final, reviewer-facing results are:

- `outputs/claim_extraction_final.json`
- `outputs/claim_extraction_final.csv`
- `outputs/claim_alignment_final.json`
- `outputs/claim_alignment_final.csv`

`unsupported` means a claim is absent from the human reference; it does not by
itself mean the claim is false or contradictory, mirroring G-Eval's
reference-unverifiable principle. `claim_coverage_rate` and `strict_alignment_rate`
are per-row descriptive metrics, not a combined score.
