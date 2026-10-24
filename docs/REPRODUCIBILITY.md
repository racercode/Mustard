# Reproduction status

The formal paper pipeline uses flow-guided keyframe selection. The research run
on 2026-08-25 used 220 uniformly sampled videos for timing/comparison; that run is
not the frame-selection baseline for the formal implementation.

## Preserved behavior

- Recovered priority-based keyframe selection and batch parameter values.
- Identical raw/trajectory selected indices.
- Qwen2.5-VL captions and real-plot caption prompts.
- Position-wise trajectory pattern extraction, concatenated without an extra LLM call.
- Flow normal-pattern and anomaly-reasoning prompts.
- Fusion of raw descriptions and two modality reasonings after score-line removal.
- PAEval prompts and per-object aggregation; pairwise prompts and answer-order swapping.
- GPT-4o judge default, confirmed by the project owner.
- Final Explanation is the default evaluated field, based on the si2025_3 exporter versions and the input/score chain matching the paper. The older 2080 Reasoning-only exporter is not used as the default.

## Explicit implementation changes

- Inputs, outputs and model hosts are configurable; legacy absolute paths are removed.
- The normal reference set is selected once per category from sorted video IDs,
  using a deterministic seed derived from the configured seed and category name, and saved in the pattern artifact. The legacy scripts
  did not preserve their per-frame reference IDs, so newly extracted patterns are
  not claimed to reproduce historical text verbatim.
- Each selected reference contributes at all eight positions. No test video enters induction.
- SciPy is required instead of silently switching peak-finding algorithms.
- Strict parsers reject missing, duplicate and out-of-range scores. Fusion also requires a nonempty Final Explanation before success is cached.
- The Phys-AD normal-directory aliases include both norm and anomaly_free.
- Pairwise outputs with both A and B verdict markers are rejected.
- A failed model call cannot become a completed cached response.
- Inputs, prompts, options and resolved model digests participate in cache identities.
- AUROC validates complete test-set coverage instead of silently dropping failed samples.
- Duplicate historical PNG filenames are avoided by selecting and extracting from videos.

## Validation performed

Offline tests cover aligned extraction, input mismatch rollback, invalid tracks,
normal-only induction, fusion score removal, cache invalidation and failure behavior,
AUROC ties/coverage, PAEval export and macro aggregation, and pairwise swapped identity.
The extracted selector also matches the legacy SciPy implementation on four motion
sequences. CLI integration checks use real synthetic videos and mocked HTTP responses to exercise all five modality recipes, both Qwen samplers, validation, resumption, AUROC, PAEval and pairwise. Live Qwen/Gemma calls, full dataset inference and paid GPT-4o requests have not been run.

## Current release scope

The current release targets Phys-AD. IPAD support and its paper experiment are deferred by the project owner; they are not a blocker for completing this release. External optical-flow/tracks/trajectory-video/plot generation also remains outside this repository.

## Remaining paper coverage

- Verify the real-model workflow in a clean environment and record dependency/model versions.
- IPAD is explicitly deferred by the project owner and is outside the current release scope. Its dataset mapping and official splits will be integrated later.
- Modality and selection recipes are implemented and tested offline. They are reconstructed recipes; the original commands cannot be recovered.
- External baseline integration is documented with an ID-preserving importer; their model/checkpoint execution remains in the original implementations.
- Clarify the license of the extracted OpticalFlow selection code before public release.

The upstream keyframe source has no LICENSE file in the local checkout. A source
notice is included; no license is invented for that code or the final project.

## API reference

The judge transport follows the official
[Chat Completions API reference](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create).
Model selection remains GPT-4o; no automatic migration to a newer judge is performed.

## Cross-version review

See [validation details](VALIDATION.md) for completed packaging/CLI checks and [version policy](VERSION_POLICY.md) for module-by-module source decisions. The Send-Physics-AD review identified baseline execution and schema mismatches; these findings are not proof that the historical paper runs used the same broken paths. Original research directories were not modified.

## Main-table provenance

Cross-checking saved detection runs has not established which full run generated the paper's main MUSTARD AUROC column. A close overall average is not treated as a match; all per-category values must agree. The matched PAEval explanation input does not contain anomaly scores and cannot substitute for that detection provenance. The original fusion experiment source cannot be obtained. The project owner has confirmed that its recovery is not a prerequisite for completing this implementation. No historical scores or experiment artifacts are bundled here.

## Recipe reconstruction

The full-model fusion prompt is preserved. Two-source fusion prompts explicitly exclude omitted modalities; single-source runs preserve their original pattern reasoning as an explicitly identified explanation. These reconstructed recipes are described in EXPERIMENTS.md and are not represented as recovered historical commands. The full Phys-AD implementation is not blocked on the missing original run script.
