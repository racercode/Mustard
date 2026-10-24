# Phys-AD experiment recipes

These are reconstructed, executable recipes for the paper's method and comparisons.
Historical main-table run scripts could not be recovered. Recipe availability
therefore does not imply that published numbers have been independently reproduced.
No results or generated patterns are bundled.

## Full method

```bash
bash scripts/run_physad.sh /work/physad/manifest.json /work/physad/full
```

This command selects normal references, extracts patterns, runs test inference,
and computes AUROC. The work directory records the exact manifest, effective
configuration, model digests, reference IDs/input hashes, runtime, predictions,
and metrics. A directory belonging to a different experiment is rejected;
repeating the same experiment resumes validated calls from its cache.

## Modality ablations

```bash
bash scripts/run_ablations.sh /work/physad/manifest.json /work/physad/ablations
```

| Recipe | Inference inputs | Induced patterns | Evaluated explanation |
|---|---|---|---|
| raw | Raw-frame descriptions | Static, position-wise | Single-source reasoning, explicitly recorded |
| trajectory | Arrow-frame descriptions | Dynamic, position-wise | Single-source reasoning, explicitly recorded |
| raw_flow | Raw descriptions and flow reasoning | Flow | Final Explanation |
| trajectory_flow | Trajectory reasoning and flow reasoning | Dynamic and flow | Final Explanation |
| full | Raw descriptions, trajectory reasoning and flow reasoning | Dynamic and flow | Final Explanation |

All five use the same keyframe configuration, frame budget, and category-specific
normal-reference selection. Reference IDs remain stable when running a category
subset or a different modality combination. Omitted modalities do not generate
model calls. External input manifests retain a shared four-input contract even
when a recipe does not consume every modality.

The full model retains the recovered fusion prompt. The two-source ablation
fusion prompts explicitly describe only their available sources. Single-source
runs use the recovered static/dynamic anomaly prompts and preserve their reasoning
as an explicit explanation record; they do not invent a Final Explanation field.
These choices are documented reconstruction decisions, not recovered historical
ablation commands.

## Frame-selection comparison

```bash
bash scripts/run_selection_comparison.sh /work/physad/manifest.json /work/physad/selection
```

The same direct-Qwen video question is applied to eight raw frames selected by
uniform sampling or flow-guided keyframe selection. This is a direct MLLM baseline
without pattern extraction, rather than a change to the full method's fusion.
The uniform rule is `step=(T-1)//7`, matching the recovered uniform extractor.

The older uniform full-pipeline profile is also available:

```bash
mustard experiment --manifest /work/physad/manifest.json \
  --config configs/benchmarks/uniform_pipeline.json \
  --work /work/physad/uniform-pipeline
```

This preserves a uniform pipeline comparison profile. It does not claim to
reproduce the old cold-cache timing methodology or timing numbers.

## Explanation metrics

Run judging explicitly after generating predictions; experiment runners do not
start paid judging automatically.

```bash
mustard paeval --predictions /work/physad/full/predictions.json \
  --labels /data/correct_ground_truth.json \
  --work /work/physad/judge-cache --out /work/physad/full/paeval.json

mustard pairwise --predictions /work/physad/full/predictions.json \
  --baseline /work/physad/selection/qwen_uniform/predictions.json \
  --labels /data/correct_ground_truth.json \
  --work /work/physad/judge-cache --out /work/physad/qwen-pairwise.json
```

Set OPENAI_API_KEY in your environment. Both metrics default to the confirmed
GPT-4o judge snapshot. The full model is evaluated using Final Explanation;
normalized single-source/baseline records explicitly identify their explanation
source. Missing full-model Final Explanation never silently falls back to Reasoning.

## Category subsets

`mustard experiment --objects ball gear ...` restricts the run and writes the
restricted manifest into the work directory. If evaluating separately, use that
saved manifest so coverage validation refers to exactly the selected categories.
