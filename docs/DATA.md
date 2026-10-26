# Data contracts

All sample manifests and generated artifacts belong in an external working directory.
The repository does not bundle sample outputs or experiment results.

## Manifest

A manifest is a JSON list. Each sample contains:

| Field | Meaning |
|---|---|
| object | Category name |
| video | Video ID, preserved as a string |
| split | `train` or `test` |
| anomaly | Test subtype; null for train |
| label | 0 normal, 1 anomalous; train entries must be 0 |
| raw_video | Original mp4 path |
| trajectory_video | Aligned arrow-overlay mp4 path |
| tracks | NumPy `(T,K,2)` float coordinate array |
| plot | External flow-magnitude PNG path |

Paths may be absolute or relative to the manifest. A sample is uniquely identified
by object, split, anomaly, and video. Labels are used only to validate normal
training data and compute evaluation; they are not included in model prompts.

IPAD integration is deferred for this release. A later adapter can map its inputs
into this explicit manifest after its official split and anomaly semantics have
been verified. No IPAD reproduction is claimed.

## Predictions

The infer command writes an object containing config, model digests, the pattern
artifact hash, and a predictions list. Each item includes object, video, anomaly,
label, score, full fusion response, raw description and modality reasoning.
Baselines use the same identifying fields. Normalized baseline and single-source
records have an explicit explanation plus explanation_source, preserving their
original text. Full fusion records have a response with Final Explanation (or
Reasoning only when explicitly chosen for a separate evaluation). An omitted
full-model Final Explanation is an error, not an instruction to use another field.

## Judge labels

The recovered PAEval format maps each object to `description_label` and
`explanation_label`. Each maps anomaly subtypes to a list of reference texts.
These are external annotations, not source code, and are not included here.

## Optical-flow preprocessing handoff

`mustard preprocess-optical-flow` can generate the tracks, trajectory videos, and
plots from the raw Phys-AD tree. Alternatively, an external provider may supply
all four inputs with frame-preserving correspondence.
Keyframe selection never uses the plot PNG to infer indices; it computes the motion
curve from the numeric tracks. The plot is independently consumed by the caption
model. The built-in command exposes grid density, TV-L1 working resolution, arrow
rendering, motion aggregation, and plot smoothing options without changing the
downstream pipeline. The tracks root also contains `preprocessing.json`, which
records those options and prevents accidental reuse under a different configuration.

Normal-directory aliases determine binary labels; they do not rename anomaly subtypes. Explanation reference libraries must use the manifest's actual object/anomaly names, or be explicitly mapped before judging.
