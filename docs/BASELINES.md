# Baseline integration

The direct-Qwen baseline is implemented here. Other methods remain external
implementations with their own checkpoints, data representations and environments.
This repository provides ID-preserving evaluation/import interfaces rather than
vendoring incompatible or broken historical checkouts.

| Paper baseline | External execution requirements |
|---|---|
| MNAD prediction/reconstruction | Separate method settings, frame representation and checkpoints |
| VADCLIP | CLIP features, verified train/test CSVs, and trained checkpoint |
| S3R | I3D features, dictionaries, category config, and trained checkpoint |
| Video-ChatGPT | Matching projection/vision weights, verified loader and question sequence |
| Video-LLaMA | Matching visual/audio weights, frame loader, and output parser |
| Video-LLaVA | Model environment and method-specific frame/question settings |
| LAVAD | Caption, index, summary and scoring workflow with an explicit experiment path |
| AnomalyRuler / O-VAD | Their original inference pipeline and checkpoint/model setup |

The local Phys-AD benchmark forks contain debug defaults, cross-checkout imports,
normal-label alias mismatches and incomplete runner/evaluator pairs. Running those
files unchanged is not an established reproduction recipe. In particular, keep
video IDs when converting results, match the chosen checkpoints and normal-label
convention, and retain each method's own sampling protocol. See [version policy](VERSION_POLICY.md).

## Import explanation records

Convert an external method's output into a JSON list of records. Each record has
object, anomaly, video (string ID), and explanation. A score in [0,1] may be included
for AUROC; it is optional for explanation-only comparisons. The exact text is
preserved without a new model call.

```bash
mustard import-baseline \
  --records /work/external-method/records.json \
  --manifest /work/physad/manifest.json \
  --source external-method-name \
  --out /work/external-method/predictions.json
```

The importer validates test coverage, duplicates and supplied labels against the
manifest. An old explanation list grouped by category and anomaly is not enough
to recover video identities. Do not infer its IDs from list position unless the
corresponding original ordering manifest is available.

Use the resulting predictions with `mustard pairwise`. If all records have scores,
use `mustard evaluate` against the same manifest. Scores and prose are external
artifacts and are not copied into this repository.
