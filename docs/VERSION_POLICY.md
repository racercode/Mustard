# Research-version policy

The assembled repository uses sources by component, rather than treating one
research directory or Git HEAD as the authoritative version for every experiment.

| Component | Source decision |
|---|---|
| Main method | Recovered MUSTARD reasoning/fusion prompts with the paper's keyframe input path |
| Keyframes | Recovered selector and batch parameters, shared between raw and trajectory videos |
| Pattern extraction | Position-wise aggregation of normal reference videos, followed by concatenation |
| PAEval field | Final Explanation, following the Send-Physics-AD/Physics-AD exporter variant |
| PAEval judge | GPT-4o snapshot used by the recovered PAEval caller |
| Pairwise judge | GPT-4o, explicitly confirmed by the project owner; the legacy GPT-5 default is not retained |
| Pairwise alignment | Explicit video IDs and recorded answer-order seed |
| Normal labels | Explicit manifest labels, with norm/normal/good/anomaly_free indexing aliases |
| External preprocessing | Tracks, trajectory videos and plots are provided by another component |
| IPAD | Deferred from this release |

Send-Physics-AD is a Phys-AD baseline benchmark fork with custom PAEval additions,
not a second complete MUSTARD implementation. Its three local copies share an
upstream Git HEAD, but have different working-tree changes and untracked files.
The current baseline entry points include partial/debug runs, format mismatches,
and machine-specific imports. They must not be copied wholesale and advertised
as the exact paper experiment recipes.

## Intentional fixes

- Do not repeat a successful judge request ten times.
- Reject pairwise responses containing more than one verdict.
- Keep video identities when exporting explanations and joining baselines.
- Validate all required fusion fields before caching a successful response.
- Require explicit normal-label mappings and complete AUROC test coverage.
- Keep invalid or empty explanation fields as errors rather than changing the
  evaluated field silently.

These changes improve the execution contract but are not evidence that historical
model responses or published scores have been reproduced. Live-model validation
and baseline experiment provenance remain separate release work.

Baseline models need their own preprocessing, feature files, model checkpoints,
and dependency environments. Preserve those interfaces when adding integrations;
do not replace every baseline's input sampler with the MUSTARD keyframe sampler.
