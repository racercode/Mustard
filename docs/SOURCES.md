# Source provenance

Keyframe selection derives from `cswagger/OpticalFlow`, local commit
`88136bb2feaa537969d1594c0955f23a82df05cb`, file `select_key_frame.py`.
Original file SHA-256: `62c68c6dacb5236cfcdabe8a04c3525d6c0c239bf3c4cb316054140dcb561295`.
The selection functions are extracted without the old command-line, plotting,
or filesystem code. SciPy is required instead of using the silent peak-finder fallback.
The optical-flow integration derives its TV-L1 grid tracking from
`OpticalFlow/tracker.py`, motion plotting from `OpticalFlow/clean_motion.py`, and
trajectory rendering from `OpticalFlow/visualize_new.py`, at the same local commit.
Hard-coded filesystem paths, experimental artifacts, mask-only branches, model
captioning scripts, and unrelated keyframe command lines are not included.

The local upstream checkout has no LICENSE file. Upstream licensing must be
clarified before public release; this draft does not assert a license for that code.

## Prompt sources

- `2080/rule_ablation/pattern.py`: SHA-256 `7b59ad13878d4873cc781cc5d986f3420b55d4473c2b8a1ebc333a217d24b9b1`
- `si2025_3/rule_ablation/MultiSourceAUC/multisource_auc.py`: SHA-256 `853a413ff87181d46fabc1e33247d04380bf70ae805e43a749c13337ed0b9d3f`
- `si2025_3/follow-the-rules-phys-ad/timing/benchmark.py`: SHA-256 `41ac605d39ef0606f03adb2679c1948d4347b08b7c1af55c59fb453b4aeecca4`
- `OpticalFlow/plot-system-prompt.txt`: SHA-256 `bfd9beaca4dbb5ac05bcf1a95b116f9dd37823ed5c23673bd109b1f10e77c892`
- `OpticalFlow/plot-user-prompt.txt`: SHA-256 `a0d25b0b8e5ae5ba49fe44406f0889bf4f20f6e8e40c2bb722953f4ac2ca52c7`

The prompt functions have been extracted without module-level clients, path constants, or historical main blocks. These are source strings, not experiment results. The upstream Follow the Rules MIT notice is preserved in THIRD_PARTY_LICENSES.md. Final project licensing has not been assigned.

## Evaluation sources

PAEval prompts and object-level aggregation follow `2080/test/Physics-AD/PAEval/text_compare.py` and `evaluator.py`. Pairwise prompt templates derive from `PAEval/pairwise_compare.py`; no API keys or outputs were copied. The 2080 two-stage exporter selects `Reasoning:`, while the si2025_3/Physics-AD and Send-Physics-AD copies select `Final Explanation:`. Cross-checking the latter input and recorded evaluation against the paper establishes Final Explanation as the default; Reasoning remains an explicit alternative.

## Reconstructed experiment adapters

Static-pattern/static-anomaly prompt functions are extracted from the same recorded 2080/rule_ablation/pattern.py source. The direct-Qwen question is extracted from the recorded timing/benchmark.py source. Reconstructed modality adapters preserve the full-method fusion prompt, explicitly identify available inputs in reduced-source fusion, and preserve single-source reasoning without an extra model rewrite. No unrecovered historical recipe is claimed.
