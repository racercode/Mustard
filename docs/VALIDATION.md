# Validation performed

The implementation was checked locally with Python 3.12.3. The following checks
passed without a GPU, real model inference, or paid API requests:

- Twenty-five offline tests, including real TV-L1 execution and file/CLI
  integration with synthetic videos.
- All five modality recipes and both direct-Qwen samplers.
- Input validation, normal-reference selection, pattern extraction, inference,
  complete-coverage AUROC, PAEval, pairwise and baseline import.
- Optical-flow tree mirroring, settings-change refusal, anti-flicker visibility,
  trajectory rendering, motion plots, and resumable output behavior.
- Resume through validated cached calls and refusal to overwrite a different experiment.
- Package wheel build, package-data inclusion, and standalone CLI imports.
- Installing the wheel with its preprocessing extra into a fresh venv, then
  running CLI help and the offline suite from outside the source directory.
- A two-video Phys-AD-shaped `ball` smoke fixture passed preprocessing,
  indexing, and decoded input validation.

GitHub Actions is configured to install the preprocessing extra and run the offline
suite on Python 3.10 and 3.12. That hosted workflow has not been executed here.

The integration test replaces HTTP responses locally. It does not bind a model
server, contact OpenAI/Ollama, load weights, or establish model quality. The
Phys-AD-shaped smoke fixture is not the official full dataset and does not establish
paper-level model results.

To check locally after installation:

```bash
python -m unittest discover -s tests -v
mustard --help
```

Official full-data verification still requires the Phys-AD videos, available model
weights, and chosen judge credentials. No claim is made that the historical paper
tables have been independently reproduced.
