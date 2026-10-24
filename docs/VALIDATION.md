# Validation performed

The implementation was checked locally with Python 3.14. The following checks
passed without a GPU, real model inference, or paid API requests:

- Eighteen offline tests, including file/CLI integration with synthetic videos.
- All five modality recipes and both direct-Qwen samplers.
- Input validation, normal-reference selection, pattern extraction, inference,
  complete-coverage AUROC, PAEval, pairwise and baseline import.
- Resume through validated cached calls and refusal to overwrite a different experiment.
- Package wheel build, package-data inclusion, and standalone CLI imports.
- Installing that wheel into a separate venv with existing system dependencies,
  then running CLI help and all eighteen tests from outside the source directory.

The wheel-install test used `--no-deps` and the venv inherited already installed
system dependencies. It is not claimed as a fresh dependency download or a test
of every supported Python version. GitHub Actions is configured to run the offline
suite on Python 3.10 and 3.12; that hosted workflow has not been executed here.

The integration test replaces HTTP responses locally. It does not bind a model
server, contact OpenAI/Ollama, load weights, or establish model quality.

To check locally after installation:

```bash
python -m unittest discover -s tests -v
mustard --help
```

Real-data verification requires the separately provided tracks, trajectory videos
and plots, available model weights, and the chosen judge credentials. No claim
is made that the historical paper tables have been independently reproduced.
