# Quick start

1. Install the package from the repository root:

   ```bash
   python -m pip install -e .
   ```

2. Obtain the external preprocessing inputs and index them:

   ```bash
   mustard index --raw-root /data/raw --trajectory-root /data/trajectory \
     --tracks-root /data/tracks --plots-root /data/plots \
     --out /work/physad/manifest.json
   ```

3. Validate alignment before running a model:

   ```bash
   mustard validate-inputs --manifest /work/physad/manifest.json \
     --out /work/physad/input-check.json
   ```

   Add --decode to count fully decoded frames rather than reading frame-count
   metadata. The fast check validates numeric track shape and the PNG header;
   actual selection also rejects non-finite coordinates. It does not estimate
   optical flow or render arrows/plots.

4. Start an external Ollama server with qwen2.5vl:7b and gemma3:12b available.
   Then run the full method:

   ```bash
   bash scripts/run_physad.sh /work/physad/manifest.json /work/physad/full
   ```

   Use --objects through `mustard experiment` for a category subset. The full
   recipe requires 30 normal training references per category. The offline tests
   use a smaller synthetic set solely to test the execution contract.

5. Run explanation judging only when requested, after setting OPENAI_API_KEY.
   See EXPERIMENTS.md for commands and all comparison recipes.

Re-running the same work directory resumes successful model calls. Use a new work
directory when changing the dataset, configuration or resolved models.
