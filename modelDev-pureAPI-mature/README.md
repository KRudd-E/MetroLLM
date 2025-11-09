# modelDev-pureAPI-mature (_AI-Generated, Human-Reviewed_)

Purpose
-------
This module implements a lightweight pipeline that classifies and extracts structured information from case-study documents using only a commercial LLM (OpenAI) via its API. It is intended as a minimal, easily auditable "pure API" alternative to fine-tuned models: retrieve textual content, build prompts, call the API, and post-process responses into CSV outputs for analysis.

This README describes the purpose, code layout, how the pieces fit together, and the minimal steps to run the pipeline locally.

What this module does (high level)
---------------------------------
- Read prepared case-study text (or a CSV of inputs).
- Format prompts per-case using templates and config.
- Call the OpenAI API (with retries/ratelimit handling) to request classifications / structured extractions.
- Parse and assemble model responses into CSV result files and archive previous runs.

Repository layout (important files)
-----------------------------------
- __main__.py
  - CLI entrypoint. Loads config, initializes the pipeline, and runs the requested action (classify, preview, dry-run, or other modes implemented in code).
- config.yaml
  - Central configuration (input/output paths, model name, batch sizes, prompt templates, parsing options). Edit this to change runtime behaviour.
- results/
  - Output CSVs and artifacts. Key files include:
    - out_full.csv / out_full_processed.csv — main outputs with model responses and parsed fields.
    - out_comparison.csv — side-by-side comparisons or analysis outputs.
    - archive/ — previous run outputs (timestamped).
- src/
  - .env
    - Optional local env file (kept in src for convenience) — typically contains OPENAI_API_KEY or similar variables.
  - ai.py
    - OpenAI API wrapper. Responsible for constructing request payloads, batching, retries, basic rate-limit handling, and converting raw responses to strings.
    - Place to add backoff behavior, change temperature, stop tokens, or switch to a different API endpoint.
  - retrieve.py
    - Input loaders and light preprocessing. Reads input sources (text files, CSVs) and returns records to the pipeline.
    - Handles basic cleaning, chunking of long documents, and de-duplication logic where present.
  - pipeline.py
    - Orchestrates the end-to-end flow: uses retrieve.py to get inputs, uses ai.py to call the model, applies post-processing/parsing rules, and writes results to results/.
    - Implements batching, per-record metadata, and result assembly.
  - utils.py
    - Utility helpers: file I/O wrappers, logging helpers, small text utilities and CSV helpers used across the module.

Design notes / how the code fits together
-----------------------------------------
- __main__.py reads config.yaml and passes it to Pipeline in src/pipeline.py.
- Pipeline asks retrieve.py for the next batch of inputs, formats each prompt (using templates or functions), and calls ai.py to get model outputs.
- ai.py performs the external API call (with configurable parameters from config.yaml) and returns raw text responses. Pipeline then uses parsing logic to convert free-text answers into structured fields.
- Final CSV outputs are saved into results/ and previous outputs are archived.

How to run (general)
--------------------------
1. Environment
   - Python 3.10+ recommended.
   - Preferred: use a virtualenv or conda env.
     Example:
     ```
     python -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt  # if present
     # otherwise:
     pip install openai pandas pyyaml python-dotenv
     ```

2. Credentials
   - Provide an OpenAI API key:
     - Option A: export in shell (recommended on Mac)
       ```
       export OPENAI_API_KEY="sk-..."
       ```
     - Option B: place in src/.env as:
       ```
       OPENAI_API_KEY=sk-...
       ```
     - The code reads env vars (via python-dotenv if installed). Confirm ai.py loads them (or set in your environment).

3. Configure
   - Edit config.yaml in this folder to set:
     - input source path (text/CSV)
     - output folder (results/)
     - model name (e.g., gpt-4o-mini, gpt-4, gpt-3.5-turbo)
     - batch size, retry limits, prompt templates
   - Example adjustments: lower batch_size for rate-limited accounts, change prompt template to tune extraction.

4. Run
   - From this directory:
     ```
     cd /path/to/modelDev-pureAPI-mature
     python __main__.py
     ```
   - If you need module-style execution (if package configured), you can also try:
     ```
     python -m modelDev-pureAPI-mature
     ```
   - CLI flags: inspect __main__.py for supported arguments (e.g., dry-run, preview, run type). Typical usage is the default action which runs the full pipeline and writes CSVs to results/.

Extending or changing behavior
------------------------------
- Prompt templates: edit config.yaml or the prompt-building logic in pipeline.py to alter what you ask the model.
- Parsing: adjust parsing rules in pipeline.py or utils.py to change how free-text responses are turned into structured fields.
- API behaviour (backoff/retries): modify ai.py for custom retry/backoff strategies, or to add batching across more robust async callers.

Where to look first in code (for a developer)
---------------------------------------------
- Start at __main__.py to see CLI and config loading.
- Open src/pipeline.py to understand orchestration and how records flow through the system.
- Inspect src/ai.py to see exact request formats and where to adjust model params.
- Check src/retrieve.py for input formats supported and sample readers.

Context: repository-level README
-------------------------------
This folder is one module of the larger MetroLLM repository. For context about upstream data processing (dataDev1, dataDev2) and how this module fits into the whole pipeline, see the top-level README at the repository root. That README documents the full DataDev → ModelDev workflow and where this pure-API module is used in evaluation and comparisons.

Troubleshooting tips
--------------------
- If outputs are empty, check config.yaml input path and OPENAI_API_KEY availability.
- If you hit rate limits, reduce batch_size or add delays in ai.py.
- Inspect results/archive for previous runs to compare behavior after changing prompts or model.

License / notes
---------------
- This module relies on the OpenAI API; be mindful of cost, rate limits, and data privacy for case-study contents.