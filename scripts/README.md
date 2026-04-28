# Run Evaluation Utility

This folder contains a utility script to evaluate a run output zip and generate report-ready summaries.

## Script

- evaluate_run.py

## What it does

Given a run zip or extracted run directory, it will:

1. Read images/run_summary.json
2. Check reliability and policy-direction consistency
3. Compute an overall verdict
4. Optionally compare against a previous run
5. Write two files in the run folder:
   - evaluation_summary.md
   - evaluation_tables.tex

## Usage

From project root:

python3 scripts/evaluate_run.py desertification_outputs_20260418_154505.zip

With previous run comparison:

python3 scripts/evaluate_run.py \
  desertification_outputs_20260418_154505.zip \
  --previous output_review/20260418_142206/images/run_summary.json

## Input options

The input and previous arguments can be any one of:

- Run zip file
- Extracted run folder
- Direct path to run_summary.json

## Outputs

For run 20260418_154505, outputs are created at:

- output_review/20260418_154505/evaluation_summary.md
- output_review/20260418_154505/evaluation_tables.tex

## LaTeX integration tip

You can insert the generated table into a report with:

\input{output_review/20260418_154505/evaluation_tables.tex}
