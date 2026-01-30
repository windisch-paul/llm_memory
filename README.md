# LLM Memory

Evaluates whether large language models "read" provided text or "remember" information from training data when classifying oncology trials.

## Overview

This project investigates LLM reliance on memorized training data versus comprehension of input text. Models classify randomized controlled oncology trials as POSITIVE (met primary endpoint) or NEGATIVE (did not meet primary endpoint) under varying input conditions.

## Models Tested

- **OpenAI:** GPT-5.2 (`gpt-5.2-2025-12-11`)
- **Google:** Gemini 3 Flash (`gemini-3-flash-preview`)
- **Anthropic:** Claude Opus 4.5 (`claude-opus-4-5-20251101`)

## Input Conditions

| Condition | Description |
|-----------|-------------|
| Baseline | Title + Abstract (full information) |
| Title only | Only the trial title |
| DOI only | Only the DOI identifier |
| Counterfactual | Modified title + abstract with inverted outcomes |
| Counterfactual + DOI | DOI combined with counterfactual content |

## Project Structure

```
├── run_evaluation.py      # Main evaluation pipeline
├── analyze_results.py     # Results analysis and metrics
├── visualizations.py      # Plot generation
├── data/trials.csv        # Source trial data (250 trials)
├── results/               # Output predictions and metrics
└── plots/                 # Generated visualizations
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation
python run_evaluation.py

# Analyze results
python analyze_results.py

# Generate plots
python visualizations.py
```

Requires a `.env` file with `OPENAI_API_KEY`, `GEMINI_API_KEY`, and `ANTHROPIC_API_KEY`.
