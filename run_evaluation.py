#!/usr/bin/env python3
"""
LLM Memory Evaluation Pipeline

Evaluates whether LLMs are "reading" vs "remembering" oncology trial information
by comparing performance across input conditions:
- baseline: Title + Abstract (full information)
- title_only: Only the trial title
- doi_only: Only the DOI identifier

Models: OpenAI GPT-5.2, Gemini 3 Flash Preview, Anthropic Claude Opus 4.5
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# API clients (imported conditionally)
openai_client = None
genai = None
anthropic_client = None


SYSTEM_PROMPT = """You will be provided with information about a randomized controlled oncology trial.
Your task will be to classify if the trial was positive, i.e. if it met its primary
endpoint, or negative, i.e. if it did not meet its primary endpoint. Your response
should be either the word POSITIVE (in all caps) or NEGATIVE (in all caps). Do not output anything else."""

# Model configurations
MODELS = {
    "openai": "gpt-5.2-2025-12-11",
    "gemini": "gemini-3-flash-preview",
    "anthropic": "claude-opus-4-5-20251101",
}

CONDITIONS = ["baseline", "title_only", "doi_only", "counterfactual", "counterfactual_with_doi"]


def init_openai():
    """Initialize OpenAI client."""
    global openai_client
    if openai_client is None:
        from openai import OpenAI
        openai_client = OpenAI()
    return openai_client


def init_gemini():
    """Initialize Gemini client using the google-genai SDK."""
    global genai
    if genai is None:
        from google import genai as genai_module
        genai = genai_module.Client(api_key=os.environ["GEMINI_API_KEY"])
    return genai


def init_anthropic():
    """Initialize Anthropic client."""
    global anthropic_client
    if anthropic_client is None:
        import anthropic
        anthropic_client = anthropic.Anthropic()
    return anthropic_client


def build_user_prompt(condition: str, title: str, abstract: str, doi: str) -> str:
    """Build user prompt based on condition."""
    if condition == "baseline":
        return f"Title: {title}\n\nAbstract: {abstract}"
    elif condition == "title_only":
        return f"Title: {title}"
    elif condition == "doi_only":
        return f"DOI: {doi}"
    elif condition == "counterfactual":
        return f"Title: {title}\n\nAbstract: {abstract}"
    elif condition == "counterfactual_with_doi":
        return f"DOI: {doi}\n\nTitle: {title}\n\nAbstract: {abstract}"
    else:
        raise ValueError(f"Unknown condition: {condition}")


def parse_output(raw_output: str | None) -> str:
    """Parse model output to POSITIVE, NEGATIVE, INVALID, or ERROR."""
    if raw_output is None:
        return "ERROR"
    text = raw_output.strip().upper()
    if text == "POSITIVE":
        return "POSITIVE"
    elif text == "NEGATIVE":
        return "NEGATIVE"
    else:
        return "INVALID"


def call_openai(model: str, system_prompt: str, user_prompt: str) -> str | None:
    """Call OpenAI API. Returns raw output or None on error."""
    try:
        client = init_openai()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_completion_tokens=500,
        )
        return response.choices[0].message.content.strip() if response.choices[0].message.content else None
    except Exception as e:
        print(f"  OpenAI API error: {e}", file=sys.stderr)
        return None


def call_gemini(model: str, system_prompt: str, user_prompt: str) -> str | None:
    """Call Gemini API using the google-genai SDK. Returns raw output or None on error."""
    try:
        from google.genai import types

        client = init_gemini()
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,
                max_output_tokens=500,
            ),
        )
        return response.text.strip() if response.text else None
    except Exception as e:
        print(f"  Gemini API error: {e}", file=sys.stderr)
        return None


def call_anthropic(model: str, system_prompt: str, user_prompt: str) -> str | None:
    """Call Anthropic API. Returns raw output or None on error."""
    try:
        client = init_anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=500,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        if response.content and len(response.content) > 0:
            return response.content[0].text.strip()
        return None
    except Exception as e:
        print(f"  Anthropic API error: {e}", file=sys.stderr)
        return None


def call_api(vendor: str, model: str, system_prompt: str, user_prompt: str) -> str | None:
    """Dispatch to appropriate API."""
    if vendor == "openai":
        return call_openai(model, system_prompt, user_prompt)
    elif vendor == "gemini":
        return call_gemini(model, system_prompt, user_prompt)
    elif vendor == "anthropic":
        return call_anthropic(model, system_prompt, user_prompt)
    else:
        raise ValueError(f"Unknown vendor: {vendor}")


def load_existing_predictions(predictions_path: Path) -> set[tuple]:
    """Load existing predictions and return set of completed keys."""
    completed = set()
    if predictions_path.exists():
        df = pd.read_csv(predictions_path)
        for _, row in df.iterrows():
            key = (row["trial_idx"], row["vendor"], row["condition"])
            completed.add(key)
    return completed


def save_prediction(
    predictions_path: Path,
    trial_idx: int,
    vendor: str,
    model: str,
    condition: str,
    raw_output: str | None,
    parsed_output: str,
    ground_truth: str,
):
    """Append a single prediction to CSV."""
    row = {
        "trial_idx": trial_idx,
        "vendor": vendor,
        "model": model,
        "condition": condition,
        "raw_output": raw_output,
        "parsed_output": parsed_output,
        "ground_truth": ground_truth,
    }

    df = pd.DataFrame([row])

    # Append mode: write header only if file doesn't exist
    write_header = not predictions_path.exists()
    df.to_csv(predictions_path, mode="a", header=write_header, index=False)


def main():
    parser = argparse.ArgumentParser(description="Run LLM memory evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Print total API calls without running")
    parser.add_argument("--max-trials", type=int, default=None, help="Limit number of trials to process")
    parser.add_argument("--condition", type=str, choices=CONDITIONS, default=None, help="Run specific condition only")
    parser.add_argument("--vendor", type=str, choices=list(MODELS.keys()), default=None, help="Run specific vendor only")
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Paths
    script_dir = Path(__file__).parent
    trials_path = script_dir / "data" / "trials.csv"
    predictions_path = script_dir / "results" / "predictions.csv"

    # Ensure results directory exists
    predictions_path.parent.mkdir(exist_ok=True)

    # Load trials
    trials_df = pd.read_csv(trials_path)
    if args.max_trials:
        trials_df = trials_df.head(args.max_trials)

    # Determine conditions and vendors to run
    conditions = [args.condition] if args.condition else CONDITIONS
    vendors = [args.vendor] if args.vendor else list(MODELS.keys())

    # Dry run mode
    if args.dry_run:
        total_calls = len(trials_df) * len(vendors) * len(conditions)
        print(f"Conditions: {len(conditions)} ({', '.join(conditions)})")
        print(f"Vendors: {len(vendors)} ({', '.join(vendors)})")
        print(f"Trials: {len(trials_df)}")
        print(f"Total API calls: {total_calls}")
        return

    # Check for API keys
    if "openai" in vendors and not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set", file=sys.stderr)
    if "gemini" in vendors and not os.environ.get("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY not set", file=sys.stderr)
    if "anthropic" in vendors and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set", file=sys.stderr)

    # Load existing predictions for resume
    completed = load_existing_predictions(predictions_path)
    print(f"Loaded {len(completed)} existing predictions")

    # Calculate totals
    total_calls = len(trials_df) * len(vendors) * len(conditions)
    remaining_calls = total_calls - len(completed)
    print(f"Total calls needed: {total_calls}, remaining: {remaining_calls}")

    # Main evaluation loop: conditions -> trials -> vendors
    call_count = 0
    for condition in conditions:
        print(f"\n=== Condition: {condition} ===\n")

        for trial_idx, row in trials_df.iterrows():
            # Get appropriate fields based on condition
            if condition in ("counterfactual", "counterfactual_with_doi"):
                title = row["counterfactual_title"]
                abstract = row["counterfactual_abstract"]
                ground_truth = row["target_label"]
            else:
                title = row["title"]
                abstract = row["abstract"]
                ground_truth = row["Annotation_accept"]
            doi = row["doi"]

            for vendor in vendors:
                # Check if already done
                key = (trial_idx, vendor, condition)
                if key in completed:
                    continue

                model = MODELS[vendor]

                # Build prompt and call API
                user_prompt = build_user_prompt(condition, title, abstract, doi)
                print(f"[{call_count + 1}/{remaining_calls}] Trial {trial_idx}, {vendor}, {condition}")

                raw_output = call_api(vendor, model, SYSTEM_PROMPT, user_prompt)
                parsed_output = parse_output(raw_output)

                # Save immediately
                save_prediction(
                    predictions_path,
                    trial_idx,
                    vendor,
                    model,
                    condition,
                    raw_output,
                    parsed_output,
                    ground_truth,
                )

                call_count += 1

                # Small delay between calls to avoid rate limiting
                time.sleep(0.5)

                # Print progress
                if parsed_output in ("INVALID", "ERROR"):
                    print(f"  -> {parsed_output}: {raw_output}")
                else:
                    print(f"  -> {parsed_output}")

    print(f"\nCompleted {call_count} API calls")
    print(f"Results saved to {predictions_path}")


if __name__ == "__main__":
    main()
