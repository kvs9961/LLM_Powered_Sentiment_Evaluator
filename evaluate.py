import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
import concurrent.futures

from mistralai.client import Mistral
from tqdm import tqdm

from datetime import datetime, timezone, timedelta

# Setup

VALID_LABELS = {"positive", "neutral", "negative"}

SYSTEM_PROMPT = """You are a sentiment classifier for customer call transcripts.
Classify the sentiment of the snippet as one of: positive, neutral, or negative.
Reply with only that one word. Nothing else."""

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Load dataset

def load_dataset(path):
    """Load examples from a CSV file."""
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")

    examples = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader, start=2):
            label = row["label"].strip().lower()
            snippet = row["snippet"].strip()

            if label not in VALID_LABELS:
                logger.warning(f"Row {i}: unknown label '{label}', skipping")
                continue
            if not snippet:
                logger.warning(f"Row {i}: empty snippet, skipping")
                continue

            examples.append({
                "id": row["id"].strip(),
                "snippet": snippet,
                "label": label
            })

    logger.info(f"Loaded {len(examples)} examples from {path}")
    return examples

# Classify a single example

def classify_one(client, example):
    """Send one snippet to Mistral and return the predicted label."""
    start = time.time()
    raw = ""
    error = None

    try:
        response = client.chat.complete(
            model="mistral-small-latest",  # free tier model
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify this snippet:\n\n{example['snippet']}"}
            ],
            max_tokens=10,
        )
        raw = response.choices[0].message.content.strip()
        predicted = parse_label(raw)

    except Exception as e:
        logger.error(f"API error on id={example['id']}: {e}")
        predicted = "unknown"
        error = str(e)

    latency_ms = int((time.time() - start) * 1000)

    return {
        "id": example["id"],
        "snippet": example["snippet"],
        "ground_truth": example["label"],
        "predicted": predicted,
        "raw_response": raw,
        "correct": predicted == example["label"],
        "error": error,
        "latency_ms": latency_ms
    }


def parse_label(raw):
    """
    Extract a valid label from the LLM response. 
    The model is asked for one word but sometimes returns 'Positive!' or 'The sentiment is negative.' — this handles those cases.
    """
    cleaned = re.sub(r"[^\w\s]", "", raw.lower()).strip()
    for word in cleaned.split():
        if word in VALID_LABELS:
            return word
    return "unknown"


def run_evaluation(examples, workers=5):

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(classify_one, client, ex): ex for ex in examples}

        with tqdm(total=len(futures), desc="Classifying") as pbar:
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                pbar.update(1)

    order = {ex["id"]: i for i, ex in enumerate(examples)}
    results.sort(key=lambda r: order.get(r["id"], 0))
    return results


# Compute metrics

def compute_metrics(results):
    """Calculate accuracy per-class."""
    valid = [r for r in results if not r["error"]]
    correct = [r for r in valid if r["correct"]]
    errors = [r for r in results if r["error"]]
    unknown = [r for r in valid if r["predicted"] == "unknown"]

    accuracy = len(correct) / len(valid) if valid else 0

    # confusion matrix
    confusion = {}
    for label in VALID_LABELS:
        confusion[label] = {l: 0 for l in VALID_LABELS}
        confusion[label]["unknown"] = 0

    for r in valid:
        gt = r["ground_truth"]
        pred = r["predicted"]
        if gt in confusion:
            confusion[gt][pred] = confusion[gt].get(pred, 0) + 1

    # per-class precision, recall, f1
    per_class = {}
    for label in VALID_LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in VALID_LABELS if other != label)
        fn = sum(v for k, v in confusion[label].items() if k != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_class[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3)
        }

    latencies = [r["latency_ms"] for r in valid]
    avg_latency = int(sum(latencies) / len(latencies)) if latencies else 0

    return {
        "total": len(results),
        "evaluated": len(valid),
        "correct": len(correct),
        "accuracy": round(accuracy, 4),
        "api_errors": len(errors),
        "unknown_responses": len(unknown),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "avg_latency_ms": avg_latency
    }

# Print report

def print_report(metrics, results):
    m = metrics
    print("\n" + "=" * 55)
    print("  SENTIMENT EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Evaluated  : {m['evaluated']} / {m['total']}")
    print(f"  Correct    : {m['correct']}")
    print(f"  Accuracy   : {m['accuracy']:.1%}")
    print(f"  API errors : {m['api_errors']}")
    print(f"  Avg latency: {m['avg_latency_ms']} ms")

    print("\n  Per-class breakdown:")
    print(f"  {'Label':<12} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    print("  " + "-" * 42)
    for label, stats in m["per_class"].items():
        print(f"  {label:<12} {stats['precision']:>10.3f} {stats['recall']:>10.3f} {stats['f1']:>8.3f}")

    print("\n  Confusion matrix (rows = actual, cols = predicted):")
    labels = sorted(VALID_LABELS)
    print(f"  {'':12}", end="")
    for l in labels:
        print(f"{l:>12}", end="")
    print(f"{'unknown':>12}")
    print("  " + "-" * (12 + 12 * (len(labels) + 1)))
    for actual in labels:
        print(f"  {actual:<12}", end="")
        for pred in labels:
            print(f"{m['confusion_matrix'][actual].get(pred, 0):>12}", end="")
        print(f"{m['confusion_matrix'][actual].get('unknown', 0):>12}")

    failures = [r for r in results if not r["correct"] and not r["error"]][:5]
    if failures:
        print("\n  Sample failures:")
        for r in failures:
            print(f"  [id={r['id']}] truth={r['ground_truth']}, predicted={r['predicted']}")
            print(f"    {r['snippet'][:80]!r}")

    print("=" * 55 + "\n")


# Save results

def save_results(results, metrics, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
        "results": results
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {output_path}")


def main():
    if not os.environ.get("MISTRAL_API_KEY"):
        print("Error: MISTRAL_API_KEY not set")
        sys.exit(1)
    ist_timezone = timezone(timedelta(hours=5, minutes=30))
    output_path = f"results/run_{datetime.now(ist_timezone).strftime('%Y%m%d_%H%M%S')}.json"

    examples = load_dataset("data/sentiment_eval_dataset.csv")
    if not examples:
        print("No valid examples found.")
        sys.exit(1)

    results = run_evaluation(examples)
    metrics = compute_metrics(results)
    print_report(metrics, results)
    save_results(results, metrics, output_path)


if __name__ == "__main__":
    main()