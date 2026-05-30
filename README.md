# Sentiment Classifier Evaluation Harness

Evaluates an LLM-based sentiment classifier against a labeled dataset of customer call snippets.

## Setup

```bash
pip install -r requirements.txt
export MISTRAL_API_KEY=your-key-here
```

## Run

```bash
python evaluate.py 

## What it does

1. Loads the CSV dataset 
2. Sends each snippet to Claude Haiku with a short system prompt asking for one word: positive, neutral, or negative
3. Parses the response — handles cases like "Positive!" or "The sentiment is negative."
4. Computes accuracy, per-class precision/recall/F1, and a confusion matrix
5. Saves everything to JSON

## Output example

```
=======================================================
  SENTIMENT EVALUATION RESULTS
=======================================================
  Evaluated  : 100 / 100
  Correct    : 91
  Accuracy   : 91.0%
  API errors : 0
  Avg latency: 430 ms

  Per-class breakdown:
  Label        Precision     Recall       F1
  ------------------------------------------
  negative         0.920      0.893    0.906
  neutral          0.880      0.913    0.896
  positive         0.930      0.939    0.934
```

## Files

```
evaluate.py              # main script
requirements.txt
README.md
data/
  sentiment_eval_dataset.csv        # the dataset
results/                 # output JSON files go here

```
