import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from config import CHECKPOINT_PATH
from pipeline import analyze

THRESHOLD = 0.25  # low threshold = high recall; rule-based classifier handles false positives

def load_model(checkpoint=CHECKPOINT_PATH):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True)
    model.eval()
    return tokenizer, model

def display(results: list[dict]):
    multi = len(results) > 1
    total_errors = sum(len(r['errors']) for r in results)

    for i, result in enumerate(results, 1):
        sent   = result['sentence']
        errors = result['errors']

        if multi:
            print(f"\nSentence {i}: \"{sent}\"")
        else:
            print(f"\nInput: \"{sent}\"")

        if not errors:
            print("  ✓ No errors found.")
        else:
            for err in errors:
                if err.span:
                    print(f"  ✗ [{err.span}]  {err.error_type}")
                else:
                    print(f"  ✗ {err.error_type}")
                if err.lesson:
                    print(f"       → {err.lesson}")

    if multi:
        s = "s" if total_errors != 1 else ""
        print(f"\n{'─' * 50}")
        print(f"{total_errors} error{s} found across {len(results)} sentence{'' if len(results) == 1 else 's'}.")

def main():
    parser = argparse.ArgumentParser(description="ESL Error Detector")
    parser.add_argument("--text", type=str, help="Sentence or paragraph to check")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    args = parser.parse_args()

    tokenizer, model = load_model(args.checkpoint)

    if args.text:
        results = analyze(args.text, tokenizer, model, THRESHOLD)
        display(results)
    else:
        print("ESL Error Detector — type a sentence or paragraph, or 'quit' to exit.")
        while True:
            text = input("\n> ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if text:
                results = analyze(text, tokenizer, model, THRESHOLD)
                display(results)

if __name__ == "__main__":
    main()
