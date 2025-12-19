from datasets import load_dataset
import pandas as pd
import os

RAW_DIR = "data/raw"
CSV_FILENAME = "wikiedits_english.csv"

def load_wikiedits_english(max_examples=10000):
    """
    Stream WikiEdits-MultiGEC, filter English examples, and save to CSV.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    csv_path = os.path.join(RAW_DIR, CSV_FILENAME)

    # Stream dataset
    ds = load_dataset("lang-uk/WikiEdits-MultiGEC", split="train", streaming=True)

    data = []
    for i, example in enumerate(ds):
        if i < 5:
            print(example)  # preview
        if example.get("language") == "english":
            data.append({
                "original_text": example.get("text"),
                "corrected_text": example.get("correction"),
                "language": "english",
                "url": example.get("url")
            })
        if len(data) >= max_examples:
            break

    if not data:
        raise RuntimeError("No English examples found in WikiEdits-MultiGEC!")

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved {len(df)} English examples to {csv_path}")
    return df


if __name__ == "__main__":
    load_wikiedits_english()
