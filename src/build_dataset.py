import pandas as pd
import argparse
from pathlib import Path

def build_dataset(input_path, output_path):
    print(f"📥 Loading raw data from: {input_path}")
    df = pd.read_csv(input_path)

    # Basic validation
    required_cols = {"original_text", "corrected_text"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input CSV must contain: {required_cols}")

    print("🔀 Shuffling dataset...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into two halves
    half = len(df) // 2
    df_err = df.iloc[:half]      # ERR examples come from original_text
    df_ok = df.iloc[half:]       # OK examples come from corrected_text

    print(f"📊 Total rows: {len(df)}")
    print(f"➡️ Using {len(df_err)} rows for ERR")
    print(f"➡️ Using {len(df_ok)} rows for OK")

    # Build the two labeled subsets
    err_examples = pd.DataFrame({
        "text": df_err["original_text"],
        "label": 1               # ERR
    })

    ok_examples = pd.DataFrame({
        "text": df_ok["corrected_text"],
        "label": 0               # OK
    })

    # Combine and reshuffle
    combined = pd.concat([err_examples, ok_examples], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print("💾 Saving final dataset...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"🎉 Done! Final dataset saved to: {output_path}")
    print(f"📈 Final dataset size: {len(combined)} rows")
    print(f"🔥 Label distribution:\n{combined['label'].value_counts()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ERR/OK classification dataset.")
    parser.add_argument("--input", required=True, help="Path to raw CSV file")
    parser.add_argument("--output", required=True, help="Path to save processed dataset")
    args = parser.parse_args()

    build_dataset(args.input, args.output)
