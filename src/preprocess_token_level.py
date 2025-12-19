import difflib
import pandas as pd
import math

def align_and_label(original, corrected):
    """Align two sentences and return tokens + ERR/OK labels."""
    orig_tokens = original.split()
    corr_tokens = corrected.split()

    matcher = difflib.SequenceMatcher(None, orig_tokens, corr_tokens)
    
    labeled_tokens = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for tok in orig_tokens[i1:i2]:
                labeled_tokens.append((tok, "OK"))
        else:
            for tok in orig_tokens[i1:i2]:
                labeled_tokens.append((tok, "ERR"))

    return labeled_tokens


def build_token_level_dataset(df):
    """Return a token-level dataframe: token | label | sentence_id"""

    rows = []

    for idx, row in df.iterrows():
        original = row["original_text"]
        corrected = row["corrected_text"]

        # --- SKIP BAD ROWS ---
        if not isinstance(original, str) or not isinstance(corrected, str):
            continue
        if original.strip() == "" or corrected.strip() == "":
            continue

        labeled = align_and_label(original, corrected)

        for token, label in labeled:
            rows.append({
                "sentence_id": idx,
                "token": token,
                "label": label
            })

    return pd.DataFrame(rows)
