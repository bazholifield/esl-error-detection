from transformers import pipeline

clf = pipeline(
    "token-classification",
    model="models/token_classifier",
    tokenizer="roberta-base",
    aggregation_strategy="simple"
)

def detect_errors(text: str):
    return clf(text)
