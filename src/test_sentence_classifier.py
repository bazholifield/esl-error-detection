from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Model checkpoint
MODEL_PATH = "../results/results/checkpoint-3000"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

# Label mapping
LABEL_LIST = ["OK", "ERR"]

# Example sentences
sentences = [
    "I has a apple.",                   # error
    "She is going to the store.",       # correct
    "They was late to the meeting.",    # error
    "He enjoys playing basketball.",   # correct
    "Me and him went to the park.",    # error
    "The cat sat on the mat.",         # correct
    "I can has cheezburger?",           # error
    "We are watching a movie tonight.", # correct
    "Him don't like vegetables.",       # error
    "This is a beautiful day."          # correct
]

# Run predictions
for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()
    print(f"Sentence: '{sentence}'")
    print(f"Prediction: {LABEL_LIST[pred]}")
    print("-" * 40)
