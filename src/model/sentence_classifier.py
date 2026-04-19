import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np

# config block
MODEL_NAME = "distilbert-base-uncased"
NUM_EPOCHS = 3
BATCH_SIZE = 8
LABEL_LIST = ["OK", "ERR"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}

# Helper functions (move this to another file later)

def load_dataset(csv_path):
    """
    Load CSV and create sentence-level classification dataset.
    Labels: 0 = OK, 1 = ERR
    """
    df = pd.read_csv(csv_path)
    
    # label sentence as error if not the same as original text
    df["label"] = (df["original_text"] != df["corrected_text"]).astype(int)
    
    # only keep relevant columns
    ds = Dataset.from_pandas(df[["original_text", "label"]])
    ds = ds.rename_column("original_text", "text")
    return ds.train_test_split(test_size=0.2)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

# train function

def train_sentence_classifier(csv_path="data/processed/sentences.csv"):
    dataset_dict = load_dataset(csv_path)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_LIST)
    )
    
    # Tokenize function
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    
    tokenized_dataset = dataset_dict.map(tokenize)
    
    # Training args
    args = TrainingArguments(
        output_dir="./results",
        learning_rate=5e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        save_total_limit=1,
        push_to_hub=False
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    print("Training complete!")

if __name__ == "__main__":
    train_sentence_classifier("data/processed/sentences.csv")
