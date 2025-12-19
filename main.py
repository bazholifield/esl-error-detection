from src.model.sentence_classifier import train_sentence_classifier

if __name__ == "__main__":
    # path to your CSV file
    csv_path = "data/processed/wikiedits_clean.csv"
    
    # Train sentence-level error detector
    train_sentence_classifier(csv_path)
