import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score

# Carichiamo lo stesso modello usato dall'API
from app.sentiment import SentimentModel


DATASET_PATH = "data/train.csv"
OUTPUT_PATH = "data/predictions_on_dataset.csv"
N_ROWS = 500   # numero di esempi da usare (puoi aumentare se vuoi)


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    # Verifichiamo le colonne importanti
    assert "text" in df.columns, "Colonna 'text' non trovata nel dataset"
    assert "sentiment" in df.columns, "Colonna 'sentiment' non trovata nel dataset"

    # Teniamo solo quello che ci serve
    df = df[["text", "sentiment"]].dropna().head(N_ROWS)

    print(f"Running inference on {len(df)} samples...")

    model = SentimentModel()

    predictions = []
    scores = []

    for text in df["text"]:
        result = model.predict(text)
        predictions.append(result["Sentiment"])
        scores.append(result["Probabilità"])

    df["predicted_sentiment"] = predictions
    df["confidence"] = scores

    # Valutazione semplice
    y_true = df["sentiment"]
    y_pred = df["predicted_sentiment"]

    print("\n=== Model performance on Kaggle dataset ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    # Salva risultati
    Path("data").mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nPredictions saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
