from app.sentiment import SentimentModel


def main():
    model = SentimentModel()

    examples = [
        "I love this product, it works great!",
        "This is the worst experience ever.",
        "It's ok, nothing special.",
    ]

    for text in examples:
        prediction = model.predict(text)
        print(f"\nTEXT: {text}")
        print(f"PRED: {prediction}")


if __name__ == "__main__":
    main()
