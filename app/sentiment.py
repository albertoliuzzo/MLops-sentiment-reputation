import os
from transformers import pipeline

class SentimentModel: #Wrapper riusabile per inferenza sentiment. Carica il modello una volta sola e lo riusa.

    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest", max_length: int = 256, device: int = -1): #Inizializza il modello di sentiment analysis con il nome del modello specificato (default: "cardiffnlp/twitter-roberta-base-sentiment-latest"), la lunghezza massima del testo (default: 256) e il dispositivo (default: -1 per CPU; invece 0 = prima GPU (se presente)).
        self.model_name = model_name
        self.pipe = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, max_length=max_length, device=device) #Crea una pipeline di sentiment analysis utilizzando il modello e il tokenizer specificati.
        
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1") #Disabilita gli avvisi sui symlink di Hugging Face Hub.

    def predict(self, text: str) -> dict: #Esegue l'inferenza sul testo in input e restituisce un dizionario con label (classe prevista (NEGATIVE, NEUTRAL, POSITIVE)) e score (confidenza (da 0 a 1)).
        result = self.pipe(text, truncation=True)[0] # la funzione pipe accetta un testo singolo o una lista di testi come input e restituisce dei dizionari come output contenente con i risultati dell'analisi del sentiment. L'argomento truncation=True assicura che il testo venga troncato se supera la lunghezza massima consentita dal modello. Con [0] si prende il primo (e unico) risultato.
        label = str(result["label"])
        score = float(result["score"])
        return {"Sentiment": label, "Probabilità": score}