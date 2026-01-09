from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.sentiment import SentimentModel

app = FastAPI(title="Sentiment Reputation API", version="0.1.0") #Definiamo l'app FastAPI

model = SentimentModel() # Carichiamo il modello, una volta sola quando parte il processo

#creo PredictRequest e PredictResponse, modelli Pydantic che FastAPI usa per validare input e documentare l’API automaticamente.
class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000) # testo da analizzare, con lunghezza minima 1 e massima 5000

class PredictResponse(BaseModel):
    Sentiment: str
    Probabilità: float

@app.get("/health")
def health(): # endpoint semplice: se risponde, il server è su
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse) # endpoint per fare predizioni di sentiment
def predict(payload: PredictRequest): #payload è un oggetto Pydantic validato
    return model.predict(payload.text)
