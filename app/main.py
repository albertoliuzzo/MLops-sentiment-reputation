from fastapi import FastAPI, Response, Request
from pydantic import BaseModel, Field
from app.sentiment import SentimentModel

"""
Middleware: intercetta ogni request e registra latenza + contatore richieste.
/metrics: espone tutte le metriche in formato Prometheus (testo).
/predict: dopo la predizione incrementa un contatore per label.
"""

# --- Prometheus metrics ---
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Metrics HTTP
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["path", "method", "status"],
)

HTTP_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["path", "method"],
)

# Metriche modello
SENTIMENT_PREDICTIONS = Counter(
    "sentiment_predictions_total",
    "Total sentiment predictions",
    ["label"],
)

app = FastAPI(title="Sentiment Reputation API", version="0.2.0")

# Carichiamo il modello una sola volta all'avvio (riutilizzato su tutte le richieste)
model = SentimentModel()


# ---- Middleware: metriche HTTP (tutte le richieste) ----
@app.middleware("http")
async def prometheus_http_metrics(request: Request, call_next):
    path = request.url.path
    method = request.method

    # Misuriamo la latenza della request
    with HTTP_LATENCY.labels(path=path, method=method).time():
        response = await call_next(request)

    # Conteggio richieste per status code
    HTTP_REQUESTS.labels(path=path, method=method, status=str(response.status_code)).inc()
    return response


# ---- Schemi Pydantic ----
#creo PredictRequest e PredictResponse, modelli Pydantic che FastAPI usa per validare input e documentare l’API automaticamente.
class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000) # testo da analizzare, con lunghezza minima 1 e massima 5000

class PredictResponse(BaseModel):
    Sentiment: str
    Probabilità: float


# ---- Endpoints ----
@app.get("/health")
def health(): # endpoint di health check: se risponde, il server è su
    return {"status": "ok"}

@app.get("/") #evita il {"detail": "Not Found"} quando apri lo Space. Chiunque apre il link capisce subito cosa fa l’API
def root():
    return {
        "message": "Sentiment Reputation Monitoring API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

@app.get("/metrics")
def metrics():
    """
    Endpoint Prometheus: restituisce tutte le metriche in formato testuale.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):  # endpoint per fare predizioni di sentiment e aggiornare metriche; payload è un oggetto Pydantic validato
    result = model.predict(payload.text)  # {"Sentiment": "...", "Probabilità": ...}; 

    # Aggiorna metriche modello
    label = result.get("Sentiment", "unknown")
    SENTIMENT_PREDICTIONS.labels(label=label).inc()

    return result
