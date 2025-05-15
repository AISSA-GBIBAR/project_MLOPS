from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time

# Prometheus Metrics ===
REQUEST_COUNT = Counter('api_requests_total', 'Total number of requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'Request latency', ['endpoint'])

# === Initialisation de l'application FastAPI ===
app = FastAPI()
import os
# === Chargement du modèle et du tokenizer ===
MODEL_PATH = "model/sentiment_model.pt"
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Charger les poids du modèle
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# === Schéma de requête ===
class CommentRequest(BaseModel):
    text: str

# === Endpoint principal ===
@app.post("/predict")
def predict_sentiment(request: CommentRequest):
    start_time = time.time()
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
    # Tokenization du texte
    inputs = tokenizer(request.text, padding=True, truncation=True, max_length=32, return_tensors="pt")
    
    # Prédiction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    REQUEST_LATENCY.labels(endpoint='/predict').observe(time.time() - start_time)
    
    # Préparation de la réponse
    label = "Positive" if prediction == 1 else "Negative"
    return {
        "text": request.text,
        "prediction": label,
        "confidence": round(confidence, 4)
    }
    
@app.get("/metrics")
def metrics():
    return generate_latest()
