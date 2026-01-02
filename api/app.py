from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_ticket

app = FastAPI(title="AI Ticket Classifier")

class TicketRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TicketRequest):
    category, confidence = predict_ticket(request.text)
    return {
        "category": category,
        "confidence": round(confidence, 2)
    }

@app.get("/health")
def health():
    return {"status": "running"}
