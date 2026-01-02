import joblib

model = joblib.load("models/model.pkl")

def predict_ticket(text: str):
    prediction = model.predict([text])[0]
    confidence = max(model.predict_proba([text])[0])
    return prediction, confidence
