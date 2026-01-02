import sys
sys.path.append(".")

import pandas as pd
import joblib
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.preprocessing import get_vectorizer

mlflow.set_experiment("Ticket_Classifier")

df = pd.read_csv("data/tickets.csv")

X = df["text"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("tfidf", get_vectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])

with mlflow.start_run():
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    
    mlflow.log_metric("accuracy", accuracy)
    joblib.dump(pipeline, "models/model.pkl")

    print(f"Model trained with accuracy: {accuracy}")
