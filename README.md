🧠 AI-Powered Support Ticket Classification System
An end-to-end AI/ML project that automatically classifies customer support tickets using Natural Language Processing (NLP) and machine learning. The system exposes predictions via REST APIs and supports containerized, cloud-ready deployment.

🚀 Key Features

NLP-based ticket classification with confidence scoring
RESTful API built using FastAPI
End-to-end ML lifecycle: preprocessing, training, inference
Experiment tracking using MLflow
Dockerized deployment for reproducibility

🛠 Tech Stack

Python
Scikit-learn
FastAPI & Uvicorn
Pandas & NumPy
MLflow
SQLite (SQLAlchemy)
Docker

📂 Project Structure

ai-support-ticket-classifier/
├── api/        # FastAPI endpoints
├── src/        # ML training & inference logic
├── data/       # Sample dataset
├── models/     # Trained models
├── scripts/    # Automation scripts

▶️ Running the Project

pip install -r requirements.txt
python src/train.py
uvicorn api.app:app --reload

Access API docs at:
👉 http://127.0.0.1:8000/docs

📌 Example API Usage

Request:-
{
  "text": "Payment deducted but order failed"
}
Response:-
{
  "category": "Billing",
  "confidence": 0.40
}

🌍 Real-World Use Cases
This system can be applied in:
Customer Support Platforms – Automatically route tickets to the correct department
E-commerce Applications – Identify billing, payment, or order-related issues
SaaS Products – Prioritize urgent technical tickets
IT Helpdesk Systems – Classify internal support requests

📊 Real-World Datasets
In production, datasets are generated from:
CRM systems (Zendesk, Freshdesk, Jira)
Live chat transcripts
Email support systems
User feedback forms

Such systems typically generate thousands to millions of tickets, enabling better model accuracy through continuous training and MLOps pipelines.
