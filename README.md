## AI-Powered Support Ticket Classification System
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-blue?style=for-the-badge)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-003B57?style=for-the-badge&logo=postgresql&logoColor=white)
![REST API](https://img.shields.io/badge/REST-API-blue?style=for-the-badge)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange?style=for-the-badge)
![Model Serving](https://img.shields.io/badge/Model-Serving-green?style=for-the-badge)

An end-to-end AI/ML project that automatically classifies customer support tickets using Natural Language Processing (NLP) and machine learning. 
The system exposes predictions via REST APIs and supports containerized, cloud-ready deployment.

## Key Features

NLP-based ticket classification with confidence scoring
RESTful API built using FastAPI
End-to-end ML lifecycle: preprocessing, training, inference
Experiment tracking using MLflow
Dockerized deployment for reproducibility

## Tech Stack

Python
Scikit-learn
FastAPI & Uvicorn
Pandas & NumPy
MLflow
SQLite (SQLAlchemy)
Docker

## Project Structure

ai-support-ticket-classifier/

├── api/        # FastAPI endpoints

├── src/        # ML training & inference logic

├── data/       # Sample dataset

├── models/     # Trained models

├── scripts/    # Automation scripts

# Running the Project

pip install -r requirements.txt
python src/train.py


![WhatsApp Image 2026-01-02 at 17 01 08](https://github.com/user-attachments/assets/082a576e-04d1-416f-a144-aa720d4fd4ec)

## Running the API Server

uvicorn api.app:app --reload


![WhatsApp Image 2026-01-02 at 17 01 08 (2)](https://github.com/user-attachments/assets/fb2fe0f6-61b6-4e60-a546-a01e7f8dd0b0)



## MLFLOW dashboard

![WhatsApp Image 2026-01-02 at 17 01 08 (1)](https://github.com/user-attachments/assets/2ec4c4d8-33e1-4dca-9f19-46153bf80c24)



## Swaager UI (Running output prediction)

![WhatsApp Image 2026-01-02 at 17 01 08 (3)](https://github.com/user-attachments/assets/97786234-bf85-4944-bbfc-fb51b686ec5a)



## Access API docs at:

 http://127.0.0.1:8000/docs

## Example API Usage

Request:-

{

  "text": "Payment deducted but order failed"
  
}


Response:-

{

  "category": "Billing",
  
  "confidence": 0.40
  
}


## Real-World Use Cases

This system can be applied in:
Customer Support Platforms – Automatically route tickets to the correct department
E-commerce Applications – Identify billing, payment, or order-related issues
SaaS Products – Prioritize urgent technical tickets
IT Helpdesk Systems – Classify internal support requests

## Real-World Datasets

In production, datasets are generated from:
CRM systems (Zendesk, Freshdesk, Jira)
Live chat transcripts
Email support systems
User feedback forms
Such systems typically generate thousands to millions of tickets, enabling better model accuracy through continuous training and MLOps pipelines.



