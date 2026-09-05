<div align="center">

# 🎫 AI Ticket Classifier

### *Automated ML-Powered Customer Support Ticket Classification & Routing Service*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005589?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

<br />

```mermaid
flowchart LR
    A["📩 Raw Ticket<br/>App crashes on startup"] -->|1. Vectorization| B["⚙️ ML Pipeline<br/>TF-IDF + LogReg"]
    B -->|2. Classification| C["🚀 FastAPI Serving<br/>POST /predict"]
    C -->|3. Output JSON| D["🏷️ Category Result<br/>Technical (Conf: 94%)"]

    style A fill:#161b22,stroke:#388bfd,stroke-width:2px,color:#c9d1d9
    style B fill:#161b22,stroke:#d29922,stroke-width:2px,color:#c9d1d9
    style C fill:#161b22,stroke:#238636,stroke-width:2px,color:#c9d1d9
    style D fill:#1b4721,stroke:#2ea043,stroke-width:2px,color:#aff5b4
```

</div>

---

## Problem Statement & Solution

### The Problem
Customer support departments in enterprise organizations receive hundreds or thousands of support tickets every day across email, web forms, and chat applications. 
* **Manual Triaging Delays:** Support agents spend substantial time reading, tagging, and manually routing incoming tickets to the appropriate teams (IT, Billing, Account Management).
* **Increased MTTR (Mean Time To Resolution):** Critical issues like application crashes or double-billing take longer to reach the right engineer due to routing bottlenecks.
* **Human Fatigue & Errors:** Misrouting tickets leads to frustration, lost productivity, and broken Service Level Agreements (SLAs).

### The Solution
The **AI Ticket Classifier** provides an end-to-end Machine Learning microservice that automatically ingests unstructured support text, applies Natural Language Processing (NLP) feature extraction, predicts the category (e.g., `Technical`, `Billing`, `Account`), calculates prediction confidence, and exposes a high-performance REST API for real-time ticket routing.

---

## System Architecture & Workflow

```mermaid
flowchart TD
    subgraph Data_Storage ["Data & Storage Layer"]
        CSV["📄 data/tickets.csv"]
        DB[("🗄️ SQLite tickets.db<br/>SQLAlchemy ORM")]
        PKL["📦 models/model.pkl<br/>Serialized Pipeline"]
        MLFLOW[("📊 MLflow Registry<br/>mlflow.db & mlruns")]
    end

    subgraph ML_Pipeline ["Machine Learning Pipeline (src/train.py)"]
        TRAIN_LOAD["1. Load Ticket Dataset"] --> SPLIT["2. Train/Test Split (80/20)"]
        SPLIT --> VECT["3. TF-IDF Feature Extraction<br/>max_features=500, English Stop Words"]
        VECT --> LOGREG["4. Multi-class Logistic Regression"]
        LOGREG --> EVAL["5. Evaluate Model Accuracy"]
        EVAL --> LOG_ML["6. Log Metrics to MLflow"]
        EVAL --> SAVE["7. Export Pipeline to model.pkl"]
    end

    subgraph API_Serving ["API Serving Layer (api/app.py & src/predict.py)"]
        REQ["📥 HTTP POST /predict<br/>Input: App crashes on startup"] --> FASTAPI["⚡ FastAPI Engine"]
        FASTAPI --> PRED_FN["🔍 predict_ticket()"]
        PKL --> PRED_FN
        PRED_FN --> RESP["📤 Response JSON<br/>Category: Technical, Confidence: 0.94"]
    end

    CSV --> TRAIN_LOAD
    SAVE --> PKL
    LOG_ML --> MLFLOW
```

---

## Repository Structure

```
ai-ticket-classifier/
├── api/
│   └── app.py              # FastAPI application & REST endpoints (/predict, /health)
├── data/
│   └── tickets.csv         # Training dataset containing ticket descriptions & categories
├── models/
│   └── model.pkl           # Trained Scikit-Learn pipeline artifact
├── scripts/
│   └── train.sh            # Shell script to run model training workflow
├── src/
│   ├── __init__.py
│   ├── database.py         # SQLAlchemy Ticket database model & SQLite configuration
│   ├── predict.py          # Inference module loading trained model & computing confidence
│   ├── preprocessing.py    # Text vectorization transformer (TF-IDF)
│   └── train.py            # Model training, split evaluation & MLflow tracking script
├── Dockerfile              # Containerization instructions for Uvicorn serving
├── mlflow.db               # SQLite database for MLflow experiment logs
├── requirements.txt        # Python dependency dependencies
└── README.md               # Project documentation
```

---

## Tech Stack & ML Techniques

* **Core Language:** Python 3.9+
* **Machine Learning Framework:** [Scikit-Learn](https://scikit-learn.org/)
  * `TfidfVectorizer`: Transforms raw text into numerical TF-IDF feature matrices (`max_features=500`, English stop-word filtering).
  * `LogisticRegression`: Multi-class classifier predicting ticket categories with probability calibration.
  * `Pipeline`: Chains feature extraction and model inference to eliminate data leakage.
* **MLOps & Experiment Tracking:** [MLflow](https://mlflow.org/) (Logs metrics, parameters, and run artifacts).
* **API & Serving:** [FastAPI](https://fastapi.tiangolo.com/) & [Uvicorn](https://www.uvicorn.org/) (Asynchronous ASGI server).
* **Database & ORM:** [SQLAlchemy](https://www.sqlalchemy.org/) & [SQLite](https://www.sqlite.org/).
* **Containerization:** [Docker](https://www.docker.com/).

---

## Local Environment Setup & Testing

Follow these step-by-step instructions to clone, set up, train, and test the project locally.

### 1. Prerequisites
Ensure you have the following installed on your local machine:
* **Python 3.9+** (`python3 --version`)
* **Git** (`git --version`)
* **Docker** *(Optional, for containerized execution)*

---

### 2. Clone the Repository
```bash
git clone https://github.com/YourUsername/ai-ticket-classifier.git
cd ai-ticket-classifier
```

---

### 3. Create & Activate Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows (Command Prompt / PowerShell):**
```cmd
python -m venv venv
venv\Scripts\activate
```

---

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 5. Train the ML Model
Run the training pipeline script. This will read `data/tickets.csv`, train the TF-IDF + Logistic Regression model, log metrics to MLflow, and export the trained model artifact to `models/model.pkl`.

```bash
# Using Python directly
python src/train.py

# OR using the training script
bash scripts/train.sh
```

*Expected Output:*
```text
Training model...
Model trained with accuracy: 1.0
Training completed
```

---

### 6. View MLflow Tracking UI (Optional)
To inspect training metrics, parameter logs, and experiment runs:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open your browser at `http://127.0.0.1:5000` to view the MLflow Dashboard.

---

### 7. Run the FastAPI Server
Start the local REST API server using Uvicorn:
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```
The API server will run at `http://127.0.0.1:8000`.

---

### 8. Testing the API Endpoints

#### A. Health Check Endpoint
```bash
curl -X GET "http://127.0.0.1:8000/health"
```
*Response:*
```json
{
  "status": "running"
}
```

#### B. Ticket Classification Endpoint (`/predict`)

**Test Case 1: Technical Issue**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "App crashes on startup"}'
```
*Response:*
```json
{
  "category": "Technical",
  "confidence": 0.44
}
```

**Test Case 2: Billing Issue**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Payment deducted twice for my subscription"}'
```
*Response:*
```json
{
  "category": "Billing",
  "confidence": 0.48
}
```

#### C. Interactive OpenAPI (Swagger) Documentation
Open your browser and navigate to:
**`http://127.0.0.1:8000/docs`** to test API requests directly via the interactive UI.

---

### 9. Run with Docker 

#### Build Docker Image
```bash
docker build -t ai-ticket-classifier .
```

#### Run Container
```bash
docker run -d -p 8000:8000 --name ticket-classifier-app ai-ticket-classifier
```

Verify the containerized API at `http://localhost:8000/health`.

---

## Future Scope & Improvements

- [ ] **Transformer Models (BERT/RoBERTa):** Upgrade from TF-IDF + Logistic Regression to pre-trained transformer embeddings for superior semantic text understanding.
- [ ] **LLM Integration (Gemini / OpenAI / Llama):** Implement zero-shot or few-shot ticket categorization and automated AI draft response generation.
- [ ] **Multi-Task Learning:** Extend the ML pipeline to predict both `category` (`Technical`, `Billing`, `Account`) and `priority` (`High`, `Medium`, `Low`) simultaneously.
- [ ] **Active Learning & Feedback Loop:** Persist low-confidence predictions to `tickets.db` for human re-annotation and automated weekly re-training loops.
- [ ] **Continuous MLOps Integration:** Implement automated model performance drift monitoring (Evidently AI) and automated CI/CD deployment via GitHub Actions.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
