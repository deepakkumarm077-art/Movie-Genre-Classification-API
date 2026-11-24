
import os
import re
from pathlib import Path

import joblib
import numpy as np
import nltk

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Project paths
BASE_DIR = Path(__file__).resolve().parent
NLTK_DATA_DIR = Path("/tmp/nltk_data")
NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
nltk.data.path.insert(0, str(NLTK_DATA_DIR))

# Download required text-processing resources
for resource in ["stopwords", "wordnet"]:
    if not nltk.download(
        resource,
        download_dir=str(NLTK_DATA_DIR),
        quiet=True
    ):
        raise RuntimeError(f"Could not download NLTK resource: {resource}")

# Load Naive Bayes artifacts
nb_model = joblib.load(BASE_DIR / "naive_bayes_model.pkl")
tfidf = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
label_encoder = joblib.load(BASE_DIR / "label_encoder.pkl")

# Load trained MLP weights without TensorFlow
weights = np.load(BASE_DIR / "mlp_weights.npz")

W1 = weights["W1"]
b1 = weights["b1"]
W2 = weights["W2"]
b2 = weights["b2"]

# Verify model dimensions
if W1.shape != (10000, 128) or W2.shape != (128, 10):
    raise RuntimeError("MLP weight dimensions do not match the trained architecture.")

# Text preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z']", " ", text)
    tokens = text.split()

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if len(word) > 1 and word not in stop_words
    ]

    return " ".join(tokens)


# NumPy MLP inference
def predict_mlp(features):
    hidden = np.maximum(features @ W1 + b1, 0)

    logits = hidden @ W2 + b2

    exp_values = np.exp(
        logits - np.max(logits, axis=1, keepdims=True)
    )

    probabilities = exp_values / np.sum(
        exp_values, axis=1, keepdims=True
    )

    return probabilities[0]


# FastAPI application
app = FastAPI(
    title="Movie Genre Classification API",
    description="Movie genre prediction using Naive Bayes and NumPy MLP",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class MovieInput(BaseModel):
    description: str


@app.get("/")
def home():
    return {
        "message": "Movie Genre Classification API is running!",
        "available_models": ["Naive Bayes", "Improved MLP"],
        "mlp_engine": "NumPy"
    }


@app.post("/predict/genre/naive-bayes")
def predict_naive_bayes(data: MovieInput):
    if not data.description.strip():
        raise HTTPException(
            status_code=400,
            detail="Description cannot be empty"
        )

    cleaned = preprocess_text(data.description)
    vector = tfidf.transform([cleaned])

    prediction = nb_model.predict(vector)[0]
    probabilities = nb_model.predict_proba(vector)[0]

    return {
        "model": "Naive Bayes",
        "predicted_genre": str(prediction),
        "score": round(float(np.max(probabilities)) * 100, 2)
    }


@app.post("/predict/genre/mlp")
def predict_mlp_endpoint(data: MovieInput):
    if not data.description.strip():
        raise HTTPException(
            status_code=400,
            detail="Description cannot be empty"
        )

    cleaned = preprocess_text(data.description)

    vector = tfidf.transform([cleaned]).toarray().astype(np.float32)

    probabilities = predict_mlp(vector)

    prediction_index = int(np.argmax(probabilities))
    prediction = label_encoder.inverse_transform(
        [prediction_index]
    )[0]

    return {
        "model": "Improved MLP",
        "predicted_genre": str(prediction),
        "score": round(float(np.max(probabilities)) * 100, 2)
    }


print("Movie Genre Classification API loaded successfully!")
print("MLP inference engine: NumPy")
