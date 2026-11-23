
import os
import re
import joblib
import numpy as np
import nltk
import tensorflow as tf

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Download required NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Load trained models
nb_model = joblib.load(BASE_DIR / "naive_bayes_model.pkl")
tfidf = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
label_encoder = joblib.load(BASE_DIR / "label_encoder.pkl")

mlp_model = tf.keras.models.load_model(
    BASE_DIR / "movie_genre_mlp.keras"
)

# Text preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z']", " ", text)
    tokens = text.split()

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if len(word) > 1 and word not in stop_words
    ]

    return " ".join(tokens)

# Create FastAPI application
app = FastAPI(
    title="Movie Genre Classification API",
    description="Movie genre prediction using Naive Bayes and Improved MLP",
    version="1.0.0"
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
        "available_models": ["Naive Bayes", "Improved MLP"]
    }

@app.post("/predict/genre/naive-bayes")
def predict_naive_bayes(data: MovieInput):
    if not data.description.strip():
        raise HTTPException(status_code=400, detail="Description cannot be empty")

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
def predict_mlp(data: MovieInput):
    if not data.description.strip():
        raise HTTPException(status_code=400, detail="Description cannot be empty")

    cleaned = preprocess_text(data.description)
    vector = tfidf.transform([cleaned]).toarray().astype("float32")

    probabilities = mlp_model.predict(vector, verbose=0)[0]

    prediction = label_encoder.inverse_transform(
        [int(np.argmax(probabilities))]
    )[0]

    return {
        "model": "Improved MLP",
        "predicted_genre": str(prediction),
        "score": round(float(np.max(probabilities)) * 100, 2)
    }

print("Movie Genre Classification API loaded successfully!")
