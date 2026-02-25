from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
LOCAL_MODEL_PATH = Path("./models/goemotions")  # keep in repo

if not LOCAL_MODEL_PATH.exists():
    raise RuntimeError(
        f"Local model not found at {LOCAL_MODEL_PATH}. "
        f"Download it first using huggingface-cli."
    )

# -------------------------
# LOAD MODEL ON STARTUP
# -------------------------
print("Loading emotion model...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
model.eval()
LABELS = model.config.id2label
print("Model loaded successfully!")

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Princesa Emotion API 🌸")

# Allow CORS for all domains (PHP frontend can access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Emotion mapping
# -------------------------
GOEMO_TO_FLOWERS = {
    "admiration": "Admiration",
    "amusement": "Amusement",
    "approval": "Approval",
    "caring": "Caring",
    "curiosity": "Curiosity",
    "desire": "Desire",
    "excitement": "Excitement",
    "gratitude": "Gratitude",
    "joy": "Joy",
    "love": "Love",
    "optimism": "Optimism",
    "pride": "Pride",
    "relief": "Relief",
    "surprise": "Surprise",
    "realization": "Realization",
    "confusion": "Confusion",
    "anger": "Anger",
    "annoyance": "Annoyance",
    "disapproval": "Disapproval",
    "disgust": "Disgust",
    "embarrassment": "Embarrassment",
    "fear": "Fear",
    "grief": "Grief",
    "nervousness": "Nervousness",
    "remorse": "Remorse",
    "sadness": "Sadness",
    "disappointment": "Disappointment"
}

FLOWER_EMOTIONS = sorted(set(GOEMO_TO_FLOWERS.values()))

def get_flower_type(emotion: str) -> str:
    flower_map = {
        "Admiration": "Orchid",
        "Amusement": "Yellow Tulip",
        "Approval": "Hydrangea",
        "Caring": "Pink Carnation",
        "Curiosity": "Blue Iris",
        "Desire": "Red Tulip",
        "Excitement": "Gerbera Daisy",
        "Gratitude": "Pink Rose",
        "Joy": "Sunflower",
        "Love": "Red Rose",
        "Optimism": "Yellow Rose",
        "Pride": "Gladiolus",
        "Relief": "Lavender",
        "Surprise": "Purple Tulip",
        "Realization": "White Rose",
        "Confusion": "Foxglove",
        "Anger": "Red Chrysanthemum",
        "Annoyance": "Orange Lily",
        "Disapproval": "Petunia",
        "Disgust": "Marigold",
        "Embarrassment": "Peony",
        "Fear": "White Poppy",
        "Grief": "White Lily",
        "Nervousness": "Daisy",
        "Remorse": "Purple Hyacinth",
        "Sadness": "Blue Hyacinth",
        "Disappointment": "Willow"
    }
    return flower_map.get(emotion, "Unknown")

# -------------------------
# Request/Response Models
# -------------------------
class TextIn(BaseModel):
    text: str
    threshold: float = 0.3
    top_k: int = 3

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict_emotion")
async def predict_emotion(data: TextIn):

    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.sigmoid(logits).cpu()[0].numpy()

    indices = np.where(probs > data.threshold)[0]
    if len(indices) == 0:
        indices = np.argsort(probs)[-data.top_k:][::-1]

    flower_scores = {}
    for idx in indices:
        label = LABELS[idx]
        score = float(probs[idx])
        flower = GOEMO_TO_FLOWERS.get(label, "Neutral/Unknown")
        flower_scores[flower] = max(score, flower_scores.get(flower, 0))

    results = [
        {
            "flower_emotion": flower,
            "score": score,
            "flower_type": get_flower_type(flower)
        }
        for flower, score in flower_scores.items()
    ]

    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:data.top_k]

    dominant = results[0] if results else {
        "flower_emotion": "Neutral/Unknown",
        "score": 0.0,
        "flower_type": "Unknown"
    }

    return {
        "text": data.text,
        "dominant_emotion": dominant["flower_emotion"],
        "dominant_score": dominant["score"],
        "all_emotions": results,
        "threshold_used": data.threshold
    }

# -------------------------
# List all flower emotions
# -------------------------
@app.get("/flower_emotions")
async def get_flower_emotions():
    return {
        "available_emotions": FLOWER_EMOTIONS,
        "count": len(FLOWER_EMOTIONS)
    }

# -------------------------
# Health check
# -------------------------
@app.get("/")
async def root():
    return {
        "message": "Princesa Emotion API 🌸",
        "flower_emotions": len(FLOWER_EMOTIONS),
        "status": "running offline"
    }