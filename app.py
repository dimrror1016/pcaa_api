from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = "monologg/bert-base-cased-goemotions-original"

tokenizer = None
model = None
LABELS = None

# -------------------------
# Lazy-load model
# -------------------------
def get_model():
    global tokenizer, model, LABELS
    if model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        print("Loading Hugging Face emotion model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()
        LABELS = model.config.id2label
        print("Model loaded successfully!")

    return tokenizer, model, LABELS

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Princesa Emotion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request model
# -------------------------
class TextIn(BaseModel):
    text: str

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict_emotion")
async def predict_emotion(data: TextIn):

    tokenizer, model, LABELS = get_model()

    # Validate input
    if not data.text or len(data.text.strip()) < 3:
        return {"error": "Text too short"}

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

    # Always return the strongest emotion
    best_idx = int(np.argmax(probs))
    best_label = LABELS[best_idx]
    best_score = float(probs[best_idx])

    return {
        "text": data.text,
        "dominant_emotion": best_label.capitalize(),
        "confidence": best_score
    }

# -------------------------
# Health check
# -------------------------
@app.get("/")
async def root():
    return {
        "message": "Princesa Emotion API running",
        "status": "ok"
    }
