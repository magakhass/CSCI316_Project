from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
 
# ----------------------------
# App Setup
# ----------------------------
app = FastAPI(
    title="Hinglish Sentiment API",
    description="Sentiment analysis for code-switched Hinglish text using XLM-R + LoRA",
    version="1.0.0"
)
 
# ----------------------------
# Model Loading
# ----------------------------
ID2LABEL   = {0: "positive", 1: "negative", 2: "neutral"}
MODEL_PATH = os.getenv("MODEL_PATH", "./hugging_peft_model_merged")
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
print(f"Loading model from: {MODEL_PATH}")
print(f"Using device: {device}")
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
 
print("Model loaded successfully.")
 
# ----------------------------
# Request / Response Schemas
# ----------------------------
class SentimentRequest(BaseModel):
    text: str
 
    class Config:
        json_schema_extra = {
            "example": {"text": "yaar aaj ka din bahut accha tha"}
        }
 
class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
 
 
# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def root():
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read())
 
 
@app.get("/health")
def health():
    return {"status": "ok"}
 
 
@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest):
    text = request.text.strip()
 
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
 
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length"
    ).to(device)
 
    with torch.no_grad():
        logits = model(**inputs).logits
 
    probs      = torch.softmax(logits, dim=-1)
    pred_id    = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_id].item()
 
    return SentimentResponse(
        text=text,
        sentiment=ID2LABEL[pred_id],
        confidence=round(confidence, 4)
    )
 