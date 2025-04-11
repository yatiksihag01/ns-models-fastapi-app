from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def id2label(idx):
    return {0: "negative", 1: "positive"}.get(idx, "unknown")


class SentimentAnalyzer:
    def __init__(self):
        model_name = "yatiksihag01/distilbert-base-uncased-news-sentiment-finetuned-english"
        print(f"[INIT] Loading sentiment model: {model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"[INIT] Model loaded to device: {self.device}")

    def predict(self, texts: List[str]):
        inputs = self.tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
            print(f"[PREDICT] Predictions generated: {predictions.tolist()}")
        return [id2label(idx) for idx in predictions.tolist()]