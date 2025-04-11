from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from services.classifier import NewsClassifier
from services.sentiment import SentimentAnalyzer

app = FastAPI()


class TextRequest(BaseModel):
    texts: List[str]

class VertexAIRequest(TextRequest):
    instances: List[TextRequest]

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    category: str


sentiment_model = SentimentAnalyzer()
classification_model = NewsClassifier()

@app.get("/health")
def root():
    return {"status": "ok"}


@app.post("/predict", response_model=List[PredictionResponse])
async def get_sentiment_with_category(request: TextRequest):
    num_texts = len(request.texts)
    print(f"{'-' * 10} Starting analysis for {num_texts} items {'-' * 10}")

    # Sentiment analysis
    print(f"{'-' * 10} Running sentiment analysis {'-' * 10}")
    sentiments = sentiment_model.predict(texts=request.texts)

    # Classification
    print(f"{'-' * 10} Running category classification {'-' * 10}")
    classifications = classification_model.classify(texts=request.texts)

    print(f"{'-' * 10} Returning prediction response {'-' * 10}")
    return [
        PredictionResponse(text=text, sentiment=sentiment, category=category)
        for text, sentiment, category in zip(request.texts, sentiments, classifications)
    ]