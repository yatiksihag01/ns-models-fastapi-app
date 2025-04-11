import torch
from transformers import pipeline


class NewsClassifier:
    def __init__(self):
        model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        print(f"[INIT] Loading classification model: {model_name}")
        self.device = 0 if torch.cuda.is_available() else -1
        self.candidate_labels = ["sports", "politics", "business", "tech", "entertainment"]
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=self.device
        )
        print(f"[INIT] Model loaded to device: {'cuda' if self.device == 0 else 'cpu'}")

    def classify(self, texts):
        results = []
        for text in texts:
            output = self.classifier(text, candidate_labels=self.candidate_labels, multi_label=False)
            label = output["labels"][0]  # top prediction
            results.append(label)
        print("[CLASSIFY] Classification generated")
        return results