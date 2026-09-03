import pickle
from pathlib import Path

MODEL_PATH = Path("models/outcome_classifier.pkl")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.pkl")

class OutcomePredictor:
    def __init__(self):
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)

    def predict(self, case_facts: str, legal_keywords: str = ""):
        text = f"{case_facts} {legal_keywords}"
        vec = self.vectorizer.transform([text])
        pred = self.model.predict(vec)[0]
        proba = self.model.predict_proba(vec)[0]
        classes = self.model.classes_
        confidence = dict(zip(classes, proba.round(3)))
        return {"predicted_outcome": pred, "confidence_scores": confidence}