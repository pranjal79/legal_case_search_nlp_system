import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.utils.mlflow_logger import mlflow_run
import mlflow

INPUT_PATH = Path("data/processed/cases_clean.parquet")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "outcome_classifier.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

def load_training_data():
    df = pd.read_parquet(INPUT_PATH, columns=["case_facts", "legal_keywords", "outcome"])
    # Drop unknown outcomes and classes too small to stratify/split meaningfully
    df = df[df["outcome"] != "unknown"]
    counts = df["outcome"].value_counts()
    valid_classes = counts[counts >= 10].index
    df = df[df["outcome"].isin(valid_classes)]

    df["text"] = df["case_facts"].fillna("") + " " + df["legal_keywords"].fillna("")
    print(f"Training data: {len(df)} labeled cases across {df['outcome'].nunique()} classes")
    print(df["outcome"].value_counts())
    return df

def train_and_compare():
    df = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["outcome"], test_size=0.2, random_state=42, stratify=df["outcome"]
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
    }

    best_model = None
    best_f1 = -1
    best_name = None

    for name, model in models.items():
        with mlflow_run(run_name=f"outcome_clf_{name}"):
            mlflow.log_param("model_type", name)
            mlflow.log_param("tfidf_max_features", 5000)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))

            model.fit(X_train_vec, y_train)
            preds = model.predict(X_test_vec)

            acc = accuracy_score(y_test, preds)
            f1_macro = f1_score(y_test, preds, average="macro")

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_macro", f1_macro)

            print(f"\n{name}: accuracy={acc:.3f}, f1_macro={f1_macro:.3f}")
            print(classification_report(y_test, preds, zero_division=0))

            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_model = model
                best_name = name

    print(f"\nBest model: {best_name} (f1_macro={best_f1:.3f})")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saved best model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")

if __name__ == "__main__":
    train_and_compare()