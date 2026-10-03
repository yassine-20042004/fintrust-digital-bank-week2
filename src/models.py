"""Model Development, Comparison, and Serialization for FinTrust Week 3."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def train_and_evaluate_all_models(X_train, y_train, X_test, y_test, preprocessor):
    """Trains 4 classification algorithms, computes metrics, error patterns, and returns candidate model."""
    
    models = {
        "Baseline Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=5, class_weight='balanced', random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.05,
            scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
            random_state=42, eval_metric='logloss'
        )
    }

    results = []
    trained_pipelines = {}

    for name, clf in models.items():
        pipe = Pipeline([
            ('prep', preprocessor),
            ('clf', clf)
        ])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "True Negatives": int(tn),
            "False Positives": int(fp),
            "False Negatives": int(fn),
            "True Positives": int(tp)
        })

        trained_pipelines[name] = pipe

    results_df = pd.DataFrame(results)

    # Candidate selection criteria: Highest F1-Score & ROC-AUC for balanced risk recall
    best_model_name = results_df.sort_values(by=["F1-Score", "ROC-AUC"], ascending=False).iloc[0]["Model"]
    candidate_pipeline = trained_pipelines[best_model_name]

    return results_df, trained_pipelines, best_model_name, candidate_pipeline


def save_candidate_model(pipeline, filepath="models/candidate_model.pkl"):
    """Serializes fitted pipeline to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"Candidate model successfully saved to {filepath}")


def load_candidate_model(filepath="models/candidate_model.pkl"):
    """Loads serialized pipeline from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")
    return joblib.load(filepath)
