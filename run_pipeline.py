"""Master CLI Runner Script for FinTrust Week 3 ML Pipeline & Model Evaluation."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.validation import validate_customer_data, validate_transaction_data, clean_datasets
from src.preprocessing import engineer_features, get_preprocessor
from src.models import train_and_evaluate_all_models, save_candidate_model


def find_data_file(filenames):
    for fn in filenames:
        if os.path.exists(fn):
            return fn
    raise FileNotFoundError(f"Could not locate data file among: {filenames}")


def generate_visualizations(results_df, candidate_pipe, X_test, y_test):
    """Generates and saves visual artifact charts for Week 3 report."""
    os.makedirs("reports/figures", exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Chart 1: Model Comparison Bar Chart (ROC-AUC & F1-Score)
    plt.figure(figsize=(10, 5))
    metrics_melt = results_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        var_name="Metric", value_name="Score"
    )
    sns.barplot(data=metrics_melt, x="Model", y="Score", hue="Metric", palette="viridis")
    plt.title("FinTrust Week 3 Model Comparison (4 Algorithms)", fontsize=14, fontweight='bold')
    plt.ylim(0.0, 1.05)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("reports/figures/model_comparison.png", dpi=300)
    plt.close()

    # Chart 2: Candidate Model Confusion Matrix
    y_pred = candidate_pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Risk', 'Risk Flag'],
                yticklabels=['No Risk', 'Risk Flag'])
    plt.title("Candidate Model Confusion Matrix", fontsize=12, fontweight='bold')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png", dpi=300)
    plt.close()

    # Chart 3: Feature Importance (for Tree Ensemble / XGBoost / Random Forest)
    try:
        clf = candidate_pipe.named_steps['clf']
        prep = candidate_pipe.named_steps['prep']

        num_cols = prep.transformers_[0][2]
        cat_encoder = prep.transformers_[1][1]
        cat_cols = cat_encoder.get_feature_names_out(prep.transformers_[1][2])
        all_feature_names = list(num_cols) + list(cat_cols)

        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(15)

            plt.figure(figsize=(10, 6))
            sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="mako")
            plt.title("Top 15 Feature Importances (Candidate Risk Model)", fontsize=14, fontweight='bold')
            plt.xlabel("Relative Importance Score")
            plt.tight_layout()
            plt.savefig("reports/figures/feature_importance.png", dpi=300)
            plt.close()
    except Exception as e:
        print(f"Feature importance chart generation skipped: {e}")

    print("Successfully generated visualization charts in reports/figures/")


def main():
    print("=" * 70)
    print("      FINTRUST DIGITAL BANK - WEEK 3 ML PIPELINE EXECUTION      ")
    print("=" * 70)

    cust_path = find_data_file([
        "data/raw/FinTrust_Customer_Data.csv",
        "data/raw/FinTrust_Customer_Data - FinTrust_Customer_Data (1).csv",
        "FinTrust_Customer_Data.csv"
    ])
    tx_path = find_data_file([
        "data/raw/FinTrust_Transaction_Data.csv",
        "data/raw/FinTrust_Transaction_Data - FinTrust_Transaction_Data (1).csv",
        "FinTrust_Transaction_Data.csv"
    ])

    df_cust = pd.read_csv(cust_path)
    df_tx = pd.read_csv(tx_path)

    print("\n[Step 1/5] Validating Data Schemas and Boundaries...")
    validate_customer_data(df_cust)
    validate_transaction_data(df_tx)
    print(" Validation passed: Zero schema or boundary exceptions.")

    print("\n[Step 2/5] Cleaning Datasets & Imputing Missing Values...")
    df_cust_clean, df_tx_clean = clean_datasets(df_cust, df_tx)
    print(f" Customers Cleaned: {len(df_cust_clean)} rows")
    print(f" Transactions Cleaned: {len(df_tx_clean)} rows")

    print("\n[Step 3/5] Engineering 14 Risk & Behavioural Features...")
    df_modelling = engineer_features(df_tx_clean, df_cust_clean)

    os.makedirs("data/processed", exist_ok=True)
    df_modelling.to_csv("data/processed/modelling_features_week3.csv", index=False)

    y = (df_modelling['Risk_Review_Flag'] == 'Yes').astype(int)
    drop_cols = [
        'Risk_Review_Flag', 'Transaction_ID', 'Customer_ID',
        'Customer_Name', 'Transaction_DateTime', 'Account_Status'
    ]
    X = df_modelling.drop(columns=drop_cols, errors='ignore')

    preprocessor, num_feats, cat_feats = get_preprocessor()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("\n[Step 4/5] Training & Cross-Evaluating 4 Classification Algorithms...")
    results_df, trained_pipes, best_name, candidate_pipe = train_and_evaluate_all_models(
        X_train, y_train, X_test, y_test, preprocessor
    )

    print("\n" + "=" * 70)
    print("                    MODEL COMPARISON MATRIX                    ")
    print("=" * 70)
    print(results_df.to_string(index=False))

    results_df.to_csv("data/processed/model_comparison_week3.csv", index=False)

    print("\n[Step 5/5] Serializing Candidate Model...")
    save_candidate_model(candidate_pipe, "models/candidate_model.pkl")

    generate_visualizations(results_df, candidate_pipe, X_test, y_test)

    print("\n" + "=" * 70)
    print(f" SUCCESS: Candidate Model Selected: {best_name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
