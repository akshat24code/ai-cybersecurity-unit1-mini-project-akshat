import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)

def train_and_evaluate_ml_models(processed_data, output_dir="outputs"):
    """
    Task 3: Trains and evaluates supervised Machine Learning models on the preprocessed threat dataset.
    Models:
    1. Logistic Regression (Linear Baseline)
    2. Decision Tree Classifier (Non-linear Rule Engine)
    3. Random Forest Classifier (Advanced Ensemble)
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n[+] --- Task 3: Machine Learning Threat Classification ---")
    
    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    X_test = processed_data['X_test']
    y_test = processed_data['y_test']
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    }
    
    results = []
    trained_models = {}
    roc_curves = {}
    
    for name, model in models.items():
        print(f"[*] Training {name}...")
        
        # Measure Training Time
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        # Measure Inference Time (Latency for 10,000 packets)
        t0 = time.time()
        y_pred = model.predict(X_test)
        inference_time = (time.time() - t0) * 1000  # in ms
        latency_per_10k = (inference_time / len(X_test)) * 10000 # ms per 10k packets
        
        # Probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
            
        # Evaluation Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        # Confusion Matrix components
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall (Detection Rate)': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'FPR (False Positive Rate)': fpr,
            'FNR (False Negative Rate)': fnr,
            'Training Time (s)': train_time,
            'Inference Latency (ms/10k)': latency_per_10k
        })
        
        trained_models[name] = model
        
        # Compute ROC Curve points
        fpr_pts, tpr_pts, _ = roc_curve(y_test, y_prob)
        roc_curves[name] = (fpr_pts, tpr_pts, auc)
        
        print(f"    -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | FPR: {fpr:.4f} | Train Time: {train_time:.2f}s")

    # Plot ROC Curves for ML models
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'Logistic Regression': '#3498db', 'Decision Tree': '#e67e22', 'Random Forest': '#2ecc71'}
    
    for name, (fpr_pts, tpr_pts, auc) in roc_curves.items():
        ax.plot(fpr_pts, tpr_pts, label=f"{name} (AUC = {auc:.4f})", color=colors[name], lw=2)
        
    ax.plot([0, 1], [0, 1], 'k--', label='Random Chance baseline', alpha=0.6)
    ax.set_title("ROC Curves: Supervised Machine Learning Threat Classifiers", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / TPR)", fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    roc_plot_path = os.path.join(output_dir, "09_ml_roc_curves.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    
    df_results = pd.DataFrame(results)
    return df_results, trained_models

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_datasets
    df_train, df_test = load_data()
    processed = preprocess_datasets(df_train, df_test)
    results, models = train_and_evaluate_ml_models(processed)
    print(results)
