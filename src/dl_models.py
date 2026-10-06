import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

def train_and_evaluate_dl_model(processed_data, output_dir="outputs"):
    """
    Task 4: Implements Deep Learning Prototype using scikit-learn MLPClassifier.
    Neural Network Architecture:
    - Layer 1: 128 hidden units (ReLU activation)
    - Layer 2: 64 hidden units (ReLU activation)
    - Optimizer: Adam (lr=0.001)
    - Early Stopping: Enabled (patience=5 iterations)
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n[+] --- Task 4: Deep Learning Prototype (Multi-Layer Perceptron) ---")
    
    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    X_test = processed_data['X_test']
    y_test = processed_data['y_test']
    
    # Instantiate MLP Classifier
    dl_model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=50,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=42
    )
    
    print("[*] Training Deep Neural Network (MLPClassifier 128x64)...")
    t0 = time.time()
    dl_model.fit(X_train, y_train)
    train_time = time.time() - t0
    
    print(f"[*] Training finished in {train_time:.2f}s across {dl_model.n_iter_} iterations.")
    
    # Measure Inference Latency
    t0 = time.time()
    y_pred = dl_model.predict(X_test)
    inference_time = (time.time() - t0) * 1000  # ms
    latency_per_10k = (inference_time / len(X_test)) * 10000 # ms per 10k packets
    
    y_prob = dl_model.predict_proba(X_test)[:, 1]
    
    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    dl_result = {
        'Model': 'Deep Learning (MLP)',
        'Accuracy': acc,
        'Precision': prec,
        'Recall (Detection Rate)': rec,
        'F1-Score': f1,
        'ROC-AUC': auc,
        'FPR (False Positive Rate)': fpr,
        'FNR (False Negative Rate)': fnr,
        'Training Time (s)': train_time,
        'Inference Latency (ms/10k)': latency_per_10k
    }
    
    print(f"    -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | FPR: {fpr:.4f}")
    return dl_result, dl_model

def plot_ml_vs_dl_comparison(df_comparison, output_dir="outputs"):
    """
    Generates comparative visualization of Machine Learning vs Deep Learning models.
    Compares: Accuracy, F1-Score, Training Time, and Inference Latency.
    """
    print("[*] Generating ML vs DL Comparative Benchmark Plot...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    
    models = df_comparison['Model']
    colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']
    
    # Subplot 1: Classification Accuracy
    bars1 = axes[0, 0].bar(models, df_comparison['Accuracy'], color=colors, edgecolor='black', alpha=0.85)
    axes[0, 0].set_title("Detection Accuracy Comparison", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel("Accuracy Score", fontsize=10)
    axes[0, 0].set_ylim(0.7, 1.0)
    for bar in bars1:
        h = bar.get_height()
        axes[0, 0].annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                            textcoords="offset points", ha='center', fontsize=9.5, fontweight='bold')
        
    # Subplot 2: F1-Score
    bars2 = axes[0, 1].bar(models, df_comparison['F1-Score'], color=colors, edgecolor='black', alpha=0.85)
    axes[0, 1].set_title("F1-Score (Harmonic Mean Prec/Rec)", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel("F1-Score", fontsize=10)
    axes[0, 1].set_ylim(0.7, 1.0)
    for bar in bars2:
        h = bar.get_height()
        axes[0, 1].annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                            textcoords="offset points", ha='center', fontsize=9.5, fontweight='bold')

    # Subplot 3: Training Time (Seconds)
    bars3 = axes[1, 0].bar(models, df_comparison['Training Time (s)'], color=colors, edgecolor='black', alpha=0.85)
    axes[1, 0].set_title("Model Training Time (Lower is Better)", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Time (Seconds)", fontsize=10)
    for bar in bars3:
        h = bar.get_height()
        axes[1, 0].annotate(f'{h:.2f}s', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                            textcoords="offset points", ha='center', fontsize=9.5, fontweight='bold')

    # Subplot 4: Inference Latency (ms per 10,000 packets)
    bars4 = axes[1, 1].bar(models, df_comparison['Inference Latency (ms/10k)'], color=colors, edgecolor='black', alpha=0.85)
    axes[1, 1].set_title("Inference Latency per 10,000 Packets (Lower is Better)", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel("Latency (ms)", fontsize=10)
    for bar in bars4:
        h = bar.get_height()
        axes[1, 1].annotate(f'{h:.2f}ms', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                            textcoords="offset points", ha='center', fontsize=9.5, fontweight='bold')

    for ax in axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=15, ha='right')

    plt.tight_layout()
    comparison_plot_path = os.path.join(output_dir, "07_ml_vs_dl_comparison.png")
    plt.savefig(comparison_plot_path, dpi=300)
    plt.close()
    print(f"    Saved: {comparison_plot_path}")

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_datasets
    df_train, df_test = load_data()
    processed = preprocess_datasets(df_train, df_test)
    dl_result, dl_model = train_and_evaluate_dl_model(processed)
    print(dl_result)
