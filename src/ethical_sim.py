import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def simulate_ethical_threshold_tradeoff(model, X_test, y_test, output_dir="outputs"):
    """
    Task 5: Simulates threshold adjustment scenario.
    In cybersecurity:
    - Low threshold (e.g. 0.2): Extremely cautious. High Recall (catches almost all threats), but HIGH FPR (blocks many legitimate users/requests).
    - High threshold (e.g. 0.8): Very permissive. Low FPR (rarely blocks legitimate users), but HIGH FNR (misses dangerous attacks).
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n[+] --- Task 5: Ethical Reflection & Operational Threshold Simulation ---")
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        print("[!] Model does not support probability predictions.")
        return None

    thresholds = np.linspace(0.05, 0.95, 37)
    precisions = []
    recalls = []
    f1s = []
    fprs = []
    
    for tau in thresholds:
        preds = (y_prob >= tau).astype(int)
        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        f = f1_score(y_test, preds, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        fprs.append(fpr)
        
    # Plot Threshold Trade-off Curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions, label='Precision (Low False Alarm)', color='#2ecc71', lw=2.5)
    ax.plot(thresholds, recalls, label='Recall / Security Coverage (Detection Rate)', color='#e74c3c', lw=2.5)
    ax.plot(thresholds, f1s, label='F1-Score', color='#3498db', lw=2, linestyle='--')
    ax.plot(thresholds, fprs, label='False Positive Rate (Legitimate Users Blocked)', color='#e67e22', lw=2, linestyle=':')
    
    # Highlight optimal threshold point (max F1)
    best_idx = np.argmax(f1s)
    best_tau = thresholds[best_idx]
    ax.axvline(x=best_tau, color='purple', linestyle='--', alpha=0.7, label=f'Optimal F1 Threshold (τ = {best_tau:.2f})')
    
    ax.set_title("Operational & Ethical Trade-off: Security vs User Disruption across Decision Thresholds", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Decision Threshold (τ)", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9.5)
    plt.tight_layout()
    
    threshold_plot_path = os.path.join(output_dir, "08_threshold_tradeoff.png")
    plt.savefig(threshold_plot_path, dpi=300)
    plt.close()
    print(f"[*] Optimal Decision Threshold (Max F1): tau = {best_tau:.2f} -> Prec: {precisions[best_idx]:.4f}, Rec: {recalls[best_idx]:.4f}, FPR: {fprs[best_idx]:.4f}")
    
    return {
        'thresholds': thresholds,
        'precisions': precisions,
        'recalls': recalls,
        'f1s': f1s,
        'fprs': fprs,
        'best_tau': best_tau
    }

def analyze_per_category_bias(model, X_test, y_test, attack_cat_test, output_dir="outputs"):
    """
    Evaluates model performance across individual attack categories to uncover dataset bias & rare class neglect.
    """
    print("\n[*] Analyzing Dataset Bias & Per-Category Attack Detection Performance...")
    
    if attack_cat_test is None:
        print("[!] No attack category labels found.")
        return None
        
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)
        
    preds = (y_prob >= 0.5).astype(int)
    
    df_eval = pd.DataFrame({
        'attack_cat': [str(x).strip() for x in attack_cat_test],
        'true_label': y_test,
        'predicted': preds
    })
    
    cat_summary = []
    for cat, group in df_eval.groupby('attack_cat'):
        total_samples = len(group)
        if cat.lower() == 'normal':
            # For Normal, evaluate False Positive Rate (percentage of Normal wrongly flagged as attack)
            false_pos = (group['predicted'] == 1).sum()
            fp_rate = false_pos / total_samples
            cat_summary.append({
                'Category': cat,
                'Sample Count': total_samples,
                'Type': 'Legitimate Traffic',
                'Metric (Recall/Accuracy)': 1.0 - fp_rate,
                'Misclassified Count': false_pos,
                'Notes': f'FPR = {fp_rate:.2%} (Blocked Users)'
            })
        else:
            # For Threat categories, evaluate Detection Rate (Recall)
            detected = (group['predicted'] == 1).sum()
            recall = detected / total_samples
            cat_summary.append({
                'Category': cat,
                'Sample Count': total_samples,
                'Type': 'Cyber Attack',
                'Metric (Recall/Accuracy)': recall,
                'Misclassified Count': total_samples - detected,
                'Notes': f'Detection Rate = {recall:.2%}'
            })
            
    df_cat_summary = pd.DataFrame(cat_summary).sort_values(by='Sample Count', ascending=False)
    print(df_cat_summary.to_string(index=False))
    
    # Save table to CSV
    csv_path = os.path.join(output_dir, "ethical_bias_attack_category_performance.csv")
    df_cat_summary.to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path}")
    return df_cat_summary

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_datasets
    from ml_models import train_and_evaluate_ml_models
    df_train, df_test = load_data()
    processed = preprocess_datasets(df_train, df_test)
    _, models = train_and_evaluate_ml_models(processed)
    simulate_ethical_threshold_tradeoff(models['Random Forest'], processed['X_test'], processed['y_test'])
    analyze_per_category_bias(models['Random Forest'], processed['X_test'], processed['y_test'], processed['attack_cat_test'])
