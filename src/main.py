import os
import pandas as pd
import joblib

from data_preprocessing import load_data, preprocess_datasets
from eda_analysis import run_exploratory_data_analysis, generate_feature_importance_plot
from ml_models import train_and_evaluate_ml_models
from dl_models import train_and_evaluate_dl_model, plot_ml_vs_dl_comparison
from ethical_sim import simulate_ethical_threshold_tradeoff, analyze_per_category_bias

def run_end_to_end_cybersecurity_pipeline():
    """
    Main Orchestrator for the End-to-End AI-Driven Cyber Threat Awareness & Detection Prototype.
    Runs Tasks 1, 2, 3, 4, and 5 sequentially.
    """
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("  K.R. MANGALAM UNIVERSITY - FACULTY OF ENGINEERING & TECHNOLOGY")
    print("  AI IN CYBER SECURITY LAB (ENSP355) - END-TO-END MINI PROJECT")
    print("  AI-Driven Cyber Threat Awareness and Detection Prototype")
    print("="*80)
    
    # Task 1 & 2: Load Data and Run EDA
    df_train, df_test = load_data(data_dir="data")
    run_exploratory_data_analysis(df_train, output_dir=output_dir)
    
    # Task 3: Preprocess Datasets
    processed = preprocess_datasets(df_train, df_test)
    
    # Feature Importance Plot
    generate_feature_importance_plot(processed['X_train'], processed['y_train'], 
                                   processed['feature_names'], output_dir=output_dir)
    
    # Task 3: Supervised Machine Learning Models
    df_ml_results, ml_models = train_and_evaluate_ml_models(processed, output_dir=output_dir)
    
    # Task 4: Deep Learning Model Prototype
    dl_result, dl_model = train_and_evaluate_dl_model(processed, output_dir=output_dir)
    
    # Combine ML and DL Results for Comparative Analysis
    df_all_results = pd.concat([df_ml_results, pd.DataFrame([dl_result])], ignore_index=True)
    
    print("\n" + "="*80)
    print("  SUMMARY: MACHINE LEARNING vs DEEP LEARNING MODEL BENCHMARK")
    print("="*80)
    print(df_all_results.to_string(index=False))
    
    # Save benchmark table to CSV
    csv_results_path = os.path.join(output_dir, "model_comparison_results.csv")
    df_all_results.to_csv(csv_results_path, index=False)
    print(f"\n[*] Model comparison matrix saved to: {csv_results_path}")
    
    # Plot Comparative Visualizations
    plot_ml_vs_dl_comparison(df_all_results, output_dir=output_dir)
    
    # Task 5: Ethical Reflection and Operational Threshold Simulation
    best_model = ml_models['Random Forest']
    simulate_ethical_threshold_tradeoff(best_model, processed['X_test'], processed['y_test'], output_dir=output_dir)
    analyze_per_category_bias(best_model, processed['X_test'], processed['y_test'], processed['attack_cat_test'], output_dir=output_dir)
    
    # Save best model to outputs
    model_save_path = os.path.join(output_dir, "best_threat_detector_rf.joblib")
    joblib.dump(best_model, model_save_path)
    print(f"[*] Best Threat Detection model (Random Forest) saved to: {model_save_path}")
    
    print("\n" + "="*80)
    print("  [SUCCESS] END-TO-END CYBER THREAT DETECTION PIPELINE EXECUTED CLEANLY!")
    print("="*80)

if __name__ == "__main__":
    run_end_to_end_cybersecurity_pipeline()
