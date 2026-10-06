import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier

# Set global aesthetic styling for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0

def run_exploratory_data_analysis(df_train_input, output_dir="outputs"):
    """
    Performs comprehensive Exploratory Data Analysis (EDA) on the UNSW-NB15 dataset.
    Identifies common threat indicators and generates multi-panel figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n[+] --- Task 2: Cyber Security Threat Landscape Analysis (EDA) ---")
    
    df_train = df_train_input.copy()
    
    # 1. Target Label Distribution (Normal vs Suspicious)
    print("[*] Analyzing Binary Target Distribution...")
    fig, ax = plt.subplots(figsize=(8, 5))
    label_counts = df_train['label'].value_counts()
    label_names = {0: 'Normal Traffic (0)', 1: 'Suspicious / Threat (1)'}
    labels_mapped = [label_names.get(x, str(x)) for x in label_counts.index]
    
    colors = ['#2ecc71', '#e74c3c'] # Green for normal, Red for threat
    bars = ax.bar(labels_mapped, label_counts.values, color=colors, width=0.5, edgecolor='black', alpha=0.85)
    
    total = len(df_train)
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        ax.annotate(f'{height:,}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    ax.set_title("UNSW-NB15 Target Distribution: Normal vs Suspicious Behavior", fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel("Event Count", fontsize=11)
    ax.set_ylim(0, max(label_counts.values) * 1.18)
    plt.tight_layout()
    plot_1_path = os.path.join(output_dir, "01_class_distribution.png")
    plt.savefig(plot_1_path, dpi=300)
    plt.close()
    print(f"    Saved: {plot_1_path}")
    
    # 2. Attack Category Breakdown
    if 'attack_cat' in df_train.columns:
        print("[*] Analyzing Attack Categories...")
        fig, ax = plt.subplots(figsize=(10, 6))
        # Strip trailing whitespace if any
        df_train['attack_cat_clean'] = df_train['attack_cat'].astype(str).str.strip()
        cat_counts = df_train['attack_cat_clean'].value_counts()
        
        palette = sns.color_palette("flare", len(cat_counts))
        bars = ax.barh(cat_counts.index[::-1], cat_counts.values[::-1], color=palette, edgecolor='black', alpha=0.85)
        
        for bar in bars:
            width = bar.get_width()
            pct = (width / total) * 100
            ax.annotate(f'{width:,} ({pct:.1f}%)',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(8, 0), textcoords="offset points",
                        ha='left', va='center', fontsize=9.5, fontweight='bold')
            
        ax.set_title("Distribution of Cyber Threat Categories (UNSW-NB15)", fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel("Frequency Count", fontsize=11)
        ax.set_xlim(0, max(cat_counts.values) * 1.25)
        plt.tight_layout()
        plot_2_path = os.path.join(output_dir, "02_attack_categories.png")
        plt.savefig(plot_2_path, dpi=300)
        plt.close()
        print(f"    Saved: {plot_2_path}")

    # 3. Top Network Protocols and Application Services by Threat Class
    print("[*] Visualizing Protocol & Service Distributions...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Top 5 Protocols
    top_protos = df_train['proto'].value_counts().head(5).index
    proto_df = df_train[df_train['proto'].isin(top_protos)]
    sns.countplot(data=proto_df, x='proto', hue='label', palette=['#2ecc71', '#e74c3c'], ax=ax1, edgecolor='black')
    ax1.set_title("Threat Distribution across Top 5 Protocols", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Network Protocol", fontsize=10)
    ax1.set_ylabel("Event Count", fontsize=10)
    ax1.legend(title="Class", labels=['Normal', 'Suspicious'])
    
    # Top 6 Services (excluding '-')
    service_clean = df_train[df_train['service'] != '-']
    top_services = service_clean['service'].value_counts().head(6).index
    service_df = service_clean[service_clean['service'].isin(top_services)]
    sns.countplot(data=service_df, x='service', hue='label', palette=['#2ecc71', '#e74c3c'], ax=ax2, edgecolor='black')
    ax2.set_title("Threat Distribution across Top Application Services", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Application Service", fontsize=10)
    ax2.set_ylabel("Event Count", fontsize=10)
    ax2.legend(title="Class", labels=['Normal', 'Suspicious'])
    
    plt.tight_layout()
    plot_3_path = os.path.join(output_dir, "03_top_protocols_services.png")
    plt.savefig(plot_3_path, dpi=300)
    plt.close()
    print(f"    Saved: {plot_3_path}")
    
    # 4. Distribution of Key Threat Feature Indicators
    print("[*] Generating Key Threat Indicator Feature Distributions...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    # Feature 1: Source Time To Live (sttl)
    sns.kdeplot(data=df_train, x='sttl', hue='label', palette=['#2ecc71', '#e74c3c'], fill=True, common_norm=False, ax=axes[0,0], alpha=0.4)
    axes[0,0].set_title("Source Time-To-Live (sttl) Distribution", fontsize=11, fontweight='bold')
    axes[0,0].set_xlabel("sttl Value", fontsize=10)
    
    # Feature 2: Log Flow Rate (rate)
    df_train['log_rate'] = np.log1p(df_train['rate'])
    sns.kdeplot(data=df_train, x='log_rate', hue='label', palette=['#2ecc71', '#e74c3c'], fill=True, common_norm=False, ax=axes[0,1], alpha=0.4)
    axes[0,1].set_title("Log Flow Packet Rate (log1p rate)", fontsize=11, fontweight='bold')
    axes[0,1].set_xlabel("log1p(rate)", fontsize=10)
    
    # Feature 3: Source Mean Packet Size (smean)
    sns.kdeplot(data=df_train, x='smean', hue='label', palette=['#2ecc71', '#e74c3c'], fill=True, common_norm=False, ax=axes[1,0], alpha=0.4)
    axes[1,0].set_title("Source Mean Packet Size (smean)", fontsize=11, fontweight='bold')
    axes[1,0].set_xlabel("smean Bytes", fontsize=10)
    
    # Feature 4: Destination Time-To-Live (dttl)
    sns.kdeplot(data=df_train, x='dttl', hue='label', palette=['#2ecc71', '#e74c3c'], fill=True, common_norm=False, ax=axes[1,1], alpha=0.4)
    axes[1,1].set_title("Destination Time-To-Live (dttl) Distribution", fontsize=11, fontweight='bold')
    axes[1,1].set_xlabel("dttl Value", fontsize=10)
    
    plt.tight_layout()
    plot_4_path = os.path.join(output_dir, "04_feature_distributions.png")
    plt.savefig(plot_4_path, dpi=300)
    plt.close()
    print(f"    Saved: {plot_4_path}")

    # 5. Correlation Heatmap of Top Numerical Features
    print("[*] Generating Top Feature Correlation Heatmap...")
    num_cols = df_train.select_dtypes(include=[np.number]).columns
    # Drop 'id' if present
    num_cols = [c for c in num_cols if c != 'id']
    
    # Compute correlation with target label
    corr_matrix = df_train[num_cols].corr()
    top_corr_features = corr_matrix['label'].abs().sort_values(ascending=False).head(16).index
    top_corr_sub = df_train[top_corr_features].corr()
    
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(top_corr_sub, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax, cbar_kws={'shrink': 0.8}, annot_kws={"size": 8.5})
    ax.set_title("Correlation Heatmap: Top 15 Threat Indicators with Label", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plot_5_path = os.path.join(output_dir, "05_correlation_heatmap.png")
    plt.savefig(plot_5_path, dpi=300)
    plt.close()
    print(f"    Saved: {plot_5_path}")

    print("[+] EDA analysis completed successfully!")
    
def generate_feature_importance_plot(X_train, y_train, feature_names, output_dir="outputs"):
    """
    Computes and plots Tree-based Feature Importance for threat analysis.
    """
    print("[*] Computing Random Forest Feature Importance Rankings...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:15] # Top 15 features
    
    top_features = [feature_names[i] for i in indices]
    top_scores = importances[indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("Blues_r", len(top_features))
    bars = ax.barh(top_features[::-1], top_scores[::-1], color=palette, edgecolor='black', alpha=0.85)
    
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.4f}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9, fontweight='bold')
        
    ax.set_title("Top 15 Cyber Threat Indicators (Random Forest Feature Importance)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Relative Importance Score", fontsize=11)
    ax.set_xlim(0, max(top_scores) * 1.18)
    plt.tight_layout()
    plot_6_path = os.path.join(output_dir, "06_feature_importance.png")
    plt.savefig(plot_6_path, dpi=300)
    plt.close()
    print(f"    Saved: {plot_6_path}")
    return top_features, top_scores

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_datasets
    df_train, df_test = load_data()
    run_exploratory_data_analysis(df_train)
    processed = preprocess_datasets(df_train, df_test)
    generate_feature_importance_plot(processed['X_train'], processed['y_train'], processed['feature_names'])
