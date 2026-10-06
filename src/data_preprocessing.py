import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def load_data(data_dir="data"):
    """
    Loads UNSW-NB15 training and testing CSV datasets.
    """
    train_path = os.path.join(data_dir, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(data_dir, "UNSW_NB15_testing-set.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset files not found in {data_dir}. Please ensure UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv are placed in {data_dir}.")
        
    print(f"[+] Loading training data from {train_path}...")
    df_train = pd.read_csv(train_path)
    print(f"[+] Loading testing data from {test_path}...")
    df_test = pd.read_csv(test_path)
    
    print(f"[*] Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    return df_train, df_test

def preprocess_datasets(df_train, df_test):
    """
    Preprocesses UNSW-NB15 datasets:
    - Drops non-predictive columns ('id')
    - Handles missing/null values
    - Encodes categorical attributes ('proto', 'service', 'state')
    - Normalizes numerical attributes via StandardScaler
    """
    print("[+] Starting data preprocessing pipeline...")
    
    # Copy datasets to avoid mutating originals
    train = df_train.copy()
    test = df_test.copy()
    
    # Drop 'id' column if present
    if 'id' in train.columns:
        train.drop(columns=['id'], inplace=True)
    if 'id' in test.columns:
        test.drop(columns=['id'], inplace=True)
        
    # Extract target variables
    y_train = train['label'].values
    y_test = test['label'].values
    
    # Store attack category names if available
    attack_cat_train = train['attack_cat'].values if 'attack_cat' in train.columns else None
    attack_cat_test = test['attack_cat'].values if 'attack_cat' in test.columns else None
    
    # Drop targets from features
    feature_cols_train = [c for c in train.columns if c not in ['label', 'attack_cat']]
    X_train_df = train[feature_cols_train]
    X_test_df = test[feature_cols_train]
    
    # Handle categorical columns
    cat_cols = ['proto', 'service', 'state']
    num_cols = [c for c in feature_cols_train if c not in cat_cols]
    
    # Handle missing values in numerical columns if any
    X_train_df[num_cols] = X_train_df[num_cols].fillna(X_train_df[num_cols].median())
    X_test_df[num_cols] = X_test_df[num_cols].fillna(X_train_df[num_cols].median())
    
    # Label encode categorical variables consistently across train and test
    label_encoders = {}
    X_train_encoded = X_train_df.copy()
    X_test_encoded = X_test_df.copy()
    
    for col in cat_cols:
        le = LabelEncoder()
        # Fit on combined unique values to handle unseen categories gracefully
        combined = pd.concat([X_train_df[col], X_test_df[col]]).astype(str)
        le.fit(combined)
        
        X_train_encoded[col] = le.transform(X_train_df[col].astype(str))
        X_test_encoded[col] = le.transform(X_test_df[col].astype(str))
        label_encoders[col] = le

    # Scale numerical features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)
    
    feature_names = list(X_train_encoded.columns)
    
    print(f"[*] Preprocessing complete! Feature count: {len(feature_names)}")
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'attack_cat_train': attack_cat_train,
        'attack_cat_test': attack_cat_test,
        'feature_names': feature_names,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'X_train_raw': X_train_df,
        'X_test_raw': X_test_df
    }

if __name__ == "__main__":
    df_train, df_test = load_data()
    processed = preprocess_datasets(df_train, df_test)
    print("Preprocessing verification successful!")
