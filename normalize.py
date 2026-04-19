"""
Script to normalize 4 cleaned datasets using Min-Max scaling
Min-Max Scaling: X_scaled = (X - X_min) / (X_max - X_min)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

cleaned_dir = 'd:/Project related/Datasets/cleaned_datasets'
normalized_dir = 'd:/Project related/Datasets/normalized_datasets'
os.makedirs(normalized_dir, exist_ok=True)

# List of cleaned datasets
datasets = {
    'CIC-IDS-2017': 'cleaned_cic_ids_merged.csv',
    'DDOS': 'cleaned_ddos_merged.csv',
    'NSL_KDD': 'cleaned_nsl_kdd_merged.csv',
    'UNSW': 'cleaned_unsw_merged.csv'
}

print("Starting dataset normalization using Min-Max Scaling...\n")

for dataset_name, filename in datasets.items():
    print(f"Normalizing {dataset_name}...")
    filepath = os.path.join(cleaned_dir, filename)
    
    try:
        # Read cleaned dataset
        df = pd.read_csv(filepath)
        print(f"  Shape: {df.shape}")
        
        # Identify label column (usually the last column or columns with specific names)
        label_col = None
        label_candidates = ['Label', 'label', 'class', 'Class', 'Attack', 'attack', 'target']
        
        for col in label_candidates:
            if col in df.columns:
                label_col = col
                break
        
        # If no label column found, assume last column is label
        if label_col is None and df.shape[1] > 1:
            # Try to detect if last column looks like a label
            last_col = df.columns[-1]
            if df[last_col].dtype == 'object' or df[last_col].nunique() < 50:
                label_col = last_col
        
        # Separate features and label
        if label_col and label_col in df.columns:
            X = df.drop(columns=[label_col])
            y = df[label_col]
            print(f"  ✓ Label column identified: {label_col}")
        else:
            X = df
            y = None
            print(f"  ! No label column found, normalizing all features")
        
        # Select only numeric columns for normalization
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if numeric_cols:
            print(f"  Numeric columns: {len(numeric_cols)}")
            print(f"  Categorical columns: {len(categorical_cols)}")
            
            # Initialize MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            
            # Normalize numeric columns
            X_numeric = X[numeric_cols].copy()
            X_normalized = scaler.fit_transform(X_numeric)
            X_normalized_df = pd.DataFrame(X_normalized, columns=numeric_cols)
            
            # Keep categorical columns unchanged
            if categorical_cols:
                X_normalized_df[categorical_cols] = X[categorical_cols].values
            
            # Combine with label if it exists
            if y is not None:
                result_df = X_normalized_df.copy()
                result_df[label_col] = y.values
            else:
                result_df = X_normalized_df
            
            print(f"  ✓ Normalization completed")
            
            # Save normalized dataset
            output_path = os.path.join(normalized_dir, f'normalized_{filename}')
            result_df.to_csv(output_path, index=False)
            print(f"  ✓ Saved to {output_path}")
            
            # Print normalization stats
            print(f"  Min values (sample): {X_normalized_df[numeric_cols[:3]].min().values}")
            print(f"  Max values (sample): {X_normalized_df[numeric_cols[:3]].max().values}\n")
        else:
            print(f"  ✗ No numeric columns found in {dataset_name}\n")
        
    except Exception as e:
        print(f"  ✗ Error normalizing {dataset_name}: {e}\n")

print("✓ All datasets normalized successfully!")
