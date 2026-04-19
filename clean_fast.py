"""
Fast chunk-based cleaning script for large datasets
Processes data in chunks to minimize memory usage
"""
import pandas as pd
import numpy as np
import os

merged_dir = 'd:/Project related/Datasets/merged_datasets'
cleaned_dir = 'd:/Project related/Datasets/cleaned_datasets'
os.makedirs(cleaned_dir, exist_ok=True)

datasets = {
    'DDOS': 'ddos_merged.csv',
    'NSL_KDD': 'nsl_kdd_merged.csv',
    'UNSW': 'unsw_merged.csv'
}

CHUNK_SIZE = 50000  # Process 50k rows at a time

print("Fast chunk-based dataset cleaning...\n")

for dataset_name, filename in datasets.items():
    print(f"Cleaning {dataset_name}...")
    filepath = os.path.join(merged_dir, filename)
    output_path = os.path.join(cleaned_dir, f'cleaned_{filename}')
    
    try:
        total_rows = 0
        first_chunk = True
        
        # Read and process in chunks
        for chunk_idx, df_chunk in enumerate(pd.read_csv(filepath, chunksize=CHUNK_SIZE)):
            print(f"  Processing chunk {chunk_idx + 1}...")
            
            # Remove duplicates
            df_chunk = df_chunk.drop_duplicates()
            
            # Drop rows with NaN
            df_chunk = df_chunk.dropna()
            
            # Handle infinite values
            numeric_cols = df_chunk.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df_chunk[col] = df_chunk[col].replace([np.inf, -np.inf], np.nan)
            df_chunk = df_chunk.dropna()
            
            # Remove constant columns (only on first chunk)
            if first_chunk:
                constant_cols = [col for col in df_chunk.columns if df_chunk[col].nunique() <= 1]
                if constant_cols:
                    print(f"  Removing constant columns: {constant_cols}")
                    df_chunk = df_chunk.drop(columns=constant_cols)
            
            # Save chunk
            if first_chunk:
                df_chunk.to_csv(output_path, index=False, mode='w')
                first_chunk = False
            else:
                df_chunk.to_csv(output_path, index=False, mode='a', header=False)
            
            total_rows += len(df_chunk)
        
        print(f"  ✓ {dataset_name} cleaned: {total_rows} rows saved\n")
        
    except Exception as e:
        print(f"  ✗ Error cleaning {dataset_name}: {e}\n")

print("✓ All remaining datasets cleaned!")
