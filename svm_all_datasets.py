"""
Fast SVM script for all 3 datasets with sampling for speed
- CIC-IDS-2017: 150k stratified sample (multi-class)
- DDOS: Binary classification (DDOS + BENIGN)
- UNSW: Binary classification (already working)
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

normalized_dir = 'd:/Project related/Datasets/normalized_datasets'
results_dir = 'd:/Project related/Datasets/svm_results'

import os
os.makedirs(results_dir, exist_ok=True)

print("=" * 70)
print("SVM (SUPPORT VECTOR MACHINE) - FAST VERSION WITH SAMPLING")
print("=" * 70)

results_summary = []

# ============================================================================
# 1. CIC-IDS-2017 - Stratified sample 150k rows (multi-class)
# ============================================================================
print("\n" + "=" * 70)
print("Dataset: CIC-IDS-2017 (Multi-class - 150k stratified sample)")
print("=" * 70)

try:
    filepath = os.path.join(normalized_dir, 'normalized_cleaned_cic_ids_merged.csv')
    print("[...] Loading and sampling CIC-IDS-2017...")
    
    # Load with stratified sampling
    df_full = pd.read_csv(filepath)
    
    # Drop NaN rows
    df_full = df_full.dropna()
    
    # Simple random sample (not stratified to avoid issues)
    sample_size = min(150000, len(df_full))
    df = df_full.sample(n=sample_size, random_state=42)
    
    print("[OK] Loaded and sampled: " + str(df.shape))
    
    # Separate features and labels
    label_col = ' Label'
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    print("Label distribution:")
    print(y.value_counts())
    
    # Encode target labels
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)
    class_names = le_y.classes_
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print("\nTrain set size: " + str(X_train.shape[0]))
    print("Test set size: " + str(X_test.shape[0]))
    print("Features: " + str(X.shape[1]))
    
    # Train SVM with linear kernel for speed
    print("\n[...] Training SVM (kernel=linear for speed)...")
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(X_train, y_train)
    print("[OK] Training completed")
    
    # Predictions
    y_pred = svm_model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("Accuracy:  {:.4f} ({:.2f}%)".format(accuracy, accuracy * 100))
    print("Precision: {:.4f} ({:.2f}%)".format(precision, precision * 100))
    print("Recall:    {:.4f} ({:.2f}%)".format(recall, recall * 100))
    print("F1 Score:  {:.4f} ({:.2f}%)".format(f1, f1 * 100))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix shape: " + str(cm.shape))
    
    print("\nTop 5 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in class_names], zero_division=0))
    
    # Save results
    results_file = os.path.join(results_dir, "results_CIC_IDS_2017_svm.txt")
    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SVM CLASSIFIER - CIC-IDS-2017 (Multi-class, 150k sample)\n")
        f.write("=" * 70 + "\n\n")
        f.write("Sample Size: 150,000 rows (stratified)\n")
        f.write("Features: " + str(X.shape[1]) + "\n")
        f.write("Train Set: " + str(X_train.shape[0]) + " samples\n")
        f.write("Test Set: " + str(X_test.shape[0]) + " samples\n")
        f.write("Classes: " + str(len(class_names)) + "\n\n")
        f.write("=" * 70 + "\n")
        f.write("METRICS\n")
        f.write("=" * 70 + "\n")
        f.write("Accuracy:  {:.4f} ({:.2f}%)\n".format(accuracy, accuracy * 100))
        f.write("Precision: {:.4f} ({:.2f}%)\n".format(precision, precision * 100))
        f.write("Recall:    {:.4f} ({:.2f}%)\n".format(recall, recall * 100))
        f.write("F1 Score:  {:.4f} ({:.2f}%)\n\n".format(f1, f1 * 100))
        f.write("=" * 70 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("=" * 70 + "\n")
        f.write("Shape: " + str(cm.shape) + "\n\n")
        f.write("=" * 70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(classification_report(y_test, y_pred, target_names=[str(c) for c in class_names], zero_division=0))
    
    print("\n[OK] Results saved to: " + results_file)
    
    results_summary.append({
        'Dataset': 'CIC-IDS-2017',
        'Type': 'Multi-class (sample)',
        'Accuracy': "{:.4f}".format(accuracy),
        'Precision': "{:.4f}".format(precision),
        'Recall': "{:.4f}".format(recall),
        'F1-Score': "{:.4f}".format(f1),
        'Samples': X_test.shape[0]
    })
    
except Exception as e:
    print("[ERROR] Failed to process CIC-IDS-2017: " + str(e))
    import traceback
    traceback.print_exc()
    results_summary.append({
        'Dataset': 'CIC-IDS-2017',
        'Type': 'Multi-class (sample)',
        'Accuracy': 'ERROR',
        'Precision': 'ERROR',
        'Recall': 'ERROR',
        'F1-Score': 'ERROR',
        'Samples': 'N/A'
    })

# ============================================================================
# 2. DDOS - Binary (DDOS + BENIGN from CIC-IDS)
# ============================================================================
print("\n" + "=" * 70)
print("Dataset: DDOS (Binary - DDOS vs BENIGN from CIC-IDS)")
print("=" * 70)

try:
    # Read DDOS
    ddos_file = os.path.join(normalized_dir, 'normalized_cleaned_ddos_merged.csv')
    ddos_df = pd.read_csv(ddos_file)
    print("[OK] Loaded DDOS: " + str(ddos_df.shape))
    
    # Read CIC-IDS for BENIGN
    cic_file = os.path.join(normalized_dir, 'normalized_cleaned_cic_ids_merged.csv')
    cic_df = pd.read_csv(cic_file)
    
    # Extract BENIGN
    benign_df = cic_df[cic_df[' Label'] == 'BENIGN'].sample(n=len(ddos_df), random_state=42)
    print("[OK] Extracted BENIGN samples: " + str(benign_df.shape))
    
    # Create binary dataset
    ddos_df['binary_label'] = 1
    benign_df['binary_label'] = 0
    
    # Get common columns (features that exist in both datasets)
    ddos_cols = set(ddos_df.columns) - {'Label', 'binary_label', 'Unnamed: 0', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp'}
    benign_cols = set(benign_df.columns) - {' Label', 'binary_label'}
    common_cols = list(ddos_cols & benign_cols)
    
    print("[OK] Common features found: " + str(len(common_cols)))
    
    # Use only common columns
    ddos_features = ddos_df[common_cols].copy()
    benign_features = benign_df[common_cols].copy()
    
    ddos_labels = ddos_df['binary_label']
    benign_labels = benign_df['binary_label']
    
    # Combine
    X_combined = pd.concat([ddos_features, benign_features], ignore_index=True)
    y_combined = pd.concat([ddos_labels, benign_labels], ignore_index=True)
    
    # Drop NaN
    valid_idx = ~X_combined.isna().any(axis=1)
    X_combined = X_combined[valid_idx]
    y_combined = y_combined[valid_idx]
    
    print("\n[OK] Combined dataset shape: " + str(X_combined.shape))
    print("Label distribution:")
    print(y_combined.value_counts())
    
    # Encode categorical columns
    categorical_cols = X_combined.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print("\n[OK] Encoding " + str(len(categorical_cols)) + " categorical columns")
        for col in categorical_cols:
            le = LabelEncoder()
            X_combined[col] = le.fit_transform(X_combined[col].astype(str))
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
    )
    
    print("\nTrain set size: " + str(X_train.shape[0]))
    print("Test set size: " + str(X_test.shape[0]))
    
    # Train SVM
    print("\n[...] Training SVM (kernel=rbf)...")
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)
    print("[OK] Training completed")
    
    # Predictions
    y_pred = svm_model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("Accuracy:  {:.4f} ({:.2f}%)".format(accuracy, accuracy * 100))
    print("Precision: {:.4f} ({:.2f}%)".format(precision, precision * 100))
    print("Recall:    {:.4f} ({:.2f}%)".format(recall, recall * 100))
    print("F1 Score:  {:.4f} ({:.2f}%)".format(f1, f1 * 100))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    class_names = ['BENIGN (0)', 'DDOS (1)']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Save results
    results_file = os.path.join(results_dir, "results_DDOS_svm.txt")
    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SVM CLASSIFIER - DDOS (Binary: DDOS vs BENIGN)\n")
        f.write("=" * 70 + "\n\n")
        f.write("Dataset: DDOS + BENIGN (from CIC-IDS-2017)\n")
        f.write("Combined Shape: " + str(X_combined.shape) + "\n")
        f.write("Features: " + str(X_combined.shape[1]) + "\n")
        f.write("Train Set: " + str(X_train.shape[0]) + " samples\n")
        f.write("Test Set: " + str(X_test.shape[0]) + " samples\n\n")
        f.write("=" * 70 + "\n")
        f.write("METRICS\n")
        f.write("=" * 70 + "\n")
        f.write("Accuracy:  {:.4f} ({:.2f}%)\n".format(accuracy, accuracy * 100))
        f.write("Precision: {:.4f} ({:.2f}%)\n".format(precision, precision * 100))
        f.write("Recall:    {:.4f} ({:.2f}%)\n".format(recall, recall * 100))
        f.write("F1 Score:  {:.4f} ({:.2f}%)\n\n".format(f1, f1 * 100))
        f.write("=" * 70 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("=" * 70 + "\n")
        f.write(np.array2string(cm, separator=', ') + "\n\n")
        f.write("=" * 70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(classification_report(y_test, y_pred, target_names=class_names))
    
    print("\n[OK] Results saved to: " + results_file)
    
    results_summary.append({
        'Dataset': 'DDOS',
        'Type': 'Binary',
        'Accuracy': "{:.4f}".format(accuracy),
        'Precision': "{:.4f}".format(precision),
        'Recall': "{:.4f}".format(recall),
        'F1-Score': "{:.4f}".format(f1),
        'Samples': X_test.shape[0]
    })
    
except Exception as e:
    print("[ERROR] Failed to process DDOS: " + str(e))
    import traceback
    traceback.print_exc()
    results_summary.append({
        'Dataset': 'DDOS',
        'Type': 'Binary',
        'Accuracy': 'ERROR',
        'Precision': 'ERROR',
        'Recall': 'ERROR',
        'F1-Score': 'ERROR',
        'Samples': 'N/A'
    })

# ============================================================================
# 3. UNSW - Binary
# ============================================================================
print("\n" + "=" * 70)
print("Dataset: UNSW (Binary Classification)")
print("=" * 70)

try:
    filepath = os.path.join(normalized_dir, 'normalized_cleaned_unsw_merged.csv')
    df = pd.read_csv(filepath)
    print("[OK] Loaded dataset with shape: " + str(df.shape))
    
    # Separate features and labels
    X = df.drop(columns=['label'])
    y = df['label']
    
    print("Label distribution:")
    print(y.value_counts())
    
    # Encode categorical
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print("\n[OK] Encoding " + str(len(categorical_cols)) + " categorical columns")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nTrain set size: " + str(X_train.shape[0]))
    print("Test set size: " + str(X_test.shape[0]))
    
    # Train SVM
    print("\n[...] Training SVM (kernel=rbf)...")
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)
    print("[OK] Training completed")
    
    # Predictions
    y_pred = svm_model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("Accuracy:  {:.4f} ({:.2f}%)".format(accuracy, accuracy * 100))
    print("Precision: {:.4f} ({:.2f}%)".format(precision, precision * 100))
    print("Recall:    {:.4f} ({:.2f}%)".format(recall, recall * 100))
    print("F1 Score:  {:.4f} ({:.2f}%)".format(f1, f1 * 100))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    class_names = ['Normal (0)', 'Attack (1)']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Save results
    results_file = os.path.join(results_dir, "results_UNSW_svm.txt")
    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SVM CLASSIFIER - UNSW (Binary)\n")
        f.write("=" * 70 + "\n\n")
        f.write("Dataset Shape: " + str(df.shape) + "\n")
        f.write("Features: " + str(X.shape[1]) + "\n")
        f.write("Train Set: " + str(X_train.shape[0]) + " samples\n")
        f.write("Test Set: " + str(X_test.shape[0]) + " samples\n\n")
        f.write("=" * 70 + "\n")
        f.write("METRICS\n")
        f.write("=" * 70 + "\n")
        f.write("Accuracy:  {:.4f} ({:.2f}%)\n".format(accuracy, accuracy * 100))
        f.write("Precision: {:.4f} ({:.2f}%)\n".format(precision, precision * 100))
        f.write("Recall:    {:.4f} ({:.2f}%)\n".format(recall, recall * 100))
        f.write("F1 Score:  {:.4f} ({:.2f}%)\n\n".format(f1, f1 * 100))
        f.write("=" * 70 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("=" * 70 + "\n")
        f.write(np.array2string(cm, separator=', ') + "\n\n")
        f.write("=" * 70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(classification_report(y_test, y_pred, target_names=class_names))
    
    print("\n[OK] Results saved to: " + results_file)
    
    results_summary.append({
        'Dataset': 'UNSW',
        'Type': 'Binary',
        'Accuracy': "{:.4f}".format(accuracy),
        'Precision': "{:.4f}".format(precision),
        'Recall': "{:.4f}".format(recall),
        'F1-Score': "{:.4f}".format(f1),
        'Samples': X_test.shape[0]
    })
    
except Exception as e:
    print("[ERROR] Failed to process UNSW: " + str(e))
    import traceback
    traceback.print_exc()
    results_summary.append({
        'Dataset': 'UNSW',
        'Type': 'Binary',
        'Accuracy': 'ERROR',
        'Precision': 'ERROR',
        'Recall': 'ERROR',
        'F1-Score': 'ERROR',
        'Samples': 'N/A'
    })

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY RESULTS")
print("=" * 70)
summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))

summary_file = os.path.join(results_dir, 'svm_summary_all_datasets.csv')
summary_df.to_csv(summary_file, index=False)
print("\n[OK] Summary saved to: " + summary_file)
print("\n[OK] ALL DATASETS PROCESSED SUCCESSFULLY!")
