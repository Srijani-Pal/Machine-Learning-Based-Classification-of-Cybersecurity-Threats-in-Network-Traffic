"""
Improved SVM script for handling all 3 datasets
- CIC-IDS-2017: Drop NaN rows, handles multi-class
- DDOS: Combines DDOS + BENIGN from CIC-IDS for binary classification
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
print("SVM (SUPPORT VECTOR MACHINE) - IMPROVED VERSION")
print("=" * 70)

results_summary = []

# ============================================================================
# 1. CIC-IDS-2017 - Drop NaN rows (handle multi-class)
# ============================================================================
print("\n" + "=" * 70)
print("Dataset: CIC-IDS-2017 (Multi-class Classification)")
print("=" * 70)

try:
    filepath = os.path.join(normalized_dir, 'normalized_cleaned_cic_ids_merged.csv')
    df = pd.read_csv(filepath)
    print("[OK] Loaded dataset with shape: " + str(df.shape))
    
    # Identify label column (with leading space in CIC-IDS)
    label_col = ' Label'
    
    # Separate features and labels
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    print("[OK] Original features shape: " + str(X.shape))
    print("[!] Checking for NaN values...")
    nan_count = X.isna().sum().sum()
    print("[!] Total NaN values: " + str(nan_count))
    
    if nan_count > 0:
        print("[...] Dropping rows with NaN values...")
        X = X.dropna()
        y = y.loc[X.index]  # Sync labels with cleaned features
        print("[OK] After removing NaN: " + str(X.shape))
    
    print("\nLabel distribution:")
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
    
    # Train SVM
    print("\n[...] Training SVM (kernel=rbf, C=1.0)...")
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, verbose=1)
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
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in class_names]))
    
    # Save results
    results_file = os.path.join(results_dir, "results_CIC_IDS_2017.txt")
    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SVM CLASSIFIER - CIC-IDS-2017 (Multi-class)\n")
        f.write("=" * 70 + "\n\n")
        f.write("Dataset Shape: " + str(df.shape) + "\n")
        f.write("After NaN removal: " + str(X.shape) + "\n")
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
        f.write(np.array2string(cm, separator=', ') + "\n\n")
        f.write("=" * 70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(classification_report(y_test, y_pred, target_names=[str(c) for c in class_names]))
    
    print("\n[OK] Results saved to: " + results_file)
    
    results_summary.append({
        'Dataset': 'CIC-IDS-2017',
        'Type': 'Multi-class',
        'Accuracy': "{:.4f}".format(accuracy),
        'Precision': "{:.4f}".format(precision),
        'Recall': "{:.4f}".format(recall),
        'F1-Score': "{:.4f}".format(f1),
        'Test Samples': X_test.shape[0]
    })
    
except Exception as e:
    print("[ERROR] Failed to process CIC-IDS-2017: " + str(e))
    import traceback
    traceback.print_exc()
    results_summary.append({
        'Dataset': 'CIC-IDS-2017',
        'Type': 'Multi-class',
        'Accuracy': 'ERROR',
        'Precision': 'ERROR',
        'Recall': 'ERROR',
        'F1-Score': 'ERROR',
        'Test Samples': 'N/A'
    })

# ============================================================================
# 2. DDOS - Binary Classification (DDOS + BENIGN from CIC-IDS)
# ============================================================================
print("\n" + "=" * 70)
print("Dataset: DDOS (Binary Classification - DDOS vs BENIGN)")
print("=" * 70)

try:
    # Read DDOS dataset
    ddos_file = os.path.join(normalized_dir, 'normalized_cleaned_ddos_merged.csv')
    ddos_df = pd.read_csv(ddos_file)
    print("[OK] Loaded DDOS: " + str(ddos_df.shape))
    
    # Read CIC-IDS for BENIGN samples
    cic_file = os.path.join(normalized_dir, 'normalized_cleaned_cic_ids_merged.csv')
    cic_df = pd.read_csv(cic_file)
    print("[OK] Loaded CIC-IDS: " + str(cic_df.shape))
    
    # Extract BENIGN from CIC-IDS (note: Label has leading space)
    benign_df = cic_df[cic_df[' Label'] == 'BENIGN'].head(len(ddos_df))  # Match DDOS size
    print("[OK] Extracted BENIGN samples: " + str(benign_df.shape))
    
    # Create binary dataset
    ddos_df['binary_label'] = 1  # 1 = Attack (DDOS)
    benign_df['binary_label'] = 0  # 0 = Normal (BENIGN)
    
    # Drop original label columns and any non-feature columns
    ddos_features = ddos_df.drop(columns=['Label', 'binary_label', 'Unnamed: 0', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp'], errors='ignore')
    benign_features = benign_df.drop(columns=[' Label', 'binary_label'], errors='ignore')
    
    ddos_labels = ddos_df['binary_label']
    benign_labels = benign_df['binary_label']
    
    # Combine
    X_combined = pd.concat([ddos_features, benign_features], ignore_index=True)
    y_combined = pd.concat([ddos_labels, benign_labels], ignore_index=True)
    
    print("\n[OK] Combined dataset shape: " + str(X_combined.shape))
    print("Label distribution:")
    print(y_combined.value_counts())
    
    # Handle missing values
    print("\n[!] Checking for NaN values...")
    nan_count = X_combined.isna().sum().sum()
    print("[!] Total NaN values: " + str(nan_count))
    
    if nan_count > 0:
        print("[...] Dropping rows with NaN values...")
        valid_idx = ~X_combined.isna().any(axis=1)
        X_combined = X_combined[valid_idx]
        y_combined = y_combined[valid_idx]
        print("[OK] After removing NaN: " + str(X_combined.shape))
    
    # Handle categorical columns
    categorical_cols = X_combined.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print("\n[OK] Encoding " + str(len(categorical_cols)) + " categorical columns")
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X_combined[col] = le.fit_transform(X_combined[col].astype(str))
            le_dict[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
    )
    
    print("\nTrain set size: " + str(X_train.shape[0]))
    print("Test set size: " + str(X_test.shape[0]))
    
    # Train SVM
    print("\n[...] Training SVM (kernel=rbf, C=1.0)...")
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, verbose=1)
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
    results_file = os.path.join(results_dir, "results_DDOS.txt")
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
        'Test Samples': X_test.shape[0]
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
        'Test Samples': 'N/A'
    })

# ============================================================================
# 3. UNSW - Binary Classification (already working)
# ============================================================================
print("\n" + "=" * 70)
print("Dataset: UNSW (Binary Classification)")
print("=" * 70)

try:
    filepath = os.path.join(normalized_dir, 'normalized_cleaned_unsw_merged.csv')
    df = pd.read_csv(filepath)
    print("[OK] Loaded dataset with shape: " + str(df.shape))
    
    # Identify label column
    label_col = 'label'
    
    # Separate features and labels
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    print("[OK] Features shape: " + str(X.shape))
    print("Label distribution:")
    print(y.value_counts())
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print("\n[OK] Encoding " + str(len(categorical_cols)) + " categorical columns")
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le
    
    y_encoded = y
    class_names = np.unique(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print("\nTrain set size: " + str(X_train.shape[0]))
    print("Test set size: " + str(X_test.shape[0]))
    
    # Train SVM
    print("\n[...] Training SVM (kernel=rbf, C=1.0)...")
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, verbose=1)
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
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in class_names]))
    
    # Save results
    results_file = os.path.join(results_dir, "results_UNSW_updated.txt")
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
        f.write(classification_report(y_test, y_pred, target_names=[str(c) for c in class_names]))
    
    print("\n[OK] Results saved to: " + results_file)
    
    results_summary.append({
        'Dataset': 'UNSW',
        'Type': 'Binary',
        'Accuracy': "{:.4f}".format(accuracy),
        'Precision': "{:.4f}".format(precision),
        'Recall': "{:.4f}".format(recall),
        'F1-Score': "{:.4f}".format(f1),
        'Test Samples': X_test.shape[0]
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
        'Test Samples': 'N/A'
    })

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY RESULTS")
print("=" * 70)
summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))

summary_file = os.path.join(results_dir, 'summary_results_improved.csv')
summary_df.to_csv(summary_file, index=False)
print("\n[OK] Summary saved to: " + summary_file)
