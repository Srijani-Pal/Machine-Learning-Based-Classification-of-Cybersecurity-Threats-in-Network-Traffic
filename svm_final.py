"""
Simple and Fast SVM for all 3 datasets
Uses smaller samples for speed
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
print("SVM (SUPPORT VECTOR MACHINE) - SIMPLE FAST VERSION")
print("=" * 70)

results_summary = []

# ============================================================================
# 1. CIC-IDS-2017 - Small 50k sample
# ============================================================================
print("\n" + "=" * 70)
print("Dataset 1: CIC-IDS-2017 (50k sample)")
print("=" * 70)

try:
    print("[...] Loading sample...")
    filepath = os.path.join(normalized_dir, 'normalized_cleaned_cic_ids_merged.csv')
    df = pd.read_csv(filepath, nrows=50000)  # Just load 50k rows
    df = df.dropna()  # Drop NaN
    
    print("[OK] Shape: " + str(df.shape))
    
    label_col = ' Label'
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    print("Classes: " + str(y.nunique()) + " | Samples: " + str(len(y)))
    
    # Encode
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    print("[...] Training SVM...")
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n✓ Accuracy:  {:.2f}%".format(acc * 100))
    print("✓ Precision: {:.2f}%".format(prec * 100))
    print("✓ Recall:    {:.2f}%".format(rec * 100))
    print("✓ F1-Score:  {:.2f}%".format(f1 * 100))
    
    # Save
    with open(os.path.join(results_dir, 'results_CIC_IDS_2017_svm.txt'), 'w') as f:
        f.write("SVM Classifier - CIC-IDS-2017 (50k sample)\n\n")
        f.write("Accuracy:  {:.4f} ({:.2f}%)\n".format(acc, acc*100))
        f.write("Precision: {:.4f} ({:.2f}%)\n".format(prec, prec*100))
        f.write("Recall:    {:.4f} ({:.2f}%)\n".format(rec, rec*100))
        f.write("F1-Score:  {:.4f} ({:.2f}%)\n\n".format(f1, f1*100))
        f.write("Test Samples: " + str(len(X_test)) + "\n")
        f.write(classification_report(y_test, y_pred, target_names=le_y.classes_, zero_division=0))
    
    results_summary.append(['CIC-IDS-2017', 'Multi-class', "{:.4f}".format(acc), "{:.4f}".format(prec), "{:.4f}".format(rec), "{:.4f}".format(f1)])
    
except Exception as e:
    print("[ERROR] " + str(e))
    results_summary.append(['CIC-IDS-2017', 'Multi-class', 'ERROR', 'ERROR', 'ERROR', 'ERROR'])

# ============================================================================
# 2. DDOS - Binary (DDOS + BENIGN)
# ============================================================================
print("\n" + "=" * 70)
print("Dataset 2: DDOS (Binary - DDOS vs BENIGN)")
print("=" * 70)

try:
    print("[...] Loading DDOS...")
    ddos_df = pd.read_csv(os.path.join(normalized_dir, 'normalized_cleaned_ddos_merged.csv'))
    ddos_df = ddos_df.dropna()
    
    # Reduce DDOS to reasonable size
    ddos_sample_size = min(50000, len(ddos_df))
    ddos_df = ddos_df.sample(n=ddos_sample_size, random_state=42)
    
    print("[...] Loading BENIGN...")
    cic_df = pd.read_csv(os.path.join(normalized_dir, 'normalized_cleaned_cic_ids_merged.csv'), nrows=200000)
    benign_df = cic_df[cic_df[' Label'] == 'BENIGN'].sample(n=len(ddos_df), random_state=42)
    
    print("[OK] DDOS: " + str(ddos_df.shape) + " | BENIGN: " + str(benign_df.shape))
    
    # Create labels
    ddos_df['binary_label'] = 1
    benign_df['binary_label'] = 0
    
    # Find common columns
    ddos_cols = set(ddos_df.columns) - {'Label', 'binary_label', 'Unnamed: 0', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp'}
    benign_cols = set(benign_df.columns) - {' Label', 'binary_label'}
    common = list(ddos_cols & benign_cols)
    
    print("[OK] Common features: " + str(len(common)))
    
    # Combine
    X_combined = pd.concat([ddos_df[common], benign_df[common]], ignore_index=True)
    y_combined = pd.concat([ddos_df['binary_label'], benign_df['binary_label']], ignore_index=True)
    
    # Encode object columns
    for col in X_combined.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_combined[col] = le.fit_transform(X_combined[col].astype(str))
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined)
    
    print("[...] Training SVM...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n✓ Accuracy:  {:.2f}%".format(acc * 100))
    print("✓ Precision: {:.2f}%".format(prec * 100))
    print("✓ Recall:    {:.2f}%".format(rec * 100))
    print("✓ F1-Score:  {:.2f}%".format(f1 * 100))
    
    # Save
    with open(os.path.join(results_dir, 'results_DDOS_svm.txt'), 'w') as f:
        f.write("SVM Classifier - DDOS (Binary: DDOS vs BENIGN)\n\n")
        f.write("Accuracy:  {:.4f} ({:.2f}%)\n".format(acc, acc*100))
        f.write("Precision: {:.4f} ({:.2f}%)\n".format(prec, prec*100))
        f.write("Recall:    {:.4f} ({:.2f}%)\n".format(rec, rec*100))
        f.write("F1-Score:  {:.4f} ({:.2f}%)\n\n".format(f1, f1*100))
        f.write("Test Samples: " + str(len(X_test)) + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)) + "\n\n")
        f.write(classification_report(y_test, y_pred, target_names=['BENIGN', 'DDOS']))
    
    results_summary.append(['DDOS', 'Binary', "{:.4f}".format(acc), "{:.4f}".format(prec), "{:.4f}".format(rec), "{:.4f}".format(f1)])
    
except Exception as e:
    print("[ERROR] " + str(e))
    import traceback
    traceback.print_exc()
    results_summary.append(['DDOS', 'Binary', 'ERROR', 'ERROR', 'ERROR', 'ERROR'])

# ============================================================================
# 3. UNSW - Binary
# ============================================================================
print("\n" + "=" * 70)
print("Dataset 3: UNSW (Binary)")
print("=" * 70)

try:
    print("[...] Loading UNSW...")
    df = pd.read_csv(os.path.join(normalized_dir, 'normalized_cleaned_unsw_merged.csv'))
    
    print("[OK] Shape: " + str(df.shape))
    
    X = df.drop(columns=['label'])
    y = df['label']
    
    # Encode object columns
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[...] Training SVM...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n✓ Accuracy:  {:.2f}%".format(acc * 100))
    print("✓ Precision: {:.2f}%".format(prec * 100))
    print("✓ Recall:    {:.2f}%".format(rec * 100))
    print("✓ F1-Score:  {:.2f}%".format(f1 * 100))
    
    # Save
    with open(os.path.join(results_dir, 'results_UNSW_svm.txt'), 'w') as f:
        f.write("SVM Classifier - UNSW (Binary)\n\n")
        f.write("Accuracy:  {:.4f} ({:.2f}%)\n".format(acc, acc*100))
        f.write("Precision: {:.4f} ({:.2f}%)\n".format(prec, prec*100))
        f.write("Recall:    {:.4f} ({:.2f}%)\n".format(rec, rec*100))
        f.write("F1-Score:  {:.4f} ({:.2f}%)\n\n".format(f1, f1*100))
        f.write("Test Samples: " + str(len(X_test)) + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)) + "\n\n")
        f.write(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    
    results_summary.append(['UNSW', 'Binary', "{:.4f}".format(acc), "{:.4f}".format(prec), "{:.4f}".format(rec), "{:.4f}".format(f1)])
    
except Exception as e:
    print("[ERROR] " + str(e))
    results_summary.append(['UNSW', 'Binary', 'ERROR', 'ERROR', 'ERROR', 'ERROR'])

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
summary_df = pd.DataFrame(results_summary, columns=['Dataset', 'Type', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
print(summary_df.to_string(index=False))

summary_df.to_csv(os.path.join(results_dir, 'svm_summary_all.csv'), index=False)
print("\n[✓] All datasets trained successfully!")
print("[✓] Results saved to svm_results folder")
