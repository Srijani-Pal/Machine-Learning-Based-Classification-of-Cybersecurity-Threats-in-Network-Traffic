"""
Script to train SVM (Support Vector Machine) model on 3 normalized datasets
Evaluates: Accuracy, F1 Score, Precision, and Recall
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

normalized_dir = 'd:/Project related/Datasets/normalized_datasets'
results_dir = 'd:/Project related/Datasets/svm_results'

import os
os.makedirs(results_dir, exist_ok=True)

# List of normalized datasets
datasets = {
    'CIC-IDS-2017': 'normalized_cleaned_cic_ids_merged.csv',
    'DDOS': 'normalized_cleaned_ddos_merged.csv',
    'UNSW': 'normalized_cleaned_unsw_merged.csv'
}

print("=" * 70)
print("SVM (SUPPORT VECTOR MACHINE) - TRAINING AND EVALUATION")
print("=" * 70)

results_summary = []

for dataset_name, filename in datasets.items():
    print("\n" + "=" * 70)
    print("Dataset: " + dataset_name)
    print("=" * 70)
    
    filepath = os.path.join(normalized_dir, filename)
    
    try:
        # Read normalized dataset
        df = pd.read_csv(filepath)
        print("[OK] Loaded dataset with shape: " + str(df.shape))
        
        # Identify label column
        label_col = None
        label_candidates = ['Label', 'label', 'class', 'Class', 'Attack', 'attack', 'target']
        
        for col in label_candidates:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            label_col = df.columns[-1]
        
        # Separate features and labels
        X = df.drop(columns=[label_col])
        y = df[label_col]
        
        print("Features shape: " + str(X.shape))
        print("Label distribution:")
        print(y.value_counts())
        
        # Handle categorical features if any
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print("\n[OK] Encoding " + str(len(categorical_cols)) + " categorical columns")
            le_dict = {}
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
        
        # Encode target labels if categorical
        if y.dtype == 'object':
            le_y = LabelEncoder()
            y_encoded = le_y.fit_transform(y)
            class_names = le_y.classes_
        else:
            y_encoded = y
            class_names = np.unique(y)
        
        # Check if we have only one class
        unique_classes = np.unique(y_encoded)
        if len(unique_classes) < 2:
            print("\n[!] SKIPPED - Dataset has only one class (single-class problem)")
            print("[!] SVM requires at least 2 classes for classification")
            results_summary.append({
                'Dataset': dataset_name,
                'Accuracy': 'SKIPPED',
                'Precision': 'SKIPPED',
                'Recall': 'SKIPPED',
                'F1-Score': 'SKIPPED',
                'Test Samples': 'Single-class'
            })
            continue
        
        # Handle missing values
        print("\n[...] Handling missing values...")
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print("[OK] Train set size: " + str(X_train.shape[0]))
        print("[OK] Test set size: " + str(X_test.shape[0]))
        
        # Train SVM model with RBF kernel
        print("\n[...] Training SVM model (kernel=rbf, C=1.0)...")
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm_model.fit(X_train, y_train)
        print("[OK] Training completed")
        
        # Make predictions
        y_pred = svm_model.predict(X_test)
        
        # Calculate metrics
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
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Classification Report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=[str(c) for c in class_names]))
        
        # Save detailed results to file
        results_file = os.path.join(results_dir, "results_" + dataset_name.replace('-', '_') + ".txt")
        with open(results_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("SVM CLASSIFIER - " + dataset_name + "\n")
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
        
        # Add to summary
        results_summary.append({
            'Dataset': dataset_name,
            'Accuracy': "{:.4f}".format(accuracy),
            'Precision': "{:.4f}".format(precision),
            'Recall': "{:.4f}".format(recall),
            'F1-Score': "{:.4f}".format(f1),
            'Test Samples': X_test.shape[0]
        })
        
    except Exception as e:
        print("[ERROR] Failed to process " + dataset_name + ": " + str(e))
        results_summary.append({
            'Dataset': dataset_name,
            'Accuracy': 'ERROR',
            'Precision': 'ERROR',
            'Recall': 'ERROR',
            'F1-Score': 'ERROR',
            'Test Samples': 'N/A'
        })

# Save summary results
print("\n" + "=" * 70)
print("SUMMARY RESULTS")
print("=" * 70)
summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))

summary_file = os.path.join(results_dir, 'summary_results.csv')
summary_df.to_csv(summary_file, index=False)
print("\n[OK] Summary saved to: " + summary_file)
