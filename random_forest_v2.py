"""
Script to train Random Forest model on 3 normalized datasets
Evaluates: Accuracy, F1 Score, Precision, and Recall
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

normalized_dir = 'd:/Project related/Datasets/normalized_datasets'
results_dir = 'd:/Project related/Datasets/rf_results'

import os
os.makedirs(results_dir, exist_ok=True)

# List of normalized datasets
datasets = {
    'CIC-IDS-2017': 'normalized_cleaned_cic_ids_merged.csv',
    'DDOS': 'normalized_cleaned_ddos_merged.csv',
    'UNSW': 'normalized_cleaned_unsw_merged.csv'
}

print("=" * 70)
print("RANDOM FOREST CLASSIFIER - TRAINING AND EVALUATION")
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
            class_names = np.array(sorted(np.unique(y_encoded))).astype(str)
        
        # Train-test split (80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print("\n[OK] Train set size: " + str(X_train.shape[0]))
        print("[OK] Test set size: " + str(X_test.shape[0]))
        
        # Train Random Forest
        print("\n[TRAINING] Training Random Forest Classifier...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        rf_model.fit(X_train, y_train)
        print("[OK] Model training completed")
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print("\n" + "=" * 70)
        print("PERFORMANCE METRICS")
        print("=" * 70)
        print("Accuracy:  {:.4f}".format(accuracy))
        print("Precision: {:.4f}".format(precision))
        print("Recall:    {:.4f}".format(recall))
        print("F1 Score:  {:.4f}".format(f1))
        
        # Per-class metrics
        print("\n" + "=" * 70)
        print("DETAILED CLASSIFICATION REPORT")
        print("=" * 70)
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Feature importance
        print("\n" + "=" * 70)
        print("TOP 10 IMPORTANT FEATURES")
        print("=" * 70)
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(feature_importance.head(10).to_string(index=False))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n" + "=" * 70)
        print("CONFUSION MATRIX")
        print("=" * 70)
        
        # Create formatted confusion matrix with class names
        if class_names is not None:
            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            print(cm_df.to_string())
        else:
            print(cm)
        
        # Store results
        results_summary.append({
            'Dataset': dataset_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Train_Samples': len(X_train),
            'Test_Samples': len(X_test)
        })
        
        # Save model details to file
        results_file = os.path.join(results_dir, 'results_' + dataset_name + '.txt')
        with open(results_file, 'w') as f:
            f.write("RANDOM FOREST RESULTS - " + dataset_name + "\n")
            f.write("=" * 70 + "\n\n")
            f.write("Dataset Shape: " + str(df.shape) + "\n")
            f.write("Train Samples: " + str(len(X_train)) + "\n")
            f.write("Test Samples: " + str(len(X_test)) + "\n\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 70 + "\n")
            f.write("Accuracy:  {:.4f}\n".format(accuracy))
            f.write("Precision: {:.4f}\n".format(precision))
            f.write("Recall:    {:.4f}\n".format(recall))
            f.write("F1 Score:  {:.4f}\n\n".format(f1))
            f.write("CLASSIFICATION REPORT\n")
            f.write("-" * 70 + "\n")
            f.write(classification_report(y_test, y_pred, zero_division=0))
            f.write("\n\nCONFUSION MATRIX\n")
            f.write("-" * 70 + "\n")
            if class_names is not None:
                cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
                f.write(cm_df.to_string())
            else:
                f.write(str(cm))
            f.write("\n\n\nTOP 10 IMPORTANT FEATURES\n")
            f.write("-" * 70 + "\n")
            f.write(feature_importance.head(10).to_string(index=False))
        
        print("\n[OK] Results saved to " + results_file)
        
    except Exception as e:
        print("\n[ERROR] Error processing " + dataset_name + ": " + str(e))
        import traceback
        traceback.print_exc()

# Summary table
print("\n\n" + "=" * 70)
print("SUMMARY - ALL DATASETS")
print("=" * 70)
if results_summary:
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_file = os.path.join(results_dir, 'summary_results.csv')
    summary_df.to_csv(summary_file, index=False)
    print("\n[OK] Summary saved to " + summary_file)

print("\n[OK] Random Forest training completed!")
