"""
Script to train CNN model on 3 normalized datasets
Uses Keras/TensorFlow for deep learning
Evaluates: Accuracy, F1 Score, Precision, and Recall
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

import os
normalized_dir = 'd:/Project related/Datasets/normalized_datasets'
results_dir = 'd:/Project related/Datasets/cnn_results'
os.makedirs(results_dir, exist_ok=True)

# List of normalized datasets
datasets = {
    'CIC-IDS-2017': 'normalized_cleaned_cic_ids_merged.csv',
    'DDOS': 'normalized_cleaned_ddos_merged.csv',
    'UNSW': 'normalized_cleaned_unsw_merged.csv'
}

print("=" * 70)
print("CNN (CONVOLUTIONAL NEURAL NETWORK) - TRAINING AND EVALUATION")
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
            num_classes = len(le_y.classes_)
            class_names = le_y.classes_
        else:
            y_encoded = y
            num_classes = len(np.unique(y))
            class_names = np.array(sorted(np.unique(y_encoded))).astype(str)
        
        # Convert labels to one-hot encoding for multi-class
        if num_classes > 2:
            y_encoded_onehot = keras.utils.to_categorical(y_encoded, num_classes=num_classes)
        else:
            y_encoded_onehot = y_encoded
        
        # Train-test split (80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y_encoded_onehot, test_size=0.2, random_state=42
        )
        
        print("\n[OK] Train set size: " + str(X_train.shape[0]))
        print("[OK] Test set size: " + str(X_test.shape[0]))
        
        # Reshape data for CNN (add channel dimension)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        print("[OK] Reshaped for CNN: Train " + str(X_train.shape) + ", Test " + str(X_test.shape))
        
        # Build CNN model
        print("\n[BUILDING] Building CNN model...")
        model = Sequential([
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            
            layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            
            layers.Conv1D(filters=256, kernel_size=3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
        ])
        
        # Add output layer based on number of classes
        if num_classes > 2:
            model.add(layers.Dense(num_classes, activation='softmax'))
            loss_fn = 'categorical_crossentropy'
        else:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss_fn = 'binary_crossentropy'
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=loss_fn,
            metrics=['accuracy']
        )
        
        print("[OK] Model compiled")
        
        # Train model with early stopping
        print("\n[TRAINING] Training CNN model...")
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )
        
        print("[OK] Model training completed (" + str(len(history.history['loss'])) + " epochs)")
        
        # Make predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        
        if num_classes > 2:
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_test_labels = np.argmax(y_test, axis=1)
        else:
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            y_test_labels = y_test
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_labels, y_pred)
        f1 = f1_score(y_test_labels, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_test_labels, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test_labels, y_pred, average='weighted', zero_division=0)
        
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
        print(classification_report(y_test_labels, y_pred, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_test_labels, y_pred)
        print("\n" + "=" * 70)
        print("CONFUSION MATRIX")
        print("=" * 70)
        
        # Create formatted confusion matrix with class names
        if class_names is not None:
            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            print(cm_df.to_string())
            cm_formatted = cm_df.to_string()
        else:
            print(cm)
            cm_formatted = str(cm)
        
        # Store results
        results_summary.append({
            'Dataset': dataset_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Train_Samples': len(X_train),
            'Test_Samples': len(X_test),
            'Epochs': len(history.history['loss'])
        })
        
        # Save model results to file
        results_file = os.path.join(results_dir, 'results_' + dataset_name + '.txt')
        with open(results_file, 'w') as f:
            f.write("CNN RESULTS - " + dataset_name + "\n")
            f.write("=" * 70 + "\n\n")
            f.write("Dataset Shape: " + str(df.shape) + "\n")
            f.write("Number of Classes: " + str(num_classes) + "\n")
            f.write("Train Samples: " + str(len(X_train)) + "\n")
            f.write("Test Samples: " + str(len(X_test)) + "\n")
            f.write("Training Epochs: " + str(len(history.history['loss'])) + "\n\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 70 + "\n")
            f.write("Accuracy:  {:.4f}\n".format(accuracy))
            f.write("Precision: {:.4f}\n".format(precision))
            f.write("Recall:    {:.4f}\n".format(recall))
            f.write("F1 Score:  {:.4f}\n\n".format(f1))
            f.write("CLASSIFICATION REPORT\n")
            f.write("-" * 70 + "\n")
            f.write(classification_report(y_test_labels, y_pred, zero_division=0))
            f.write("\n\nCONFUSION MATRIX\n")
            f.write("-" * 70 + "\n")
            f.write(cm_formatted)
        
        # Save model
        model_file = os.path.join(results_dir, 'cnn_model_' + dataset_name + '.h5')
        model.save(model_file)
        
        print("\n[OK] Results saved to " + results_file)
        print("[OK] Model saved to " + model_file)
        
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

print("\n[OK] CNN training completed!")
