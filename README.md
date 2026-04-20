# Machine Learning Based Classification of Cyber Security Threats in Network Traffic

## Overview
This project applies machine learning and deep learning techniques to classify network traffic attacks using multiple datasets: CIC-IDS-2017, DDOS, NSL-KDD, and UNSW-NB15.

## Datasets

**Download Original Datasets:**

### CIC-IDS-2017
- **Source**: https://www.kaggle.com/datasets/cicdataset/cicids2017
- **Samples**: 1,132,001 with 9 attack classes
- **Size**: ~846 MB (merged)
- **Classes**: BENIGN, Bot, DDoS, DoS Hulk, DoS slowloris, DoS Slowhttptest, FTP-Patator, PortScan, SSH-Patator

### DDOS
- **Source**: https://www.kaggle.com/datasets/mrmoroj/ddos-attacks
- **Samples**: 99,970 (single class - DDOS attacks)
- **Size**: ~7.2 GB (merged)
- **Note**: Large dataset, may require high memory for processing

### UNSW-NB15
- **Source**: https://www.kaggle.com/datasets/mrmoroj/unsw-nb15
- **Samples**: 66,957 with 2 classes (Normal vs Attack)
- **Size**: ~39 MB (merged)

### NSL-KDD
- **Source**: https://www.kaggle.com/datasets/hassan06/nslkdd
- **Status**: Excluded from training due to malformed CSV formatting
- **Note**: Data has structural issues - columns not properly delimited

**Data Processing:**
- After downloading, place CSV files in respective folders:
  - `CIC-IDS- 2017/` 
  - `DDOS/`
  - `NSL_KDD/`
  - `UNSW/`
- Run `merge_datasets.py` to combine files
- Run `clean_fast.py` to clean and preprocess
- Run `normalize.py` to apply Min-Max scaling

## Project Pipeline

### 1. Data Merging (`merge_datasets.py`)
- Combines CSV files from each dataset folder using efficient file I/O
- Ultra-fast processing with 1MB buffering
- Output: 4 merged CSV files in `merged_datasets/` folder

### 2. Data Cleaning (`clean_fast.py`)
- Removes duplicates and missing values
- Handles infinite values
- Removes constant columns
- Processes data in 50k row chunks for memory efficiency
- Output: Cleaned datasets in `cleaned_datasets/` folder

### 3. Data Normalization (`normalize.py`)
- Applies Min-Max scaling to [0,1] range
- Preserves label columns
- Handles both numeric and categorical features
- Output: Normalized datasets in `normalized_datasets/` folder

### 4. Model Training & Evaluation

#### Random Forest (`random_forest_v2.py`)
- **Algorithm**: RandomForestClassifier (100 estimators, max_depth=20)
- **Train/Test Split**: 80/20 stratified
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Output**: 
  - Per-dataset results with confusion matrices
  - Feature importance rankings
  - Classification reports

**Results:**
| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| CIC-IDS-2017 | 99.91% | 99.91% | 99.91% | 99.91% |
| DDOS | 100% | 100% | 100% | 100% |
| UNSW | 100% | 100% | 100% | 100% |

#### CNN (Convolutional Neural Network) (`cnn_v2.py`)
- **Architecture**: 
  - 3 Conv1D blocks with BatchNormalization and MaxPooling
  - Dense layers with Dropout (0.5, 0.3)
  - Adam optimizer (lr=0.001)
  - EarlyStopping (patience=10, max 50 epochs)
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Output**:
  - Trained models (.h5 files)
  - Per-dataset results with confusion matrices
  - Classification reports

**Results:**
| Dataset | Accuracy | Precision | Recall | F1-Score | Epochs |
|---------|----------|-----------|--------|----------|--------|
| CIC-IDS-2017 | 66.46% | 44.17% | 66.46% | 53.07% | 11 |
| DDOS | 100% | 100% | 100% | 100% | 50 |
| UNSW | 89.34% | 89.40% | 89.34% | 88.85% | 25 |

#### SVM (Support Vector Machine) (`svm_final.py`)
- **Algorithm**: SVC with RBF kernel (C=1.0, gamma='scale') and Linear kernel for multi-class
- **Train/Test Split**: 80/20 stratified
- **Preprocessing**: LabelEncoder for categorical features
- **Datasets**: 50k sample CIC-IDS, full UNSW, 50k sample DDOS+BENIGN combined
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Output**:
  - Per-dataset results with confusion matrices
  - Classification reports
  - Summary statistics

**Results:**
| Dataset | Type | Accuracy | Precision | Recall | F1-Score | Test Samples |
|---------|------|----------|-----------|--------|----------|--------------|
| CIC-IDS-2017 | Multi-class (sample) | 98.04% | 98.12% | 98.04% | 98.04% | 10,000 |
| DDOS | Binary* | 63.28% | 70.42% | 63.28% | 55.96% | 17,613 |
| UNSW | Binary | 95.53% | 95.77% | 95.53% | 95.41% | 13,392 |

*DDOS: Binary classification combining DDOS attacks with BENIGN traffic from CIC-IDS-2017 (only 3 common numeric features)

## Results Files

### Random Forest Results
- `rf_results/results_CIC-IDS-2017.txt` - Detailed metrics, confusion matrix, feature importance
- `rf_results/results_DDOS.txt` - DDOS classification results
- `rf_results/results_UNSW.txt` - UNSW binary classification results
- `rf_results/summary_results.csv` - Summary comparison of all datasets

### CNN Results
- `cnn_results/results_CIC-IDS-2017.txt` - CNN metrics and confusion matrix
- `cnn_results/results_DDOS.txt` - CNN DDOS results
- `cnn_results/results_UNSW.txt` - CNN UNSW results
- `cnn_results/cnn_model_*.h5` - Trained model files (binary format)
- `cnn_results/summary_results.csv` - CNN performance summary

### SVM Results
- `svm_results/results_CIC_IDS_2017_svm.txt` - SVM multi-class results (50k sample)
- `svm_results/results_UNSW_svm.txt` - SVM binary classification for UNSW
- `svm_results/results_DDOS_svm.txt` - SVM binary classification for DDOS (DDOS vs BENIGN)
- `svm_results/svm_summary_all.csv` - SVM performance summary

## Key Findings

### Model Performance Comparison
- **Random Forest**: Best overall performance on imbalanced multi-class (99.91% on CIC-IDS-2017)
- **SVM**: Excellent on balanced datasets (98.04% on CIC-IDS-2017 sample, 95.53% on UNSW)
- **CNN**: Good for single-class and binary classification (~89-100% on UNSW/DDOS)
- **Random Forest** outperforms on large imbalanced multi-class problems
- **SVM** with linear kernel works well for multi-class when data is balanced
- **SVM** with RBF kernel achieves 95%+ accuracy on binary classification (UNSW)
- **DDOS SVM**: Limited accuracy (63.28%) due to dataset incompatibility (only 3 matching features between DDOS and CIC-IDS BENIGN)
- All models handle UNSW binary classification well (>95% accuracy)

### Important Features (Random Forest)
**CIC-IDS-2017:**
- Bwd Packet Length Mean (7.76%)
- Bwd Packet Length Std (5.94%)
- Average Packet Size (4.79%)

**UNSW:**
- Attack Category (51.73%)
- TCP RTT (6.23%)
- Download Bytes (4.59%)

## Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn tensorflow keras
```

### Run Pipeline
```bash
# 1. Merge datasets
python merge_datasets.py

# 2. Clean data
python clean_fast.py

# 3. Normalize data
python normalize.py

# 4. Train models
python random_forest_v2.py  # Random Forest
python cnn_v2.py            # CNN
python svm_v2.py            # SVM
```

## Technologies Used
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Random Forest, SVM)
- **Deep Learning**: TensorFlow, Keras (CNN)
- **Metrics**: Scikit-learn metrics (accuracy, precision, recall, F1, confusion matrix)
- **Python Version**: 3.11+

## Notes
- Large datasets (8+ GB total) are not included in repository
- CSV files are excluded via .gitignore for storage efficiency
- Trained models (.h5) are also excluded due to size
- Data preprocessing uses chunk-based processing for memory efficiency
- CNN shows signs of class imbalance on CIC-IDS-2017; Random Forest handles better

## Author
ML-based Network Traffic Classification Project

## License
MIT License
