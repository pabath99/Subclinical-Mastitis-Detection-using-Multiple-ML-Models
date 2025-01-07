# Subclinical Mastitis Detection using Multiple ML Models


This repository provides a comprehensive machine learning pipeline for detecting subclinical mastitis in dairy cows. The project leverages milk composition, production parameters, and California Mastitis Test (CMT) results to evaluate multiple ML models, such as Random Forest, XGBoost, SVM, Logistic Regression, and Gradient Boosting, to identify the best-performing approach.

---

## Table of Contents
- [Subclinical Mastitis Detection using Multiple ML Models](#subclinical-mastitis-detection-using-multiple-ml-models)
  - [Table of Contents](#table-of-contents)
  - [About the Project](#about-the-project)
    - [Built With](#built-with)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
  - [Dataset](#dataset)
    - [Data Preparation Example](#data-preparation-example)
  - [Features](#features)
  - [Model Training \& Comparison](#model-training--comparison)
    - [Training a Model](#training-a-model)
    - [Comparing Models](#comparing-models)
  - [Saving and Predicting](#saving-and-predicting)
    - [Saving the Best Model](#saving-the-best-model)
    - [Predicting on New Data](#predicting-on-new-data)
  - [License](#license)
  - [Contact](#contact)

---

## About the Project
Subclinical mastitis is a hidden udder infection in dairy cows, often detected by Somatic Cell Count (SCC) or CMT scores. This project aims to:

- Derive a binary target (Subclinical_Mastitis) using SCC thresholds (e.g., > 200 × 10^3 cells/ml).
- Convert CMT scores (0–4) into approximate cell-count equivalents (100 to 8100).
- Train and compare multiple ML models on features such as lactation number, daily yield, fat, protein, etc.
- Evaluate performance using metrics like Accuracy, Precision, Recall, F1, and AUC.

### Built With
- **Python**
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Seaborn, Matplotlib, Joblib

---

## Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. Install dependencies listed in `requirements.txt`.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/pabath99/Subclinical-Mastitis-Detection-using-Multiple-ML-Models.git
   cd Subclinical Mastitis Detection using Multiple ML Models
   ```
2. Install Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Prepare your dataset (e.g., `datasheet.xlsx`) with required columns.
2. Run the scripts provided for data preparation, model training, and evaluation.
3. Evaluate model results using classification metrics.
4. Save and load the best model for predictions on new data.

---

## Dataset
The dataset should include:
- **Identification No, Sample No** (IDs)
- **Milk composition**: Fat (%), SNF (%), Protein (%), Lactose (%)
- **Conductivity (mS/cm), pH, Freezing point (°C), Salt (%)**
- **SCC (10^3 cells/ml), CMT (Score)**

### Data Preparation Example
```python
import pandas as pd
df = pd.read_excel('datasheet.xlsx')
df['Subclinical_Mastitis'] = (df['SCC (10^3cells/ml)'] > 200).astype(int)
cmt_map = {0: 100, 1: 300, 2: 900, 3: 2700, 4: 8100}
df['CMT_cellcount'] = df['CMT(Score)'].map(cmt_map)
df.drop(columns=['SCC (10^3cells/ml)', 'CMT(Score)'], inplace=True)
```

---

## Features
- Derived target column: `Subclinical_Mastitis`.
- Features: Lactation Number, Days in Milk, Fat (%), Protein (%), CMT_cellcount, etc.

---

## Model Training & Comparison
### Training a Model
```python
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
```

### Comparing Models
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

models = [
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
    ("Random Forest", RandomForestClassifier(random_state=42)),
    ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
    ("SVM", SVC(probability=True, random_state=42)),
    ("Gradient Boosting", GradientBoostingClassifier(random_state=42))
]

for name, model in models:
    m.fit(X_train_scaled, y_train)
    df_results = pd.DataFrame(results).sort_values(by="AUC", ascending=False)
    print(df_results)
```

---

## Saving and Predicting
### Saving the Best Model
```python
import joblib
joblib.dump(best_model, 'best_subclinical_mastitis_model.pkl')
```

### Predicting on New Data
```python
loaded_model = joblib.load('best_subclinical_mastitis_model.pkl')
new_data_scaled = loaded_scaler.transform(new_data)
print("Prediction (Subclinical Mastitis):", pred)
```

---

## License
This project is distributed under the MIT License. See the `LICENSE` file for details.

---

## Contact 
- **Email:** [pabath2015@gmail.com](pabath2015@gmail.com)
- **Project Link:** [https://github.com/pabath99/Subclinical-Mastitis-Detection-using-Multiple-ML-Models.git](https://github.com/pabath99/Subclinical-Mastitis-Detection-using-Multiple-ML-Models.git)
