# 🧬 Prostate Cancer Classification Tool

This project uses logistic regression to predict the **risk level** of prostate cancer, classifying tumors as either **Primary Tumor (High Risk)** or **Metastatic Tumor (Low Risk)** based on patient data.

---

## 🔍 Project Overview

Prostate cancer is a serious health concern for men worldwide. 
Using machine learning, this tool aims to assist in the early classification of prostate tumors based on clinical and demographic data.

We use logistic regression to analyze patient features and predict whether the cancer is likely a *Primary Tumor* or a *Metastatic Tumor*.

---

## 📊 Dataset

- **File used**: `data.csv`
- The dataset includes:
  - Demographic data: `RACE`, `SMOKING`, `Age`
  - Clinical features: `intermediate_dimension`, `shortest_dimension`, `longest_dimension`, `PSA`, `Clinical_Gleason_sum`, `Residual_tumor`
  - Diagnosis metadata: `year_of_diagnosis`
  - **Target**: `sample_type` (`Primary Tumor` or `Metastatic`)

> Note: All missing values are removed during preprocessing.

---

## ⚙️ How It Works

1. **Data Preprocessing**
   - Missing values dropped.
   - Label encoding applied to categorical variables (`RACE`, `SMOKING`, `sample_type`).

2. **Model Training**
   - Logistic Regression model trained using `sklearn`.

3. **Evaluation**
   - Accuracy scores
   - Classification report
   - Confusion matrix
   - Feature importance by model coefficients

4. **Prediction Tool**
   - Custom input data is provided to test the model on real-world examples.
   - Outputs the predicted cancer risk level.

---
## 📚 Table of Contents

1. [Installation](#installation)  
2. [Usage](#usage)  

---

## Installation

To run this project locally, ensure you have Python installed along with the necessary libraries:

### Required Libraries:
- `numpy`  
- `pandas`  
- `seaborn`  
- `matplotlib`  
- `scikit-learn`  

### Installation Steps:

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/sgr308/PRAD_risk_prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd PRAD_risk_prediction
    ```
3. Install the required dependencies:
    ```bash
    pip install numpy pandas seaborn scikit-learn matplotlib
    ```
---

## Usage

1. **Ensure Dataset Availability**  
   Place the dataset file (`data.csv`) in the same directory as the script.

2. **Run the Script**  
   Execute the script using your Python interpreter:
    ```bash
    python PRAD_PRED.py
    ```

3. **Test the Predictive System**  
   Modify the `input_data` section in the script with new patient data and run the corresponding section to obtain predictions.

## 🧪 Sample Prediction Examples

### 🔢 Data Input for Example for LOW risk patient:
```python
input_data = ('Asian', 'Never', 1, 0.2, 1.7, 2008, 50, 1, 1, 1)

Output:
✅ Training Accuracy: 95.70%
✅ Test Accuracy: 94.59%

 Classification Report (Test Data):
                     precision    recall  f1-score   support

      Primary Tumor       0.96      0.98      0.97        99
Solid Tissue Normal       0.80      0.67      0.73        12

           accuracy                           0.95       111
          macro avg       0.88      0.82      0.85       111
       weighted avg       0.94      0.95      0.94       111

Feature Importance (by absolute coefficient values):
longest_dimension         2.409357
shortest_dimension        1.407170
intermediate_dimension    1.040950
Clinical_Gleason_sum      0.978637
Residual_tumor            0.347743
PSA                       0.305350
RACE                      0.192756
SMOKING                   0.137552
year_of_diagnosis         0.080759
Age                       0.021562
dtype: float64

Prediction Result:
The Prostate cancer is at LOW risk.
```
---

### 🔢 Data Input for Example for HIGH risk patient:
```python
input_data = ('White', 'Never', 0.7, 0.6, 1.2, 2010, 53, 90, 7, 1)

Output:
✅ Training Accuracy: 95.70%
✅ Test Accuracy: 94.59%

 Classification Report (Test Data):
                     precision    recall  f1-score   support

      Primary Tumor       0.96      0.98      0.97        99
Solid Tissue Normal       0.80      0.67      0.73        12

           accuracy                           0.95       111
          macro avg       0.88      0.82      0.85       111
       weighted avg       0.94      0.95      0.94       111


Feature Importance (by absolute coefficient values):
longest_dimension         2.409357
shortest_dimension        1.407170
intermediate_dimension    1.040950
Clinical_Gleason_sum      0.978637
Residual_tumor            0.347743
PSA                       0.305350
RACE                      0.192756
SMOKING                   0.137552
year_of_diagnosis         0.080759
Age                       0.021562
dtype: float64

Prediction Result:
The Prostate cancer is at HIGH risk.
```
---
