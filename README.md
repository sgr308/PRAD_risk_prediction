# PRAD_risk_prediction
# **Prostate Cancer Risk Prediction**

This project is a machine learning application built to predict prostate cancer risk based on specific input features using Logistic Regression. 
The dataset used includes sample types and additional features for training and evaluating the model's performance.

---

## **Table of Contents**

1. [Installation](#installation)
2. [Usage](#usage)
---

## **Installation**

To run this project locally, ensure you have Python installed along with the necessary libraries:

### **Required Libraries:**
- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `scikit-learn`

### **Installation Steps:**

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
    pip install numpy pandas seaborn matplotlib scikit-learn
    ```

---

## **Usage**

1. **Ensure Dataset Availability**:
   Place the dataset file (`data.csv`) in the same directory as the script.

2. **Run the Script**:
   Execute the script using your Python interpreter:
    ```bash
    python prostate_cancer_prediction.py
    ```

3. **Test the Predictive System**:
   Modify the `input_data` section in the script with new patient data and run the corresponding section to obtain predictions.

**Example:**
```python
input_data = (1.2, 0.5, 1.7, 2008, 1, 0, 61, 0, 0, 0, 0, 0)

Output:
The Prostate Cancer is at low risk

