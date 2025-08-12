# Import dependencies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
data_frame = pd.read_csv("data.csv")

# Drop rows with missing values
data_frame.dropna(inplace=True)

# Encode categorical features: sample_type, RACE, SMOKING
label_encoders = {}

for column in ['sample_type', 'RACE', 'SMOKING']:
    le = LabelEncoder()
    data_frame[column] = le.fit_transform(data_frame[column])
    label_encoders[column] = le  # Save encoder for later decoding

# Split features and target
X = data_frame.drop(columns='sample_type')
Y = data_frame['sample_type']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Evaluate the model
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# Accuracy
train_accuracy = accuracy_score(Y_train, Y_train_pred)
test_accuracy = accuracy_score(Y_test, Y_test_pred)

print(f"✅ Training Accuracy: {train_accuracy:.2f}")
print(f"✅ Test Accuracy: {test_accuracy:.2f}")

# Detailed evaluation
print("\n Classification Report (Test Data):")
print(classification_report(Y_test, Y_test_pred, target_names=label_encoders['sample_type'].classes_))

# Confusion matrix
cm = confusion_matrix(Y_test, Y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoders['sample_type'].classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Optional: Feature importance from model coefficients
feature_importance = pd.Series(model.coef_[0], index=X.columns)
feature_importance = feature_importance.abs().sort_values(ascending=False)

print("\n📊 Feature Importance (by absolute coefficient values):")
print(feature_importance)

# -------------------------------
# Prediction System
# -------------------------------
# Example input (replace values to test others)
input_data = ('White', 'Never', 0.7, 0.6, 1.2, 2010, 53, 90, 7, 1)

# Convert to DataFrame and encode categorical values
input_df = pd.DataFrame([input_data], columns=['RACE', 'SMOKING', 'intermediate_dimension',
                                               'shortest_dimension', 'longest_dimension',
                                               'year_of_diagnosis', 'Age', 'PSA',
                                               'Clinical_Gleason_sum', 'Residual_tumor'])

for column in ['RACE', 'SMOKING']:
    input_df[column] = label_encoders[column].transform(input_df[column])

# Predict
prediction = model.predict(input_df)[0]
predicted_label = label_encoders['sample_type'].inverse_transform([prediction])[0]

# Output
print("\n Prediction Result:")
if predicted_label == 'Primary Tumor':
    print("⚠️ The Prostate cancer is at HIGH risk.")
else:
    print("✅ The Prostate cancer is at LOW risk.")
