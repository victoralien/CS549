import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv("filtered_data.csv")

# Encode the target variable (message_start) as categorical labels
label_encoder = LabelEncoder()
data["message start"] = label_encoder.fit_transform(data["message start"])
num_classes = len(label_encoder.classes_)

features_to_drop = ["message start"]
X = data.drop(columns=features_to_drop)
y = data["message start"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovo')  # 'ovr' for one-vs-rest multi-class
svm_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test_scaled)

# Print evaluation metrics
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
target_names = [str(label) for label in label_encoder.classes_]
print(classification_report(y_test, y_pred, target_names=target_names))

