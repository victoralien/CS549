import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, classification_report

# Load the dataset
data = pd.read_csv("filtered_data.csv")

# Prepare features (X) and target (y)
X = data.drop(columns=["Freq/Channel","message start"])
y = data["message start"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create polynomial features
poly_degree = 6 # Change this to experiment with degrees of the polynomial
poly = PolynomialFeatures(degree=poly_degree)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict the message start
y_pred = model.predict(X_test_poly)

# Map predictions to the closest known start times
unique_message_starts = np.unique(y_train)  # Known message start classes
y_pred_mapped = [min(unique_message_starts, key=lambda x: abs(x - pred)) for pred in y_pred]

# Evaluate the model
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred_mapped))

# Optionally print a confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(y_test, y_pred_mapped, labels=unique_message_starts)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_message_starts, yticklabels=unique_message_starts)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
