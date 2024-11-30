import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
data = pd.read_csv("filtered_data.csv")

# Replace missing 'message start' values with 0
data["message start"] = data["message start"].fillna(0)

# Encode the target variable (message start) as categorical labels
label_encoder = LabelEncoder()
data["message start"] = label_encoder.fit_transform(data["message start"])
num_classes = len(label_encoder.classes_)

# Split features and target
X = data.drop(columns=["Freq/Channel","CFAR Low","CFAR High","message start"])
y = data["message start"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = Sequential()

# Input layer
model.add(Dense(32, activation='relu', input_dim=X_train_scaled.shape[1]))
model.add(BatchNormalization())  # Add batch normalization for stability
#model.add(Dropout(0.3))          # Dropout to prevent overfitting

# Hidden layers
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.4))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.3))

# Output layer
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict on test data
y_pred_probs = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Classification metrics
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

print("\nClassification Report:")
target_names = [str(label) for label in label_encoder.classes_]
print(classification_report(y_test, y_pred_classes, target_names=target_names))

