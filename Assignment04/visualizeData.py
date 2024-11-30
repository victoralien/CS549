import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("updated_data.csv")

# Create `is_noise` column (1 for noise, 0 otherwise)
data["is_noise"] = data["message start"].isna().astype(int)

# Fill missing values for visualization purposes
data_filled = data.fillna(0)

# Basic stats
print("Basic Statistics:")
print(data.describe())

# 1. Plot the distribution of the `message start` column (valid vs noise)
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x="message start", bins=50, kde=True, color="blue")
plt.title("Distribution of `message start`")
plt.xlabel("Message Start")
plt.ylabel("Frequency")
plt.show()

# 2. Countplot for noise vs valid samples
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x="is_noise", palette="coolwarm")
plt.title("Count of Noise vs Valid Samples")
plt.xlabel("Is Noise (1 = Noise, 0 = Valid)")
plt.ylabel("Count")
plt.xticks([0, 1], labels=["Valid", "Noise"])
plt.show()

# 3. Correlation heatmap (numerical features)
plt.figure(figsize=(12, 8))
numerical_features = data_filled.select_dtypes(include=["float64", "int64"])
correlation_matrix = numerical_features.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# 4. Pairplot of features (valid vs noise)
plt.figure(figsize=(15, 12))
sns.pairplot(data, hue="is_noise", palette="Set1", diag_kind="kde", corner=True)
plt.suptitle("Pairplot of Features (Colored by Noise)", y=1.02)
plt.show()

# 5. Boxplot of features grouped by noise
plt.figure(figsize=(12, 8))
sns.boxplot(data=data_filled, x="is_noise", y="message start", palette="Set2")
plt.title("Boxplot of `message start` by Noise Classification")
plt.xlabel("Is Noise")
plt.ylabel("Message Start")
plt.xticks([0, 1], labels=["Valid", "Noise"])
plt.show()

# 6. Scatterplot of two chosen features (adjust as needed)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x=data.columns[0], y=data.columns[1], hue="is_noise", palette="coolwarm")
plt.title("Scatterplot of Feature 1 vs Feature 2")
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])
plt.legend(title="Is Noise")
plt.show()
