import pandas as pd

# Load the original dataset
data = pd.read_csv("data.csv")

# Replace missing values in 'message start' with 0 (random noise)
data["message start"] = data["message start"].fillna(0)

# Save the updated dataset
updated_file = "updated_data.csv"
data.to_csv(updated_file, index=False)

print(f"Updated dataset saved to {updated_file}")