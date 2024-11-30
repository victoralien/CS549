import pandas as pd

# Load the original dataset
input_file = "data.csv"  # Replace with your file name
data = pd.read_csv(input_file)

# Drop rows where 'message start' is NaN or empty
filtered_data = data.dropna(subset=["message start"])  # Remove rows with NaN in 'message start'

# Save the filtered data to a new CSV file
output_file = "filtered_data.csv" 
filtered_data.to_csv(output_file, index=False)

print(f"Filtered data saved to {output_file}")