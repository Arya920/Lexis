import pandas as pd
import os

# File path (update if needed)
input_file = "datatset_consumer_complaints.csv"

# Read CSV
df = pd.read_csv(input_file)

# Total rows before filtering
total_rows = len(df)
print(f"Total rows in original dataset: {total_rows}")

# Filter rows where Product == "Mortgage"
filtered_df = df[df["Product"] == "Mortgage"]

# Total rows after filtering
filtered_rows = len(filtered_df)
print(f"Total rows after filtering (Product == 'Mortgage'): {filtered_rows}")

# Output file path (same directory)
output_file = os.path.join(os.path.dirname(input_file), "Mortgage_Consumer_Complaints.csv")

# Save filtered data
filtered_df.to_csv(output_file, index=False)

print(f"Filtered data saved to: {output_file}")