import pandas as pd
import os

# Path to your CSV files
csv_folder_path = "output_folder/"

# Output list to store reformatted times from all files
formatted_times = []

# Loop through all CSV files in the folder
for filename in os.listdir(csv_folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_folder_path, filename)

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Ensure 'start' and 'end' columns exist
        if 'start' in df.columns and 'end' in df.columns:
            # Format the times
            formatted = [f'("{start}","{end}"),' for start, end in zip(df['start'], df['end'])]
            formatted_times.extend(formatted)
        else:
            print(f"Skipping {filename}: 'start' or 'end' column missing.")

# Print or save the results
for entry in formatted_times:
    print(entry)
