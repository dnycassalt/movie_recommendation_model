import pandas as pd
import os

file_path = 'data/movie_data.csv'
print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")

# Try to read just the first few rows with different encodings
encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']

for encoding in encodings:
    try:
        print(f"\nTrying encoding: {encoding}")
        df = pd.read_csv(file_path, encoding=encoding, nrows=5)
        print("Success!")
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nFirst few rows:")
        print(df)
        break
    except Exception as e:
        print(f"Failed with error: {str(e)}")
