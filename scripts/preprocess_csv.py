import pandas as pd
import os


def preprocess_csv(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Replace "null" strings with empty strings
    df = df.replace('null', '')

    # Save the processed file
    df.to_csv(output_file, index=False)
    print(f"Processed {input_file} -> {output_file}")


def main():
    # Create processed directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)

    # Process each CSV file
    files = [
        ('data/movie_data.csv', 'data/processed/movies.csv'),
        ('data/ratings_export.csv', 'data/processed/ratings.csv'),
        ('data/users_export.csv', 'data/processed/users.csv')
    ]

    for input_file, output_file in files:
        preprocess_csv(input_file, output_file)


if __name__ == '__main__':
    main()
