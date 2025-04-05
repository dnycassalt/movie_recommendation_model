import pandas as pd
import ast


def safe_eval(x):
    """Safely evaluate string representation of a list."""
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []


def clean_movie_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the movie dataset by:
    1. Removing duplicates
    2. Handling missing values
    3. Converting data types
    4. Removing outliers
    5. Standardizing column names
    """
    print("\nStarting data cleaning...")

    # Create a copy to avoid modifying the original
    df_clean = df.copy()

    # Convert genres to string representation for duplicate checking
    df_clean['genres_str'] = df_clean['genres'].apply(str)

    # Remove duplicates based on all columns except genres
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates(
        subset=df_clean.columns.difference(['genres']))
    print(f"Removed {initial_rows - len(df_clean)} duplicate rows")

    # Handle missing values
    missing_counts = df_clean.isnull().sum()
    print("\nMissing values before cleaning:")
    print(missing_counts[missing_counts > 0])

    # Fill missing values with appropriate defaults
    df_clean['runtime'] = df_clean['runtime'].fillna(
        df_clean['runtime'].median())
    df_clean['vote_average'] = df_clean['vote_average'].fillna(0)
    df_clean['vote_count'] = df_clean['vote_count'].fillna(0)
    df_clean['overview'] = df_clean['overview'].fillna('No overview available')
    df_clean['genres'] = df_clean['genres'].fillna('[]')

    # Convert data types
    df_clean['runtime'] = pd.to_numeric(df_clean['runtime'], errors='coerce')
    df_clean['vote_average'] = pd.to_numeric(
        df_clean['vote_average'], errors='coerce')
    df_clean['vote_count'] = pd.to_numeric(
        df_clean['vote_count'], errors='coerce')

    # Remove outliers in runtime (movies longer than 4 hours or shorter than 1 min)
    df_clean = df_clean[
        (df_clean['runtime'] >= 1) &
        (df_clean['runtime'] <= 240)
    ]

    # Remove movies with no votes
    df_clean = df_clean[df_clean['vote_count'] > 0]

    # Convert genres back to lists if they were converted to strings
    df_clean['genres'] = df_clean['genres'].apply(safe_eval)

    # Remove the temporary genres_str column
    df_clean = df_clean.drop('genres_str', axis=1)

    # Standardize column names (lowercase, replace spaces with underscores)
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')

    print("\nData cleaning completed!")
    print(f"Final dataset shape: {df_clean.shape}")
    print("\nMissing values after cleaning:")
    print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])

    return df_clean


def clean_user_data(df):
    """
    Clean user data by removing duplicates, handling missing values,
    and converting data types.
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()

    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    print(f"Removed {duplicates_removed} duplicate rows")

    # Convert data types
    df_clean['num_ratings_pages'] = pd.to_numeric(
        df_clean['num_ratings_pages'], errors='coerce')
    df_clean['num_reviews'] = pd.to_numeric(
        df_clean['num_reviews'], errors='coerce')

    # Handle missing values
    df_clean['display_name'] = df_clean['display_name'].fillna('Unknown')
    df_clean['username'] = df_clean['username'].fillna('unknown')
    df_clean['num_ratings_pages'] = df_clean['num_ratings_pages'].fillna(0)
    df_clean['num_reviews'] = df_clean['num_reviews'].fillna(0)

    # Standardize column names
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')

    # Print the shape of the cleaned dataset
    print(f"Cleaned dataset shape: {df_clean.shape}")

    return df_clean


def clean_rating_data(df):
    """
    Clean rating data by removing duplicates, handling missing values,
    and converting data types.
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()

    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    print(f"Removed {duplicates_removed} duplicate rows")

    # Convert data types
    df_clean['rating_val'] = pd.to_numeric(
        df_clean['rating_val'], errors='coerce')

    # Handle missing values
    df_clean['rating_val'] = df_clean['rating_val'].fillna(0)
    df_clean['movie_id'] = df_clean['movie_id'].fillna('unknown')
    df_clean['user_id'] = df_clean['user_id'].fillna('unknown')

    # Standardize column names
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')

    # Print the shape of the cleaned dataset
    print(f"Cleaned dataset shape: {df_clean.shape}")

    return df_clean
