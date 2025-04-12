import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import DataLoader
from data_cleaner import clean_movie_data, safe_eval

# Set the style for better visualizations
plt.style.use('default')  # Using default style instead of seaborn
sns.set_theme()  # This will set seaborn's default styling
sns.set_palette("husl")

# Print current working directory and check if file exists
print(f"Current working directory: {os.getcwd()}")
file_path = 'data/movie_data.csv'
print(f"Checking if file exists at: {file_path}")
print(f"File exists: {os.path.exists(file_path)}")


def process_genres(chunk):
    """Process genres column in a chunk of data."""
    chunk['genres'] = chunk['genres'].apply(safe_eval)
    return chunk


# Initialize data loader
loader = DataLoader(
    file_path=file_path,
    chunk_size=250,
    checkpoint_interval=50000
)

try:
    # Load data with genre processing
    df = loader.load_data(
        chunk_processor=process_genres,
        encoding='utf-8',
        on_bad_lines='skip',
        low_memory=False,
        lineterminator='\n'
    )

    # Clean the data
    df = clean_movie_data(df)

    # Display basic information about the dataset
    print("\nDataset Info:")
    print(df.info())

    print("\nFirst few rows of the dataset:")
    print(df.head())

    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe())

    # Create visualizations
    plt.figure(figsize=(15, 10))

    # 1. Distribution of ratings
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='vote_average', bins=30)
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')

    # 2. Top 10 genres
    plt.subplot(2, 2, 2)
    # Process genres
    all_genres = []
    for genres in df['genres']:
        all_genres.extend(genres)
    top_genres = pd.Series(all_genres).value_counts().head(10)
    sns.barplot(x=top_genres.values, y=top_genres.index)
    plt.title('Top 10 Movie Genres')
    plt.xlabel('Count')

    # 3. Average rating by year
    plt.subplot(2, 2, 3)
    yearly_ratings = df.groupby('year_released')[
        'vote_average'].mean().reset_index()
    sns.lineplot(data=yearly_ratings, x='year_released', y='vote_average')
    plt.title('Average Rating by Year')
    plt.xlabel('Year')
    plt.ylabel('Average Rating')

    # 4. Rating vs Runtime
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df, x='runtime', y='vote_average', alpha=0.5)
    plt.title('Rating vs Runtime')
    plt.xlabel('Runtime (minutes)')
    plt.ylabel('Rating')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # Print some interesting statistics
    print("\nInteresting Statistics:")
    print(f"Total number of movies: {len(df)}")
    print(f"Average movie rating: {df['vote_average'].mean():.2f}")
    print(f"Most common genre: {pd.Series(all_genres).mode()[0]}")
    print(f"Average runtime: {df['runtime'].mean():.2f} minutes")

except Exception as e:
    print(f"Error in main script: {str(e)}")
    exit(1)
