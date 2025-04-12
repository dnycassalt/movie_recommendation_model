import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import DataLoader
from data_cleaner import clean_rating_data

# Set the style for better visualizations
plt.style.use('default')
sns.set_theme()
sns.set_palette("husl")

# Print current working directory and check if file exists
print(f"Current working directory: {os.getcwd()}")
file_path = 'data/ratings_export.csv'
# Different checkpoint file for ratings
checkpoint_file = 'ratings_data_checkpoint.pkl'
print(f"Checking if file exists at: {file_path}")
print(f"File exists: {os.path.exists(file_path)}")

# Initialize data loader with custom checkpoint file
loader = DataLoader(
    file_path=file_path,
    checkpoint_file=checkpoint_file,  # Use different checkpoint file
    chunk_size=250,
    checkpoint_interval=50000
)

try:
    # Load data
    df = loader.load_data(
        encoding='utf-8',
        on_bad_lines='skip',
        low_memory=False,
        lineterminator='\n'
    )

    # Clean the data
    df = clean_rating_data(df)

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
    sns.histplot(data=df, x='rating_val', bins=30)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating Value')
    plt.ylabel('Count')

    # 2. Average rating by user
    plt.subplot(2, 2, 2)
    user_ratings = df.groupby('user_id')['rating_val'].mean()
    user_ratings.plot(kind='line')
    plt.title('Average Rating by User')
    plt.xlabel('User ID')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45)

    # 3. Rating distribution by movie
    plt.subplot(2, 2, 3)
    movie_ratings = df.groupby('movie_id')['rating_val'].mean()
    movie_ratings.plot(kind='box')
    plt.title('Rating Distribution by Movie')
    plt.xlabel('Movie ID')
    plt.ylabel('Rating Value')
    plt.xticks(rotation=45)

    # 4. Number of ratings per user
    plt.subplot(2, 2, 4)
    ratings_per_user = df['user_id'].value_counts()
    sns.histplot(data=ratings_per_user, bins=30)
    plt.title('Distribution of Ratings per User')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Count')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # Print some interesting statistics
    print("\nInteresting Statistics:")
    print(f"Total number of ratings: {len(df)}")
    print(f"Average rating: {df['rating_val'].mean():.2f}")
    print(f"Number of unique users: {df['user_id'].nunique()}")
    print(f"Number of unique movies: {df['movie_id'].nunique()}")
    print(f"Average ratings per user: {len(df) / df['user_id'].nunique():.2f}")

except Exception as e:
    print(f"Error in main script: {str(e)}")
    exit(1)
