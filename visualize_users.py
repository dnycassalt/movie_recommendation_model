import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import DataLoader
from data_cleaner import clean_user_data

# Set the style for better visualizations
plt.style.use('default')
sns.set_theme()
sns.set_palette("husl")

# Print current working directory and check if file exists
print(f"Current working directory: {os.getcwd()}")
file_path = 'data/users_export.csv'
checkpoint_file = 'users_checkpoint.pkl'  # Different checkpoint file for users
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
    df = clean_user_data(df)

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

    # 1. Distribution of number of ratings pages per user
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='num_ratings_pages', bins=30)
    plt.title('Distribution of Number of Rating Pages per User')
    plt.xlabel('Number of Rating Pages')
    plt.ylabel('Count')

    # 2. Distribution of number of reviews per user
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='num_reviews', bins=30)
    plt.title('Distribution of Number of Reviews per User')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Count')

    # 3. Scatter plot of ratings pages vs reviews
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='num_ratings_pages', y='num_reviews')
    plt.title('Number of Reviews vs Number of Rating Pages')
    plt.xlabel('Number of Rating Pages')
    plt.ylabel('Number of Reviews')

    # 4. Distribution of usernames length
    plt.subplot(2, 2, 4)
    df['username_length'] = df['username'].str.len()
    sns.histplot(data=df, x='username_length', bins=30)
    plt.title('Distribution of Username Lengths')
    plt.xlabel('Username Length (characters)')
    plt.ylabel('Count')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # Print some interesting statistics
    print("\nInteresting Statistics:")
    print(f"Total number of users: {len(df)}")
    avg_ratings = df['num_ratings_pages'].mean()
    print(f"Average number of rating pages per user: {avg_ratings:.2f}")
    avg_reviews = df['num_reviews'].mean()
    print(f"Average number of reviews per user: {avg_reviews:.2f}")
    avg_username = df['username_length'].mean()
    print(f"Average username length: {avg_username:.2f} chars")
    mode_username = df['username_length'].mode().iloc[0]
    print(f"Most common username length: {mode_username} chars")

except Exception as e:
    print(f"Error in main script: {str(e)}")
    exit(1)
