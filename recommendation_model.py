# Import required libraries
import torch  # PyTorch for deep learning
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
from torch.utils.data import Dataset, DataLoader  # Data handling utilities
import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
from sklearn.model_selection import train_test_split  # For splitting dataset
from data_loader import DataLoader as MovieDataLoader  # Custom data loader
from data_cleaner import clean_rating_data  # Data cleaning utility
import matplotlib.pyplot as plt  # For plotting results


class MovieRatingDataset(Dataset):
    """Custom Dataset for loading movie ratings.

    This class prepares the data for PyTorch by:
    1. Converting user and movie IDs to numerical indices
    2. Converting ratings to tensors
    3. Providing iteration capabilities over the dataset
    """

    def __init__(self, ratings_df, user_mapping, movie_mapping):
        """
        Args:
            ratings_df (pd.DataFrame): DataFrame containing rating data
            user_mapping (dict): Mapping of user IDs to numerical indices
            movie_mapping (dict): Mapping of movie IDs to numerical indices
        """
        # Convert user IDs to numerical indices using the mapping
        self.users = torch.tensor(
            ratings_df['user_id'].map(user_mapping).values,
            dtype=torch.long
        )
        # Convert movie IDs to numerical indices using the mapping
        self.movies = torch.tensor(
            ratings_df['movie_id'].map(movie_mapping).values,
            dtype=torch.long
        )
        # Convert ratings to float tensors
        self.ratings = torch.tensor(
            ratings_df['rating_val'].values,
            dtype=torch.float32
        )

    def __len__(self):
        """Return the total number of ratings in the dataset."""
        return len(self.ratings)

    def __getitem__(self, idx):
        """Return a single (user, movie, rating) tuple for training."""
        return (
            self.users[idx],
            self.movies[idx],
            self.ratings[idx]
        )


class CollaborativeFiltering(nn.Module):
    """Matrix Factorization model for collaborative filtering.

    This model learns latent representations (embeddings) for both users and movies.
    The rating prediction is computed as the dot product of user and movie embeddings,
    plus user and movie bias terms.
    """

    def __init__(self, num_users, num_movies, embedding_dim=50):
        """
        Args:
            num_users (int): Total number of unique users
            num_movies (int): Total number of unique movies
            embedding_dim (int): Size of the embedding vectors
        """
        super().__init__()
        # Create embedding layers for users and movies
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)
        # Create bias terms for users and movies
        self.user_biases = nn.Embedding(num_users, 1)
        self.movie_biases = nn.Embedding(num_movies, 1)

        # Initialize weights with small random values and zero biases
        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.movie_embeddings.weight, std=0.1)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.movie_biases.weight)

    def forward(self, user_ids, movie_ids):
        """
        Compute rating predictions for given user-movie pairs.

        The prediction is computed as: 
        rating = (user_embedding * movie_embedding).sum() + user_bias + movie_bias
        """
        # Get embeddings for the batch of users and movies
        user_embeds = self.user_embeddings(user_ids)
        movie_embeds = self.movie_embeddings(movie_ids)
        # Get bias terms
        user_bias = self.user_biases(user_ids).squeeze()
        movie_bias = self.movie_biases(movie_ids).squeeze()
        # Compute dot product of embeddings
        dot_products = (user_embeds * movie_embeds).sum(dim=1)
        # Return final prediction
        return dot_products + user_bias + movie_bias


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=10, device='cpu'):
    """Train the model and return training history.

    Args:
        model: The CollaborativeFiltering model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function (MSE)
        optimizer: Optimization algorithm (Adam)
        num_epochs: Number of training epochs
        device: Device to run the training on (CPU/GPU)

    Returns:
        Lists of training and validation losses per epoch
    """
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Set model to training mode
        total_train_loss = 0
        for users, movies, ratings in train_loader:
            # Move data to appropriate device (CPU/GPU)
            users, movies, ratings = (users.to(device), movies.to(device),
                                      ratings.to(device))

            # Forward pass
            optimizer.zero_grad()  # Clear previous gradients
            predictions = model(users, movies)  # Get predictions
            loss = criterion(predictions, ratings)  # Compute loss

            # Backward pass
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights

            total_train_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0
        with torch.no_grad():  # Disable gradient computation
            for users, movies, ratings in val_loader:
                users, movies, ratings = (users.to(device), movies.to(device),
                                          ratings.to(device))
                predictions = model(users, movies)
                loss = criterion(predictions, ratings)
                total_val_loss += loss.item()

        # Calculate average validation loss for the epoch
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}\n')

    return train_losses, val_losses


def main():
    """Main function to run the training pipeline."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load and clean the ratings data
    loader = MovieDataLoader(
        file_path='data/ratings_export.csv',
        checkpoint_file='ratings_data_checkpoint.pkl',
        chunk_size=250,
        checkpoint_interval=50000
    )

    df = loader.load_data(
        encoding='utf-8',
        on_bad_lines='skip',
        low_memory=False,
        lineterminator='\n'
    )
    df = clean_rating_data(df)

    # Create mappings from user/movie IDs to numerical indices
    user_mapping = {uid: idx for idx, uid in enumerate(df['user_id'].unique())}
    movie_mapping = {mid: idx for idx, mid in
                     enumerate(df['movie_id'].unique())}

    # Split data into training and validation sets (80/20 split)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Create PyTorch datasets
    train_dataset = MovieRatingDataset(train_df, user_mapping, movie_mapping)
    val_dataset = MovieRatingDataset(val_df, user_mapping, movie_mapping)

    # Create data loaders with batching
    batch_size = 1024  # Process 1024 ratings at once
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True  # Shuffle training data
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )

    # Initialize the model
    model = CollaborativeFiltering(
        num_users=len(user_mapping),
        num_movies=len(movie_mapping),
        embedding_dim=50  # Size of latent factors
    ).to(device)

    # Set up training parameters
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
    num_epochs = 10  # Number of training epochs

    # Train the model
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        device=device
    )

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.show()

    # Save the trained model and mappings
    torch.save({
        'model_state_dict': model.state_dict(),
        'user_mapping': user_mapping,
        'movie_mapping': movie_mapping,
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'recommendation_model.pth')

    print("Model saved successfully!")


if __name__ == "__main__":
    main()
