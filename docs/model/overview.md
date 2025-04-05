# Model Overview

Our movie recommendation system uses collaborative filtering through matrix factorization, implemented in PyTorch. The model learns latent representations (embeddings) for both users and movies to predict ratings.

## Architecture

```python
class CollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50):
        super().__init__()
        # Create embedding layers for users and movies
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)
        # Create bias terms for users and movies
        self.user_biases = nn.Embedding(num_users, 1)
        self.movie_biases = nn.Embedding(num_movies, 1)
```

### Components

1. **User Embeddings**
    - Each user gets a vector of 50 numbers
    - Represents user preferences/taste
    - Learned during training

2. **Movie Embeddings**
    - Each movie gets a vector of 50 numbers
    - Represents movie characteristics
    - Learned during training

3. **Bias Terms**
    - User bias: Captures if a user tends to rate high/low
    - Movie bias: Captures if a movie tends to be rated high/low

## Making Predictions

The model predicts ratings using:
```python
def forward(self, user_ids, movie_ids):
    # Get embeddings
    user_embeds = self.user_embeddings(user_ids)
    movie_embeds = self.movie_embeddings(movie_ids)
    
    # Get bias terms
    user_bias = self.user_biases(user_ids).squeeze()
    movie_bias = self.movie_biases(movie_ids).squeeze()
    
    # Compute rating prediction
    dot_products = (user_embeds * movie_embeds).sum(dim=1)
    return dot_products + user_bias + movie_bias
```

### Prediction Process

1. **Look up embeddings**
    - Get user's taste vector
    - Get movie's characteristic vector

2. **Compute similarity**
    - Multiply corresponding numbers
    - Sum the products

3. **Add biases**
    - Add user's rating tendency
    - Add movie's rating tendency

## Example

```python
# User 123's taste vector
user_embedding = [0.5, -0.2, 0.8, ...]  # 50 numbers

# Movie 456's characteristic vector
movie_embedding = [0.3, 0.1, 0.7, ...]  # 50 numbers

# Prediction calculation
dot_product = sum([0.5 * 0.3, -0.2 * 0.1, 0.8 * 0.7, ...])
user_bias = 0.2
movie_bias = 0.1

final_prediction = dot_product + user_bias + movie_bias
```

## Model Features

1. **Learned Representations**
    - Model discovers important features automatically
    - No need for manual feature engineering
    - Can capture complex patterns

2. **Scalability**
    - Efficient matrix operations
    - Batch processing
    - GPU support

3. **Flexibility**
    - Can adjust embedding size
    - Can modify architecture
    - Can add additional features

## Usage

```python
# Initialize model
model = CollaborativeFiltering(
    num_users=len(user_mapping),
    num_movies=len(movie_mapping),
    embedding_dim=50
).to(device)

# Make predictions
predictions = model(user_ids, movie_ids)
``` 