# Model Implementation

This document explains the implementation details of our collaborative filtering recommendation model.

## Model Architecture

### CollaborativeFiltering Class
```python
class CollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50):
        super(CollaborativeFiltering, self).__init__()
        
        # User and movie embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)
        
        # User and movie biases
        self.user_biases = nn.Embedding(num_users, 1)
        self.movie_biases = nn.Embedding(num_movies, 1)
        
        # Global bias
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values"""
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.movie_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.movie_biases.weight)
    
    def forward(self, user_ids, movie_ids):
        """
        Forward pass to predict ratings
        """
        # Get embeddings and biases
        user_emb = self.user_embeddings(user_ids)
        movie_emb = self.movie_embeddings(movie_ids)
        
        user_bias = self.user_biases(user_ids).squeeze()
        movie_bias = self.movie_biases(movie_ids).squeeze()
        
        # Calculate dot product of embeddings
        dot_product = (user_emb * movie_emb).sum(dim=1)
        
        # Add biases
        predictions = dot_product + user_bias + movie_bias + self.global_bias
        
        # Clip predictions to rating range [0, 5]
        predictions = torch.clamp(predictions, 0, 5)
        
        return predictions
```

## Key Components

### 1. Embeddings
- **User Embeddings**: Learn user preferences in a latent space
- **Movie Embeddings**: Learn movie characteristics in a latent space
- **Embedding Dimension**: Controls model capacity (default: 50)

### 2. Biases
- **User Biases**: Capture user-specific rating tendencies
- **Movie Biases**: Capture movie-specific rating tendencies
- **Global Bias**: Captures overall rating average

### 3. Forward Pass
1. Look up user and movie embeddings
2. Calculate dot product of embeddings
3. Add user, movie, and global biases
4. Clip predictions to valid rating range

## Training Process

### Loss Function
```python
criterion = nn.MSELoss()
```

### Optimizer
```python
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```

### Training Loop
```python
def train_epoch(model, train_data, optimizer, criterion, batch_size, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for i in range(0, len(train_data), batch_size):
        # Prepare batch
        batch = train_data[i:i + batch_size]
        users = torch.tensor([x[0] for x in batch]).to(device)
        movies = torch.tensor([x[1] for x in batch]).to(device)
        ratings = torch.tensor([x[2] for x in batch], dtype=torch.float32).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(users, movies)
        loss = criterion(predictions, ratings)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches
```

## Model Parameters

### Hyperparameters
```python
default_params = {
    'embedding_dim': 50,      # Size of embedding vectors
    'learning_rate': 0.001,   # Learning rate for optimizer
    'batch_size': 1024,       # Number of ratings per batch
    'num_epochs': 10          # Number of training epochs
}
```

### Trainable Parameters
- User embeddings: `num_users × embedding_dim`
- Movie embeddings: `num_movies × embedding_dim`
- User biases: `num_users × 1`
- Movie biases: `num_movies × 1`
- Global bias: `1`

## Implementation Details

### 1. Data Handling
```python
def prepare_batch(batch, device):
    """
    Convert batch data to tensors
    """
    users = torch.tensor([x[0] for x in batch]).to(device)
    movies = torch.tensor([x[1] for x in batch]).to(device)
    ratings = torch.tensor([x[2] for x in batch], dtype=torch.float32).to(device)
    return users, movies, ratings
```

### 2. Model Saving
```python
def save_model(model, path):
    """
    Save model state and mappings
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'user_mapping': user_mapping,
        'movie_mapping': movie_mapping
    }, path)
```

### 3. Model Loading
```python
def load_model(path, device):
    """
    Load model state and mappings
    """
    checkpoint = torch.load(path, map_location=device)
    model = CollaborativeFiltering(
        num_users=len(checkpoint['user_mapping']),
        num_movies=len(checkpoint['movie_mapping']),
        embedding_dim=50
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
```

## Best Practices

### 1. Weight Initialization
- Use small random values for embeddings
- Initialize biases to zero
- Helps prevent early convergence to poor solutions

### 2. Regularization
```python
# Add L2 regularization to optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.01  # L2 regularization strength
)
```

### 3. Gradient Clipping
```python
# Clip gradients during training
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 4. Learning Rate Scheduling
```python
# Reduce learning rate during training
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2,
    verbose=True
)
```

## Common Issues and Solutions

### 1. Memory Issues
- Use appropriate batch sizes
- Clear GPU cache regularly
- Use gradient checkpointing for large models

### 2. Training Instability
- Normalize input data
- Use gradient clipping
- Adjust learning rate

### 3. Overfitting
- Add dropout layers
- Increase regularization
- Use early stopping

## Example Usage

```python
# Initialize model
model = CollaborativeFiltering(
    num_users=num_users,
    num_movies=num_movies,
    embedding_dim=50
).to(device)

# Train model
for epoch in range(num_epochs):
    train_loss = train_epoch(
        model, train_data, optimizer, criterion, batch_size, device
    )
    val_loss = evaluate_model(model, val_data, device)
    
    print(f"Epoch {epoch + 1}:")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")

# Save model
save_model(model, 'model.pth')

# Load model
loaded_model = load_model('model.pth', device)
``` 