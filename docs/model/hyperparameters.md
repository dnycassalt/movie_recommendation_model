# Hyperparameters

Hyperparameters are configuration settings used to control the learning process. Unlike model parameters (which are learned during training), hyperparameters are set before training begins.

## Key Hyperparameters in Our Model

### 1. Embedding Dimension
```python
class CollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50):
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)
```

- **Value**: 50
- **Purpose**: Determines the size of user and movie embedding vectors
- **Impact**:
  - Larger = More expressive but more parameters to learn
  - Smaller = Faster training but might miss patterns
  - 50 is a good balance for most recommendation systems

### 2. Batch Size
```python
batch_size = 1024
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True
)
```

- **Value**: 1024
- **Purpose**: Number of ratings processed in one training step
- **Impact**:
  - Larger = Faster training, more stable gradients, but more memory
  - Smaller = Less memory, but noisier gradients and slower training
  - 1024 works well with modern GPUs

### 3. Learning Rate
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

- **Value**: 0.001
- **Purpose**: Controls how much parameters change in each update
- **Impact**:
  - Too high = Unstable training, might miss optimal values
  - Too low = Very slow training
  - 0.001 is a good default for Adam optimizer

### 4. Number of Epochs
```python
num_epochs = 10
```

- **Value**: 10
- **Purpose**: Number of complete passes through the training data
- **Impact**:
  - Too few = Underfitting (model hasn't learned enough)
  - Too many = Overfitting (model memorizes training data)
  - Monitor validation loss to determine optimal number

### 5. Train-Test Split Ratio
```python
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
```

- **Value**: test_size=0.2 (80% train, 20% validation)
- **Purpose**: Determines how much data to use for validation
- **Impact**:
  - Too little validation data = Unreliable performance estimates
  - Too much validation data = Not enough training data
  - 80/20 split is a common default

## Tuning Hyperparameters

### Grid Search Example
```python
embedding_dims = [32, 50, 64]
batch_sizes = [512, 1024, 2048]
learning_rates = [0.0001, 0.001, 0.01]

best_params = None
best_val_loss = float('inf')

for dim in embedding_dims:
    for batch in batch_sizes:
        for lr in learning_rates:
            # Initialize model with these hyperparameters
            model = CollaborativeFiltering(
                num_users=len(user_mapping),
                num_movies=len(movie_mapping),
                embedding_dim=dim
            )
            # Train and validate...
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = {
                    'embedding_dim': dim,
                    'batch_size': batch,
                    'learning_rate': lr
                }
```

## Current Settings

```python
hyperparameters = {
    'embedding_dim': 50,
    'batch_size': 1024,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'test_size': 0.2
}
```

## Tips for Hyperparameter Tuning

1. **Start with Defaults**
   - Use common default values first
   - Make sure model works before tuning

2. **One at a Time**
   - Change one hyperparameter at a time
   - Easier to understand impact

3. **Monitor Validation Loss**
   ```python
   plt.plot(val_losses, label='Validation Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.show()
   ```

4. **Use Cross-Validation**
   - More reliable performance estimates
   - Helps prevent overfitting to validation set

5. **Consider Resource Constraints**
   - Larger models need more memory
   - Longer training needs more compute time 