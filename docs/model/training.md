# Training Process

This document explains how our recommendation model is trained, including data preparation, training loop, and validation process.

## Data Preparation

### 1. Loading and Cleaning
```python
# Load data using custom DataLoader
loader = MovieDataLoader(
    file_path='data/ratings_export.csv',
    checkpoint_file='ratings_data_checkpoint.pkl',
    chunk_size=250,
    checkpoint_interval=50000
)

# Clean the data
df = clean_rating_data(df)
```

### 2. Creating Train/Validation Split
```python
# Split data 80% training, 20% validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
```

### 3. Creating DataLoaders
```python
# Create PyTorch datasets
train_dataset = MovieRatingDataset(train_df, user_mapping, movie_mapping)
val_dataset = MovieRatingDataset(val_df, user_mapping, movie_mapping)

# Create data loaders with batching
batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
```

## Training Loop

### 1. Setup
```python
# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
num_epochs = 10
```

### 2. Training Process
```python
for epoch in range(num_epochs):
    # Training phase
    model.train()
    total_train_loss = 0
    
    for users, movies, ratings in train_loader:
        # Move data to device
        users = users.to(device)
        movies = movies.to(device)
        ratings = ratings.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(users, movies)
        loss = criterion(predictions, ratings)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
```

## Validation Process

During each epoch, we validate the model:
```python
# Validation phase
model.eval()
total_val_loss = 0

with torch.no_grad():
    for users, movies, ratings in val_loader:
        # Move data to device
        users = users.to(device)
        movies = movies.to(device)
        ratings = ratings.to(device)
        
        # Make predictions
        predictions = model(users, movies)
        loss = criterion(predictions, ratings)
        total_val_loss += loss.item()
```

## Progress Tracking

### 1. Loss Monitoring
```python
# Print progress
print(f'Epoch {epoch+1}/{num_epochs}:')
print(f'Training Loss: {avg_train_loss:.4f}')
print(f'Validation Loss: {avg_val_loss:.4f}')
```

### 2. Visualization
```python
# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.show()
```

## Model Saving

After training, we save the model and mappings:
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'user_mapping': user_mapping,
    'movie_mapping': movie_mapping,
    'optimizer_state_dict': optimizer.state_dict(),
}, 'recommendation_model.pth')
```

## Training Tips

1. **Batch Size**
   - Larger batch size = faster training but more memory
   - Smaller batch size = slower training but less memory
   - Current setting: 1024 ratings per batch

2. **Learning Rate**
   - Current setting: 0.001
   - Too high = unstable training
   - Too low = slow training

3. **Number of Epochs**
   - Current setting: 10
   - Monitor validation loss to prevent overfitting
   - Stop when validation loss plateaus 