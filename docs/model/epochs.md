# Understanding Epochs

An epoch is one complete pass through the entire training dataset. This document explains how epochs work in our recommendation model and why we use multiple epochs.

## What is an Epoch?

### Basic Definition
```python
num_epochs = 10  # We'll pass through the data 10 times

for epoch in range(num_epochs):  # Outer loop: epochs
    for batch in train_loader:   # Inner loop: batches
        # Process one batch of data
        pass
```

### Example with Numbers
If we have:
- 100,000 ratings in our training set
- batch_size = 1024
Then one epoch means:
- Processing all 100,000 ratings once
- ≈98 batches (100,000 ÷ 1024)
- Last batch might be smaller

## Learning Process Across Epochs

### Epoch 1: Initial Learning
```python
# Start of training
user_embedding = [random numbers]     # Random initialization
movie_embedding = [random numbers]    # Random initialization

# End of first epoch
user_embedding = [slightly better numbers]  # First patterns learned
movie_embedding = [slightly better numbers] # First patterns learned
```

### Epoch 5: Refinement
```python
# Model has learned:
user_embedding = [better numbers]     # More refined patterns
movie_embedding = [better numbers]    # More refined patterns

# Example patterns:
- User likes action movies
- Movie has strong drama elements
```

### Epoch 10: Fine-tuning
```python
# Model has learned:
user_embedding = [well-tuned numbers]  # Detailed patterns
movie_embedding = [well-tuned numbers] # Detailed patterns

# Complex patterns:
- User likes action movies but not sci-fi
- Movie combines drama and comedy elements
```

## Progress Tracking

### Loss Values Example
```python
Epoch 1/10:
Training Loss: 2.5000
Validation Loss: 2.4800

Epoch 2/10:
Training Loss: 1.8000
Validation Loss: 1.7900

Epoch 3/10:
Training Loss: 1.2000
Validation Loss: 1.1900
```

### Visualization
```python
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.show()
```

## Why Multiple Epochs?

### 1. Learning Takes Time
Like studying for an exam:
- First pass: Basic understanding
- Second pass: Catch things you missed
- Third pass: Deeper understanding
- And so on...

### 2. Gradual Improvement
```python
# Epoch 1: Rough predictions
prediction = 3.0  # Actual rating: 4.0
error = 1.0

# Epoch 5: Better predictions
prediction = 3.7  # Actual rating: 4.0
error = 0.3

# Epoch 10: Refined predictions
prediction = 3.9  # Actual rating: 4.0
error = 0.1
```

### 3. Complex Patterns
The model needs time to learn:
- User preferences
- Movie characteristics
- Interactions between features
- General rating patterns

## Choosing Number of Epochs

### Factors to Consider
1. **Dataset Size**
   - Larger datasets might need fewer epochs
   - Smaller datasets might need more epochs

2. **Model Complexity**
   - More complex models need more epochs
   - Simple models might converge faster

3. **Convergence**
   - Monitor validation loss
   - Stop when improvement plateaus

### Signs to Stop Training
1. Validation loss stops decreasing
2. Validation loss starts increasing (overfitting)
3. Improvements are negligible

## Tips for Epoch Training

1. **Monitor Both Losses**
   ```python
   print(f'Training Loss: {train_loss:.4f}')
   print(f'Validation Loss: {val_loss:.4f}')
   ```

2. **Save Best Model**
   ```python
   if val_loss < best_val_loss:
       best_val_loss = val_loss
       torch.save(model.state_dict(), 'best_model.pth')
   ```

3. **Early Stopping**
   - Stop if validation loss hasn't improved for several epochs
   - Helps prevent overfitting 