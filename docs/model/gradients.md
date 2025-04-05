# Understanding Gradients

A gradient is the rate of change that tells us how adjusting a parameter affects the prediction error. This document explains how gradients work in our recommendation model.

## Basic Concept

### Simple Example
```python
# Simple prediction
prediction = user_bias + movie_bias
error = actual_rating - prediction

# Example values
user_bias = 0.5
movie_bias = 0.3
prediction = 0.8    # (0.5 + 0.3)
actual_rating = 1.0
error = 0.2        # (1.0 - 0.8)
```

### Testing Parameter Changes
```python
# Increase user_bias by 0.1
new_prediction = (0.5 + 0.1) + 0.3 = 0.9
new_error = 1.0 - 0.9 = 0.1  # Error decreased!

# Decrease user_bias by 0.1
new_prediction = (0.5 - 0.1) + 0.3 = 0.7
new_error = 1.0 - 0.7 = 0.3  # Error increased!
```

## Gradients in Our Model

### Model Parameters
```python
class CollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50):
        # Each user has:
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.user_biases = nn.Embedding(num_users, 1)
        
        # Each movie has:
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)
        self.movie_biases = nn.Embedding(num_movies, 1)
```

### Computing Gradients
```python
# During training
predictions = model(users, movies)
loss = criterion(predictions, ratings)
loss.backward()  # Compute gradients
```

## Visual Example

### Gradient Descent
```
    Error
    ^
    |     You are here
    |        •
    |       / \
    |      /   \
    |     /     \
    |    /       \
    |   /         \
    |  /           \
    | /             \
    |/               \
    +-----------------> Parameter Value
```

The gradient tells us:
- Which direction to move (up/down)
- How big of a step to take

## Real Example with Numbers

### Initial State
```python
# User 123's taste vector
user_embedding = [0.5, -0.2, 0.8, ...]  # 50 numbers

# Movie 456's feature vector
movie_embedding = [0.3, 0.1, 0.7, ...]  # 50 numbers

# Current prediction: 3.8
# Actual rating: 4.0
# Error: 0.2
```

### Gradient Update
```python
# If gradient = 0.1 and learning_rate = 0.01:
old_value = 0.5
new_value = old_value - (learning_rate * gradient)
         = 0.5 - (0.01 * 0.1)
         = 0.499
```

## How Gradients Are Used

### Training Loop
```python
for epoch in range(num_epochs):
    for users, movies, ratings in train_loader:
        # 1. Forward pass
        predictions = model(users, movies)
        loss = criterion(predictions, ratings)
        
        # 2. Compute gradients
        loss.backward()
        
        # 3. Update weights using gradients
        optimizer.step()
        
        # 4. Reset gradients
        optimizer.zero_grad()
```

### Optimizer's Role
The Adam optimizer:
1. Tracks gradient history
2. Adjusts learning rates automatically
3. Handles different scales of gradients
4. Makes updates more stable

## Tips for Working with Gradients

### 1. Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
- Prevents exploding gradients
- Stabilizes training

### 2. Learning Rate
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
- Too high: Unstable training
- Too low: Slow training
- Just right: Steady improvement

### 3. Monitoring Gradients
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name} grad norm: {param.grad.norm()}")
```

## Common Issues

### 1. Vanishing Gradients
- Gradients become too small
- Learning slows down
- Solution: Use appropriate activation functions

### 2. Exploding Gradients
- Gradients become too large
- Training becomes unstable
- Solution: Use gradient clipping

### 3. Dead Neurons
- Gradients become zero
- Parts of model stop learning
- Solution: Check activation functions and learning rate 