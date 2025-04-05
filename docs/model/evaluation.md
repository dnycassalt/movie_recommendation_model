# Model Evaluation

This guide explains how to evaluate the effectiveness and quality of the trained recommendation model using various metrics and techniques.

## Key Metrics

### 1. Rating Prediction Metrics

#### Mean Squared Error (MSE)
```python
from sklearn.metrics import mean_squared_error

def calculate_mse(model, test_data, device):
    model.eval()
    with torch.no_grad():
        users = torch.tensor([x[0] for x in test_data]).to(device)
        movies = torch.tensor([x[1] for x in test_data]).to(device)
        actual_ratings = torch.tensor([x[2] for x in test_data]).to(device)
        
        predicted_ratings = model(users, movies)
        mse = mean_squared_error(actual_ratings.cpu(), predicted_ratings.cpu())
    return mse
```

#### Root Mean Squared Error (RMSE)
```python
def calculate_rmse(model, test_data, device):
    mse = calculate_mse(model, test_data, device)
    return np.sqrt(mse)
```

#### Mean Absolute Error (MAE)
```python
from sklearn.metrics import mean_absolute_error

def calculate_mae(model, test_data, device):
    model.eval()
    with torch.no_grad():
        users = torch.tensor([x[0] for x in test_data]).to(device)
        movies = torch.tensor([x[1] for x in test_data]).to(device)
        actual_ratings = torch.tensor([x[2] for x in test_data]).to(device)
        
        predicted_ratings = model(users, movies)
        mae = mean_absolute_error(actual_ratings.cpu(), predicted_ratings.cpu())
    return mae
```

### 2. Ranking Metrics

#### Normalized Discounted Cumulative Gain (NDCG)
```python
def calculate_ndcg(model, user_id, actual_items, k=10, device='cuda'):
    """
    Calculate NDCG@k for a user's recommendations
    """
    model.eval()
    with torch.no_grad():
        # Get all movie predictions for the user
        all_movies = torch.arange(model.num_movies).to(device)
        user_tensor = torch.full_like(all_movies, user_id)
        predictions = model(user_tensor, all_movies)
        
        # Get top k predictions
        _, indices = torch.topk(predictions, k)
        recommended_items = indices.cpu().numpy()
        
        # Calculate DCG
        dcg = 0
        for i, item in enumerate(recommended_items):
            if item in actual_items:
                dcg += 1 / np.log2(i + 2)
        
        # Calculate IDCG
        idcg = 0
        for i in range(min(len(actual_items), k)):
            idcg += 1 / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0
        return ndcg
```

#### Precision and Recall at K
```python
def calculate_precision_recall_at_k(model, user_id, actual_items, k=10, device='cuda'):
    """
    Calculate Precision@k and Recall@k for a user's recommendations
    """
    model.eval()
    with torch.no_grad():
        # Get all movie predictions for the user
        all_movies = torch.arange(model.num_movies).to(device)
        user_tensor = torch.full_like(all_movies, user_id)
        predictions = model(user_tensor, all_movies)
        
        # Get top k predictions
        _, indices = torch.topk(predictions, k)
        recommended_items = set(indices.cpu().numpy())
        actual_items = set(actual_items)
        
        # Calculate metrics
        num_relevant = len(recommended_items & actual_items)
        precision = num_relevant / k
        recall = num_relevant / len(actual_items) if actual_items else 0
        
        return precision, recall
```

### Understanding Balanced Precision and Recall

Precision and recall measure different aspects of recommendation quality:

#### Precision
- Measures how many of our recommended items are actually relevant
- Formula: (Number of relevant items recommended) / (Total number of items recommended)
- Example: If we recommend 10 movies and 7 are ones the user actually likes, precision is 7/10 = 0.7
- High precision means fewer irrelevant recommendations

#### Recall
- Measures how many of the relevant items we successfully recommended
- Formula: (Number of relevant items recommended) / (Total number of relevant items)
- Example: If a user likes 20 movies, and we recommended 7 of them, recall is 7/20 = 0.35
- High recall means we're not missing many good recommendations

#### The Trade-off
```python
# Example showing precision-recall trade-off
def demonstrate_tradeoff(model, user_id, actual_items, ks=[5, 10, 20, 50]):
    """
    Show how precision and recall change with different k values
    """
    results = []
    for k in ks:
        precision, recall = calculate_precision_recall_at_k(
            model, user_id, actual_items, k=k
        )
        results.append({
            'k': k,
            'precision': precision,
            'recall': recall
        })
    
    # Print results
    for r in results:
        print(f"At k={r['k']}:")
        print(f"  Precision: {r['precision']:.4f}")
        print(f"  Recall: {r['recall']:.4f}")
```

#### Balanced Performance
A balanced model should:
1. Have reasonable values for both metrics (e.g., both > 0.3)
2. Show appropriate trade-offs as k increases:
   - Precision typically decreases (harder to be accurate with more recommendations)
   - Recall typically increases (more chances to find relevant items)
3. Maintain consistency across different user groups:
   - Active users (many ratings)
   - Casual users (few ratings)
   - Different genre preferences

#### Example Analysis
```python
def analyze_precision_recall_balance(model, test_data, user_item_matrix, device):
    """
    Analyze precision-recall balance across different user groups
    """
    # Group users by activity level
    user_activity = {
        user_id: len(user_item_matrix[user_id].nonzero()[0])
        for user_id in set(x[0] for x in test_data)
    }
    
    # Categorize users
    active_users = [u for u, c in user_activity.items() if c >= 50]
    casual_users = [u for u, c in user_activity.items() if 10 <= c < 50]
    new_users = [u for u, c in user_activity.items() if c < 10]
    
    # Analyze each group
    for group_name, users in [
        ('Active Users', active_users),
        ('Casual Users', casual_users),
        ('New Users', new_users)
    ]:
        print(f"\n{group_name} Analysis:")
        
        # Sample users from group
        sample = random.sample(users, min(10, len(users)))
        
        # Calculate metrics for different k values
        for k in [5, 10, 20]:
            p_scores = []
            r_scores = []
            
            for user_id in sample:
                actual_items = user_item_matrix[user_id].nonzero()[0]
                precision, recall = calculate_precision_recall_at_k(
                    model, user_id, actual_items, k=k, device=device
                )
                p_scores.append(precision)
                r_scores.append(recall)
            
            print(f"k={k}:")
            print(f"  Avg Precision: {np.mean(p_scores):.4f}")
            print(f"  Avg Recall: {np.mean(r_scores):.4f}")
```

#### Interpreting Balance
1. **Good Balance**:
   - Precision@10 ≈ 0.4-0.6
   - Recall@10 ≈ 0.3-0.5
   - Both metrics degrade gracefully as k changes

2. **Precision-Heavy**:
   - High precision (> 0.7)
   - Low recall (< 0.2)
   - Model is too conservative, recommending only very safe choices

3. **Recall-Heavy**:
   - Low precision (< 0.3)
   - High recall (> 0.6)
   - Model is too aggressive, recommending too many items

4. **Poor Balance**:
   - Both metrics < 0.2
   - Model needs improvement in overall quality

## Using the Metrics

### Basic Evaluation
```python
def evaluate_model(model, test_data, device):
    """
    Comprehensive model evaluation
    """
    metrics = {}
    
    # Calculate rating prediction metrics
    metrics['mse'] = calculate_mse(model, test_data, device)
    metrics['rmse'] = calculate_rmse(model, test_data, device)
    metrics['mae'] = calculate_mae(model, test_data, device)
    
    print("Rating Prediction Metrics:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    
    return metrics
```

### Advanced Evaluation
```python
def evaluate_model_detailed(model, test_data, user_item_matrix, device, k=10):
    """
    Detailed model evaluation including ranking metrics
    """
    metrics = evaluate_model(model, test_data, device)
    
    # Calculate ranking metrics for a sample of users
    ndcg_scores = []
    precision_scores = []
    recall_scores = []
    
    # Sample users for evaluation
    unique_users = list(set(x[0] for x in test_data))
    sample_users = random.sample(unique_users, min(100, len(unique_users)))
    
    for user_id in sample_users:
        # Get actual items for user
        actual_items = user_item_matrix[user_id].nonzero()[0]
        
        # Calculate ranking metrics
        ndcg = calculate_ndcg(model, user_id, actual_items, k, device)
        precision, recall = calculate_precision_recall_at_k(
            model, user_id, actual_items, k, device
        )
        
        ndcg_scores.append(ndcg)
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    # Average the metrics
    metrics['ndcg@k'] = np.mean(ndcg_scores)
    metrics['precision@k'] = np.mean(precision_scores)
    metrics['recall@k'] = np.mean(recall_scores)
    
    print(f"\nRanking Metrics (k={k}):")
    print(f"NDCG@{k}: {metrics['ndcg@k']:.4f}")
    print(f"Precision@{k}: {metrics['precision@k']:.4f}")
    print(f"Recall@{k}: {metrics['recall@k']:.4f}")
    
    return metrics
```

## Visualizing Results

### Rating Distribution
```python
def plot_rating_distribution(model, test_data, device):
    """
    Plot actual vs predicted rating distributions
    """
    model.eval()
    with torch.no_grad():
        users = torch.tensor([x[0] for x in test_data]).to(device)
        movies = torch.tensor([x[1] for x in test_data]).to(device)
        actual_ratings = torch.tensor([x[2] for x in test_data]).cpu().numpy()
        predicted_ratings = model(users, movies).cpu().numpy()
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(actual_ratings, bins=20, alpha=0.5, label='Actual')
        plt.hist(predicted_ratings, bins=20, alpha=0.5, label='Predicted')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.title('Rating Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.scatter(actual_ratings, predicted_ratings, alpha=0.1)
        plt.plot([0, 5], [0, 5], 'r--')  # Perfect prediction line
        plt.xlabel('Actual Ratings')
        plt.ylabel('Predicted Ratings')
        plt.title('Actual vs Predicted Ratings')
        
        plt.tight_layout()
        plt.show()
```

### Error Analysis
```python
def analyze_errors(model, test_data, device):
    """
    Analyze prediction errors
    """
    model.eval()
    with torch.no_grad():
        users = torch.tensor([x[0] for x in test_data]).to(device)
        movies = torch.tensor([x[1] for x in test_data]).to(device)
        actual_ratings = torch.tensor([x[2] for x in test_data]).cpu().numpy()
        predicted_ratings = model(users, movies).cpu().numpy()
        
        errors = predicted_ratings - actual_ratings
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(errors, bins=50)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title('Error Distribution')
        
        plt.subplot(1, 2, 2)
        plt.boxplot(errors)
        plt.ylabel('Prediction Error')
        plt.title('Error Box Plot')
        
        plt.tight_layout()
        plt.show()
        
        print("\nError Statistics:")
        print(f"Mean Error: {np.mean(errors):.4f}")
        print(f"Std Error: {np.std(errors):.4f}")
        print(f"Median Error: {np.median(errors):.4f}")
```

## Best Practices

### 1. Cross-Validation
```python
from sklearn.model_selection import KFold

def cross_validate(model_class, train_data, k_folds=5, **model_params):
    """
    Perform k-fold cross-validation
    """
    kf = KFold(n_splits=k_folds, shuffle=True)
    metrics_list = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
        # Initialize model
        model = model_class(**model_params)
        
        # Train model
        train_fold = [train_data[i] for i in train_idx]
        val_fold = [train_data[i] for i in val_idx]
        
        # Train the model...
        
        # Evaluate
        metrics = evaluate_model(model, val_fold, device)
        metrics_list.append(metrics)
        
        print(f"\nFold {fold + 1} Results:")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
    
    # Average metrics across folds
    avg_metrics = {
        key: np.mean([m[key] for m in metrics_list])
        for key in metrics_list[0].keys()
    }
    
    print("\nAverage Metrics Across Folds:")
    print(f"RMSE: {avg_metrics['rmse']:.4f}")
    print(f"MAE: {avg_metrics['mae']:.4f}")
    
    return avg_metrics
```

### 2. Time-Based Evaluation
```python
def evaluate_temporal(model, test_data, timestamp_key='timestamp'):
    """
    Evaluate model performance over time
    """
    # Sort test data by timestamp
    sorted_data = sorted(test_data, key=lambda x: x[timestamp_key])
    
    # Create time windows
    windows = np.array_split(sorted_data, 5)  # Split into 5 time periods
    
    for i, window in enumerate(windows):
        print(f"\nTime Window {i + 1}:")
        metrics = evaluate_model(model, window, device)
```

## Example Usage

```python
# Load trained model
model = load_model('best_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Load test data
test_data = load_test_data()

# Basic evaluation
metrics = evaluate_model(model, test_data, device)

# Detailed evaluation with ranking metrics
detailed_metrics = evaluate_model_detailed(model, test_data, user_item_matrix, device, k=10)

# Visualize results
plot_rating_distribution(model, test_data, device)
analyze_errors(model, test_data, device)

# Cross-validation
cv_metrics = cross_validate(CollaborativeFiltering, train_data, k_folds=5,
                          num_users=num_users, num_movies=num_movies,
                          embedding_dim=50)
```

## Interpreting Results

### Rating Prediction Metrics
- **RMSE**: Lower is better. Typical values range from 0.8 to 1.2
- **MAE**: Lower is better. Usually slightly lower than RMSE
- **MSE**: Lower is better. Square of RMSE

### Ranking Metrics
- **NDCG@k**: Ranges from 0 to 1, higher is better
- **Precision@k**: Ranges from 0 to 1, higher is better
- **Recall@k**: Ranges from 0 to 1, higher is better

### What Makes a Good Model?
1. Low RMSE and MAE (< 1.0 is good)
2. High NDCG@k (> 0.5 is good)
3. Balanced Precision and Recall
4. Consistent performance across different user groups
5. Stable performance over time

## Common Issues and Solutions

1. **High Error for Certain Users**
   - Check user activity level
   - Consider user cold-start problem
   - Analyze user rating patterns

2. **Poor Ranking Performance**
   - Increase embedding dimension
   - Add implicit feedback
   - Consider adding user/item biases

3. **Inconsistent Performance**
   - Use cross-validation
   - Increase training data
   - Add regularization 