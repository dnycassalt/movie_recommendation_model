# Running the Model

This guide explains how to run the recommendation model and what to expect during execution.

## Prerequisites

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

Required packages:
- torch (PyTorch)
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

2. **Data Files**
Ensure you have the following files in your `data` directory:
- `ratings_export.csv`
- `users_export.csv`

## Running the Script

Execute the script using Python:
```bash
python recommendation_model.py
```

## What to Expect

### 1. Initial Output
```
Using device: cuda  # If you have a GPU, otherwise 'cpu'
Loading data...
Cleaned dataset shape: (XXXX, 4)  # Shows number of ratings
```

### 2. Training Progress
For each epoch, you'll see:
```
Epoch 1/10:
Training Loss: X.XXXX
Validation Loss: X.XXXX

Epoch 2/10:
Training Loss: X.XXXX
Validation Loss: X.XXXX
...
```

### 3. Visualizations
The script will display:
- Training vs Validation loss plot
- Model will be saved as 'recommendation_model.pth'

## Customizing the Run

### Modifying Hyperparameters
Edit these values in the script:
```python
# Model hyperparameters
embedding_dim = 50
batch_size = 1024
learning_rate = 0.001
num_epochs = 10
```

### Using GPU/CPU
The script automatically detects and uses GPU if available:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch_size
   - Reduce embedding_dim
   ```python
   batch_size = 512  # Try smaller batch size
   ```

2. **Slow Training**
   - Increase batch_size if using GPU
   - Reduce number of epochs
   ```python
   num_epochs = 5  # Try fewer epochs
   ```

3. **Poor Performance**
   - Increase embedding_dim
   - Increase number of epochs
   - Adjust learning rate
   ```python
   embedding_dim = 64  # Try larger embedding
   num_epochs = 15    # Train for longer
   ```

## Saving and Loading Models

### Saving
The model is automatically saved after training:
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'user_mapping': user_mapping,
    'movie_mapping': movie_mapping,
    'optimizer_state_dict': optimizer.state_dict(),
}, 'recommendation_model.pth')
```

### Loading for Predictions
```python
# Load the saved model
checkpoint = torch.load('recommendation_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
user_mapping = checkpoint['user_mapping']
movie_mapping = checkpoint['movie_mapping']
```

## Monitoring Training

### Watch for:
1. Decreasing training and validation losses
2. Gap between training and validation loss
3. Any error messages in the console

### Good Training Signs:
- Validation loss decreases steadily
- Small gap between training and validation loss
- No sudden spikes in loss values 