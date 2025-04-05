# Running on Google Colab

This guide explains how to run the recommendation model using Google Colab, which provides free GPU access.

## Setup Steps

### 1. Create a New Colab Notebook
- Go to [Google Colab](https://colab.research.google.com)
- Create a new notebook

### 2. Install Dependencies
```python
!pip install torch pandas numpy scikit-learn matplotlib seaborn
```

### 3. Mount Google Drive (Optional)
If your data is stored in Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Clone the Repository
```python
!git clone https://github.com/yourusername/movie_recommendation.git
%cd movie_recommendation
```

### 5. Upload Data Files
Option 1 - Direct upload:
```python
from google.colab import files
uploaded = files.upload()  # Upload ratings_export.csv and users_export.csv
```

Option 2 - From Google Drive:
```python
# Assuming files are in your Google Drive
!cp /content/drive/MyDrive/path_to_data/ratings_export.csv data/
!cp /content/drive/MyDrive/path_to_data/users_export.csv data/
```

## Running the Model

### 1. Verify GPU Access
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### 2. Copy Model Files
Create all necessary Python files in Colab:

```python
%%writefile recommendation_model.py
# Paste the contents of recommendation_model.py here
```

```python
%%writefile data_cleaner.py
# Paste the contents of data_cleaner.py here
```

```python
%%writefile data_loader.py
# Paste the contents of data_loader.py here
```

### 3. Run the Training
```python
!python recommendation_model.py
```

## Colab-Specific Tips

### 1. Preventing Session Timeouts
Add this to your notebook to prevent disconnections:
```javascript
function ClickConnect(){
    console.log("Working"); 
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)
```

### 2. Saving Results
Save the model to Google Drive:
```python
model_path = '/content/drive/MyDrive/models/recommendation_model.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'user_mapping': user_mapping,
    'movie_mapping': movie_mapping,
    'optimizer_state_dict': optimizer.state_dict(),
}, model_path)
```

### 3. Monitoring Training
```python
# Display live training progress
from IPython.display import clear_output

def plot_losses(train_losses, val_losses):
    clear_output(wait=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.show()
```

## Advantages of Using Colab

1. **Free GPU Access**
   - Faster training with NVIDIA GPUs
   - No local GPU required

2. **Pre-installed Libraries**
   - Many ML libraries pre-installed
   - Easy to install additional packages

3. **Collaborative Features**
   - Share notebooks easily
   - Real-time collaboration

4. **Persistent Storage**
   - Save to Google Drive
   - Access from anywhere

## Limitations

1. **Session Limits**
   - Runtime disconnects after 12 hours
   - Idle timeout after 90 minutes

2. **Resource Limits**
   - Limited GPU memory
   - Shared GPU resources

3. **Storage Constraints**
   - Limited disk space
   - Need to reupload data each session

## Best Practices

1. **Save Checkpoints**
```python
# Save checkpoints periodically
if epoch % 5 == 0:
    checkpoint_path = f'/content/drive/MyDrive/checkpoints/model_epoch_{epoch}.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, checkpoint_path)
```

2. **Monitor GPU Memory**
```python
# Check GPU memory usage
!nvidia-smi
```

3. **Clear Memory**
```python
import gc
torch.cuda.empty_cache()
gc.collect()
```

## Model Saving and Loading

### Setting Up Storage

Before saving models, set up your Google Drive storage:

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Create directories for models and checkpoints
!mkdir -p /content/drive/MyDrive/models
!mkdir -p /content/drive/MyDrive/checkpoints
```

### Saving Models

There are several ways to save your model:

1. **Basic Model Saving**
```python
# Save just the model state
model_path = '/content/drive/MyDrive/models/model_basic.pth'
torch.save(model.state_dict(), model_path)
```

2. **Complete Checkpoint Saving**
```python
# Save complete training state
checkpoint_path = '/content/drive/MyDrive/models/model_complete.pth'
checkpoint = {
    'epoch': current_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss_history,
    'val_loss': val_loss_history,
    'user_mapping': user_mapping,
    'movie_mapping': movie_mapping,
    'hyperparameters': {
        'embedding_dim': embedding_dim,
        'learning_rate': learning_rate,
        'batch_size': batch_size
    }
}
torch.save(checkpoint, checkpoint_path)
```

3. **Automatic Periodic Saving**
```python
def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, is_best=False):
    # Regular checkpoint
    checkpoint_path = f'/content/drive/MyDrive/checkpoints/model_epoch_{epoch}.pth'
    
    # Best model checkpoint
    best_model_path = '/content/drive/MyDrive/models/best_model.pth'
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    
    # Save regular checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model if this is the best performance
    if is_best:
        torch.save(checkpoint, best_model_path)
        print(f"Saved new best model with validation loss: {val_loss:.4f}")

# Use during training
best_val_loss = float('inf')
for epoch in range(num_epochs):
    # ... training code ...
    
    # Save checkpoint every 5 epochs
    if epoch % 5 == 0:
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss)
    
    # Save best model when validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, is_best=True)
```

### Loading Models

1. **Loading Basic Model**
```python
# Initialize model first
model = CollaborativeFiltering(num_users, num_movies, embedding_dim)
model.load_state_dict(torch.load('/content/drive/MyDrive/models/model_basic.pth'))
model.eval()  # Set to evaluation mode
```

2. **Loading Complete Checkpoint**
```python
# Initialize model and optimizer first
model = CollaborativeFiltering(num_users, num_movies, embedding_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Load checkpoint
checkpoint = torch.load('/content/drive/MyDrive/models/model_complete.pth')

# Restore state
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
train_loss_history = checkpoint['train_loss']
val_loss_history = checkpoint['val_loss']
user_mapping = checkpoint['user_mapping']
movie_mapping = checkpoint['movie_mapping']

# Set model to evaluation mode if you're going to make predictions
model.eval()
```

3. **Resuming Training from Checkpoint**
```python
def load_checkpoint_and_resume(checkpoint_path):
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Starting from scratch.")
        return None
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Initialize model and optimizer
    model = CollaborativeFiltering(
        num_users=len(checkpoint['user_mapping']),
        num_movies=len(checkpoint['movie_mapping']),
        embedding_dim=checkpoint['hyperparameters']['embedding_dim']
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=checkpoint['hyperparameters']['learning_rate']
    )
    
    # Load state
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Resumed from epoch {checkpoint['epoch']}")
    return model, optimizer, checkpoint

# Usage
checkpoint_path = '/content/drive/MyDrive/models/model_complete.pth'
loaded_model, loaded_optimizer, checkpoint = load_checkpoint_and_resume(checkpoint_path)
```

### Best Practices for Model Saving

1. **Regular Checkpointing**
   - Save checkpoints every few epochs
   - Keep track of best performing models
   - Include all necessary information for resuming training

2. **Version Control**
```python
# Save with timestamp
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = f'/content/drive/MyDrive/models/model_{timestamp}.pth'
torch.save(checkpoint, model_path)
```

3. **Cleanup Old Checkpoints**
```python
def cleanup_old_checkpoints(directory, keep_last_n=5):
    """Keep only the n most recent checkpoints"""
    checkpoints = sorted(
        [f for f in os.listdir(directory) if f.endswith('.pth')],
        key=lambda x: os.path.getmtime(os.path.join(directory, x))
    )
    
    # Remove old checkpoints
    for checkpoint in checkpoints[:-keep_last_n]:
        os.remove(os.path.join(directory, checkpoint))
        print(f"Removed old checkpoint: {checkpoint}")
```

4. **Verify Saved Models**
```python
def verify_saved_model(model_path):
    """Verify that the saved model can be loaded"""
    try:
        checkpoint = torch.load(model_path)
        print("Model loaded successfully!")
        print(f"Saved at epoch: {checkpoint['epoch']}")
        print(f"Training loss: {checkpoint['train_loss']:.4f}")
        print(f"Validation loss: {checkpoint['val_loss']:.4f}")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

# Use after saving
verify_saved_model('/content/drive/MyDrive/models/model_complete.pth')
```

### Common Issues and Solutions

1. **Out of Memory**
```python
# If you run out of memory while saving large models
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, model_path, _use_new_zipfile_serialization=False)  # Use old PyTorch format
```

2. **Drive Space Management**
```python
# Check available space
from google.colab import drive
drive.mount('/content/drive')
!df -h /content/drive/MyDrive
```

3. **Loading Models on Different Devices**
```python
# Load model to specific device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
``` 