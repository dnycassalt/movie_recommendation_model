# Training with Google Colab

This guide explains how to use our custom `colab_train.py` script to train the recommendation model using Google Colab's free GPU resources.

## Setup Instructions

### 1. Create a New Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Select Runtime > Change runtime type and choose GPU as the hardware accelerator

### 2. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

This will prompt you to authorize Colab to access your Google Drive. Follow the authorization steps when prompted.

### 3. Set Up Project Files

Clone the repository and copy necessary files:

```python
# Clone the repository
!git clone https://github.com/yourusername/movie_recommendation.git
%cd movie_recommendation

# Create the colab_train.py script
%%writefile colab_train.py
# [Paste the contents of colab_train.py here]
```

### 4. Copy Data Files

Ensure your data files are accessible. You have two options:

Option 1 - Upload through Colab:
```python
from google.colab import files
uploaded = files.upload()  # Upload ratings_export.csv and users_export.csv
```

Option 2 - Copy from Google Drive:
```python
# Assuming files are in your Google Drive
!cp /content/drive/MyDrive/path_to_data/ratings_export.csv data/
!cp /content/drive/MyDrive/path_to_data/users_export.csv data/
```

## Training the Model

### Basic Training

To start training with default parameters:

```python
from colab_train import ColabTrainer

trainer = ColabTrainer(
    embedding_dim=50,
    learning_rate=0.001,
    batch_size=1024,
    num_epochs=10
)

trainer.train()
```

### Customizing Training Parameters

You can customize the training by adjusting the parameters:

```python
trainer = ColabTrainer(
    embedding_dim=100,     # Larger embedding dimension
    learning_rate=0.0005,  # Slower learning rate
    batch_size=2048,      # Larger batch size
    num_epochs=20         # More epochs
)
```

### Resuming Training

To resume training from a previous checkpoint:

```python
trainer = ColabTrainer()
trainer.train(resume_from='/content/drive/MyDrive/checkpoints/previous_checkpoint.pth')
```

## Model Storage

The trainer automatically manages model storage in your Google Drive:

1. **Regular Checkpoints**: 
   - Saved every 5 epochs
   - Located in `/content/drive/MyDrive/checkpoints/`
   - Named `checkpoint_epoch_{epoch}_{timestamp}.pth`

2. **Best Model**:
   - Saved whenever validation loss improves
   - Located in `/content/drive/MyDrive/models/best_model.pth`

Each checkpoint contains:
- Model state
- Optimizer state
- Training history
- User and movie mappings
- Hyperparameters

## Monitoring Training

The script provides real-time monitoring:

1. **Loss Plots**: 
   - Updates after each epoch
   - Shows training and validation loss
   - Automatically clears and updates in the notebook

2. **Console Output**:
   - Current epoch progress
   - Training loss
   - Validation loss
   - Best model saves

## Managing Resources

### Checkpoint Cleanup

The trainer automatically manages checkpoints:
- Keeps the 5 most recent checkpoints
- Automatically removes older checkpoints
- Always preserves the best model

### Memory Management

If you encounter memory issues:
```python
# Clear GPU memory
import torch
torch.cuda.empty_cache()

# Clear other memory
import gc
gc.collect()
```

## Best Practices

1. **Save Frequently**:
   - Default checkpoint saving every 5 epochs
   - Best model saved automatically
   - All saved to Google Drive for persistence

2. **Monitor Resources**:
   - Watch GPU memory usage
   - Keep track of Drive storage space
   - Use appropriate batch sizes for your GPU

3. **Prevent Session Timeouts**:
```javascript
function ClickConnect(){
    console.log("Working"); 
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)
```

## Troubleshooting

1. **Drive Mount Issues**:
   - Ensure you've authorized Colab
   - Try remounting the drive
   - Check path permissions

2. **GPU Memory Errors**:
   - Reduce batch size
   - Clear memory cache
   - Restart runtime

3. **Training Interruption**:
   - Use the resume feature
   - Check the latest checkpoint
   - Verify saved model integrity 