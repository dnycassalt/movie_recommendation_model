# Add matplotlib inline magic command at the top
from data_loader import DataLoader
from recommendation_model import CollaborativeFiltering
from sklearn.model_selection import train_test_split
import pandas as pd
import shutil
from IPython.display import clear_output
import matplotlib.pyplot as plt
from google.colab import drive
from datetime import datetime
import torch.nn as nn
import torch
import os
%matplotlib inline


class ColabTrainer:
    def __init__(self, embedding_dim=50, learning_rate=0.001, batch_size=1024, num_epochs=10):
        # Mount Google Drive
       # drive.mount('/content/drive')

        # Create necessary directories
        self.model_dir = '/content/drive/MyDrive/movie_recommendation_data/models'
        self.checkpoint_dir = '/content/drive/MyDrive/movie_recommendation_data/checkpoints'
        self.backup_dir = '/content/drive/MyDrive/movie_recommendation_data/backups'

        for dir_path in [self.model_dir, self.checkpoint_dir, self.backup_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Training parameters
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Initialize lists to store metrics
        self.train_losses = []
        self.val_losses = []
        self.train_precisions = []
        self.train_recalls = []
        self.val_precisions = []
        self.val_recalls = []

        # Set device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def load_data(self):
        """Load and prepare data"""
        # Define ratings file path
        ratings_file = '/content/drive/MyDrive/movie_recommendation_data/ratings_export.csv'

        # Initialize data loader with file paths
        data_loader = DataLoader(
            file_path=ratings_file,
            checkpoint_file='ratings_data_checkpoint.pkl',
            chunk_size=20050,
            checkpoint_interval=500000
        )

        # Load the data
        df = data_loader.load_data(
            encoding='utf-8',
            on_bad_lines='skip',
            low_memory=False,
            lineterminator='\n'
        )

        # Create mappings from user/movie IDs to numerical indices
        self.user_mapping = {uid: idx for idx,
                             uid in enumerate(df['user_id'].unique())}
        self.movie_mapping = {mid: idx for idx,
                              mid in enumerate(df['movie_id'].unique())}

        # Convert data to list of tuples (user_idx, movie_idx, rating)
        data = [
            (self.user_mapping[row['user_id']],
             self.movie_mapping[row['movie_id']],
             row['rating_val'])
            for _, row in df.iterrows()
        ]

        # Split data into training and validation sets (80/20 split)
        train_data, val_data = train_test_split(
            data, test_size=0.2, random_state=42)

        return train_data, val_data

    def initialize_model(self):
        """Initialize the model and optimizer"""
        self.model = CollaborativeFiltering(
            num_users=len(self.user_mapping),
            num_movies=len(self.movie_mapping),
            embedding_dim=self.embedding_dim
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        """Save model checkpoint with versioning and verification"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_loss_history': self.train_losses,
            'val_loss_history': self.val_losses,
            'user_mapping': self.user_mapping,
            'movie_mapping': self.movie_mapping,
            'hyperparameters': {
                'embedding_dim': self.embedding_dim,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size
            },
            'timestamp': timestamp,
            'device': str(self.device)
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f'checkpoint_epoch_{epoch}_{timestamp}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save best model if this is the best performance
        if is_best:
            # Create versioned best model
            best_model_path = os.path.join(
                self.model_dir, f'best_model_{timestamp}.pth')
            torch.save(checkpoint, best_model_path)

            # Update latest best model
            latest_best_path = os.path.join(self.model_dir, 'best_model.pth')
            shutil.copy2(best_model_path, latest_best_path)

            print(f"Saved new best model with validation loss: {val_loss:.4f}")

            # Create backup of best model
            backup_path = os.path.join(
                self.backup_dir, f'best_model_backup_{timestamp}.pth')
            shutil.copy2(best_model_path, backup_path)
            print(f"Created backup at {backup_path}")

        # Verify the saved checkpoint
        self.verify_checkpoint(checkpoint_path)

    def verify_checkpoint(self, checkpoint_path):
        """Verify that the saved checkpoint can be loaded and contains all required data"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Check required keys
            required_keys = [
                'model_state_dict', 'optimizer_state_dict', 'epoch',
                'train_loss', 'val_loss', 'user_mapping', 'movie_mapping'
            ]
            missing_keys = [
                key for key in required_keys if key not in checkpoint]

            if missing_keys:
                print(f"Warning: Checkpoint missing keys: {missing_keys}")
                return False

            # Verify model can be loaded
            test_model = CollaborativeFiltering(
                num_users=len(checkpoint['user_mapping']),
                num_movies=len(checkpoint['movie_mapping']),
                embedding_dim=checkpoint['hyperparameters']['embedding_dim']
            ).to(self.device)

            test_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Successfully verified checkpoint at {checkpoint_path}")
            return True

        except Exception as e:
            print(f"Error verifying checkpoint: {str(e)}")
            return False

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        if not os.path.exists(checkpoint_path):
            print("No checkpoint found. Starting from scratch.")
            return 0, float('inf')

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load training history
        self.train_losses = checkpoint['train_loss_history']
        self.val_losses = checkpoint['val_loss_history']

        print(f"Resumed from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'], checkpoint['val_loss']

    def calculate_precision_recall(self, predictions, ratings, threshold=3.5):
        """Calculate precision and recall for a batch of predictions"""
        # Convert predictions and ratings to binary (like/dislike)
        pred_binary = (predictions >= threshold).float()
        true_binary = (ratings >= threshold).float()

        # Calculate true positives, false positives, and false negatives
        true_positives = (pred_binary * true_binary).sum().item()
        false_positives = (pred_binary * (1 - true_binary)).sum().item()
        false_negatives = ((1 - pred_binary) * true_binary).sum().item()

        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)

        return precision, recall

    def plot_losses(self):
        """Plot training and validation losses and metrics"""
        clear_output(wait=True)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot losses
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training History')
        ax1.legend()
        ax1.grid(True)

        # Plot precision and recall
        ax2.plot(self.train_precisions,
                 label='Training Precision', color='green')
        ax2.plot(self.train_recalls, label='Training Recall', color='blue')
        ax2.plot(self.val_precisions, label='Validation Precision',
                 color='green', linestyle='--')
        ax2.plot(self.val_recalls, label='Validation Recall',
                 color='blue', linestyle='--')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision and Recall')
        ax2.legend()
        ax2.grid(True)

        # Use IPython's display to show the plot
        from IPython.display import display
        display(fig)
        plt.close(fig)  # Close the figure to free memory

    def cleanup_old_checkpoints(self, keep_last_n=5):
        """Keep only the n most recent checkpoints and backups"""
        # Clean up checkpoints
        checkpoints = sorted(
            [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')],
            key=lambda x: os.path.getmtime(
                os.path.join(self.checkpoint_dir, x))
        )

        # Clean up backups
        backups = sorted(
            [f for f in os.listdir(self.backup_dir) if f.endswith('.pth')],
            key=lambda x: os.path.getmtime(os.path.join(self.backup_dir, x))
        )

        # Remove old checkpoints
        for checkpoint in checkpoints[:-keep_last_n]:
            os.remove(os.path.join(self.checkpoint_dir, checkpoint))
            print(f"Removed old checkpoint: {checkpoint}")

        # Keep only the 3 most recent backups
        for backup in backups[:-3]:
            os.remove(os.path.join(self.backup_dir, backup))
            print(f"Removed old backup: {backup}")

    def train(self, resume_from=None):
        """Main training loop"""
        print("Loading data...")
        train_data, val_data = self.load_data()

        print("Initializing model...")
        self.initialize_model()

        # Resume from checkpoint if specified
        start_epoch = 0
        best_val_loss = float('inf')
        if resume_from:
            start_epoch, best_val_loss = self.load_checkpoint(resume_from)

        print("Starting training...")
        for epoch in range(start_epoch, self.num_epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            total_train_precision = 0
            total_train_recall = 0
            num_batches = 0

            for i in range(0, len(train_data), self.batch_size):
                batch = train_data[i:i + self.batch_size]
                users = torch.tensor([x[0] for x in batch]).to(self.device)
                movies = torch.tensor([x[1] for x in batch]).to(self.device)
                ratings = torch.tensor(
                    [x[2] for x in batch], dtype=torch.float32).to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(users, movies)
                loss = self.criterion(predictions, ratings)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Calculate metrics
                precision, recall = self.calculate_precision_recall(
                    predictions, ratings)
                total_train_precision += precision
                total_train_recall += recall
                total_train_loss += loss.item()
                num_batches += 1

            avg_train_loss = total_train_loss / num_batches
            avg_train_precision = total_train_precision / num_batches
            avg_train_recall = total_train_recall / num_batches
            self.train_losses.append(avg_train_loss)
            self.train_precisions.append(avg_train_precision)
            self.train_recalls.append(avg_train_recall)

            # Validation phase
            self.model.eval()
            total_val_loss = 0
            total_val_precision = 0
            total_val_recall = 0
            num_val_batches = 0

            with torch.no_grad():
                for i in range(0, len(val_data), self.batch_size):
                    batch = val_data[i:i + self.batch_size]
                    users = torch.tensor([x[0] for x in batch]).to(self.device)
                    movies = torch.tensor([x[1]
                                          for x in batch]).to(self.device)
                    ratings = torch.tensor(
                        [x[2] for x in batch], dtype=torch.float32).to(self.device)

                    predictions = self.model(users, movies)
                    loss = self.criterion(predictions, ratings)

                    # Calculate metrics
                    precision, recall = self.calculate_precision_recall(
                        predictions, ratings)
                    total_val_precision += precision
                    total_val_recall += recall
                    total_val_loss += loss.item()
                    num_val_batches += 1

            avg_val_loss = total_val_loss / num_val_batches
            avg_val_precision = total_val_precision / num_val_batches
            avg_val_recall = total_val_recall / num_val_batches
            self.val_losses.append(avg_val_loss)
            self.val_precisions.append(avg_val_precision)
            self.val_recalls.append(avg_val_recall)

            # Plot progress
            self.plot_losses()

            # Save checkpoint
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss

            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, avg_train_loss,
                                     avg_val_loss, is_best)
                self.cleanup_old_checkpoints()

            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Training Precision: {avg_train_precision:.4f}")
            print(f"Training Recall: {avg_train_recall:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Validation Precision: {avg_val_precision:.4f}")
            print(f"Validation Recall: {avg_val_recall:.4f}")
            print("-" * 50)


if __name__ == "__main__":
    # Initialize trainer with hyperparameters
    trainer = ColabTrainer(
        embedding_dim=50,
        learning_rate=0.001,
        batch_size=1024,
        num_epochs=10
    )

    # Start training (or resume from a checkpoint)
    # trainer.train(resume_from='/content/drive/MyDrive/checkpoints/previous_checkpoint.pth')
    trainer.train()
