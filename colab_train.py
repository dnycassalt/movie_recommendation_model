import os
import torch
import torch.nn as nn
from datetime import datetime
from google.colab import drive
import matplotlib.pyplot as plt
from IPython.display import clear_output
import shutil

from recommendation_model import CollaborativeFiltering
from data_loader import DataLoader


class ColabTrainer:
    def __init__(self, embedding_dim=50, learning_rate=0.001, batch_size=1024, num_epochs=10):
        # Mount Google Drive
        drive.mount('/content/drive')

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

        # Initialize lists to store losses
        self.train_losses = []
        self.val_losses = []

        # Set device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def load_data(self):
        """Load and prepare data"""
        data_loader = DataLoader()
        train_data, val_data = data_loader.load_and_split_data()
        self.user_mapping = data_loader.user_mapping
        self.movie_mapping = data_loader.movie_mapping

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

    def plot_losses(self):
        """Plot training and validation losses"""
        clear_output(wait=True)
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.show()

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

                total_train_loss += loss.item()
                num_batches += 1

            avg_train_loss = total_train_loss / num_batches
            self.train_losses.append(avg_train_loss)

            # Validation phase
            self.model.eval()
            total_val_loss = 0
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

                    total_val_loss += loss.item()
                    num_val_batches += 1

            avg_val_loss = total_val_loss / num_val_batches
            self.val_losses.append(avg_val_loss)

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
            print(f"Validation Loss: {avg_val_loss:.4f}")
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
