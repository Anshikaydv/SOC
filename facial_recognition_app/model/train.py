import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import os
import json
from typing import Dict, List, Tuple, Optional
import time

try:
    from siamese_model import SiameseNetwork, ContrastiveLoss, TripletLoss, cosine_similarity_score
except ImportError:
    from .siamese_model import SiameseNetwork, ContrastiveLoss, TripletLoss, cosine_similarity_score

try:
    from utils.data_loader import SiamesePairDataset, get_data_loaders
except ImportError:
    try:
        from ..utils.data_loader import SiamesePairDataset, get_data_loaders
    except ImportError:
        # Handle case when running as standalone script
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.data_loader import SiamesePairDataset, get_data_loaders


class SiameseTrainer:
    """
    Trainer class for Siamese Neural Network.
    """
    
    def __init__(
        self, 
        model: SiameseNetwork,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 0.001,
        loss_type: str = "contrastive",
        margin: float = 1.0,
        device: str = "auto"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Siamese network model
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate for optimizer
            loss_type: Type of loss ("contrastive" or "triplet")
            margin: Margin for loss function
            device: Device to use for training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # Initialize loss function
        if loss_type == "contrastive":
            self.criterion = ContrastiveLoss(margin=margin)
        elif loss_type == "triplet":
            self.criterion = TripletLoss(margin=margin)
        else:
            raise ValueError("loss_type must be 'contrastive' or 'triplet'")
        
        self.loss_type = loss_type
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_aucs = []
        
    def train_epoch(self) -> float:
        """
        Train the model for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            if self.loss_type == "contrastive":
                img1, img2, labels = batch
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                
                # Forward pass
                emb1, emb2 = self.model(img1, img2)
                loss = self.criterion(emb1, emb2, labels.float())
                
            elif self.loss_type == "triplet":
                anchor, positive, negative = batch
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                # Forward pass
                anchor_emb = self.model.forward_one(anchor)
                positive_emb = self.model.forward_one(positive)
                negative_emb = self.model.forward_one(negative)
                
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        return running_loss / num_batches
    
    def validate(self) -> Tuple[float, float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, accuracy, auc_score)
        """
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_similarities = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                if self.loss_type == "contrastive":
                    img1, img2, labels = batch
                    img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    emb1, emb2 = self.model(img1, img2)
                    loss = self.criterion(emb1, emb2, labels.float())
                    
                    # Calculate similarity scores
                    similarities = cosine_similarity_score(emb1, emb2)
                    
                    all_labels.extend(labels.cpu().numpy())
                    all_similarities.extend(similarities.cpu().numpy())
                    
                running_loss += loss.item()
                num_batches += 1
        
        avg_loss = running_loss / num_batches
        
        # Calculate metrics
        all_labels = np.array(all_labels)
        all_similarities = np.array(all_similarities)
        
        # For accuracy, use threshold of 0.5 on similarity scores
        predictions = (all_similarities > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, predictions)
        
        # Calculate AUC
        auc_score = roc_auc_score(all_labels, all_similarities)
        
        return avg_loss, accuracy, auc_score
    
    def train(self, num_epochs: int, save_path: str = "model_checkpoints") -> Dict:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_path: Path to save model checkpoints
            
        Returns:
            Training history dictionary
        """
        os.makedirs(save_path, exist_ok=True)
        
        best_auc = 0.0
        best_model_path = os.path.join(save_path, "best_model.pth")
        
        print(f"Training on device: {self.device}")
        print(f"Total epochs: {num_epochs}")
        print("-" * 50)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss, val_accuracy, val_auc = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            self.val_aucs.append(val_auc)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.2f}s)")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            print(f"Val AUC: {val_auc:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 50)
            
            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_accuracy': val_accuracy,
                }, best_model_path)
                print(f"New best model saved with AUC: {val_auc:.4f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'val_auc': val_auc,
                }, checkpoint_path)
        
        # Save final model
        final_model_path = os.path.join(save_path, "final_model.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_aucs': self.val_aucs,
        }, final_model_path)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_aucs': self.val_aucs,
        }
        
        with open(os.path.join(save_path, "training_history.json"), 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.val_accuracies, label='Val Accuracy', color='green')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC plot
        axes[1, 0].plot(self.val_aucs, label='Val AUC', color='red')
        axes[1, 0].set_title('Validation AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Combined metrics
        axes[1, 1].plot(self.val_accuracies, label='Accuracy', color='green')
        axes[1, 1].plot(self.val_aucs, label='AUC', color='red')
        axes[1, 1].set_title('Validation Metrics')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def main():
    """
    Main training script.
    """
    # Configuration
    config = {
        'data_dir': '../data/processed',
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'embedding_dim': 128,
        'loss_type': 'contrastive',  # or 'triplet'
        'margin': 1.0,
        'train_split': 0.8,
        'image_size': (224, 224),
        'num_workers': 4,
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 50)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        train_split=config['train_split'],
        image_size=config['image_size'],
        num_workers=config['num_workers'],
        loss_type=config['loss_type']
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print("-" * 50)
    
    # Initialize model
    model = SiameseNetwork(embedding_dim=config['embedding_dim'])
    
    # Initialize trainer
    trainer = SiameseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['learning_rate'],
        loss_type=config['loss_type'],
        margin=config['margin']
    )
    
    # Train model
    history = trainer.train(
        num_epochs=config['num_epochs'],
        save_path='../model/checkpoints'
    )
    
    # Plot training history
    trainer.plot_training_history(save_path='../model/training_history.png')
    
    print("Training completed!")
    print(f"Best validation AUC: {max(history['val_aucs']):.4f}")
    print(f"Final validation accuracy: {history['val_accuracies'][-1]:.4f}")


if __name__ == "__main__":
    main()
