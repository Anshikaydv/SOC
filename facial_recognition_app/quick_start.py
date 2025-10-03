"""
Quick start training script for the facial recognition system.
This script can work with minimal data or create synthetic data for testing.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import json
from typing import List, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model.siamese_model import SiameseNetwork, ContrastiveLoss
except ImportError:
    print("Error: Could not import model. Make sure you're in the correct directory.")
    sys.exit(1)


class MinimalDataset(Dataset):
    """Minimal dataset for testing with synthetic data."""
    
    def __init__(self, num_samples=100, image_size=(224, 224)):
        self.num_samples = num_samples
        self.image_size = image_size
        self.data = self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """Create synthetic image pairs for testing."""
        data = []
        
        for i in range(self.num_samples):
            # Create two random images
            img1 = torch.randn(3, *self.image_size)
            
            # 50% chance of same person (positive pair)
            if i % 2 == 0:
                # Same person - add small noise to img1
                img2 = img1 + torch.randn(3, *self.image_size) * 0.1
                label = 1
            else:
                # Different person - completely different image
                img2 = torch.randn(3, *self.image_size)
                label = 0
            
            data.append((img1, img2, label))
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img1, img2, label = self.data[idx]
        return img1, img2, torch.tensor(label, dtype=torch.float32)


def create_dummy_training_data():
    """Create dummy training data to test the system."""
    print("Creating synthetic training data...")
    
    # Create train and validation datasets
    train_dataset = MinimalDataset(num_samples=200)
    val_dataset = MinimalDataset(num_samples=50)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    return train_loader, val_loader


def quick_train_model(num_epochs=10):
    """Quick training function with minimal epochs."""
    print("🚀 Starting quick model training...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = SiameseNetwork(embedding_dim=128)
    model.to(device)
    
    # Create optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = ContrastiveLoss(margin=1.0)
    
    # Get training data
    train_loader, val_loader = create_dummy_training_data()
    
    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (img1, img2, labels) in enumerate(train_loader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                
                emb1, emb2 = model(img1, img2)
                loss = criterion(emb1, emb2, labels)
                val_loss += loss.item()
                num_val_batches += 1
                
                # Calculate accuracy (using simple distance threshold)
                distances = torch.nn.functional.pairwise_distance(emb1, emb2)
                predictions = (distances < 1.0).float()  # Simple threshold
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = val_loss / num_val_batches
        val_accuracy = correct / total
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        print("-" * 40)
    
    # Save the model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, "model", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model_save_path = os.path.join(checkpoint_dir, "best_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracies[-1],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
    }, model_save_path)
    
    print(f"✅ Model saved to: {model_save_path}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
    }
    
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✅ Training history saved to: {history_path}")
    print(f"🎉 Quick training completed! Final accuracy: {val_accuracies[-1]:.4f}")
    
    return model, history


def main():
    """Main function."""
    print("🔧 Quick Start Training for Facial Recognition System")
    print("=" * 60)
    print()
    print("This script will create a basic trained model so you can test the system.")
    print("For real use, you should collect actual face images using the web interface.")
    print()
    
    # Check if model already exists
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model", "checkpoints", "best_model.pth")
    if os.path.exists(model_path):
        print(f"⚠️  Model already exists at: {model_path}")
        response = input("Do you want to retrain? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Quick training
    try:
        model, history = quick_train_model(num_epochs=15)
        
        print()
        print("🎯 Next Steps:")
        print("1. Refresh your Streamlit app - the model should now load!")
        print("2. Use 'Register New User' to collect real face images")
        print("3. Retrain with real data using: python train.py")
        print("4. Test face verification in the web interface")
        print()
        print("The current model uses synthetic data, so verification won't work")
        print("with real faces until you train with actual face images.")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
