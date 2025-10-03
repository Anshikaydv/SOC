#!/usr/bin/env python3
"""
Improved model retraining script to fix false positive issues.
This script addresses the core problem of insufficient training data and poor negative sampling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import glob
import random
import json
from typing import List, Tuple, Dict, Optional
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from model.siamese_model import SiameseNetwork, ContrastiveLoss, cosine_similarity_score
from utils.image_utils import FaceDetector, ImagePreprocessor
from utils.data_loader import SiamesePairDataset


class ImprovedDataAugmentation:
    """Enhanced data augmentation for better model training."""
    
    def __init__(self):
        self.augmentation_transforms = [
            transforms.RandomRotation((-15, 15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        ]
    
    def augment_user_images(self, user_dir: str, target_count: int = 20) -> int:
        """
        Augment images for a user to reach target count.
        
        Args:
            user_dir: Directory containing user images
            target_count: Target number of images per user
            
        Returns:
            Number of augmented images created
        """
        if not os.path.exists(user_dir):
            return 0
        
        # Get existing images
        existing_images = glob.glob(os.path.join(user_dir, "*.jpg")) + \
                         glob.glob(os.path.join(user_dir, "*.png"))
        
        if len(existing_images) >= target_count:
            return 0
        
        needed_count = target_count - len(existing_images)
        created_count = 0
        
        print(f"Augmenting {user_dir}: {len(existing_images)} -> {target_count} images")
        
        for i in range(needed_count):
            # Select random source image
            source_img_path = random.choice(existing_images)
            
            try:
                # Load and augment image
                image = Image.open(source_img_path).convert('RGB')
                
                # Apply random augmentation
                augment_transform = random.choice(self.augmentation_transforms)
                augmented_image = augment_transform(image)
                
                # Save augmented image
                base_name = os.path.splitext(os.path.basename(source_img_path))[0]
                aug_name = f"{base_name}_aug_{i+1}_{int(time.time())}.jpg"
                aug_path = os.path.join(user_dir, aug_name)
                
                augmented_image.save(aug_path, quality=95)
                created_count += 1
                
            except Exception as e:
                print(f"Error augmenting {source_img_path}: {e}")
                continue
        
        return created_count


class NegativeDataGenerator:
    """Generate negative data for training when insufficient users exist."""
    
    def __init__(self, face_detector: FaceDetector):
        self.face_detector = face_detector
        
    def create_synthetic_negative_user(self, data_dir: str, negative_user_name: str = "negative_samples") -> str:
        """
        Create synthetic negative user from random face images.
        
        Args:
            data_dir: Base data directory
            negative_user_name: Name for the negative user directory
            
        Returns:
            Path to created negative user directory
        """
        negative_dir = os.path.join(data_dir, negative_user_name)
        os.makedirs(negative_dir, exist_ok=True)
        
        # Download or generate random face images
        # For now, we'll create placeholder instructions
        readme_path = os.path.join(negative_dir, "README.txt")
        with open(readme_path, 'w') as f:
            f.write("NEGATIVE SAMPLES DIRECTORY\n")
            f.write("========================\n\n")
            f.write("This directory should contain face images of people who are NOT authorized users.\n")
            f.write("To improve model accuracy:\n\n")
            f.write("1. Add 10-20 images of different people's faces\n")
            f.write("2. Use clear, well-lit photos\n")
            f.write("3. Include diverse ages, genders, and ethnicities\n")
            f.write("4. Avoid images similar to your authorized users\n\n")
            f.write("Supported formats: .jpg, .jpeg, .png\n")
            f.write("Recommended image size: 224x224 or larger\n")
        
        print(f"Created negative samples directory: {negative_dir}")
        print("Please add negative sample images as instructed in the README.txt file.")
        
        return negative_dir


class ImprovedTrainer:
    """Improved trainer with better validation and metrics."""
    
    def __init__(self, model: SiameseNetwork, device: torch.device):
        self.model = model
        self.device = device
        self.face_detector = FaceDetector()
        self.augmenter = ImprovedDataAugmentation()
        self.negative_generator = NegativeDataGenerator(self.face_detector)
        
    def prepare_enhanced_dataset(self, data_dir: str) -> Tuple[List[Tuple], int, int]:
        """
        Prepare enhanced dataset with proper negative sampling.
        
        Args:
            data_dir: Directory containing user data
            
        Returns:
            Tuple of (pairs, num_positive, num_negative)
        """
        print("Preparing enhanced dataset...")
        
        # Get user directories
        user_dirs = [d for d in os.listdir(data_dir) 
                    if os.path.isdir(os.path.join(data_dir, d)) and d != "negative_samples"]
        
        print(f"Found {len(user_dirs)} user directories: {user_dirs}")
        
        # Augment existing user data
        for user_dir in user_dirs:
            user_path = os.path.join(data_dir, user_dir)
            self.augmenter.augment_user_images(user_path, target_count=20)
        
        # Create negative samples if needed
        if len(user_dirs) < 2:
            print("Insufficient users for negative pair generation. Creating synthetic negative samples...")
            self.negative_generator.create_synthetic_negative_user(data_dir)
            
            # Check if negative samples were added
            negative_dir = os.path.join(data_dir, "negative_samples")
            negative_images = glob.glob(os.path.join(negative_dir, "*.jpg")) + \
                            glob.glob(os.path.join(negative_dir, "*.png"))
            
            if len(negative_images) == 0:
                print("WARNING: No negative samples found. Model may still have false positive issues.")
                print("Please add negative sample images to improve accuracy.")
        
        # Generate pairs
        pairs = self._generate_enhanced_pairs(data_dir)
        
        # Count positive and negative pairs
        positive_pairs = [p for p in pairs if p[2] == 1]
        negative_pairs = [p for p in pairs if p[2] == 0]
        
        print(f"Generated {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs")
        
        return pairs, len(positive_pairs), len(negative_pairs)
    
    def _generate_enhanced_pairs(self, data_dir: str, max_pairs_per_combo: int = 10) -> List[Tuple]:
        """Generate enhanced training pairs with better balancing."""
        pairs = []
        
        # Get all user directories
        user_dirs = [d for d in os.listdir(data_dir) 
                    if os.path.isdir(os.path.join(data_dir, d))]
        
        if len(user_dirs) < 2:
            print("ERROR: Need at least 2 user directories (including negative samples)")
            return pairs
        
        # Get images for each user
        user_images = {}
        for user in user_dirs:
            user_path = os.path.join(data_dir, user)
            images = glob.glob(os.path.join(user_path, "*.jpg")) + \
                    glob.glob(os.path.join(user_path, "*.png"))
            if len(images) >= 2:  # Need at least 2 images
                user_images[user] = images
        
        users = list(user_images.keys())
        print(f"Users with sufficient images: {users}")
        
        # Generate positive pairs (same person)
        positive_pairs = []
        for user in users:
            user_imgs = user_images[user]
            # Limit positive pairs to prevent imbalance
            pair_count = 0
            for i in range(len(user_imgs)):
                for j in range(i + 1, len(user_imgs)):
                    if pair_count < max_pairs_per_combo:
                        positive_pairs.append((user_imgs[i], user_imgs[j], 1))
                        pair_count += 1
        
        # Generate negative pairs (different persons)
        negative_pairs = []
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                user1_imgs = user_images[users[i]]
                user2_imgs = user_images[users[j]]
                
                # Limit negative pairs
                pair_count = 0
                for img1 in user1_imgs:
                    for img2 in user2_imgs:
                        if pair_count < max_pairs_per_combo:
                            negative_pairs.append((img1, img2, 0))
                            pair_count += 1
        
        # Balance the dataset
        min_pairs = min(len(positive_pairs), len(negative_pairs))
        if min_pairs > 0:
            selected_positive = random.sample(positive_pairs, min_pairs)
            selected_negative = random.sample(negative_pairs, min_pairs)
            pairs = selected_positive + selected_negative
        
        random.shuffle(pairs)
        return pairs
    
    def train_with_validation(self, train_loader: DataLoader, val_loader: DataLoader, 
                            epochs: int = 50, learning_rate: float = 0.001) -> Dict:
        """
        Train model with proper validation and early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            
        Returns:
            Training history dictionary
        """
        criterion = ContrastiveLoss(margin=1.0)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, (img1, img2, labels) in enumerate(tqdm(train_loader, desc="Training")):
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device).float()
                
                optimizer.zero_grad()
                
                emb1, emb2 = self.model(img1, img2)
                loss = criterion(emb1, emb2, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            
            # Validation phase
            val_metrics = self._validate_model(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_metrics['loss'])
            
            # Store history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            history['val_f1'].append(val_metrics['f1'])
            
            # Print epoch results
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save best model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy']
                }
                
                model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "checkpoints")
                os.makedirs(model_dir, exist_ok=True)
                torch.save(checkpoint, os.path.join(model_dir, "best_model.pth"))
                print("Saved new best model!")
                
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        return history
    
    def _validate_model(self, val_loader: DataLoader, criterion) -> Dict:
        """Validate model and return metrics."""
        self.model.eval()
        val_loss = 0.0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device).float()
                
                emb1, emb2 = self.model(img1, img2)
                loss = criterion(emb1, emb2, labels)
                val_loss += loss.item()
                
                # Calculate similarities for predictions
                similarities = cosine_similarity_score(emb1, emb2)
                preds = (similarities >= 0.85).float()  # Use same threshold as app
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        return {
            'loss': val_loss / len(val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


def main():
    """Main training function."""
    print("🔧 Improved Facial Recognition Model Training")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = SiameseNetwork(embedding_dim=128)
    model.to(device)
    
    # Data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")
    
    # Initialize trainer
    trainer = ImprovedTrainer(model, device)
    
    # Prepare dataset
    pairs, num_pos, num_neg = trainer.prepare_enhanced_dataset(data_dir)
    
    if len(pairs) == 0:
        print("❌ No training pairs generated. Please ensure you have:")
        print("1. At least 2 user directories with images")
        print("2. At least 2 images per user")
        print("3. Consider adding negative samples")
        return
    
    print(f"Total pairs: {len(pairs)} (Positive: {num_pos}, Negative: {num_neg})")
    
    # Create datasets and loaders
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Split data
    split_idx = int(0.8 * len(pairs))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    train_dataset = SiamesePairDataset(data_dir, train_pairs, transform)
    val_dataset = SiamesePairDataset(data_dir, val_pairs, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Train model
    print("\n🚀 Starting training...")
    history = trainer.train_with_validation(train_loader, val_loader, epochs=50, learning_rate=0.001)
    
    # Save training history
    history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "checkpoints", "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("✅ Training completed!")
    print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
    print(f"Best validation F1: {max(history['val_f1']):.4f}")
    print("\n📊 To improve accuracy further:")
    print("1. Add more diverse training images")
    print("2. Include negative samples from different people")
    print("3. Ensure good lighting and image quality")
    print("4. Consider retraining with more epochs if needed")


if __name__ == "__main__":
    main()
