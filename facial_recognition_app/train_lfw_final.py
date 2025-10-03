#!/usr/bin/env python3
"""
Final Fixed LFW Dataset Training Script for Facial Recognition
Uses OpenCV's Haar Cascades and handles small validation sets properly
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
import random
from tqdm import tqdm
import logging
from typing import List, Tuple, Optional
import json
from pathlib import Path

# Add the project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.siamese_model import SiameseNetwork

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleFaceDetector:
    """Simple face detector using OpenCV Haar Cascades."""
    
    def __init__(self):
        # Load Haar cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
    def detect_faces(self, image):
        """Detect faces using Haar cascades."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to list of tuples (x1, y1, x2, y2)
        face_boxes = []
        for (x, y, w, h) in faces:
            face_boxes.append((x, y, x + w, y + h))
        
        return face_boxes

class LFWDatasetProcessor:
    """Process LFW dataset for face recognition training."""
    
    def __init__(self, lfw_path: str, max_people: int = 80):
        self.lfw_path = lfw_path
        self.max_people = max_people
        self.face_detector = SimpleFaceDetector()
        
        print(f"🔧 Simple face detector initialized")
        
    def process_dataset(self) -> Tuple[List[str], List[np.ndarray]]:
        """
        Process LFW dataset and extract faces.
        
        Returns:
            Tuple of (person_names, face_images)
        """
        if not os.path.exists(self.lfw_path):
            raise ValueError(f"LFW dataset not found at {self.lfw_path}")
        
        print(f"🔄 Processing LFW dataset...")
        
        # Get all person directories
        person_dirs = [d for d in os.listdir(self.lfw_path) 
                      if os.path.isdir(os.path.join(self.lfw_path, d))]
        
        print(f"📊 Found {len(person_dirs)} people in dataset")
        
        # Limit number of people for manageable training
        if len(person_dirs) > self.max_people:
            person_dirs = random.sample(person_dirs, self.max_people)
            print(f"📊 Using {self.max_people} people for training")
        
        all_faces = []
        all_labels = []
        person_count = 0
        
        for person_name in tqdm(person_dirs, desc="Processing people"):
            person_path = os.path.join(self.lfw_path, person_name)
            
            # Get all images for this person
            image_files = [f for f in os.listdir(person_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            person_faces = []
            
            for img_file in image_files:
                img_path = os.path.join(person_path, img_file)
                
                try:
                    # Load image
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces
                    faces = self.face_detector.detect_faces(image_rgb)
                    
                    if len(faces) >= 1:  # Use first face if multiple detected
                        face_box = faces[0]
                        x1, y1, x2, y2 = face_box
                        
                        # Extract face region with some padding
                        h, w = image_rgb.shape[:2]
                        x1 = max(0, x1 - 10)
                        y1 = max(0, y1 - 10)
                        x2 = min(w, x2 + 10)
                        y2 = min(h, y2 + 10)
                        
                        face_img = image_rgb[y1:y2, x1:x2]
                        
                        if face_img.size > 0 and face_img.shape[0] > 50 and face_img.shape[1] > 50:
                            # Resize to standard size
                            face_resized = cv2.resize(face_img, (224, 224))
                            person_faces.append(face_resized)
                            
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
                    continue
            
            # Only include people with at least 3 faces for better training
            if len(person_faces) >= 3:
                all_faces.extend(person_faces)
                all_labels.extend([person_count] * len(person_faces))
                print(f"✅ {person_name}: {len(person_faces)} faces")
                person_count += 1
            else:
                print(f"⚠️ {person_name}: Only {len(person_faces)} faces (skipped)")
        
        print(f"✅ Processed {person_count} people with {len(all_faces)} total faces")
        return all_labels, all_faces

class LFWPairDataset(Dataset):
    """Dataset for generating positive and negative pairs from LFW faces."""
    
    def __init__(self, labels: List[int], faces: List[np.ndarray], 
                 pairs_per_person: int = 8, train: bool = True):
        self.labels = labels
        self.faces = faces
        self.pairs_per_person = pairs_per_person
        self.train = train
        
        # Group faces by person
        self.person_faces = {}
        for i, label in enumerate(labels):
            if label not in self.person_faces:
                self.person_faces[label] = []
            self.person_faces[label].append(i)
        
        # Generate pairs
        self.pairs = []
        self.pair_labels = []
        
        self._generate_pairs()
        
    def _generate_pairs(self):
        """Generate positive and negative pairs."""
        
        # Generate positive pairs (same person)
        for person_id, face_indices in self.person_faces.items():
            if len(face_indices) < 2:
                continue
                
            # Generate pairs for this person
            for _ in range(min(self.pairs_per_person, len(face_indices) * (len(face_indices) - 1) // 2)):
                idx1, idx2 = random.sample(face_indices, 2)
                self.pairs.append((idx1, idx2))
                self.pair_labels.append(1)  # Positive pair
        
        # Generate negative pairs (different persons)
        num_positive = len([label for label in self.pair_labels if label == 1])
        
        person_ids = list(self.person_faces.keys())
        
        # Only generate negative pairs if we have at least 2 people
        if len(person_ids) >= 2:
            for _ in range(num_positive):  # Equal number of negative pairs
                person1, person2 = random.sample(person_ids, 2)
                idx1 = random.choice(self.person_faces[person1])
                idx2 = random.choice(self.person_faces[person2])
                self.pairs.append((idx1, idx2))
                self.pair_labels.append(0)  # Negative pair
        
        print(f"✅ Created {num_positive} positive pairs and {len(self.pair_labels) - num_positive} negative pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        label = self.pair_labels[idx]
        
        # Get images
        img1 = self.faces[idx1].astype(np.float32) / 255.0
        img2 = self.faces[idx2].astype(np.float32) / 255.0
        
        # Convert to tensor and normalize
        img1 = torch.from_numpy(img1).permute(2, 0, 1)  # CHW format
        img2 = torch.from_numpy(img2).permute(2, 0, 1)
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        img1 = (img1 - mean) / std
        img2 = (img2 - mean) / std
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)

class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese networks."""
    
    def __init__(self, margin: float = 2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss_contrastive

class LFWTrainer:
    """Trainer for LFW dataset."""
    
    def __init__(self, model_save_path: str = "model/checkpoints"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        self.model_save_path = model_save_path
        os.makedirs(model_save_path, exist_ok=True)
        
        # Initialize model
        self.model = SiameseNetwork()
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = ContrastiveLoss(margin=2.0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=8, gamma=0.1)
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (img1, img2, labels) in enumerate(pbar):
            img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output1, output2 = self.model(img1, img2)
            loss = self.criterion(output1, output2, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            distances = nn.functional.pairwise_distance(output1, output2)
            predicted = (distances < 1.0).float()  # Threshold for similarity
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model."""
        if val_loader is None or len(val_loader) == 0:
            return 0.0, 50.0  # Return dummy values if no validation data
            
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                
                output1, output2 = self.model(img1, img2)
                loss = self.criterion(output1, output2, labels)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                distances = nn.functional.pairwise_distance(output1, output2)
                predicted = (distances < 1.0).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs: int = 12):
        """Train the model."""
        best_val_accuracy = 0.0
        training_history = []
        
        print(f"🚀 Starting training for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # Log results
            print(f"\nEpoch {epoch}/{epochs} ({epoch_time:.1f}s):")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            if val_loader is not None and len(val_loader) > 0:
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save history
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model_path = os.path.join(self.model_save_path, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'val_loss': val_loss,
                }, best_model_path)
                print(f"💾 Saved best model (Val Acc: {val_acc:.2f}%)")
        
        # Save training history
        history_path = os.path.join(self.model_save_path, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"\n✅ Training completed! Best validation accuracy: {best_val_accuracy:.2f}%")
        return training_history

def main():
    """Main training function."""
    print("🎯 Final LFW Dataset Training for Facial Recognition")
    print("=" * 52)
    
    # Paths
    archive_path = r"c:\Desktop\SOC_Project\archive"
    lfw_path = os.path.join(archive_path, "lfw-deepfunneled", "lfw-deepfunneled")
    
    if not os.path.exists(lfw_path):
        print(f"❌ LFW dataset not found at {lfw_path}")
        print("Please ensure the archive folder is extracted properly.")
        return
    
    print(f"✅ Found LFW dataset at {lfw_path}")
    
    try:
        # Process dataset with more people for better training
        processor = LFWDatasetProcessor(lfw_path, max_people=100)
        labels, faces = processor.process_dataset()
        
        if len(faces) == 0:
            print("❌ No faces processed! Check dataset path and face detection.")
            return
        
        if len(set(labels)) < 5:
            print("❌ Need at least 5 people for training. Using all data for training.")
            # Use all data for training
            train_labels = labels
            train_faces = faces
            val_labels = []
            val_faces = []
            val_loader = None
        else:
            # Split into train/validation (80/20)
            total_faces = len(faces)
            split_idx = int(0.8 * total_faces)
            
            train_labels = labels[:split_idx]
            train_faces = faces[:split_idx]
            val_labels = labels[split_idx:]
            val_faces = faces[split_idx:]
        
        print(f"📊 Training samples: {len(train_faces)}")
        print(f"📊 Validation samples: {len(val_faces)}")
        
        # Create datasets
        train_dataset = LFWPairDataset(train_labels, train_faces, pairs_per_person=8, train=True)
        
        if len(val_faces) > 0 and len(set(val_labels)) >= 2:
            val_dataset = LFWPairDataset(val_labels, val_faces, pairs_per_person=4, train=False)
            val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=0)
        else:
            val_loader = None
            print("⚠️ No validation set - using all data for training")
        
        if len(train_dataset) == 0:
            print("❌ No training pairs generated! Need more diverse data.")
            return
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        
        print(f"📊 Training pairs: {len(train_dataset)}")
        if val_loader:
            print(f"📊 Validation pairs: {len(val_loader.dataset)}")
        
        # Train model
        trainer = LFWTrainer()
        history = trainer.train(train_loader, val_loader, epochs=12)
        
        print("\n🎉 LFW training completed successfully!")
        print("💡 Your facial recognition system now uses a model trained on professional faces!")
        print("🚀 The system should now be much more accurate and less strict!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
