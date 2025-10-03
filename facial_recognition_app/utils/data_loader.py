import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import random
from PIL import Image
from typing import List, Tuple, Optional, Union
import glob
import json
from sklearn.model_selection import train_test_split


class SiamesePairDataset(Dataset):
    """
    Dataset class for Siamese network training with image pairs.
    """
    
    def __init__(
        self, 
        data_dir: str,
        pairs: List[Tuple],
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing processed images
            pairs: List of image pairs with labels
            transform: Image transformations
            image_size: Target image size
        """
        self.data_dir = data_dir
        self.pairs = pairs
        self.transform = transform
        self.image_size = image_size
        
        if self.transform is None:
            self.transform = self._get_default_transform()
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transformations."""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a pair of images and their label.
        
        Args:
            idx: Index of the pair
            
        Returns:
            Tuple of (image1, image2, label)
        """
        img1_path, img2_path, label = self.pairs[idx]
        
        # Load images
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)
        
        # Apply transformations
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label
    
    def _load_image(self, img_path: str) -> Image.Image:
        """Load and convert image to RGB."""
        try:
            img = Image.open(img_path).convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            return Image.new('RGB', self.image_size, (0, 0, 0))


class TripletDataset(Dataset):
    """
    Dataset class for triplet loss training.
    """
    
    def __init__(
        self,
        data_dir: str,
        person_dirs: List[str],
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (224, 224),
        triplets_per_person: int = 10
    ):
        """
        Initialize the triplet dataset.
        
        Args:
            data_dir: Directory containing person folders
            person_dirs: List of person directory names
            transform: Image transformations
            image_size: Target image size
            triplets_per_person: Number of triplets to generate per person
        """
        self.data_dir = data_dir
        self.person_dirs = person_dirs
        self.transform = transform
        self.image_size = image_size
        self.triplets_per_person = triplets_per_person
        
        if self.transform is None:
            self.transform = self._get_default_transform()
        
        # Generate triplets
        self.triplets = self._generate_triplets()
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transformations."""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _generate_triplets(self) -> List[Tuple[str, str, str]]:
        """Generate triplets (anchor, positive, negative)."""
        triplets = []
        
        # Get all images for each person
        person_images = {}
        for person in self.person_dirs:
            person_path = os.path.join(self.data_dir, person)
            if os.path.exists(person_path):
                images = glob.glob(os.path.join(person_path, "*.jpg")) + \
                        glob.glob(os.path.join(person_path, "*.png"))
                if len(images) >= 2:  # Need at least 2 images for positive pairs
                    person_images[person] = images
        
        persons = list(person_images.keys())
        
        for person in persons:
            person_imgs = person_images[person]
            
            for _ in range(self.triplets_per_person):
                # Select anchor and positive (same person)
                if len(person_imgs) >= 2:
                    anchor_img, positive_img = random.sample(person_imgs, 2)
                else:
                    continue
                
                # Select negative (different person)
                other_persons = [p for p in persons if p != person]
                if other_persons:
                    negative_person = random.choice(other_persons)
                    negative_img = random.choice(person_images[negative_person])
                    
                    triplets.append((anchor_img, positive_img, negative_img))
        
        return triplets
    
    def __len__(self) -> int:
        return len(self.triplets)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a triplet of images.
        
        Args:
            idx: Index of the triplet
            
        Returns:
            Tuple of (anchor, positive, negative)
        """
        anchor_path, positive_path, negative_path = self.triplets[idx]
        
        # Load images
        anchor = self._load_image(anchor_path)
        positive = self._load_image(positive_path)
        negative = self._load_image(negative_path)
        
        # Apply transformations
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative
    
    def _load_image(self, img_path: str) -> Image.Image:
        """Load and convert image to RGB."""
        try:
            img = Image.open(img_path).convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return Image.new('RGB', self.image_size, (0, 0, 0))


def generate_pairs_from_directory(data_dir: str, num_pairs: int = 1000) -> List[Tuple[str, str, int]]:
    """
    Generate positive and negative pairs from a directory structure.
    
    Args:
        data_dir: Directory containing person folders
        num_pairs: Number of pairs to generate
        
    Returns:
        List of (img1_path, img2_path, label) tuples
    """
    pairs = []
    
    # Get all person directories
    person_dirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    
    if len(person_dirs) < 2:
        raise ValueError("Need at least 2 person directories to generate pairs")
    
    # Get all images for each person
    person_images = {}
    for person in person_dirs:
        person_path = os.path.join(data_dir, person)
        images = glob.glob(os.path.join(person_path, "*.jpg")) + \
                glob.glob(os.path.join(person_path, "*.png"))
        if len(images) >= 2:  # Need at least 2 images for positive pairs
            person_images[person] = images
    
    persons = list(person_images.keys())
    
    # Generate positive pairs (same person)
    positive_pairs = []
    for person in persons:
        person_imgs = person_images[person]
        if len(person_imgs) >= 2:
            # Generate all possible pairs for this person
            for i in range(len(person_imgs)):
                for j in range(i + 1, len(person_imgs)):
                    positive_pairs.append((person_imgs[i], person_imgs[j], 1))
    
    # Generate negative pairs (different persons)
    negative_pairs = []
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            person1_imgs = person_images[persons[i]]
            person2_imgs = person_images[persons[j]]
            
            # Generate pairs between different persons
            for img1 in person1_imgs:
                for img2 in person2_imgs:
                    negative_pairs.append((img1, img2, 0))
    
    # Balance positive and negative pairs
    num_positive = min(len(positive_pairs), num_pairs // 2)
    num_negative = min(len(negative_pairs), num_pairs // 2)
    
    selected_positive = random.sample(positive_pairs, num_positive)
    selected_negative = random.sample(negative_pairs, num_negative)
    
    pairs = selected_positive + selected_negative
    random.shuffle(pairs)
    
    print(f"Generated {len(selected_positive)} positive pairs and {len(selected_negative)} negative pairs")
    
    return pairs


def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    loss_type: str = "contrastive",
    num_pairs: int = 2000
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        data_dir: Directory containing processed images
        batch_size: Batch size for data loaders
        train_split: Fraction of data to use for training
        image_size: Target image size
        num_workers: Number of worker processes for data loading
        loss_type: Type of loss function ("contrastive" or "triplet")
        num_pairs: Number of pairs to generate for contrastive loss
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1))),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if loss_type == "contrastive":
        # Generate pairs
        pairs = generate_pairs_from_directory(data_dir, num_pairs)
        
        # Split pairs into train and validation
        train_pairs, val_pairs = train_test_split(
            pairs, train_size=train_split, random_state=42, stratify=[p[2] for p in pairs]
        )
        
        # Create datasets
        train_dataset = SiamesePairDataset(
            data_dir=data_dir,
            pairs=train_pairs,
            transform=train_transform,
            image_size=image_size
        )
        
        val_dataset = SiamesePairDataset(
            data_dir=data_dir,
            pairs=val_pairs,
            transform=val_transform,
            image_size=image_size
        )
        
    elif loss_type == "triplet":
        # Get person directories
        person_dirs = [d for d in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, d))]
        
        # Split persons into train and validation
        train_persons, val_persons = train_test_split(
            person_dirs, train_size=train_split, random_state=42
        )
        
        # Create datasets
        train_dataset = TripletDataset(
            data_dir=data_dir,
            person_dirs=train_persons,
            transform=train_transform,
            image_size=image_size
        )
        
        val_dataset = TripletDataset(
            data_dir=data_dir,
            person_dirs=val_persons,
            transform=val_transform,
            image_size=image_size
        )
    
    else:
        raise ValueError("loss_type must be 'contrastive' or 'triplet'")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def save_pairs_to_file(pairs: List[Tuple], file_path: str):
    """Save generated pairs to a JSON file."""
    pairs_data = [{"img1": p[0], "img2": p[1], "label": p[2]} for p in pairs]
    with open(file_path, 'w') as f:
        json.dump(pairs_data, f, indent=2)


def load_pairs_from_file(file_path: str) -> List[Tuple]:
    """Load pairs from a JSON file."""
    with open(file_path, 'r') as f:
        pairs_data = json.load(f)
    return [(p["img1"], p["img2"], p["label"]) for p in pairs_data]


if __name__ == "__main__":
    # Test the data loader
    data_dir = "../data/processed"
    
    if os.path.exists(data_dir):
        try:
            train_loader, val_loader = get_data_loaders(
                data_dir=data_dir,
                batch_size=4,
                num_pairs=100
            )
            
            print(f"Training samples: {len(train_loader.dataset)}")
            print(f"Validation samples: {len(val_loader.dataset)}")
            
            # Test one batch
            for batch in train_loader:
                img1, img2, labels = batch
                print(f"Batch shape - img1: {img1.shape}, img2: {img2.shape}, labels: {labels.shape}")
                break
                
        except Exception as e:
            print(f"Error creating data loaders: {e}")
    else:
        print(f"Data directory {data_dir} does not exist")
