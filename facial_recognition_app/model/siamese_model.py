import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple


class SiameseNetwork(nn.Module):
    """
    Siamese Neural Network for face verification.
    Uses a CNN backbone to extract features and computes similarity between two images.
    """
    
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super(SiameseNetwork, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        # Add custom embedding layers
        self.embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # L2 normalize embeddings
        self.l2_norm = nn.functional.normalize
        
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one image to get embedding.
        
        Args:
            x: Input image tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Normalized embedding tensor of shape (batch_size, embedding_dim)
        """
        features = self.backbone(x)
        embedding = self.embedding(features)
        # L2 normalize the embedding
        normalized_embedding = self.l2_norm(embedding, p=2, dim=1)
        return normalized_embedding
    
    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both images.
        
        Args:
            input1: First image tensor
            input2: Second image tensor
            
        Returns:
            Tuple of normalized embeddings for both images
        """
        embedding1 = self.forward_one(input1)
        embedding2 = self.forward_one(input2)
        return embedding1, embedding2


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function for training Siamese networks.
    """
    
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embedding1: First embedding tensor
            embedding2: Second embedding tensor
            label: Binary label (1 for same person, 0 for different)
            
        Returns:
            Contrastive loss value
        """
        # Euclidean distance between embeddings
        euclidean_distance = F.pairwise_distance(embedding1, embedding2, keepdim=True)
        
        # Contrastive loss formula
        loss = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss function for training Siamese networks.
    """
    
    def __init__(self, margin: float = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embedding
            positive: Positive embedding (same person as anchor)
            negative: Negative embedding (different person from anchor)
            
        Returns:
            Triplet loss value
        """
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        
        loss = torch.mean(torch.clamp(pos_distance - neg_distance + self.margin, min=0.0))
        return loss


def cosine_similarity_score(embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        
    Returns:
        Cosine similarity score
    """
    return F.cosine_similarity(embedding1, embedding2)


def euclidean_distance_score(embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized Euclidean distance score between two embeddings.
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        
    Returns:
        Distance score (lower means more similar)
    """
    distance = F.pairwise_distance(embedding1, embedding2)
    # Convert to similarity score (higher means more similar)
    similarity = 1 / (1 + distance)
    return similarity


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork(embedding_dim=128).to(device)
    
    # Create dummy input
    batch_size = 4
    img1 = torch.randn(batch_size, 3, 224, 224).to(device)
    img2 = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Forward pass
    emb1, emb2 = model(img1, img2)
    print(f"Embedding 1 shape: {emb1.shape}")
    print(f"Embedding 2 shape: {emb2.shape}")
    
    # Test loss functions
    labels = torch.randint(0, 2, (batch_size,)).float().to(device)
    contrastive_loss = ContrastiveLoss()
    loss = contrastive_loss(emb1, emb2, labels)
    print(f"Contrastive loss: {loss.item()}")
    
    # Test similarity scores
    cos_sim = cosine_similarity_score(emb1, emb2)
    euc_sim = euclidean_distance_score(emb1, emb2)
    print(f"Cosine similarity: {cos_sim}")
    print(f"Euclidean similarity: {euc_sim}")
