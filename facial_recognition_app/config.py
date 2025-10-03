"""
Configuration file for the facial recognition application.
"""

import os
from typing import Dict, Any

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "model")
CHECKPOINTS_DIR = os.path.join(MODEL_DIR, "checkpoints")

# Model configuration
MODEL_CONFIG = {
    "embedding_dim": 128,
    "pretrained_backbone": True,
    "dropout_rate": 0.2,
    "normalize_embeddings": True,
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "num_epochs": 50,
    "loss_type": "contrastive",  # or "triplet"
    "margin": 1.0,
    "train_split": 0.8,
    "num_workers": 4,
    "pin_memory": True,
    "drop_last": True,
}

# Data configuration
DATA_CONFIG = {
    "image_size": (224, 224),
    "num_pairs": 2000,
    "triplets_per_person": 10,
    "augmentation": True,
    "face_detection": True,
    "face_margin": 0.2,
    "min_face_size": 40,
}

# Webcam configuration
WEBCAM_CONFIG = {
    "camera_id": 0,
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,
    "capture_delay": 1.0,  # seconds between captures
    "auto_capture": False,
    "preview_enabled": True,
}

# Application configuration
APP_CONFIG = {
    "similarity_threshold": 0.85,  # Increased from 0.7 for better accuracy
    "relative_threshold": 0.1,     # Minimum difference from other users
    "max_verification_history": 100,
    "auto_save_captures": True,
    "enable_gpu": True,
    "model_path": os.path.join(CHECKPOINTS_DIR, "best_model.pth"),
    "history_file": os.path.join(CHECKPOINTS_DIR, "training_history.json"),
}

# Face detection configuration
FACE_DETECTION_CONFIG = {
    "detector_type": "mtcnn",  # or "opencv"
    "confidence_threshold": 0.95,  # Increased from 0.9 for better face detection
    "min_face_size": 60,           # Increased from 40 for better quality
    "scale_factor": 0.709,
    "selection_method": "largest",  # or "first", "center"
}

# Image preprocessing configuration
PREPROCESSING_CONFIG = {
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "resize_method": "bilinear",
    "apply_clahe": False,  # Contrast Limited Adaptive Histogram Equalization
    "gamma_correction": 1.0,
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    "horizontal_flip_prob": 0.5,
    "rotation_range": 10,  # degrees
    "brightness_range": 0.2,
    "contrast_range": 0.2,
    "saturation_range": 0.2,
    "hue_range": 0.1,
    "gaussian_noise_std": 10,
    "blur_kernel_size": (3, 3),
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "metrics": ["accuracy", "auc", "eer", "precision", "recall", "f1"],
    "threshold_range": (0.0, 1.0),
    "threshold_step": 0.01,
    "cross_validation_folds": 5,
    "bootstrap_samples": 1000,
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": os.path.join(BASE_DIR, "logs", "app.log"),
    "max_log_size": 10 * 1024 * 1024,  # 10 MB
    "backup_count": 5,
}

# Security configuration
SECURITY_CONFIG = {
    "enable_liveness_check": False,
    "max_face_angle": 30,  # degrees
    "min_face_quality": 0.5,
    "enable_spoofing_detection": False,
    "rate_limiting": True,
    "max_requests_per_minute": 60,
}

# Performance configuration
PERFORMANCE_CONFIG = {
    "enable_mixed_precision": True,
    "gradient_clipping": 1.0,
    "optimize_for_inference": True,
    "batch_inference": True,
    "cache_embeddings": True,
    "parallel_processing": True,
}

# Development configuration
DEV_CONFIG = {
    "debug_mode": False,
    "save_debug_images": False,
    "profile_performance": False,
    "enable_tensorboard": True,
    "checkpoint_frequency": 5,  # epochs
    "early_stopping_patience": 10,
}


def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration dictionary.
    
    Returns:
        Dictionary containing all configuration parameters
    """
    return {
        "model": MODEL_CONFIG,
        "training": TRAINING_CONFIG,
        "data": DATA_CONFIG,
        "webcam": WEBCAM_CONFIG,
        "app": APP_CONFIG,
        "face_detection": FACE_DETECTION_CONFIG,
        "preprocessing": PREPROCESSING_CONFIG,
        "augmentation": AUGMENTATION_CONFIG,
        "evaluation": EVALUATION_CONFIG,
        "logging": LOGGING_CONFIG,
        "security": SECURITY_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "dev": DEV_CONFIG,
        "paths": {
            "base": BASE_DIR,
            "data": DATA_DIR,
            "raw_data": RAW_DATA_DIR,
            "processed_data": PROCESSED_DATA_DIR,
            "model": MODEL_DIR,
            "checkpoints": CHECKPOINTS_DIR,
        }
    }


def update_config(updates: Dict[str, Any]) -> None:
    """
    Update configuration with new values.
    
    Args:
        updates: Dictionary of configuration updates
    """
    config = get_config()
    
    def update_nested_dict(d: Dict, u: Dict) -> Dict:
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    update_nested_dict(config, updates)
    
    # Update global variables
    globals().update(config)


def create_directories() -> None:
    """Create necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        CHECKPOINTS_DIR,
        os.path.join(BASE_DIR, "logs"),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def validate_config() -> bool:
    """
    Validate configuration parameters.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Check required paths
        create_directories()
        
        # Validate model config
        assert MODEL_CONFIG["embedding_dim"] > 0, "Embedding dimension must be positive"
        assert 0 <= MODEL_CONFIG["dropout_rate"] <= 1, "Dropout rate must be between 0 and 1"
        
        # Validate training config
        assert TRAINING_CONFIG["batch_size"] > 0, "Batch size must be positive"
        assert TRAINING_CONFIG["learning_rate"] > 0, "Learning rate must be positive"
        assert TRAINING_CONFIG["num_epochs"] > 0, "Number of epochs must be positive"
        assert 0 < TRAINING_CONFIG["train_split"] < 1, "Train split must be between 0 and 1"
        
        # Validate data config
        assert len(DATA_CONFIG["image_size"]) == 2, "Image size must be (height, width)"
        assert all(s > 0 for s in DATA_CONFIG["image_size"]), "Image dimensions must be positive"
        
        # Validate app config
        assert 0 <= APP_CONFIG["similarity_threshold"] <= 1, "Similarity threshold must be between 0 and 1"
        
        return True
        
    except AssertionError as e:
        print(f"Configuration validation error: {e}")
        return False
    except Exception as e:
        print(f"Configuration error: {e}")
        return False


# Initialize configuration on import
if not validate_config():
    raise RuntimeError("Invalid configuration detected")

create_directories()

# Export main configuration
__all__ = [
    "get_config",
    "update_config",
    "create_directories",
    "validate_config",
    "MODEL_CONFIG",
    "TRAINING_CONFIG",
    "DATA_CONFIG",
    "WEBCAM_CONFIG",
    "APP_CONFIG",
    "FACE_DETECTION_CONFIG",
    "PREPROCESSING_CONFIG",
    "AUGMENTATION_CONFIG",
    "EVALUATION_CONFIG",
    "LOGGING_CONFIG",
    "SECURITY_CONFIG",
    "PERFORMANCE_CONFIG",
    "DEV_CONFIG",
]
