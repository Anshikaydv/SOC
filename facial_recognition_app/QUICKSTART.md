# 🚀 Quick Start Guide - Facial Recognition App (FIXED VERSION)

## ⚠️ CRITICAL: Fixing False Positive Issues

**Your model is currently matching every face because of insufficient training data!**

### � Immediate Steps to Fix:

#### 1. Diagnose the Problem
```powershell
# Check current registered users
python register_users.py
# Choose option 3 to see current status
```

#### 2. Register Multiple Users
```powershell
# You need AT LEAST 2 different people
python register_users.py
# Choose option 1, register yourself with 15+ images
# Then register a friend/family member with 15+ images
```

#### 3. Collect Negative Samples
```powershell
# These are faces that should NOT be recognized
python register_users.py
# Choose option 2, collect 20+ images of different people
```

#### 4. Retrain with Fixed Algorithm
```powershell
# This uses improved training with proper negative sampling
python retrain_improved_model.py
```

#### 5. Test the Fix
```powershell
# Launch the improved app
streamlit run app/main.py
```

---

## 🚀 Getting Started (Detailed Setup)

### Step 1: Installation

1. **Setup Environment**:
   ```powershell
   # Navigate to the project directory
   cd facial_recognition_app
   
   # Run the setup script
   python setup.py
   ```

2. **Install Dependencies**:
   ```powershell
   # Install PyTorch (choose based on your system)
   # For CPU only:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   
   # For CUDA 11.8:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Install other requirements
   pip install -r requirements.txt
   ```

### Step 2: Test Installation

```powershell
# Run the demo script to verify everything works
python demo.py
```

### Step 3: Collect Data

1. **Register Users via Webcam**:
   ```powershell
   # Start the Streamlit app
   cd app
   streamlit run main.py
   ```
   
2. **Or manually capture images**:
   ```python
   from utils.image_utils import WebcamCapture
   
   webcam = WebcamCapture()
   webcam.capture_face_images(
       person_name="john_doe",
       output_dir="../data/raw",
       num_images=15
   )
   ```

### Step 4: Train the Model

```powershell
cd model
python train.py
```

**Training Configuration** (edit in `train.py`):
```python
config = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 50,
    'loss_type': 'contrastive',  # or 'triplet'
    'embedding_dim': 128,
}
```

### Step 5: Run Face Verification

```powershell
cd app
streamlit run main.py
```

**The web interface provides**:
- ✅ Face verification (1:1 matching)
- 👤 User registration via webcam
- 📊 System information and metrics
- 📈 Training history visualization

---

## 🛠️ Alternative Setup Methods

### Method 1: Manual Installation

1. **Create Virtual Environment**:
   ```powershell
   python -m venv face_env
   face_env\Scripts\activate  # Windows
   # source face_env/bin/activate  # macOS/Linux
   ```

2. **Install PyTorch First**:
   ```powershell
   # Visit: https://pytorch.org/get-started/locally/
   # Choose your configuration and copy the command
   ```

3. **Install Dependencies**:
   ```powershell
   pip install opencv-python streamlit numpy pillow matplotlib scikit-learn mtcnn facenet-pytorch tqdm pandas seaborn albumentations
   ```

### Method 2: Conda Environment

```powershell
# Create conda environment
conda create -n face_recognition python=3.9
conda activate face_recognition

# Install PyTorch via conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other packages
pip install streamlit opencv-python mtcnn facenet-pytorch
pip install numpy pillow matplotlib scikit-learn tqdm pandas seaborn albumentations
```

---

## 📊 Usage Examples

### 1. Basic Face Verification

```python
import cv2
from model.siamese_model import SiameseNetwork
from utils.image_utils import FaceDetector, ImagePreprocessor

# Load model
model = SiameseNetwork()
model.load_state_dict(torch.load('model/checkpoints/best_model.pth'))

# Load and preprocess images
img1 = cv2.imread('person1.jpg')
img2 = cv2.imread('person2.jpg')

# Get similarity score
similarity = verify_faces(img1, img2, model)
is_same_person = similarity > 0.7
```

### 2. Register New User

```python
from utils.image_utils import WebcamCapture

webcam = WebcamCapture()
success = webcam.capture_face_images(
    person_name="alice",
    output_dir="data/raw",
    num_images=20
)
```

### 3. Batch Processing

```python
from utils.image_utils import process_raw_images

# Process all raw images
process_raw_images(
    input_dir="data/raw",
    output_dir="data/processed"
)
```

---

## ⚙️ Configuration

### Model Parameters

```python
# In model/siamese_model.py
MODEL_CONFIG = {
    "embedding_dim": 128,      # Size of face embeddings
    "pretrained": True,        # Use pretrained ResNet
    "dropout_rate": 0.2,       # Dropout for regularization
}
```

### Training Parameters

```python
# In model/train.py
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "loss_type": "contrastive",
    "margin": 1.0,
}
```

### App Settings

```python
# In config.py
APP_CONFIG = {
    "similarity_threshold": 0.7,  # Verification threshold
    "max_history": 100,           # Max verification history
    "enable_gpu": True,           # Use GPU if available
}
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```powershell
# Solution: Install missing packages
pip install package_name

# Or reinstall all requirements
pip install -r requirements.txt --force-reinstall
```

#### 2. CUDA/GPU Issues
```powershell
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CPU-only version if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 3. Webcam Access
- **Windows**: Check camera permissions in Settings > Privacy > Camera
- **macOS**: Grant camera access in System Preferences > Security & Privacy
- **Linux**: Ensure user is in `video` group: `sudo usermod -a -G video $USER`

#### 4. Face Detection Issues
- Ensure good lighting conditions
- Keep face clearly visible and centered
- Try different camera angles
- Check if MTCNN is properly installed: `pip install mtcnn`

#### 5. Low Training Accuracy
- Increase dataset size (more images per person)
- Improve image quality
- Adjust learning rate and batch size
- Try different loss functions (contrastive vs triplet)

### Performance Optimization

#### 1. Faster Training
```python
# Use mixed precision training
config["enable_mixed_precision"] = True

# Increase batch size (if memory allows)
config["batch_size"] = 64

# Use more workers for data loading
config["num_workers"] = 8
```

#### 2. Faster Inference
```python
# Use GPU inference
model = model.cuda()

# Batch multiple verifications
embeddings = model(batch_of_images)
```

---

## 🔍 System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14+, or Linux
- **Python**: 3.8 or higher
- **RAM**: 8 GB (16 GB recommended)
- **Storage**: 2 GB free space
- **Camera**: Built-in or USB webcam

### Recommended Requirements
- **OS**: Windows 11, macOS 12+, or Ubuntu 20.04+
- **Python**: 3.9 or 3.10
- **RAM**: 16 GB or more
- **GPU**: NVIDIA GPU with 4+ GB VRAM (for faster training)
- **Storage**: 10 GB free space (for datasets)
- **Camera**: HD webcam (1080p preferred)

### GPU Support
- **NVIDIA**: GTX 1060 6GB or better
- **CUDA**: Version 11.8 or 12.1
- **cuDNN**: Compatible version
- **VRAM**: 4 GB minimum, 8 GB recommended

---

## 📞 Support

### Getting Help

1. **Check the README**: Detailed documentation available
2. **Run Diagnostics**: Use `python demo.py` to test system
3. **Common Issues**: See troubleshooting section above
4. **GitHub Issues**: Report bugs and feature requests

### Useful Commands

```powershell
# Test installation
python demo.py

# Check package versions
pip list | findstr "torch\|opencv\|streamlit"

# Test webcam
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"

# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Performance Monitoring

```python
# In the Streamlit app, check:
# - System Info page for hardware details
# - Training History for model performance
# - Verification History for accuracy trends
```

---

**Happy Face Recognition! 🎭**
