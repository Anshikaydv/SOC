# Facial Recognition App using Siamese Neural Network

A deep learning-based facial recognition application that can verify whether two input images belong to the same person using a Siamese Neural Network with contrastive/triplet loss.

## Features

- **Custom Data Collection**: Capture face images directly from webcam
- **Data Preprocessing**: Automatic face detection, resizing, and normalization
- **Model Training**: Siamese network with contrastive or triplet loss
- **Real-time Verification**: Interactive Streamlit web interface
- **User Registration**: Register new users through webcam capture
- **Performance Metrics**: Accuracy, ROC-AUC, and similarity scores

## Project Structure

```
facial_recognition_app/
├── data/
│   ├── raw/               # Raw images from webcam (labeled folders per user)
│   └── processed/         # Preprocessed pairs of images
├── model/
│   ├── siamese_model.py   # Siamese network architecture
│   └── train.py           # Training script with contrastive loss
├── utils/
│   ├── data_loader.py     # Data loading & preprocessing utils
│   └── image_utils.py     # For resizing, face detection, etc.
├── app/
│   └── main.py            # Streamlit app for verification
├── requirements.txt
└── README.md
```

## Technology Stack

- **Python 3.8+**
- **PyTorch 2.0+** - Deep learning framework
- **OpenCV 4.8+** - Computer vision and webcam handling
- **Streamlit 1.28+** - Web interface
- **MTCNN** - Face detection
- **scikit-learn** - Metrics and evaluation
- **matplotlib/seaborn** - Visualization

## Installation

1. **Clone the repository:**
   ```bash
   cd facial_recognition_app
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv face_recognition_env
   # On Windows:
   face_recognition_env\Scripts\activate
   # On macOS/Linux:
   source face_recognition_env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Data Collection

Collect face images for users:

```python
from utils.image_utils import WebcamCapture

# Initialize webcam capture
webcam = WebcamCapture()

# Capture images for a user
webcam.capture_face_images(
    person_name="john_doe",
    output_dir="data/raw",
    num_images=20
)
```

### 2. Data Preprocessing

Process raw images and detect faces:

```python
from utils.image_utils import process_raw_images

# Process all raw images
process_raw_images(
    input_dir="data/raw",
    output_dir="data/processed"
)
```

### 3. Train the Model

Train the Siamese network:

```bash
cd model
python train.py
```

### 4. Run the Application

Launch the Streamlit web interface:

```bash
cd app
streamlit run main.py
```

## Model Architecture

The Siamese Neural Network consists of:

- **Backbone**: ResNet-18 (pre-trained on ImageNet)
- **Embedding Layer**: Fully connected layers (512 → 256 → 128)
- **Loss Function**: Contrastive Loss or Triplet Loss
- **Similarity Metric**: Cosine similarity or Euclidean distance

### Contrastive Loss

```python
loss = label * d² + (1 - label) * max(margin - d, 0)²
```

Where:
- `d` = Euclidean distance between embeddings
- `label` = 1 for same person, 0 for different person
- `margin` = margin parameter (default: 1.0)

### Triplet Loss

```python
loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
```

Where:
- `d(a, p)` = distance between anchor and positive
- `d(a, n)` = distance between anchor and negative

## Usage

### Training Configuration

Modify training parameters in `model/train.py`:

```python
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
}
```

### Web Application Features

1. **Face Verification**: Upload image or capture from webcam to verify against registered users
2. **User Registration**: Register new users by capturing multiple face images
3. **System Information**: View model details, dataset statistics, and system requirements
4. **Training History**: Visualize training curves and performance metrics

### API Usage

```python
from model.siamese_model import SiameseNetwork
from utils.image_utils import FaceDetector, ImagePreprocessor

# Load trained model
model = SiameseNetwork(embedding_dim=128)
model.load_state_dict(torch.load('model/checkpoints/best_model.pth'))

# Verify two images
similarity_score = verify_faces(image1, image2, model)
is_same_person = similarity_score > 0.7  # threshold
```

## Performance Metrics

The model is evaluated using:

- **Accuracy**: Percentage of correct classifications
- **ROC-AUC**: Area under the ROC curve
- **Similarity Score**: Cosine similarity between embeddings
- **Equal Error Rate (EER)**: Point where FAR = FRR

## Best Practices

### Data Collection

- Capture 15-20 images per person
- Include different lighting conditions
- Vary facial expressions and angles
- Ensure high-quality, clear face images
- Maintain consistent image resolution

### Training Tips

- Use data augmentation (rotation, flip, brightness)
- Monitor validation metrics to prevent overfitting
- Experiment with different embedding dimensions
- Try both contrastive and triplet loss functions
- Use learning rate scheduling

### Deployment

- Set appropriate similarity threshold (0.6-0.8)
- Implement face quality checks
- Add liveness detection for security
- Monitor system performance in production
- Regular model retraining with new data

## Troubleshooting

### Common Issues

1. **Webcam not detected**:
   - Check camera permissions
   - Try different camera indices (0, 1, 2...)
   - Ensure no other application is using the camera

2. **Face detection fails**:
   - Improve lighting conditions
   - Ensure face is clearly visible
   - Check image quality and resolution

3. **Low accuracy**:
   - Increase dataset size
   - Improve data quality
   - Tune hyperparameters
   - Try different loss functions

4. **Memory issues during training**:
   - Reduce batch size
   - Use smaller image resolution
   - Enable gradient checkpointing

### Dependencies Issues

If you encounter package conflicts:

```bash
# Create fresh environment
conda create -n face_recognition python=3.8
conda activate face_recognition

# Install PyTorch first
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other requirements
pip install -r requirements.txt
```

## Future Enhancements

- [ ] Add liveness detection
- [ ] Implement face recognition (1:N identification)
- [ ] Support for multiple face verification
- [ ] Add REST API endpoints
- [ ] Mobile app integration
- [ ] Real-time video stream processing
- [ ] Advanced data augmentation techniques
- [ ] Model quantization for edge deployment

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the Siamese Networks for One-shot Learning paper
- Uses MTCNN for robust face detection
- Inspired by FaceNet and DeepFace architectures
- Built with PyTorch and Streamlit frameworks

## Contact

For questions or support, please open an issue on GitHub or contact the development team.

---

**Note**: This system is designed for educational and research purposes. For production use, consider additional security measures, privacy compliance, and robustness testing.
