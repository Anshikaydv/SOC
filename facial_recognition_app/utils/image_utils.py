import cv2
import numpy as np
from PIL import Image
import os
import glob
from typing import List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import torch
import torchvision.transforms as transforms


class FaceDetector:
    """
    Face detection utility using MTCNN.
    """
    
    def __init__(self, min_face_size: int = 20, confidence_threshold: float = 0.5):
        """
        Initialize face detector.
        
        Args:
            min_face_size: Minimum size of detected faces (used for filtering)
            confidence_threshold: Minimum confidence for face detection
        """
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold
        
        # Initialize MTCNN with optimized parameters for better detection
        try:
            if torch.cuda.is_available():
                device_str = 'cuda:0'
            else:
                device_str = 'cpu'
            
            self.detector = MTCNN(
                device=device_str,
                min_face_size=min_face_size,
                thresholds=[0.5, 0.6, 0.6],  # Lower thresholds for better detection
                factor=0.709,  # Default scaling factor
                post_process=True
            )
        except Exception:
            # Fallback to basic initialization
            self.detector = MTCNN(device=device_str if torch.cuda.is_available() else 'cpu')
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image with improved robustness.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of bounding boxes [(x, y, width, height)]
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Ensure image is in correct format
            if rgb_image.dtype != np.uint8:
                rgb_image = rgb_image.astype(np.uint8)
            
            # Detect faces
            result = self.detector.detect_faces(rgb_image)
            
            faces = []
            for detection in result:
                if isinstance(detection, dict) and 'box' in detection and 'confidence' in detection:
                    bbox = detection['box']
                    confidence = detection['confidence']
                    
                    # Filter by confidence threshold
                    if confidence > self.confidence_threshold:
                        x, y, w, h = bbox
                        
                        # Ensure positive dimensions
                        if w > 0 and h > 0:
                            # Filter by minimum face size
                            if w >= self.min_face_size and h >= self.min_face_size:
                                # Ensure coordinates are within image bounds
                                x = max(0, int(x))
                                y = max(0, int(y))
                                w = min(int(w), image.shape[1] - x)
                                h = min(int(h), image.shape[0] - y)
                                
                                if w > 0 and h > 0:  # Final check
                                    faces.append((x, y, w, h))
            
            return faces
            
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []
    
    def extract_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                    margin: float = 0.2) -> Optional[np.ndarray]:
        """
        Extract face from image using bounding box.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, width, height)
            margin: Margin to add around the face
            
        Returns:
            Cropped face image or None if extraction fails
        """
        x, y, w, h = bbox
        
        # Add margin
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        # Extract face
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
        
        return face


class ImagePreprocessor:
    """
    Image preprocessing utilities for face recognition.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target size for processed images
        """
        self.target_size = target_size
        self.face_detector = FaceDetector()
    
    def resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            size: Target size (width, height)
            
        Returns:
            Resized image
        """
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image values to [0, 1].
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def preprocess_image(self, image: np.ndarray, detect_face: bool = True) -> Optional[np.ndarray]:
        """
        Preprocess image for face recognition.
        
        Args:
            image: Input image
            detect_face: Whether to detect and crop face first
            
        Returns:
            Preprocessed image or None if face detection fails
        """
        if detect_face:
            # Detect faces
            faces = self.face_detector.detect_faces(image)
            
            if not faces:
                return None
            
            # Use the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            face_img = self.face_detector.extract_face(image, largest_face)
            
            if face_img is None:
                return None
        else:
            face_img = image.copy()
        
        # Resize to target size
        resized = self.resize_image(face_img, self.target_size)
        
        # Normalize
        normalized = self.normalize_image(resized)
        
        return normalized
    
    def augment_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Apply data augmentation to image.
        
        Args:
            image: Input image
            
        Returns:
            List of augmented images
        """
        augmented = []
        
        # Original image
        augmented.append(image.copy())
        
        # Horizontal flip
        flipped = cv2.flip(image, 1)
        augmented.append(flipped)
        
        # Brightness adjustment
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=20)
        augmented.append(bright)
        
        # Slight rotation
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, 5, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        augmented.append(rotated)
        
        # Gaussian noise
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise)
        augmented.append(noisy)
        
        return augmented


class WebcamCapture:
    """
    Webcam capture utility for collecting face images.
    """
    
    def __init__(self, camera_id: int = 0):
        """
        Initialize webcam capture.
        
        Args:
            camera_id: ID of the camera to use
        """
        self.camera_id = camera_id
        self.cap = None
        self.face_detector = FaceDetector()
        self.preprocessor = ImagePreprocessor()
    
    def start_capture(self) -> bool:
        """
        Start webcam capture.
        
        Returns:
            True if successful, False otherwise
        """
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        return True
    
    def stop_capture(self):
        """Stop webcam capture."""
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
    
    def capture_face_images(self, person_name: str, output_dir: str, 
                          num_images: int = 20, show_preview: bool = True) -> int:
        """
        Capture face images for a person.
        
        Args:
            person_name: Name of the person
            output_dir: Directory to save images
            num_images: Number of images to capture
            show_preview: Whether to show camera preview
            
        Returns:
            Number of images successfully captured
        """
        if not self.start_capture():
            return 0
        
        # Create output directory
        person_dir = os.path.join(output_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        captured_count = 0
        frame_count = 0
        
        print(f"Capturing images for {person_name}")
        print("Press SPACE to capture, 'q' to quit")
        
        while captured_count < num_images:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect faces in current frame
            faces = self.face_detector.detect_faces(frame)
            
            # Draw bounding boxes
            display_frame = frame.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Face {len(faces)}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add capture information
            cv2.putText(display_frame, f"Captured: {captured_count}/{num_images}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture, 'q' to quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if show_preview:
                cv2.imshow('Face Capture', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Capture image on space key
            if key == ord(' ') and faces:
                # Use the largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                face_img = self.face_detector.extract_face(frame, largest_face)
                
                if face_img is not None:
                    # Save original image
                    img_filename = f"{person_name}_{captured_count:03d}.jpg"
                    img_path = os.path.join(person_dir, img_filename)
                    cv2.imwrite(img_path, face_img)
                    
                    # Apply augmentation and save additional images
                    augmented_images = self.preprocessor.augment_image(face_img)
                    for i, aug_img in enumerate(augmented_images[1:], 1):  # Skip original
                        aug_filename = f"{person_name}_{captured_count:03d}_aug_{i}.jpg"
                        aug_path = os.path.join(person_dir, aug_filename)
                        cv2.imwrite(aug_path, aug_img)
                    
                    captured_count += 1
                    print(f"Captured image {captured_count}/{num_images}")
            
            # Quit on 'q' key
            elif key == ord('q'):
                break
        
        self.stop_capture()
        print(f"Capture complete. Saved {captured_count} images for {person_name}")
        return captured_count
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get current frame from webcam.
        
        Returns:
            Current frame or None if capture fails
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        return frame if ret else None


def process_raw_images(input_dir: str, output_dir: str, target_size: Tuple[int, int] = (224, 224)):
    """
    Process raw images and save preprocessed versions.
    
    Args:
        input_dir: Directory containing raw images
        output_dir: Directory to save processed images
        target_size: Target size for processed images
    """
    preprocessor = ImagePreprocessor(target_size)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each person's directory
    for person_dir in os.listdir(input_dir):
        person_path = os.path.join(input_dir, person_dir)
        
        if not os.path.isdir(person_path):
            continue
        
        print(f"Processing images for {person_dir}...")
        
        # Create output directory for this person
        output_person_dir = os.path.join(output_dir, person_dir)
        os.makedirs(output_person_dir, exist_ok=True)
        
        # Get all image files
        image_files = glob.glob(os.path.join(person_path, "*.jpg")) + \
                     glob.glob(os.path.join(person_path, "*.png")) + \
                     glob.glob(os.path.join(person_path, "*.jpeg"))
        
        processed_count = 0
        
        for img_file in image_files:
            try:
                # Load image
                image = cv2.imread(img_file)
                if image is None:
                    continue
                
                # Preprocess image
                processed_img = preprocessor.preprocess_image(image, detect_face=True)
                
                if processed_img is not None:
                    # Convert back to uint8 for saving
                    processed_img = (processed_img * 255).astype(np.uint8)
                    
                    # Save processed image
                    base_name = os.path.splitext(os.path.basename(img_file))[0]
                    output_file = os.path.join(output_person_dir, f"{base_name}_processed.jpg")
                    cv2.imwrite(output_file, processed_img)
                    
                    processed_count += 1
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
        
        print(f"Processed {processed_count} images for {person_dir}")


def visualize_face_detection(image_path: str):
    """
    Visualize face detection on an image.
    
    Args:
        image_path: Path to the image file
    """
    detector = FaceDetector()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Detect faces
    faces = detector.detect_faces(image)
    
    # Draw bounding boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display image
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected {len(faces)} faces")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Test webcam capture
    print("Testing webcam capture...")
    
    webcam = WebcamCapture()
    if webcam.start_capture():
        print("Webcam started successfully")
        
        # Capture a test frame
        frame = webcam.get_frame()
        if frame is not None:
            print(f"Frame shape: {frame.shape}")
        
        webcam.stop_capture()
    else:
        print("Failed to start webcam")
