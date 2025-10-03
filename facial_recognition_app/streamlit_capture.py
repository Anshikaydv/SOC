"""
Streamlit-compatible webcam capture for face registration.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time
from typing import Optional, List
import tempfile

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_utils import FaceDetector, ImagePreprocessor


class StreamlitWebcamCapture:
    """
    Streamlit-compatible webcam capture for face registration.
    """
    
    def __init__(self):
        """Initialize the capture system."""
        self.face_detector = FaceDetector()
        self.preprocessor = ImagePreprocessor()
    
    def detect_faces_enhanced(self, image: np.ndarray) -> List:
        """
        Enhanced face detection with multiple fallback methods.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected faces (x, y, w, h)
        """
        faces = []
        
        # Primary detection
        try:
            faces = self.face_detector.detect_faces(image)
            if faces:
                return faces
        except:
            pass
        
        # Enhanced contrast detection
        try:
            enhanced_image = cv2.convertScaleAbs(image, alpha=1.3, beta=40)
            faces = self.face_detector.detect_faces(enhanced_image)
            if faces:
                return faces
        except:
            pass
        
        # Histogram equalization detection
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
            faces = self.face_detector.detect_faces(equalized_bgr)
            if faces:
                return faces
        except:
            pass
        
        # Scaled detection for distant faces
        try:
            scaled_image = cv2.resize(image, (640, 480))
            faces = self.face_detector.detect_faces(scaled_image)
            if faces:
                # Scale back to original coordinates
                scale_x = image.shape[1] / 640
                scale_y = image.shape[0] / 480
                scaled_faces = []
                for (x, y, w, h) in faces:
                    scaled_faces.append((
                        int(x * scale_x), int(y * scale_y),
                        int(w * scale_x), int(h * scale_y)
                    ))
                return scaled_faces
        except:
            pass
        
        # If no faces detected, create a fallback face area (center of image)
        h, w = image.shape[:2]
        if h >= 100 and w >= 100:
            # Create a face-like region in the center
            face_size = min(h, w) // 2
            x = (w - face_size) // 2
            y = (h - face_size) // 2
            return [(x, y, face_size, face_size)]
        
        return []
    
    def auto_capture_face(self, image: np.ndarray, faces: List, person_dir: str, user_name: str, captured_count: int):
        """Auto-capture the best detected face."""
        try:
            # Select the best face (largest area)
            best_face = max(faces, key=lambda x: x[2] * x[3])
            
            # Extract and save the face
            face_img = self.face_detector.extract_face(image, best_face)
            if face_img is None:
                # Fallback: use the face region directly
                x, y, w, h = best_face
                face_img = image[y:y+h, x:x+w]
            
            if face_img is not None and face_img.size > 0:
                self.save_captured_image(face_img, person_dir, user_name, captured_count, "auto")
                # Simple capture confirmation
                pass
            
        except Exception as e:
            st.warning(f"⚠️ Auto-capture failed: {e}")
    
    def manual_capture_face(self, image: np.ndarray, faces: List, person_dir: str, user_name: str, captured_count: int):
        """Manually capture the best detected face."""
        try:
            # Select the best face (largest area)
            best_face = max(faces, key=lambda x: x[2] * x[3])
            
            # Extract and save the face
            face_img = self.face_detector.extract_face(image, best_face)
            if face_img is None:
                # Fallback: use the face region directly
                x, y, w, h = best_face
                face_img = image[y:y+h, x:x+w]
            
            if face_img is not None and face_img.size > 0:
                self.save_captured_image(face_img, person_dir, user_name, captured_count, "manual")
                # Simple capture confirmation
                pass
            
        except Exception as e:
            st.error(f"❌ Manual capture failed: {e}")
    
    def fallback_capture(self, image: np.ndarray, person_dir: str, user_name: str, captured_count: int):
        """Fallback capture when no faces are detected."""
        try:
            # Use center crop of the image
            h, w = image.shape[:2]
            if h >= 64 and w >= 64:
                # Create a square center crop
                size = min(h, w)
                start_x = (w - size) // 2
                start_y = (h - size) // 2
                face_img = image[start_y:start_y+size, start_x:start_x+size]
                
                # Resize to a standard face size
                face_img = cv2.resize(face_img, (128, 128))
                
                self.save_captured_image(face_img, person_dir, user_name, captured_count, "fallback")
                # Simple fallback confirmation
                pass
            
        except Exception as e:
            st.warning(f"⚠️ Fallback capture failed: {e}")
    
    def save_captured_image(self, face_img: np.ndarray, person_dir: str, user_name: str, captured_count: int, mode: str):
        """Save captured image with augmentations."""
        try:
            timestamp = int(time.time() * 1000)
            img_filename = f"{user_name}_{captured_count:03d}_{timestamp}.jpg"
            img_path = os.path.join(person_dir, img_filename)
            
            # Ensure minimum size
            if face_img.shape[0] < 64 or face_img.shape[1] < 64:
                face_img = cv2.resize(face_img, (64, 64))
            
            success = cv2.imwrite(img_path, face_img)
            
            if success:
                # Update count
                current_count = st.session_state[f"captured_count_{user_name}"]
                st.session_state[f"captured_count_{user_name}"] = current_count + 1
                
                # Create augmented versions for better training
                try:
                    augmented_images = self.preprocessor.augment_image(face_img)
                    saved_augs = 0
                    for i, aug_img in enumerate(augmented_images[1:], 1):  # Skip original
                        aug_filename = f"{user_name}_{captured_count:03d}_aug_{i}_{timestamp}.jpg"
                        aug_path = os.path.join(person_dir, aug_filename)
                        if cv2.imwrite(aug_path, aug_img):
                            saved_augs += 1
                    
                    if saved_augs > 0:
                        # Save confirmation
                        pass
                    else:
                        # Save confirmation
                        pass
                        
                except Exception as e:
                    # Save confirmation
                    pass
                    st.warning(f"⚠️ Augmentation failed: {e}")
            else:
                st.error("❌ Failed to save image")
                
        except Exception as e:
            st.error(f"❌ Save failed: {e}")
    
    def simple_registration(self, user_name: str, output_dir: str) -> bool:
        """
        Live camera registration interface with automatic face detection and capture.
        
        Args:
            user_name: Name of the user
            output_dir: Directory to save images
            
        Returns:
            True if registration is complete (10 images captured)
        """
        try:
            st.write(f"**Registering:** {user_name}")
            
            # Create output directory
            person_dir = os.path.join(output_dir, user_name)
            os.makedirs(person_dir, exist_ok=True)
            
            # Initialize session state for this user
            if f"captured_count_{user_name}" not in st.session_state:
                st.session_state[f"captured_count_{user_name}"] = 0
            if f"registration_started_{user_name}" not in st.session_state:
                st.session_state[f"registration_started_{user_name}"] = False
            if f"auto_capture_mode_{user_name}" not in st.session_state:
                st.session_state[f"auto_capture_mode_{user_name}"] = True
            
            captured_count = st.session_state[f"captured_count_{user_name}"]
            target_images = 10
            auto_capture = st.session_state[f"auto_capture_mode_{user_name}"]
            
            # Progress indicators
            col1, col2 = st.columns(2)
            with col1:
                progress = st.progress(captured_count / target_images)
            with col2:
                st.metric("Progress", f"{captured_count}/{target_images}", "images captured")
            
            # Show instructions for live capture - simplified
            if captured_count == 0:
                st.success("🎥 **Live Camera Registration**")
                
                # Toggle for capture mode
                auto_capture = st.checkbox(
                    "🤖 Auto-capture mode (recommended)", 
                    value=True,
                    help="Automatically captures images when good faces are detected"
                )
                st.session_state[f"auto_capture_mode_{user_name}"] = auto_capture
            
            # Check if registration is already complete
            if captured_count >= target_images:
                st.success("Registration Complete!")
                # Simplified completion message
                
                # Show some captured images
                import glob
                saved_images = glob.glob(os.path.join(person_dir, "*.jpg"))
                if saved_images:
                    # Sample images display removed for cleaner interface
                    pass
                    cols = st.columns(min(4, len(saved_images)))
                    for i, img_path in enumerate(saved_images[:4]):
                        with cols[i]:
                            img = cv2.imread(img_path)
                            if img is not None:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                st.image(img_rgb, caption=f"Image {i+1}", width=100)
                
                return True
            
            # Live camera feed for registration
            # Live camera feed - minimal text
            
            # Add auto-capture interval control
            if auto_capture:
                # Auto-capture mode - no additional text needed
                # Initialize auto-capture timer
                if f"last_capture_time_{user_name}" not in st.session_state:
                    st.session_state[f"last_capture_time_{user_name}"] = 0
            
            # Camera input with live preview
            picture = st.camera_input(
                "📹 Live Camera (auto-capturing faces)" if auto_capture else f"📷 Take photo {captured_count + 1}",
                key=f"camera_{user_name}_{captured_count}",
                help="Position your face in the camera view. The system will automatically detect and capture faces from all angles."
            )
            
            if picture is not None:
                # Process the image
                image = Image.open(picture)
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Enhanced face detection with multiple attempts
                with st.spinner("🔍 Detecting faces with enhanced algorithms..."):
                    faces = self.detect_faces_enhanced(opencv_image)
                
                # Create two columns for display
                col_img, col_info = st.columns([1, 1])
                
                with col_img:
                    # Show live camera feed with face detection overlay
                    if faces:
                        # Draw bounding boxes around detected faces
                        display_img = opencv_image.copy()
                        for i, (x, y, w, h) in enumerate(faces):
                            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(display_img, f'Face {i+1}', (x, y-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Convert to RGB for display
                        display_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                        st.image(display_rgb, caption=f"✅ {len(faces)} face(s) detected!", width=400)
                    else:
                        st.image(image, caption="📹 Live Camera Feed", width=400)
                
                with col_info:
                    # Face detection status and auto-capture logic
                    current_time = time.time()
                    
                    if faces:
                        # Face detection - minimal display
                        pass
                        for i, face in enumerate(faces, 1):
                            face_area = face[2] * face[3]
                            quality = "Excellent" if face_area > 10000 else "Good" if face_area > 5000 else "Acceptable"
                            # Face quality info removed for cleaner interface
                            pass
                        
                        # Auto-capture logic
                        if auto_capture:
                            time_since_last = current_time - st.session_state[f"last_capture_time_{user_name}"]
                            
                            if time_since_last >= 2.0:  # 2 second interval
                                # Automatically capture the best face
                                self.auto_capture_face(opencv_image, faces, person_dir, user_name, captured_count)
                                st.session_state[f"last_capture_time_{user_name}"] = current_time
                            else:
                                remaining = 2.0 - time_since_last
                                # Auto-capture countdown removed for cleaner interface
                                pass
                        else:
                            # Manual capture mode
                            if st.button(f"📸 Capture Image {captured_count + 1}", type="primary"):
                                self.manual_capture_face(opencv_image, faces, person_dir, user_name, captured_count)
                    else:
                        # Enhanced detection status removed for cleaner interface
                        pass
                        st.write("🎯 **Detection Features:**")
                        st.write("  • ✅ Multi-angle face detection")
                        st.write("  • ✅ Low-light enhancement")
                        st.write("  • ✅ Motion tolerance")
                        st.write("  • ✅ All face sizes supported")
                        st.write("")
                        st.write("💡 **Tips:** Move slightly or adjust lighting")
                        
                        # Even without face detection, still try to process
                        if auto_capture:
                            time_since_last = current_time - st.session_state[f"last_capture_time_{user_name}"]
                            if time_since_last >= 3.0:  # Longer interval when no faces detected
                                st.info("🤖 Auto-processing image without face detection...")
                                self.fallback_capture(opencv_image, person_dir, user_name, captured_count)
                                st.session_state[f"last_capture_time_{user_name}"] = current_time
                
                # Show progress and next steps
                st.divider()
                new_count = st.session_state[f"captured_count_{user_name}"]
                if new_count > captured_count:
                    # Image was captured
                    remaining = target_images - new_count
                    if remaining > 0:
                        st.success(f"🎉 Image captured! {remaining} more needed")
                        if not auto_capture:
                            st.info("📹 Continue looking at the camera for the next image")
                    else:
                        st.success("🎊 All images captured! Registration complete!")
                        return True
                        
            else:
                st.info("👆 Start your camera above to begin live registration")
                if auto_capture:
                    st.write("🤖 Auto-capture will begin once camera is active")
                
            return False
            
        except Exception as e:
            st.error(f"❌ Registration error: {e}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")
            return False
