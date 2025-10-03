#!/usr/bin/env python3
"""
Quick user registration script - Register yourself and create negative samples for testing.
"""

import cv2
import os
import time
import numpy as np
from typing import Tuple, List


class QuickRegistration:
    """Quick registration system for testing purposes."""
    
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")
        os.makedirs(self.data_dir, exist_ok=True)
    
    def capture_user_images(self, user_name: str, num_images: int = 10) -> bool:
        """Capture images for a user."""
        user_dir = os.path.join(self.data_dir, user_name)
        os.makedirs(user_dir, exist_ok=True)
        
        print(f"\n📷 Capturing {num_images} images for {user_name}")
        print("Instructions:")
        print("- Look directly at the camera")
        print("- Ensure good lighting")
        print("- Press SPACE to capture, ESC to quit")
        print("- Try slight variations in expression/angle")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return False
        
        captured = 0
        
        while captured < num_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Add capture info
            cv2.putText(frame, f"Capturing for: {user_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Images: {captured}/{num_images}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: Capture | ESC: Quit", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(f"Register {user_name}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space to capture
                timestamp = int(time.time() * 1000)
                filename = f"{user_name}_{captured:03d}_{timestamp}.jpg"
                filepath = os.path.join(user_dir, filename)
                
                # Resize and save
                face_image = cv2.resize(frame, (224, 224))
                cv2.imwrite(filepath, face_image)
                captured += 1
                print(f"✅ Captured {captured}/{num_images}")
                
                # Brief pause
                time.sleep(0.5)
            
            elif key == 27:  # ESC to quit
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if captured >= num_images:
            print(f"✅ Successfully captured {captured} images for {user_name}")
            return True
        else:
            print(f"⚠️ Only captured {captured}/{num_images} images")
            return False
    
    def create_test_negative_samples(self) -> bool:
        """Create simple negative samples for testing."""
        negative_dir = os.path.join(self.data_dir, "negative_samples")
        os.makedirs(negative_dir, exist_ok=True)
        
        print(f"\n🚫 Creating negative samples")
        print("Instructions:")
        print("- Have a friend look at the camera")
        print("- Or use different people")
        print("- These should NOT be the registered user")
        print("- Press SPACE to capture, ESC to quit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return False
        
        captured = 0
        target_count = 15
        
        while captured < target_count:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            cv2.putText(frame, "Negative Samples (Different Person)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Samples: {captured}/{target_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "SPACE: Capture | ESC: Quit", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Negative Samples", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                timestamp = int(time.time() * 1000)
                filename = f"negative_{captured:03d}_{timestamp}.jpg"
                filepath = os.path.join(negative_dir, filename)
                
                face_image = cv2.resize(frame, (224, 224))
                cv2.imwrite(filepath, face_image)
                captured += 1
                print(f"✅ Captured negative sample {captured}/{target_count}")
                time.sleep(0.5)
            
            elif key == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if captured > 0:
            print(f"✅ Created {captured} negative samples")
            return True
        return False
    
    def show_status(self):
        """Show current registration status."""
        print("\n📊 Current Registration Status:")
        print("=" * 40)
        
        if not os.path.exists(self.data_dir):
            print("❌ No data directory found")
            return
        
        users = []
        negative_count = 0
        
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path):
                if item == "negative_samples":
                    import glob
                    negatives = glob.glob(os.path.join(item_path, "*.jpg"))
                    negative_count = len(negatives)
                else:
                    import glob
                    images = glob.glob(os.path.join(item_path, "*.jpg"))
                    users.append((item, len(images)))
        
        print(f"Registered Users: {len(users)}")
        for user, count in users:
            status = "✅" if count >= 10 else "⚠️"
            print(f"  {status} {user}: {count} images")
        
        print(f"Negative Samples: {negative_count}")
        
        print(f"\nStatus: ", end="")
        if len(users) >= 2 and negative_count >= 10:
            print("✅ Ready for training!")
        elif len(users) >= 1 and negative_count >= 5:
            print("⚠️ Minimum data available - should work but recommend more")
        else:
            print("❌ Insufficient data for reliable verification")


def main():
    """Main function."""
    reg = QuickRegistration()
    
    print("🚀 Quick Facial Recognition Setup")
    print("=" * 40)
    
    reg.show_status()
    
    while True:
        print("\n🔧 Quick Setup Options:")
        print("1. Register yourself (primary user)")
        print("2. Register another person")
        print("3. Create negative samples (different people)")
        print("4. Show current status")
        print("5. Exit")
        
        choice = input("\nChoice (1-5): ").strip()
        
        if choice == '1':
            name = input("Enter your name: ").strip()
            if name:
                reg.capture_user_images(name, 15)
        
        elif choice == '2':
            name = input("Enter person's name: ").strip()
            if name:
                reg.capture_user_images(name, 15)
        
        elif choice == '3':
            reg.create_test_negative_samples()
        
        elif choice == '4':
            reg.show_status()
        
        elif choice == '5':
            print("👋 Setup complete!")
            break
        
        else:
            print("❌ Invalid choice")
    
    # Final status
    reg.show_status()
    
    print("\n🎯 Next Steps:")
    print("1. If you have sufficient data, retrain: python retrain_improved_model.py")
    print("2. Test the app: streamlit run app/main.py")
    print("3. The app now has much stricter verification - should reject unknown faces")


if __name__ == "__main__":
    main()
