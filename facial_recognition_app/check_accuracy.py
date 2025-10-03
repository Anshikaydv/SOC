#!/usr/bin/env python3
"""
Check the accuracy and performance metrics of the trained LFW model.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path

def check_training_accuracy():
    """Check the accuracy from training history."""
    # Check training history file
    history_file = os.path.join("model", "checkpoints", "training_history.json")
    
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            if history:
                # Get best validation accuracy
                best_val_acc = max([epoch['val_accuracy'] for epoch in history])
                print(f"accuracy = {best_val_acc:.2f}%")
                
            else:
                print("No training data")
                
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Model not trained")

def main():
    """Main function to check accuracy."""
    try:
        check_training_accuracy()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
