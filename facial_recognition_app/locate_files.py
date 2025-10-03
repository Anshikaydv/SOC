import json
import os

# Check both training history files
file1 = "model/checkpoints/training_history.json"
file2 = "../model/checkpoints/training_history.json"

print("=== TRAINING HISTORY FILE LOCATIONS ===")
print()

print(f"FILE 1: {os.path.abspath(file1)}")
if os.path.exists(file1):
    with open(file1, 'r') as f:
        history1 = json.load(f)
    max_acc1 = max([e["val_accuracy"] for e in history1])
    print(f"  ✅ EXISTS - Max accuracy: {max_acc1:.2f}%")
    print(f"  📊 Total epochs: {len(history1)}")
else:
    print("  ❌ NOT FOUND")

print()

print(f"FILE 2: {os.path.abspath(file2)}")
if os.path.exists(file2):
    with open(file2, 'r') as f:
        history2 = json.load(f)
    max_acc2 = max([e["val_accuracy"] for e in history2])
    print(f"  ✅ EXISTS - Max accuracy: {max_acc2:.2f}%")
    print(f"  📊 Total epochs: {len(history2)}")
else:
    print("  ❌ NOT FOUND")

print()
print("=== WHICH FILE HAS 65.62%? ===")
if os.path.exists(file1):
    for epoch in history1:
        if abs(epoch["val_accuracy"] - 65.62) < 0.01:
            print(f"🎯 FILE 1 - Epoch {epoch['epoch']}: {epoch['val_accuracy']:.2f}%")

if os.path.exists(file2):
    for epoch in history2:
        if abs(epoch["val_accuracy"] - 65.62) < 0.01:
            print(f"🎯 FILE 2 - Epoch {epoch['epoch']}: {epoch['val_accuracy']:.2f}%")
