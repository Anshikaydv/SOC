#!/usr/bin/env python3
"""
Script to completely clear all registered users and reset the facial recognition system.
This script removes all user data, session files, and resets the system to a clean state.
"""

import os
import shutil
import glob
import json

def clear_all_users():
    """Clear all registered users and associated data."""
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths to clean
    processed_dir = os.path.join(base_dir, "data", "processed")
    raw_dir = os.path.join(base_dir, "data", "raw")
    logs_dir = os.path.join(base_dir, "logs")
    
    print("🗑️  Clearing all registered users...")
    
    # Clear processed data (registered users)
    if os.path.exists(processed_dir):
        for item in os.listdir(processed_dir):
            item_path = os.path.join(processed_dir, item)
            if os.path.isdir(item_path) and item not in [".", ".."]:
                try:
                    shutil.rmtree(item_path)
                    print(f"✅ Deleted user directory: {item}")
                except Exception as e:
                    print(f"❌ Error deleting {item}: {e}")
    
    # Clear any raw user data
    if os.path.exists(raw_dir):
        for item in os.listdir(raw_dir):
            item_path = os.path.join(raw_dir, item)
            if os.path.isdir(item_path) and item not in [".", ".."]:
                try:
                    shutil.rmtree(item_path)
                    print(f"✅ Deleted raw user directory: {item}")
                except Exception as e:
                    print(f"❌ Error deleting raw {item}: {e}")
    
    # Clear log files
    if os.path.exists(logs_dir):
        log_files = glob.glob(os.path.join(logs_dir, "*.log"))
        for log_file in log_files:
            try:
                os.remove(log_file)
                print(f"✅ Deleted log file: {os.path.basename(log_file)}")
            except Exception as e:
                print(f"❌ Error deleting log {os.path.basename(log_file)}: {e}")
    
    # Clear any cached session files
    cache_patterns = [
        os.path.join(base_dir, "*.cache"),
        os.path.join(base_dir, "*.session"),
        os.path.join(base_dir, "*.tmp"),
        os.path.join(base_dir, ".streamlit", "*.cache"),
    ]
    
    for pattern in cache_patterns:
        for cache_file in glob.glob(pattern):
            try:
                if os.path.isfile(cache_file):
                    os.remove(cache_file)
                    print(f"✅ Deleted cache file: {os.path.basename(cache_file)}")
            except Exception as e:
                print(f"❌ Error deleting cache {os.path.basename(cache_file)}: {e}")
    
    print("\n🎉 All registered users have been successfully deleted!")
    print("📝 Summary:")
    print("   • All user directories removed")
    print("   • All training images deleted")
    print("   • Log files cleared")
    print("   • Cache files removed")
    print("\n💡 The facial recognition system is now reset to a clean state.")
    print("🔄 You can now register new users or restart the application.")

if __name__ == "__main__":
    # Confirmation prompt
    print("⚠️  WARNING: This will delete ALL registered users and their data!")
    print("📊 This action cannot be undone.")
    
    response = input("\n❓ Are you sure you want to proceed? (type 'yes' to confirm): ")
    
    if response.lower() == 'yes':
        clear_all_users()
    else:
        print("❌ Operation cancelled. No data was deleted.")
