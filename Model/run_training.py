import os
import subprocess
import sys

def run_pipeline():
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('processed_data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Step 1: Run data augmentation
    print("Starting data augmentation...")
    try:
        subprocess.run([sys.executable, 'data_augmentation.py'], check=True)
        print("Data augmentation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during data augmentation: {e}")
        return
    
    # Step 2: Run model training
    print("\nStarting model training...")
    try:
        subprocess.run([sys.executable, 'cnn_model_training.py'], check=True)
        print("Model training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during model training: {e}")
        return

    print("\nComplete pipeline finished successfully!")

if __name__ == "__main__":
    run_pipeline()