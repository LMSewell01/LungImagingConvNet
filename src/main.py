import argparse
import sys
import os

# Import functions from our modular scripts
from train import main as train_main # Renamed to avoid conflict with main() in this script
from evaluation import main as evaluate_main
from predict_and_visualize import main as predict_main

def main():
    parser = argparse.ArgumentParser(description="Run lung abnormality segmentation pipeline.")

    # Define command-line arguments
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate', 'predict'],
                        help="Mode to run: 'train', 'evaluate', or 'predict'")
    
    args = parser.parse_args()

    # Create necessary directories if they don't exist
    os.makedirs('data/raw/images', exist_ok=True)
    os.makedirs('data/raw/masks', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    if args.mode == 'train':
        print("--- Running Training Mode ---")
        train_main()
    elif args.mode == 'evaluate':
        print("--- Running Evaluation Mode ---")
        evaluate_main()
    elif args.mode == 'predict':
        print("--- Running Prediction and Visualization Mode ---")
        predict_main()
    else:
        print(f"Error: Invalid mode '{args.mode}'. Choose from 'train', 'evaluate', 'predict'.")
        sys.exit(1)

if __name__ == "__main__":
    main()