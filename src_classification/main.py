import argparse
import os
import sys

# Import the main functions from your scripts directly (assuming they are in the same 'src' directory)
from .train import main as train_main
from .evaluate import main as evaluate_main
from .predict import main as predict_main

def main():
    """
    Main entry point for the COVID-19 X-ray Classification project.
    Allows running different tasks (train, evaluate, predict) via command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run different tasks for the COVID-19 X-ray Classification project."
    )
    
    # Define command-line arguments for each task
    parser.add_argument('--train', action='store_true', help='Run the model training script.')
    parser.add_argument('--evaluate', action='store_true', help='Run the model evaluation script on the test set.')
    parser.add_argument('--predict', action='store_true', help='Run the prediction and visualization script.')

    args = parser.parse_args()

    # Determine which script to run based on arguments
    if args.train:
        print("--- Running Training Script ---")
        train_main()
    elif args.evaluate:
        print("--- Running Evaluation Script ---")
        evaluate_main()
    elif args.predict:
        print("--- Running Prediction and Visualization Script ---")
        predict_main()
    else:
        print("No task specified. Use --train, --evaluate, or --predict.")
        parser.print_help()

if __name__ == '__main__':
    main()
