#!/bin/bash
set -e

echo "[INFO] Start TRAINING with Data Preparation..."
python -m src.training.data_prep --input_dir /data/training/raw --output_dir /data/training/processed
echo "[INFO] Data preparation completed successfully."
echo "[INFO] Training model..."
python -m src.training.train_model --input_dir /data/training/processed --output_dir /output
echo "[INFO] Training pipeline completed successfully."
