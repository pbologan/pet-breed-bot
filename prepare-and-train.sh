#!/bin/bash
# prepare_and_train.sh

set -e

echo "Step 1: Downloading datasets..."
python scripts/download_datasets.py --dataset all

echo "Step 2: Preparing data..."
python scripts/prepare_data.py --create-label-map --stats

echo "Step 3: Training model..."
python -m app.ml.train --data-dir data/processed --epochs 20

echo "Step 4: Evaluating model..."
python -m app.ml.eval --data-dir data/processed/val

echo "Done!"
