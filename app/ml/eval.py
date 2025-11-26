#!/usr/bin/env python3
"""
Evaluation script for pet breed classification model
"""
import sys
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings
from app.ml.model import PetBreedClassifier
from app.ml.preprocessing import get_transforms


def evaluate_model(model, dataloader, device, class_names):
    """Evaluate model on dataset"""
    model.model.eval()
    all_preds = []
    all_labels = []

    logger.info("Evaluating model...")

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            outputs = model.model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate metrics
    logger.info("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    ))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{cm}")

    # Overall accuracy
    accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    logger.info(f"\nOverall Accuracy: {accuracy:.2f}%")

    return accuracy, cm


def main():
    parser = argparse.ArgumentParser(description="Evaluate pet breed classification model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.data_dir / "processed" / "val",
        help="Directory with validation data"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=settings.model_path,
        help="Path to trained model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )

    args = parser.parse_args()

    # Check if model exists
    if not args.model_path.exists():
        logger.error(f"Model not found at {args.model_path}")
        sys.exit(1)

    # Check if data directory exists
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("Loading model...")
    model = PetBreedClassifier(
        model_path=args.model_path,
        label_map_path=settings.label_map_path,
        device=str(device)
    )

    # Load dataset
    logger.info("Loading dataset...")
    val_transform = get_transforms(train=False)
    val_dataset = datasets.ImageFolder(args.data_dir, transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    logger.info(f"Number of validation samples: {len(val_dataset)}")
    logger.info(f"Number of classes: {len(val_dataset.classes)}")

    # Evaluate
    accuracy, cm = evaluate_model(
        model,
        val_loader,
        device,
        val_dataset.classes
    )

    logger.info("\nEvaluation completed!")


if __name__ == "__main__":
    main()
