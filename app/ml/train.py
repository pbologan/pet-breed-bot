#!/usr/bin/env python3
"""
Training script for pet breed classification model

This is a template script. Customize it based on your dataset and requirements.
"""
import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings
from app.ml.model import create_model_for_training
from app.ml.preprocessing import get_transforms
from app.ml.labels import LabelManager, create_sample_label_map


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} "
                f"Acc: {100. * correct / total:.2f}%"
            )

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description="Train pet breed classification model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.data_dir / "processed",
        help="Directory with train/val subdirectories"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=settings.model_path,
        help="Path to save trained model"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained weights"
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Check data directory
    train_dir = args.data_dir / "train"
    val_dir = args.data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        logger.error(f"Train or val directory not found in {args.data_dir}")
        logger.error("Please prepare your dataset first using scripts/prepare_data.py")
        sys.exit(1)

    # Get transforms
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    num_classes = len(train_dataset.classes)
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Create model
    logger.info("Creating model...")
    model = create_model_for_training(num_classes, pretrained=args.pretrained)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training loop
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch [{epoch + 1}/{args.epochs}]")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info(f"New best validation accuracy: {best_val_acc:.2f}%")
            logger.info(f"Saving model to {args.model_output}")

            # Create output directory
            args.model_output.parent.mkdir(parents=True, exist_ok=True)

            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, args.model_output)

        # Update learning rate
        scheduler.step()

    logger.info("\nTraining completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Save label map
    logger.info("Saving label map...")
    label_map = {
        "idx_to_label": {str(i): cls for i, cls in enumerate(train_dataset.classes)},
        "label_to_idx": {cls: i for i, cls in enumerate(train_dataset.classes)}
    }

    import json
    with open(settings.label_map_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    logger.info(f"Label map saved to {settings.label_map_path}")


if __name__ == "__main__":
    main()
