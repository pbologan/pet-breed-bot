#!/usr/bin/env python3
"""
Script to prepare and preprocess datasets for training

This script:
1. Organizes images from various dataset formats
2. Creates train/val splits
3. Generates label mappings
4. Provides dataset statistics
"""
import sys
from pathlib import Path
import argparse
import json
import shutil
import random
from collections import defaultdict
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from loguru import logger
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings


def get_image_files(directory: Path, extensions: List[str] = None) -> List[Path]:
    """Get all image files in directory"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

    image_files = []
    for ext in extensions:
        image_files.extend(directory.glob(f'**/*{ext}'))
        image_files.extend(directory.glob(f'**/*{ext.upper()}'))

    return sorted(image_files)


def organize_oxford_pets(input_dir: Path, output_dir: Path, train_split: float = 0.8):
    """
    Organize Oxford-IIIT Pet Dataset
    Expected structure: input_dir/oxford-pets/images/*.jpg
    """
    logger.info("Organizing Oxford-IIIT Pet Dataset...")

    oxford_dir = input_dir / "oxford-pets"
    images_dir = oxford_dir / "images"

    if not images_dir.exists():
        logger.warning(f"Oxford-IIIT images directory not found: {images_dir}")
        return

    # Group images by breed (filename format: Breed_name_123.jpg)
    breed_images = defaultdict(list)

    for img_path in get_image_files(images_dir):
        # Parse filename to get breed name
        # Format: "Abyssinian_1.jpg" -> "Abyssinian"
        filename = img_path.stem
        parts = filename.split('_')
        if len(parts) >= 2:
            # Remove the last number part
            breed_name = '_'.join(parts[:-1])
            breed_images[breed_name].append(img_path)

    logger.info(f"Found {len(breed_images)} breeds in Oxford-IIIT dataset")

    # Split and copy images
    for breed_name, images in breed_images.items():
        train_imgs, val_imgs = train_test_split(
            images,
            train_size=train_split,
            random_state=42
        )

        # Create breed directories
        train_breed_dir = output_dir / "train" / breed_name
        val_breed_dir = output_dir / "val" / breed_name
        train_breed_dir.mkdir(parents=True, exist_ok=True)
        val_breed_dir.mkdir(parents=True, exist_ok=True)

        # Copy training images
        for img_path in train_imgs:
            dst = train_breed_dir / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)

        # Copy validation images
        for img_path in val_imgs:
            dst = val_breed_dir / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)

        logger.info(f"  {breed_name}: {len(train_imgs)} train, {len(val_imgs)} val")


def organize_stanford_dogs(input_dir: Path, output_dir: Path, train_split: float = 0.8):
    """
    Organize Stanford Dogs Dataset
    Expected structure: input_dir/stanford-dogs/Images/n02085620-Chihuahua/*.jpg
    """
    logger.info("Organizing Stanford Dogs Dataset...")

    stanford_dir = input_dir / "stanford-dogs"
    images_dir = stanford_dir / "Images"

    if not images_dir.exists():
        # Try alternative location
        images_dir = stanford_dir / "images"

    if not images_dir.exists():
        logger.warning(f"Stanford Dogs images directory not found: {images_dir}")
        return

    # Each subdirectory is a breed
    breed_dirs = [d for d in images_dir.iterdir() if d.is_dir()]

    logger.info(f"Found {len(breed_dirs)} breeds in Stanford Dogs dataset")

    for breed_dir in breed_dirs:
        # Clean breed name (remove ImageNet ID prefix)
        # Format: "n02085620-Chihuahua" -> "Chihuahua"
        breed_name = breed_dir.name.split('-', 1)[-1] if '-' in breed_dir.name else breed_dir.name
        breed_name = breed_name.replace('_', ' ')

        images = get_image_files(breed_dir)

        if not images:
            continue

        train_imgs, val_imgs = train_test_split(
            images,
            train_size=train_split,
            random_state=42
        )

        # Create breed directories
        train_breed_dir = output_dir / "train" / breed_name
        val_breed_dir = output_dir / "val" / breed_name
        train_breed_dir.mkdir(parents=True, exist_ok=True)
        val_breed_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        for img_path in train_imgs:
            dst = train_breed_dir / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)

        for img_path in val_imgs:
            dst = val_breed_dir / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)

        logger.info(f"  {breed_name}: {len(train_imgs)} train, {len(val_imgs)} val")


def organize_cub_birds(input_dir: Path, output_dir: Path, train_split: float = 0.8):
    """
    Organize CUB-200-2011 Birds Dataset
    Expected structure: input_dir/cub-200-2011/CUB_200_2011/images/001.Black_footed_Albatross/*.jpg
    """
    logger.info("Organizing CUB-200-2011 Birds Dataset...")

    cub_dir = input_dir / "cub-200-2011"
    images_dir = cub_dir / "CUB_200_2011" / "images"

    if not images_dir.exists():
        logger.warning(f"CUB-200-2011 images directory not found: {images_dir}")
        return

    # Each subdirectory is a bird species
    species_dirs = [d for d in images_dir.iterdir() if d.is_dir()]

    logger.info(f"Found {len(species_dirs)} bird species in CUB-200-2011 dataset")

    for species_dir in species_dirs:
        # Clean species name (remove number prefix)
        # Format: "001.Black_footed_Albatross" -> "Black footed Albatross"
        species_name = species_dir.name.split('.', 1)[-1] if '.' in species_dir.name else species_dir.name
        species_name = species_name.replace('_', ' ')

        images = get_image_files(species_dir)

        if not images:
            continue

        train_imgs, val_imgs = train_test_split(
            images,
            train_size=train_split,
            random_state=42
        )

        # Create species directories
        train_species_dir = output_dir / "train" / species_name
        val_species_dir = output_dir / "val" / species_name
        train_species_dir.mkdir(parents=True, exist_ok=True)
        val_species_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        for img_path in train_imgs:
            dst = train_species_dir / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)

        for img_path in val_imgs:
            dst = val_species_dir / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)

        logger.info(f"  {species_name}: {len(train_imgs)} train, {len(val_imgs)} val")


def organize_images(
    input_dir: Path,
    output_dir: Path,
    train_split: float = 0.8,
    val_split: float = 0.2
):
    """
    Organize images into train/val directories from all available datasets

    Args:
        input_dir: Directory containing raw dataset
        output_dir: Output directory for organized dataset
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
    """
    logger.info("=" * 60)
    logger.info(f"Organizing images from {input_dir} to {output_dir}")
    logger.info("=" * 60)

    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Organize different datasets
    datasets_organized = []

    if (input_dir / "oxford-pets").exists():
        organize_oxford_pets(input_dir, output_dir, train_split)
        datasets_organized.append("Oxford-IIIT Pet Dataset")

    if (input_dir / "stanford-dogs").exists():
        organize_stanford_dogs(input_dir, output_dir, train_split)
        datasets_organized.append("Stanford Dogs Dataset")

    if (input_dir / "cub-200-2011").exists():
        organize_cub_birds(input_dir, output_dir, train_split)
        datasets_organized.append("CUB-200-2011 Birds Dataset")

    if datasets_organized:
        logger.success(f"\nOrganized {len(datasets_organized)} datasets:")
        for dataset in datasets_organized:
            logger.success(f"  ✓ {dataset}")
    else:
        logger.warning("No recognized datasets found in input directory")
        logger.info("Supported datasets: oxford-pets, stanford-dogs, cub-200-2011")


def create_label_map(dataset_dir: Path, output_path: Path):
    """
    Create label mapping JSON from dataset

    Args:
        dataset_dir: Directory containing organized dataset
        output_path: Path to save label map JSON
    """
    logger.info(f"Creating label map from {dataset_dir}")

    # Get all class directories
    class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    class_names = sorted([d.name for d in class_dirs])

    # Create mappings
    idx_to_label = {i: name for i, name in enumerate(class_names)}
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    # Save to JSON
    label_map = {
        "idx_to_label": idx_to_label,
        "label_to_idx": label_to_idx
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    logger.info(f"Label map saved to {output_path}")
    logger.info(f"Total classes: {len(class_names)}")


def check_image_valid(image_path: Path) -> Tuple[bool, str]:
    """Check if image is valid and readable"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True, "OK"
    except Exception as e:
        return False, str(e)


def generate_statistics(dataset_dir: Path):
    """
    Generate and display dataset statistics

    Args:
        dataset_dir: Directory containing organized dataset
    """
    logger.info("=" * 60)
    logger.info("Generating dataset statistics...")
    logger.info("=" * 60)

    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"

    if not train_dir.exists() and not val_dir.exists():
        logger.error(f"No train or val directories found in {dataset_dir}")
        return

    # Statistics containers
    stats = {
        'train': defaultdict(int),
        'val': defaultdict(int)
    }
    image_sizes = []
    corrupted_images = []
    total_images = {'train': 0, 'val': 0}

    # Analyze train set
    if train_dir.exists():
        logger.info("\nAnalyzing training set...")
        class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]

        for class_dir in tqdm(class_dirs, desc="Train classes"):
            class_name = class_dir.name
            images = get_image_files(class_dir)
            stats['train'][class_name] = len(images)
            total_images['train'] += len(images)

            # Sample image sizes from first 10 images per class
            for img_path in images[:10]:
                valid, msg = check_image_valid(img_path)
                if valid:
                    try:
                        with Image.open(img_path) as img:
                            image_sizes.append(img.size)
                    except:
                        corrupted_images.append(str(img_path))
                else:
                    corrupted_images.append(str(img_path))

    # Analyze validation set
    if val_dir.exists():
        logger.info("Analyzing validation set...")
        class_dirs = [d for d in val_dir.iterdir() if d.is_dir()]

        for class_dir in tqdm(class_dirs, desc="Val classes"):
            class_name = class_dir.name
            images = get_image_files(class_dir)
            stats['val'][class_name] = len(images)
            total_images['val'] += len(images)

    # Display statistics
    logger.info("\n" + "=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 60)

    # Overall counts
    logger.info(f"\n📊 Overall Statistics:")
    logger.info(f"  Training images: {total_images['train']:,}")
    logger.info(f"  Validation images: {total_images['val']:,}")
    logger.info(f"  Total images: {total_images['train'] + total_images['val']:,}")
    logger.info(f"  Number of classes: {len(stats['train'])}")

    # Train/Val split
    total = total_images['train'] + total_images['val']
    if total > 0:
        train_pct = (total_images['train'] / total) * 100
        val_pct = (total_images['val'] / total) * 100
        logger.info(f"  Train/Val split: {train_pct:.1f}% / {val_pct:.1f}%")

    # Class distribution
    logger.info(f"\n📈 Class Distribution:")
    if stats['train']:
        min_train = min(stats['train'].values())
        max_train = max(stats['train'].values())
        avg_train = sum(stats['train'].values()) / len(stats['train'])
        logger.info(f"  Training images per class:")
        logger.info(f"    Min: {min_train}, Max: {max_train}, Avg: {avg_train:.1f}")

    # Top 10 classes by count
    if stats['train']:
        logger.info(f"\n🔝 Top 10 Classes (by training images):")
        sorted_classes = sorted(stats['train'].items(), key=lambda x: x[1], reverse=True)
        for i, (class_name, count) in enumerate(sorted_classes[:10], 1):
            val_count = stats['val'].get(class_name, 0)
            logger.info(f"  {i:2d}. {class_name:30s} - Train: {count:4d}, Val: {val_count:4d}")

    # Bottom 10 classes
    if len(stats['train']) > 10:
        logger.info(f"\n⚠️  Bottom 10 Classes (smallest training sets):")
        sorted_classes = sorted(stats['train'].items(), key=lambda x: x[1])
        for i, (class_name, count) in enumerate(sorted_classes[:10], 1):
            val_count = stats['val'].get(class_name, 0)
            logger.info(f"  {i:2d}. {class_name:30s} - Train: {count:4d}, Val: {val_count:4d}")

    # Image size statistics
    if image_sizes:
        logger.info(f"\n🖼️  Image Size Statistics (sampled):")
        widths = [size[0] for size in image_sizes]
        heights = [size[1] for size in image_sizes]
        logger.info(f"  Width:  Min: {min(widths):4d}, Max: {max(widths):4d}, Avg: {sum(widths)/len(widths):.0f}")
        logger.info(f"  Height: Min: {min(heights):4d}, Max: {max(heights):4d}, Avg: {sum(heights)/len(heights):.0f}")

    # Corrupted images
    if corrupted_images:
        logger.warning(f"\n⚠️  Found {len(corrupted_images)} corrupted/unreadable images:")
        for img_path in corrupted_images[:10]:
            logger.warning(f"  - {img_path}")
        if len(corrupted_images) > 10:
            logger.warning(f"  ... and {len(corrupted_images) - 10} more")
    else:
        logger.success(f"\n✓ No corrupted images found")

    # Save statistics to file
    stats_file = dataset_dir / "dataset_statistics.json"
    stats_data = {
        'total_train': total_images['train'],
        'total_val': total_images['val'],
        'num_classes': len(stats['train']),
        'train_classes': dict(stats['train']),
        'val_classes': dict(stats['val']),
        'corrupted_images': corrupted_images
    }

    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)

    logger.info(f"\n💾 Statistics saved to: {stats_file}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Organize all datasets and create label map
  python prepare_data.py --create-label-map

  # Organize with custom split and generate statistics
  python prepare_data.py --train-split 0.85 --stats

  # Full pipeline: organize, create labels, and generate stats
  python prepare_data.py --create-label-map --stats

  # Custom directories
  python prepare_data.py --input-dir /path/to/raw --output-dir /path/to/processed
        """
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=settings.data_dir / "raw",
        help="Input directory with raw datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.data_dir / "processed",
        help="Output directory for processed datasets"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Proportion of data for training (default: 0.8)"
    )
    parser.add_argument(
        "--create-label-map",
        action="store_true",
        help="Create label map JSON after organizing"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Generate dataset statistics"
    )
    parser.add_argument(
        "--dataset",
        choices=["oxford-pets", "stanford-dogs", "cub-birds", "all"],
        default="all",
        help="Specific dataset to organize (default: all)"
    )

    args = parser.parse_args()

    try:
        logger.info("\n" + "=" * 60)
        logger.info("DATASET PREPARATION PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Input directory: {args.input_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Train/Val split: {args.train_split:.0%} / {1-args.train_split:.0%}")
        logger.info("=" * 60 + "\n")

        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Check if input directory exists
        if not args.input_dir.exists():
            logger.error(f"Input directory not found: {args.input_dir}")
            logger.info("Please download datasets first using:")
            logger.info("  python scripts/download_datasets.py --dataset all")
            return

        # Organize images
        logger.info("Step 1: Organizing images...")
        organize_images(
            args.input_dir,
            args.output_dir,
            train_split=args.train_split,
            val_split=1 - args.train_split
        )

        # Create label map
        if args.create_label_map:
            logger.info("\nStep 2: Creating label map...")
            train_dir = args.output_dir / "train"
            if train_dir.exists() and any(train_dir.iterdir()):
                create_label_map(
                    train_dir,
                    settings.label_map_path
                )
                logger.success(f"Label map created at {settings.label_map_path}")
            else:
                logger.error(f"Train directory is empty or not found: {train_dir}")
                logger.info("Make sure images were organized successfully")

        # Generate statistics
        if args.stats:
            logger.info("\nStep 3: Generating statistics...")
            generate_statistics(args.output_dir)

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.success("✓ Data preparation completed successfully!")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("1. Review the organized data structure:")
        logger.info(f"   ls -la {args.output_dir}/train")
        logger.info("2. Train the model:")
        logger.info(f"   python -m app.ml.train --data-dir {args.output_dir}")
        logger.info("\n")

    except KeyboardInterrupt:
        logger.warning("\n\nData preparation interrupted by user")
    except Exception as e:
        logger.error(f"\n\nError during data preparation: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
