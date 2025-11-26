#!/usr/bin/env python3
"""
Helper script to download datasets from Kaggle

This script requires Kaggle API credentials.
Setup: https://github.com/Kaggle/kaggle-api#api-credentials
"""
import sys
from pathlib import Path
import argparse
import subprocess
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings


def check_kaggle_installed():
    """Check if Kaggle CLI is installed"""
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_kaggle_credentials():
    """Check if Kaggle credentials are configured"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def download_kaggle_dataset(dataset_id: str, output_dir: Path):
    """
    Download dataset from Kaggle

    Args:
        dataset_id: Kaggle dataset ID (e.g., 'username/dataset-name')
        output_dir: Output directory
    """
    logger.info(f"Downloading Kaggle dataset: {dataset_id}")

    try:
        cmd = ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(output_dir), "--unzip"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.success(f"Successfully downloaded {dataset_id}")
            logger.info(result.stdout)
        else:
            logger.error(f"Failed to download {dataset_id}")
            logger.error(result.stderr)
            return False

        return True

    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download datasets from Kaggle")
    parser.add_argument(
        "--dataset-id",
        type=str,
        help="Kaggle dataset ID (e.g., 'ma7555/cat-breeds-dataset')"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.data_dir / "raw",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--list-popular",
        action="store_true",
        help="List popular pet breed datasets on Kaggle"
    )

    args = parser.parse_args()

    # Check if Kaggle is installed
    if not check_kaggle_installed():
        logger.error("Kaggle CLI is not installed!")
        logger.info("Install it with: pip install kaggle")
        sys.exit(1)

    # Check credentials
    if not check_kaggle_credentials():
        logger.error("Kaggle credentials not found!")
        logger.info("Setup instructions:")
        logger.info("1. Go to https://www.kaggle.com/account")
        logger.info("2. Scroll to 'API' section and click 'Create New API Token'")
        logger.info("3. Move downloaded kaggle.json to ~/.kaggle/")
        logger.info("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)

    if args.list_popular:
        logger.info("\nPopular pet breed datasets on Kaggle:")
        logger.info("\nDogs:")
        logger.info("  • stanford-dogs/dog-breed-identification")
        logger.info("  • gpiosenka/70-dog-breedsimage-data-set")
        logger.info("\nCats:")
        logger.info("  • ma7555/cat-breeds-dataset")
        logger.info("  • shawngregory/cat-breeds-dataset-with-photos")
        logger.info("\nMixed Pets:")
        logger.info("  • tanlikesmath/the-oxfordiiit-pet-dataset")
        logger.info("\nTo download:")
        logger.info("  python scripts/download_kaggle_datasets.py --dataset-id ma7555/cat-breeds-dataset")
        return

    if not args.dataset_id:
        logger.error("Please provide --dataset-id or use --list-popular")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset
    success = download_kaggle_dataset(args.dataset_id, args.output_dir)

    if success:
        logger.success("\nDownload completed!")
        logger.info(f"Dataset saved to: {args.output_dir}")
    else:
        logger.error("\nDownload failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
