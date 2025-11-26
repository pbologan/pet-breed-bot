#!/usr/bin/env python3
"""
Script to download pet breed datasets

This script downloads popular pet breed datasets:
- Oxford-IIIT Pet Dataset (dogs and cats)
- Stanford Dogs Dataset
- CUB-200-2011 Birds Dataset
"""
import sys
from pathlib import Path
import argparse
import urllib.request
import tarfile
import zipfile
import shutil
from tqdm import tqdm
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, description: str = "Downloading"):
    """
    Download file with progress bar

    Args:
        url: URL to download from
        output_path: Path to save the file
        description: Description for progress bar
    """
    logger.info(f"Downloading from {url}")
    logger.info(f"Saving to {output_path}")

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=description) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)

    logger.info(f"Download completed: {output_path}")


def extract_tar(tar_path: Path, extract_to: Path):
    """Extract tar archive"""
    logger.info(f"Extracting {tar_path.name}...")
    with tarfile.open(tar_path, 'r:*') as tar:
        tar.extractall(extract_to)
    logger.info(f"Extraction completed to {extract_to}")


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract zip archive"""
    logger.info(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    logger.info(f"Extraction completed to {extract_to}")


def download_oxford_pets(output_dir: Path):
    """
    Download Oxford-IIIT Pet Dataset
    URL: https://www.robots.ox.ac.uk/~vgg/data/pets/

    This dataset contains 37 category pet dataset with roughly 200 images for each class.
    The images have a large variations in scale, pose and lighting.
    """
    logger.info("=" * 60)
    logger.info("Downloading Oxford-IIIT Pet Dataset...")
    logger.info("=" * 60)

    dataset_dir = output_dir / "oxford-pets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # URLs for Oxford-IIIT Pet Dataset
    images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

    try:
        # Download images
        images_tar = dataset_dir / "images.tar.gz"
        if not images_tar.exists():
            download_file(images_url, images_tar, "Oxford-IIIT Images")
            extract_tar(images_tar, dataset_dir)
            logger.info("Images extracted successfully")
        else:
            logger.info("Images already downloaded, skipping...")

        # Download annotations
        annotations_tar = dataset_dir / "annotations.tar.gz"
        if not annotations_tar.exists():
            download_file(annotations_url, annotations_tar, "Oxford-IIIT Annotations")
            extract_tar(annotations_tar, dataset_dir)
            logger.info("Annotations extracted successfully")
        else:
            logger.info("Annotations already downloaded, skipping...")

        logger.success(f"Oxford-IIIT Pet Dataset downloaded to {dataset_dir}")
        logger.info("This dataset contains 37 pet breeds (25 dogs, 12 cats)")

    except Exception as e:
        logger.error(f"Error downloading Oxford-IIIT Pet Dataset: {e}")
        logger.warning("You can download manually from:")
        logger.warning("https://www.robots.ox.ac.uk/~vgg/data/pets/")


def download_stanford_dogs(output_dir: Path):
    """
    Download Stanford Dogs Dataset
    URL: http://vision.stanford.edu/aditya86/ImageNetDogs/

    This dataset contains 120 breeds of dogs with about 150-200 images per breed.
    Total: ~20,000 images
    """
    logger.info("=" * 60)
    logger.info("Downloading Stanford Dogs Dataset...")
    logger.info("=" * 60)

    dataset_dir = output_dir / "stanford-dogs"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # URLs for Stanford Dogs Dataset
    images_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    annotations_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"
    lists_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"

    try:
        # Download images
        images_tar = dataset_dir / "images.tar"
        if not images_tar.exists():
            download_file(images_url, images_tar, "Stanford Dogs Images")
            extract_tar(images_tar, dataset_dir)
            logger.info("Images extracted successfully")
        else:
            logger.info("Images already downloaded, skipping...")

        # Download annotations
        annotations_tar = dataset_dir / "annotation.tar"
        if not annotations_tar.exists():
            download_file(annotations_url, annotations_tar, "Stanford Dogs Annotations")
            extract_tar(annotations_tar, dataset_dir)
            logger.info("Annotations extracted successfully")
        else:
            logger.info("Annotations already downloaded, skipping...")

        # Download lists (train/test splits)
        lists_tar = dataset_dir / "lists.tar"
        if not lists_tar.exists():
            download_file(lists_url, lists_tar, "Stanford Dogs Lists")
            extract_tar(lists_tar, dataset_dir)
            logger.info("Lists extracted successfully")
        else:
            logger.info("Lists already downloaded, skipping...")

        logger.success(f"Stanford Dogs Dataset downloaded to {dataset_dir}")
        logger.info("This dataset contains 120 dog breeds with ~20,000 images")

    except Exception as e:
        logger.error(f"Error downloading Stanford Dogs Dataset: {e}")
        logger.warning("You can download manually from:")
        logger.warning("http://vision.stanford.edu/aditya86/ImageNetDogs/")


def download_cats_dataset(output_dir: Path):
    """
    Download Cats Dataset

    Note: This uses a publicly available cat breeds dataset.
    For Kaggle datasets, you need to use Kaggle API with authentication.
    """
    logger.info("=" * 60)
    logger.info("Downloading Cats Dataset...")
    logger.info("=" * 60)

    dataset_dir = output_dir / "cats"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.warning("Cat breeds datasets typically require Kaggle API access.")
    logger.info("To download from Kaggle:")
    logger.info("1. Install kaggle: pip install kaggle")
    logger.info("2. Setup API credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
    logger.info("3. Use commands like:")
    logger.info("   kaggle datasets download -d ma7555/cat-breeds-dataset")
    logger.info("   kaggle datasets download -d shawngregory/cat-breeds-dataset-with-photos")

    # Alternative: Try to download from a public source
    logger.info("\nAlternative: Using Oxford-IIIT Pet Dataset which includes 12 cat breeds")
    logger.info("The Oxford-IIIT dataset already contains cat breeds if downloaded above.")

    # Create a placeholder file with instructions
    instructions_file = dataset_dir / "README.txt"
    with open(instructions_file, 'w') as f:
        f.write("Cat Breeds Dataset Download Instructions\n")
        f.write("=" * 50 + "\n\n")
        f.write("Option 1: Use Kaggle API\n")
        f.write("1. Install: pip install kaggle\n")
        f.write("2. Setup API key from https://www.kaggle.com/account\n")
        f.write("3. Download: kaggle datasets download -d ma7555/cat-breeds-dataset\n\n")
        f.write("Option 2: Manual Download\n")
        f.write("Visit https://www.kaggle.com/datasets and search for 'cat breeds'\n\n")
        f.write("Option 3: Use Oxford-IIIT Pet Dataset\n")
        f.write("Already includes 12 cat breeds (downloaded separately)\n")

    logger.info(f"Instructions saved to {instructions_file}")


def download_birds_dataset(output_dir: Path):
    """
    Download CUB-200-2011 Birds Dataset
    URL: http://www.vision.caltech.edu/datasets/cub_200_2011/

    This dataset contains 11,788 images of 200 bird species.
    """
    logger.info("=" * 60)
    logger.info("Downloading CUB-200-2011 Birds Dataset...")
    logger.info("=" * 60)

    dataset_dir = output_dir / "cub-200-2011"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # URL for CUB-200-2011 Dataset
    images_url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"

    try:
        # Download dataset
        dataset_tar = dataset_dir / "CUB_200_2011.tgz"
        if not dataset_tar.exists():
            download_file(images_url, dataset_tar, "CUB-200-2011 Birds")
            extract_tar(dataset_tar, dataset_dir)
            logger.info("Dataset extracted successfully")
        else:
            logger.info("Dataset already downloaded, skipping...")

        logger.success(f"CUB-200-2011 Birds Dataset downloaded to {dataset_dir}")
        logger.info("This dataset contains 200 bird species with 11,788 images")
        logger.info("Each species has about 60 images")

    except Exception as e:
        logger.error(f"Error downloading CUB-200-2011 Birds Dataset: {e}")
        logger.warning("You can download manually from:")
        logger.warning("https://data.caltech.edu/records/65de6-vp158")
        logger.warning("or http://www.vision.caltech.edu/datasets/cub_200_2011/")


def main():
    parser = argparse.ArgumentParser(description="Download pet breed datasets")
    parser.add_argument(
        "--dataset",
        choices=["oxford-pets", "stanford-dogs", "cats", "birds", "all"],
        default="all",
        help="Dataset to download"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.data_dir / "raw",
        help="Output directory for datasets"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Download datasets
    try:
        if args.dataset == "all":
            download_oxford_pets(args.output_dir)
            download_stanford_dogs(args.output_dir)
            download_cats_dataset(args.output_dir)
            download_birds_dataset(args.output_dir)
        elif args.dataset == "oxford-pets":
            download_oxford_pets(args.output_dir)
        elif args.dataset == "stanford-dogs":
            download_stanford_dogs(args.output_dir)
        elif args.dataset == "cats":
            download_cats_dataset(args.output_dir)
        elif args.dataset == "birds":
            download_birds_dataset(args.output_dir)

        logger.success("\n" + "=" * 60)
        logger.success("Download script completed!")
        logger.success("=" * 60)
        logger.info(f"\nAll datasets are in: {args.output_dir}")
        logger.info("\nNext steps:")
        logger.info("1. Run: python scripts/prepare_data.py")
        logger.info("2. Train model: python -m app.ml.train")

    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user")
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        logger.info("Some datasets may need to be downloaded manually")


if __name__ == "__main__":
    main()
