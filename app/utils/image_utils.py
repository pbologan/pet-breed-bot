import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
from loguru import logger


def validate_image(image_path: str) -> bool:
    """
    Validate if file is a valid image

    Args:
        image_path: Path to image file

    Returns:
        True if valid image, False otherwise
    """
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception as e:
        logger.error(f"Invalid image: {e}")
        return False


def get_image_info(image_path: str) -> dict:
    """
    Get image information

    Args:
        image_path: Path to image file

    Returns:
        Dictionary with image information
    """
    try:
        img = Image.open(image_path)
        return {
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "width": img.width,
            "height": img.height
        }
    except Exception as e:
        logger.error(f"Error getting image info: {e}")
        return {}


def resize_image(
    image_path: str,
    output_path: str,
    max_size: Tuple[int, int] = (800, 800),
    quality: int = 85
) -> bool:
    """
    Resize and compress image

    Args:
        image_path: Path to input image
        output_path: Path to save resized image
        max_size: Maximum dimensions (width, height)
        quality: JPEG quality (1-100)

    Returns:
        True if successful, False otherwise
    """
    try:
        img = Image.open(image_path)

        # Calculate new size maintaining aspect ratio
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Save with compression
        img.save(output_path, optimize=True, quality=quality)
        logger.info(f"Image resized and saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return False


def crop_center(image_path: str, output_path: str, size: int = 224) -> bool:
    """
    Crop center of image to square

    Args:
        image_path: Path to input image
        output_path: Path to save cropped image
        size: Size of square crop

    Returns:
        True if successful, False otherwise
    """
    try:
        img = Image.open(image_path)
        width, height = img.size

        # Calculate crop box
        left = (width - size) / 2
        top = (height - size) / 2
        right = (width + size) / 2
        bottom = (height + size) / 2

        # Crop and save
        img_cropped = img.crop((left, top, right, bottom))
        img_cropped.save(output_path)
        logger.info(f"Image cropped and saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error cropping image: {e}")
        return False


def enhance_image(image_path: str, output_path: str) -> bool:
    """
    Apply basic image enhancement

    Args:
        image_path: Path to input image
        output_path: Path to save enhanced image

    Returns:
        True if successful, False otherwise
    """
    try:
        # Read image
        img = cv2.imread(image_path)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Denoise
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

        # Save
        cv2.imwrite(output_path, enhanced)
        logger.info(f"Image enhanced and saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        return False


def convert_to_rgb(image_path: str, output_path: Optional[str] = None) -> bool:
    """
    Convert image to RGB format

    Args:
        image_path: Path to input image
        output_path: Path to save converted image (if None, overwrite original)

    Returns:
        True if successful, False otherwise
    """
    try:
        img = Image.open(image_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')
            output = output_path or image_path
            img.save(output)
            logger.info(f"Image converted to RGB and saved to {output}")

        return True

    except Exception as e:
        logger.error(f"Error converting image to RGB: {e}")
        return False
