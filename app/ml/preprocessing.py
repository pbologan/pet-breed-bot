from torchvision import transforms
from PIL import Image
import torch


def get_transforms(train: bool = False):
    """
    Get image transformations for training or inference

    Args:
        train: If True, return training transforms with augmentation
               If False, return inference transforms
    """
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Preprocess an image for model inference

    Args:
        image_path: Path to the image file

    Returns:
        Preprocessed image tensor with batch dimension
    """
    image = Image.open(image_path).convert('RGB')
    transform = get_transforms(train=False)
    image_tensor = transform(image)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def preprocess_pil_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL Image for model inference

    Args:
        image: PIL Image object

    Returns:
        Preprocessed image tensor with batch dimension
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = get_transforms(train=False)
    image_tensor = transform(image)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor
