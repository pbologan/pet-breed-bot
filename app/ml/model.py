import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image

from app.ml.preprocessing import preprocess_image, preprocess_pil_image
from app.ml.labels import LabelManager
from loguru import logger


class PetBreedClassifier:
    """Wrapper class for pet breed classification model"""

    def __init__(
        self,
        model_path: Path,
        label_map_path: Path,
        device: Optional[str] = None
    ):
        """
        Initialize the classifier

        Args:
            model_path: Path to trained model weights
            label_map_path: Path to label mapping JSON
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = model_path
        self.label_map_path = label_map_path

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load label manager
        self.label_manager = LabelManager(label_map_path)
        logger.info(f"Loaded {self.label_manager.num_classes()} classes")

        # Load model
        self.model = self._load_model()
        self.model.eval()
        logger.info("Model loaded successfully")

    def _load_model(self) -> nn.Module:
        """Load the trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Initialize model architecture (ResNet50 as default)
        model = models.resnet50(weights=None)
        num_classes = self.label_manager.num_classes()

        # Replace classifier
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

        # Load weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        return model

    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Predict breed/species from image file

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (predicted_label, confidence)
        """
        try:
            # Preprocess image
            image_tensor = preprocess_image(image_path)
            image_tensor = image_tensor.to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            # Get label
            predicted_label = self.label_manager.get_label(predicted_idx.item())
            confidence_value = confidence.item()

            logger.info(f"Prediction: {predicted_label} (confidence: {confidence_value:.3f})")

            return predicted_label, confidence_value

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def predict_pil(self, image: Image.Image) -> Tuple[str, float]:
        """
        Predict breed/species from PIL Image

        Args:
            image: PIL Image object

        Returns:
            Tuple of (predicted_label, confidence)
        """
        try:
            # Preprocess image
            image_tensor = preprocess_pil_image(image)
            image_tensor = image_tensor.to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            # Get label
            predicted_label = self.label_manager.get_label(predicted_idx.item())
            confidence_value = confidence.item()

            logger.info(f"Prediction: {predicted_label} (confidence: {confidence_value:.3f})")

            return predicted_label, confidence_value

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def predict_top_k(self, image_path: str, k: int = 3) -> list[Tuple[str, float]]:
        """
        Get top K predictions

        Args:
            image_path: Path to image file
            k: Number of top predictions to return

        Returns:
            List of (label, confidence) tuples
        """
        try:
            image_tensor = preprocess_image(image_path)
            image_tensor = image_tensor.to(self.device)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probabilities, k, dim=1)

            results = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                label = self.label_manager.get_label(idx.item())
                results.append((label, prob.item()))

            return results

        except Exception as e:
            logger.error(f"Error during top-k prediction: {e}")
            raise


def create_model_for_training(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Create a model for training

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights

    Returns:
        PyTorch model
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # Replace classifier
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model
