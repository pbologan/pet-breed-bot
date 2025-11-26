from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import tempfile
import os
from loguru import logger

from app.ml.model import PetBreedClassifier
from app.config import settings

router = APIRouter()

# Global model instance (will be set in main.py)
model_instance: PetBreedClassifier = None


def set_model_instance(model: PetBreedClassifier):
    """Set the global model instance"""
    global model_instance
    model_instance = model


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": model_instance is not None
    }


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict pet breed from uploaded image

    Args:
        file: Image file (JPEG, PNG)

    Returns:
        JSON with predicted class and probability
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    # Create temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            # Write uploaded file to temp
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        logger.info(f"Processing image: {file.filename}")

        # Make prediction
        predicted_label, confidence = model_instance.predict(temp_path)

        logger.info(f"Prediction: {predicted_label} (confidence: {confidence:.3f})")

        return {
            "class": predicted_label,
            "probability": float(confidence),
            "filename": file.filename
        }

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Failed to delete temp file: {e}")


@router.post("/predict-top-k")
async def predict_top_k(file: UploadFile = File(...), k: int = 3):
    """
    Get top K predictions from uploaded image

    Args:
        file: Image file (JPEG, PNG)
        k: Number of top predictions to return

    Returns:
        JSON with top K predictions
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    # Validate k
    if k < 1 or k > 10:
        raise HTTPException(
            status_code=400,
            detail="k must be between 1 and 10"
        )

    # Create temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        logger.info(f"Processing image for top-{k}: {file.filename}")

        # Make prediction
        predictions = model_instance.predict_top_k(temp_path, k=k)

        logger.info(f"Top-{k} predictions: {predictions}")

        return {
            "predictions": [
                {"class": label, "probability": float(prob)}
                for label, prob in predictions
            ],
            "filename": file.filename
        }

    except Exception as e:
        logger.error(f"Error during top-k prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Failed to delete temp file: {e}")


@router.get("/classes")
async def get_classes():
    """
    Get list of supported classes

    Returns:
        JSON with list of class names
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    classes = model_instance.label_manager.get_all_labels()

    return {
        "classes": classes,
        "num_classes": len(classes)
    }
