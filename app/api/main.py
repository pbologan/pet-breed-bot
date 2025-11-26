import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import settings
from app.logging_config import setup_logging
from app.ml.model import PetBreedClassifier
from app.ml.labels import create_sample_label_map
from app.api.routes import router, set_model_instance

# Setup logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title="Pet Breed Classification API",
    description="API for classifying pet breeds from images using deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="", tags=["prediction"])


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Starting Pet Breed Classification API...")

    # Create sample label map if it doesn't exist
    if not settings.label_map_path.exists():
        logger.warning(f"Label map not found at {settings.label_map_path}")
        logger.info("Creating sample label map for testing...")
        create_sample_label_map(settings.label_map_path)

    # Check if model exists
    if not settings.model_path.exists():
        logger.error(f"Model not found at {settings.model_path}")
        logger.error("Please train a model first or provide a valid model path")
        logger.warning("API will start but predictions will fail")
        return

    # Load model
    try:
        logger.info("Loading model...")
        model = PetBreedClassifier(
            model_path=settings.model_path,
            label_map_path=settings.label_map_path
        )
        set_model_instance(model)
        logger.info("Model loaded successfully")
        logger.info(f"Model supports {model.label_manager.num_classes()} classes")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API will start but predictions will fail")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Pet Breed Classification API...")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Pet Breed Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_top_k": "/predict-top-k",
            "classes": "/classes",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
