import asyncio
import sys
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import settings
from app.logging_config import setup_logging
from app.bot.handlers import start, photo

# Import model if not using API service
if not settings.use_api_service:
    from app.ml.model import PetBreedClassifier
    from app.ml.labels import create_sample_label_map


async def main():
    """Main function to run the bot"""

    # Setup logging
    setup_logging()
    logger.info("Starting Pet Breed Bot...")

    # Check configuration
    if not settings.telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN is not set in environment variables")
        sys.exit(1)

    # Initialize bot and dispatcher
    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()

    # Include routers
    dp.include_router(start.router)
    dp.include_router(photo.router)

    logger.info("Routers registered")

    # Initialize model if not using API service
    if not settings.use_api_service:
        logger.info("Initializing local model...")

        # Create sample label map if it doesn't exist
        if not settings.label_map_path.exists():
            logger.warning(f"Label map not found at {settings.label_map_path}")
            logger.info("Creating sample label map for testing...")
            create_sample_label_map(settings.label_map_path)

        # Check if model exists
        if not settings.model_path.exists():
            logger.error(f"Model not found at {settings.model_path}")
            logger.error("Please train a model first or provide a valid model path")
            logger.error("For testing, you can skip this by setting USE_API_SERVICE=true")
            sys.exit(1)

        try:
            model = PetBreedClassifier(
                model_path=settings.model_path,
                label_map_path=settings.label_map_path
            )
            photo.set_model_instance(model)
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            sys.exit(1)
    else:
        logger.info(f"Using API service at {settings.api_url}")

    # Start polling
    logger.info("Bot started. Press Ctrl+C to stop.")
    try:
        await dp.start_polling(bot)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    finally:
        await bot.session.close()
        logger.info("Bot session closed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped")
