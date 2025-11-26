import os
from pathlib import Path
from aiogram import Router, F
from aiogram.types import Message
from loguru import logger
import httpx

from app.config import settings

router = Router()

# Global variable to store model instance (will be set in main.py)
model_instance = None


def set_model_instance(model):
    """Set the global model instance"""
    global model_instance
    model_instance = model


async def predict_via_api(image_path: str) -> tuple[str, float]:
    """
    Make prediction using FastAPI service

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (predicted_label, confidence)
    """
    try:
        async with httpx.AsyncClient(timeout=settings.api_timeout) as client:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = await client.post(
                    f"{settings.api_url}/predict",
                    files=files
                )
                response.raise_for_status()
                data = response.json()
                return data['class'], data['probability']
    except Exception as e:
        logger.error(f"Error calling API service: {e}")
        raise


async def predict_via_model(image_path: str) -> tuple[str, float]:
    """
    Make prediction using local model

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (predicted_label, confidence)
    """
    if model_instance is None:
        raise RuntimeError("Model instance not initialized")

    return model_instance.predict(image_path)


def format_breed_name(breed: str) -> str:
    """Format breed name for display"""
    # Replace underscores with spaces and capitalize words
    return breed.replace('_', ' ').title()


@router.message(F.photo)
async def handle_photo(message: Message):
    """Handle photo messages"""
    logger.info(f"Received photo from user {message.from_user.id}")

    # Send processing message
    processing_msg = await message.answer("🔄 Обрабатываю фотографию...")

    try:
        # Get the largest photo
        photo = message.photo[-1]
        file_id = photo.file_id

        # Download photo
        file = await message.bot.get_file(file_id)
        file_path = file.file_path

        # Create temp directory if not exists
        temp_dir = settings.temp_dir
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Save photo locally
        local_file_path = temp_dir / f"{file_id}.jpg"
        await message.bot.download_file(file_path, local_file_path)

        logger.info(f"Photo saved to {local_file_path}")

        # Make prediction
        if settings.use_api_service:
            predicted_label, confidence = await predict_via_api(str(local_file_path))
        else:
            predicted_label, confidence = await predict_via_model(str(local_file_path))

        # Format breed name
        breed_display = format_breed_name(predicted_label)

        # Prepare response based on confidence
        if confidence >= settings.confidence_threshold:
            response_text = (
                f"✅ <b>Результат:</b>\n\n"
                f"🐾 Это похоже на <b>{breed_display}</b>\n"
                f"📊 Уверенность: <b>{confidence*100:.1f}%</b>\n\n"
                f"Модель достаточно уверена в своём ответе!"
            )
        else:
            response_text = (
                f"🤔 <b>Результат:</b>\n\n"
                f"🐾 Возможно, это <b>{breed_display}</b>\n"
                f"📊 Уверенность: <b>{confidence*100:.1f}%</b>\n\n"
                f"⚠️ Модель не очень уверена. Попробуйте другое фото с лучшим качеством."
            )

        # Delete processing message
        await processing_msg.delete()

        # Send result
        await message.answer(response_text, parse_mode="HTML")

        # Clean up temporary file
        try:
            os.remove(local_file_path)
            logger.info(f"Cleaned up temporary file: {local_file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temporary file: {e}")

    except Exception as e:
        logger.error(f"Error processing photo: {e}")
        await processing_msg.delete()
        await message.answer(
            "❌ Произошла ошибка при обработке фотографии. "
            "Пожалуйста, попробуйте ещё раз или отправьте другое фото."
        )


@router.message(F.document)
async def handle_document(message: Message):
    """Handle document messages (reject them)"""
    await message.answer(
        "⚠️ Пожалуйста, отправьте изображение как фото, а не как файл.\n"
        "Нажмите на значок скрепки и выберите 'Фото' вместо 'Файл'."
    )


@router.message(F.text)
async def handle_text(message: Message):
    """Handle text messages that are not commands"""
    # Skip if message is a button text (already handled by start.py)
    if message.text in ["ℹ️ Помощь", "📊 О проекте"]:
        return

    await message.answer(
        "📸 Отправьте мне фотографию питомца, "
        "и я определю его породу!\n\n"
        "Используйте /help для получения дополнительной информации."
    )
