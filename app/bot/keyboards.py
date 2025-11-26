from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton


def get_main_keyboard() -> ReplyKeyboardMarkup:
    """Get main reply keyboard"""
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="ℹ️ Помощь")],
            [KeyboardButton(text="📊 О проекте")]
        ],
        resize_keyboard=True,
        input_field_placeholder="Отправьте фото питомца..."
    )
    return keyboard


def get_help_inline_keyboard() -> InlineKeyboardMarkup:
    """Get inline keyboard for help message"""
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📝 Как использовать", callback_data="help_usage")],
            [InlineKeyboardButton(text="🐕 Поддерживаемые породы", callback_data="help_breeds")]
        ]
    )
    return keyboard
