from aiogram.types import User, Message
from datetime import datetime
from loguru import logger


def format_user_info(user: User) -> str:
    """
    Format user information for logging

    Args:
        user: Telegram User object

    Returns:
        Formatted user info string
    """
    info = f"User(id={user.id}, username={user.username or 'None'}"
    if user.first_name:
        info += f", first_name={user.first_name}"
    if user.last_name:
        info += f", last_name={user.last_name}"
    info += ")"
    return info


def get_message_timestamp(message: Message) -> str:
    """
    Get formatted timestamp from message

    Args:
        message: Telegram Message object

    Returns:
        Formatted timestamp string
    """
    if message.date:
        return message.date.strftime("%Y-%m-%d %H:%M:%S")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_message(message: Message, action: str = "received"):
    """
    Log message information

    Args:
        message: Telegram Message object
        action: Action description (e.g., "received", "processed")
    """
    user_info = format_user_info(message.from_user)
    timestamp = get_message_timestamp(message)

    if message.photo:
        logger.info(f"Photo message {action} from {user_info} at {timestamp}")
    elif message.text:
        logger.info(f"Text message {action} from {user_info} at {timestamp}: {message.text[:50]}")
    elif message.document:
        logger.info(f"Document message {action} from {user_info} at {timestamp}")
    else:
        logger.info(f"Message {action} from {user_info} at {timestamp}")


def escape_markdown(text: str) -> str:
    """
    Escape special characters for Markdown

    Args:
        text: Input text

    Returns:
        Escaped text
    """
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length

    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
