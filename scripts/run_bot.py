#!/usr/bin/env python3
"""
Script to run the Pet Breed Bot
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.bot.main import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
