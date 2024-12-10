import asyncio

from emotion_detection.tg_bot.config import setup_logging
from emotion_detection.tg_bot.bot import start_bot

if __name__ == "__main__":
    setup_logging()
    asyncio.run(start_bot())