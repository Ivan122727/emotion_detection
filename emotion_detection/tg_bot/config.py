import os
from os import getenv
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.types import Update
from aiogram.client.default import DefaultBotProperties
import logging
from logging.handlers import RotatingFileHandler

current_directory = os.getcwd()
env_path = os.path.join(current_directory, '.env')

# Загрузка переменных окружения из файла .env
load_dotenv(dotenv_path=env_path, override=True)
BOT_TOKEN = getenv("BOT_TOKEN")
ADMIN_ID = getenv("ADMIN_ID")


# Создание директории для логов, если она не существует
os.makedirs("logs", exist_ok=True)
log_file = "logs/bot.log"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
        ]
    )


setup_logging()


# Проверка, что BOT_TOKEN не является None
if BOT_TOKEN is None:
    raise ValueError("BOT_TOKEN is not set in the environment variables")

dp = Dispatcher()
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

async def on_error(update: Update, exception: Exception):
    logging.error(f"Update {update} caused error {exception}")
    await bot.send_message(chat_id=ADMIN_ID, text=f"Update {update} caused error {exception}")s