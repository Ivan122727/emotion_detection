from aiogram.methods import DeleteWebhook

from emotion_detection.admin_handler import admin_router
from emotion_detection.config import ADMIN_ID, bot, dp
from emotion_detection.user_handler import user_router
from emotion_detection.config import on_error


async def start_bot() -> None:
    await bot(DeleteWebhook(drop_pending_updates=True))
    dp.include_routers(admin_router, user_router)
    dp.errors.register(on_error)
    await bot.send_message(chat_id=ADMIN_ID, text="Бот запущен!")
    await dp.start_polling(bot)