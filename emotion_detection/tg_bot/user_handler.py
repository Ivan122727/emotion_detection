import logging
import os
from aiogram import Router, html, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message

from emotion_detection.ai.config import load_model
from emotion_detection.ai.detect import predict_image, predict_transforms
from emotion_detection.tg_bot.config import bot


device, model = load_model()
user_router = Router()

@user_router.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(f"Привет, {html.bold(message.from_user.full_name)}!\nЯ нейросеть для распознавания эмоций.\n\nОтправь мне фотографию и я вынесу свой вердикт.\n\n/credits - информация")
    logging.info(f"User {message.from_user.full_name} (@{message.from_user.username}|id:{message.from_user.id}) used the command /start")

@user_router.message(Command('credits'))
async def help_handler(message: Message) -> None:
    await message.answer(f'Проект разработан для дисциплины "{html.italic("Распознавание образов и машинное обучение")}".\n\nРазработчики: @Ivan122727 @ilyakhakimov03')
    logging.info(f"User {message.from_user.full_name} (@{message.from_user.username}|id:{message.from_user.id}) used the command /credits")

@user_router.message(F.document)
async def document_handler(message: Message) -> None:
    file_id = message.document.file_id
    file_info = await bot.get_file(file_id)
    file_path = file_info.file_path
    file_extension = os.path.splitext(file_path)[1]
    file_name = f"{message.chat.id}_{message.message_id}{file_extension}"
    destination = f"photos/{file_name}"
    # Скачиваем документ
    await bot.download_file(file_path, destination)
    # Анализируем документ с помощью нашей модели
    label = predict_image(destination, model, predict_transforms, device)
    # Удаляем временный файл
    os.remove(destination)
    # Отправляем результат пользователю
    await message.reply(f"Результат работы: {label}")
    logging.info(f"User {message.from_user.full_name} (@{message.from_user.username}|id:{message.from_user.id}) used the predict method with document {file_id}")

@user_router.message()
async def other_handler(message: Message) -> None:
    await message.answer("Я могу обрабатывать только фотографии и документы с изображениями.")
    logging.info(f"User {message.from_user.full_name} (@{message.from_user.username}|id:{message.from_user.id}) used the other method")