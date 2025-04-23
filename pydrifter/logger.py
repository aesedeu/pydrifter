import logging
import os

import pendulum
from termcolor import colored

# # Создание директории для логов, если ее нет
# if "logs" not in os.listdir():
#     os.mkdir("logs")


class CustomConsoleLogger(logging.Formatter):
    FMT = "[{levelname:^10}] - {asctime}: {message}"
    FORMATS = {
        logging.DEBUG: colored(FMT, "light_grey"),
        logging.INFO: colored(FMT),
        logging.WARNING: colored(FMT, "light_yellow"),
        logging.ERROR: colored(FMT, "light_red"),
        logging.CRITICAL: colored(FMT, "red"),
    }

    def format(self, record):
        # Формат для консоли с цветами
        log_fmt = self.FORMATS.get(record.levelno, self.FMT)
        formatter = logging.Formatter(log_fmt, style="{", datefmt="%H:%M:%S")
        return formatter.format(record)


class CustomFileLogger(logging.Formatter):
    FMT = "[{levelname:^10}] - {asctime}: {message}"

    def format(self, record):
        # Формат для файла
        formatter = logging.Formatter(self.FMT, style="{", datefmt="%H:%M:%S")
        return formatter.format(record)


# Настройка обработчика для консоли
console_handler = logging.StreamHandler()
console_handler.setFormatter(CustomConsoleLogger())

# Настройка обработчика для файла
file_handler = logging.FileHandler(f"logs/{pendulum.now().to_date_string()}.log")
file_handler.setFormatter(CustomFileLogger())
file_handler.setLevel(logging.INFO)  # Устанавливаем уровень логирования для файла


def create_logger(
    level: str,
    name: str | None = None,
):
    logs = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    assert level in list(logs), "Неверный уровень логирования (debug/info/warning/error/critical)"

    log = logging.getLogger()
    log.addHandler(console_handler)
    # log.addHandler(file_handler)

    log.setLevel(logs[level])

    return log
