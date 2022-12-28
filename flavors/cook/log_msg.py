import string
from enum import Enum


class LogLevel(Enum):
    INFO = 1
    WARNING = 2
    ERROR = 3


def PackLogMsg(loglevel: LogLevel, message: string) -> object:
    return {"level": loglevel.name, "message": message}


def UnPackLogMsg(log: object):
    return log["level"], log["message"]
