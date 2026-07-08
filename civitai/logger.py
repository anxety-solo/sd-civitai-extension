import logging
import sys


_PREFIX = '[CivitAI-Extension]'


def _is_verbose() -> bool:
    try:
        from modules.shared import opts
        return bool(getattr(opts, 'ce_verbose', False))
    except Exception:
        return True


_BLUE = '\033[34m'
_RESET = '\033[0m'
_COLORS = {
    'WARNING': '\033[33m',
    'ERROR':   '\033[31m',
}


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord):
        prefix = f"{_BLUE}{_PREFIX}{_RESET}"
        if record.levelno == logging.INFO:
            return f"{prefix} - {record.getMessage()}"

        color = _COLORS.get(record.levelname, '')

        if color:
            level = f"{color}{record.levelname}{_RESET}"
        else:
            level = record.levelname

        return f"{prefix} - {level} - {record.getMessage()}"


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_Formatter())

_log = logging.getLogger('civitai_extension')
_log.setLevel(logging.DEBUG)
_log.addHandler(_handler)
_log.propagate = False


class _LogWrapper:
    def info(self, msg: str):
        _log.info(msg)

    def success(self, msg: str):
        if _is_verbose():
            _log.info(msg)

    def warning(self, msg: str):
        _log.warning(msg)

    def error(self, msg: str):
        _log.error(msg)


log = _LogWrapper()
