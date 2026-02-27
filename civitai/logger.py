class Logger:
    _COLORS = {
        'info':    '\033[34m',
        'success': '\033[32m',
        'warning': '\033[33m',
        'error':   '\033[31m',
    }
    _RESET = '\033[0m'

    def _log(self, level: str, msg: str, verbose: bool = True):
        """Format and print a colored log message, gated by verbose for info/success."""
        if not verbose and level in ('info', 'success'):
            return
        color = self._COLORS[level]
        tag = f" [{level.upper()}]:" if level in ('warning', 'error') else ''
        print(f"{color}[CivitAI-Extension]:{self._RESET}{tag} {msg}")

    def info(self, msg: str, verbose: bool = True):    self._log('info', msg, verbose)
    def success(self, msg: str, verbose: bool = True): self._log('success', msg, verbose)
    def warning(self, msg: str):                       self._log('warning', msg)
    def error(self, msg: str):                         self._log('error', msg)

log = Logger()