import logging
from colorama import init, Fore, Style

init(autoreset=True)
import logging
from colorama import init, Fore, Style


def setup_logger():
    init(autoreset=True)

    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            colors = {
                logging.DEBUG: Fore.CYAN,
                logging.INFO: Fore.GREEN,
                logging.WARNING: Fore.YELLOW,
                logging.ERROR: Fore.RED,
                logging.CRITICAL: Fore.RED,
            }
            levelname_color = colors.get(record.levelno, Fore.WHITE)
            record.levelname = (
                f"{levelname_color}{record.levelname}{Style.RESET_ALL}"
            )
            return super().format(record)

    # Initialize logger
    logger = logging.getLogger("galactic")
    logger.setLevel(logging.INFO)

    # Initialize and set handler and formatter
    colored_formatter = ColoredFormatter("%(levelname)s: %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(colored_formatter)
    logger.addHandler(console_handler)

    # Suppress third-party logs
    third_party_loggers = [
        "urllib3",
        "filelock",
        "git",
        "tensorflow",
        "jax",
        "h5py",
        "jaxlib",
        "hpack",
        "asyncio",
        "modal-utils",
        "fsspec",
        "httpx",
    ]
    for log_name in third_party_loggers:
        logging.getLogger(log_name).setLevel(logging.WARNING)
