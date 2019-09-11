from __future__ import print_function
import logging

WARNING = logging.WARNING
ERROR = logging.ERROR
DEBUG = logging.DEBUG
INFO = logging.INFO
CRITICAL = logging.CRITICAL


def init_logger(f_ref, f_log='fiw_error.log', str_ref='Parse FIW'):
    logger = setup_custom_logger(f_ref, f_log)
    logger.debug(str_ref)
    return logger


def setup_custom_logger(name, f_log='fiw_error.log', level=logging.WARNING,
                        log_fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s'):
    formatter = logging.Formatter(fmt=log_fmt)

    handler = logging.FileHandler(f_log)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    if level == INFO:
        logger.setLevel(INFO)
    elif level == WARNING:
        logger.setLevel(WARNING)
    elif level == ERROR:
        logger.setLevel(ERROR)
    elif level == DEBUG:
        logger.setLevel(DEBUG)
    else:
        logger.setLevel(CRITICAL)

    logger.addHandler(handler)
    return logger
