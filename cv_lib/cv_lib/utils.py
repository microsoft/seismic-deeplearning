import os
import logging


def load_log_configuration(log_config_file):
    """
    Loads logging configuration from the given configuration file.
    """
    if not os.path.exists(log_config_file) or not os.path.isfile(log_config_file):
        msg = "%s configuration file does not exist!", log_config_file
        logging.getLogger(__name__).error(msg)
        raise ValueError(msg)
    try:
        logging.config.fileConfig(log_config_file, disable_existing_loggers=False)
        logging.getLogger(__name__).info(
            "%s configuration file was loaded.", log_config_file
        )
    except Exception as e:
        logging.getLogger(__name__).error(
            "Failed to load configuration from %s!", log_config_file
        )
        logging.getLogger(__name__).debug(str(e), exc_info=True)
        raise e
