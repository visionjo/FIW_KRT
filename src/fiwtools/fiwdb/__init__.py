# from os import path, remove
# import logging
# import logging.config
# # from .fiw import tri_subjects
#
# # If applicable, delete the existing log file to generate a fresh log file during each execution
# # if path.isfile("python_logging.log"):
# #     remove("python_logging.log")
#
# # Create the Logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
#
# # Create the Handler for logging data to a file
# logger_handler = logging.FileHandler('fiw_1.log')
#
# logger_handler.setLevel(logging.WARNING)
#
# # Create a Formatter for formatting the log messages
# logger_formatter = logging.Formatter('%(asctime)s:%(name)s - %(levelname)s - %(message)s')
#
# # Add the Formatter to the Handler
# logger_handler.setFormatter(logger_formatter)
#
# # Add the Handler to the Logger
# logger.addHandler(logger_handler)
# logger.info('Completed configuring logger()!')
