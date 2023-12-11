# Logging in Python is a means of tracking events that happen when software runs
# It allows for a centralized way to collect errors, warnings, and debug information, which can be invaluable for troubleshooting and analysis.

import logging

# these are log levels
logging.debug("This is a debug message")
logging.info("This is an info message")
logging.warning("This is a warning")
logging.error("This is an error message")
logging.critical("This is a critical message")

# by default, only logging levels of warining and above are printed
# its best practice to create your own module, importing the logger module, add the framework, and import back the newly created module to your program
# hieracrhy for logger, [__root__, __ new module___ ]

# lock handlers
logger = logging.getLogger(__name__)
# create handler
stream_h = logging.StreamHandler()
# file hanlder
file_h = logging.FileHandler('file.log')
# level and format
stream_h.setLevel(logging.WARNING)
file_h.setLevel(logging.ERROR)
# speificay formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_h.setFormatter(formatter)
file_h.setFormatter(formatter)

logger.addHandler(stream_h)
logger.addHandler(file_h)

logger.warning('this is a warning')
logger.error('this is an error')

#this created a new file called "file.log" and shows the error in that log
#rotating file handler
#timing rotating handler.