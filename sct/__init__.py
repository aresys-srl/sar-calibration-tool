"""
SAR Calibration Tool
"""

_MIN_PYTHON_VERSION = "3.5"

import sys  # noqa

assert sys.version_info >= tuple(
    (int(v) for v in _MIN_PYTHON_VERSION.split("."))
), "SAR Calibration Tool requires Python {} or higher".format(_MIN_PYTHON_VERSION)

from datetime import datetime  # noqa
import logging  # noqa
import os  # noqa

# log_file = os.path.join(
#     os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
#     datetime.now().strftime(__name__ + "_%Y%m%dT%H%M%S.log"),
# )
logger = logging.getLogger(__name__)  # create logger
logger.setLevel(logging.DEBUG)
# fh = logging.FileHandler(log_file)  # create file handler which logs even debug messages
# fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()  # create console handler with a higher log level
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)  # create formatter and add it to the handlers
# fh.setFormatter(formatter)
ch.setFormatter(formatter)
# logger.addHandler(fh)  # add the handlers to the logger
logger.addHandler(ch)

# codedef_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'resources/sar_products_specs/sentinel1/')
# os.putenv('CODA_DEFINITION', codedef_file)

__version__ = "1.0.2"
