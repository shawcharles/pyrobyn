import os
import logging
from datetime import datetime
import robyn

current_datetime = datetime.now()

# Get the directory of the current module
module_dir = os.path.dirname(robyn.__file__)

# Set up basic logging configuration to log to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# If you have any specific loggers you want to configure, you can do so here
logger = logging.getLogger(__name__)
logger.info("Logging is set up to console only.")

# Import core classes
from robyn.robyn import Robyn
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters, ChannelHyperparameters
from robyn.data.entities.enums import AdstockType, DependentVarType
from robyn.export_manager import ExportManager

__all__ = [
    "Robyn",
    "MMMData",
    "HolidaysData",
    "Hyperparameters",
    "ChannelHyperparameters",
    "AdstockType",
    "DependentVarType",
    "ExportManager",
]
