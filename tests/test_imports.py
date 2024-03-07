import scipy
from rich.logging import RichHandler
from rich.table import Table
from rich.console import Console
import rich
import logging
from logging import Formatter
import numpy as np
import pandas as pd
import scipy.signal
from scipy.fft import rfft, rfftfreq, irfft
from scipy.interpolate import interp1d
from scipy.misc import electrocardiogram
import requests
import numpy as np
import wfdb
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
from matplotlib.colors import rgb2hex
from datetime import datetime
import sys
import os
import time
from time import strftime
from collections import deque, Counter
from rad_ecg.scripts import utils

# Logger formatting code
logger = utils.load_logger(__name__)

def test_libs():
	logger.warning(f'Testing main library imports')
	logger.info(f"{scipy.__name__}\t: {scipy.__version__}")
	logger.info(f"{np.__name__}\t: {np.__version__}")
	logger.info(f"{pd.__name__}\t: {pd.__version__}")
	logger.info(f"{mpl.__name__}\t: {mpl.__version__}")
	logger.info(f"{wfdb.__name__}\t: {wfdb.__version__}")
	logger.info(f"{rich.__name__}\t: {rich.__package__}")