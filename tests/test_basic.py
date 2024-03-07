import scipy.signal as ss
from rad_ecg.scripts import utils
from rad_ecg.scripts import setup_globals
import numpy as np

logger = utils.load_logger(__name__)

def test_segmentation():
	global ecg_data, wave, fs
	ecg_data, wave, fs = setup_globals.load_structures("test", logger)
	test_sects = utils.segment_ECG(wave, fs)
	assert len(test_sects) == 37, "Incorrect number of sections"

def test_section_finder():
	start = 54000
	end = 82000
	start_sect = utils.section_finder(start, wave, fs)
	assert start_sect == 18, "Start section incorrect"
	end_sect = utils.section_finder(end, wave, fs)
	assert end_sect == 28, "End section incorrect"

def test_rolling_med():
	rolled_med = utils.roll_med(wave[:len(wave)//4])
	rolled_med = rolled_med.reshape(-1, 1)
	assert rolled_med.size == 27000, "Shape of rolled median is incorrect"
	assert np.mean(rolled_med) == -0.2510440740740741, "Rolling median mean incorrect"
	assert np.std(rolled_med) == 0.5800210841490816, "Rolling median std dev incorrect"

