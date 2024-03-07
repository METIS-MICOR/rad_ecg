
from rad_ecg.scripts import peak_detect_v3
from rad_ecg.scripts import utils
from rad_ecg.scripts import setup_globals

logger = utils.load_logger(__name__)

def test_detect():
	ecg_data, wave, fs = setup_globals.load_structures("test", logger)
	ecg_data = peak_detect_v3.main_peak_search(
		False,
		False, 
		(ecg_data, wave, fs)
	)

