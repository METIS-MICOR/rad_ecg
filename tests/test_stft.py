from rad_ecg.scripts import utils
from rad_ecg.scripts import setup_globals
from rad_ecg.scripts.peak_detect_v3 import STFT
import scipy.signal as ss
import numpy as np

logger = utils.load_logger(__name__)

def test_STFT():
	wstart = 92_000
	wend = 108_000
	_, wave, fs = setup_globals.init("test", logger)
	wave = wave.reshape(-1, 1)
	wave_chunk = wave[wstart:wend]
	rolled_med = utils.roll_med(wave_chunk).astype(np.float32)

	R_peaks, peak_info = ss.find_peaks(
		wave_chunk.flatten(), 
		prominence = np.percentile(wave_chunk, 99),
		height = np.percentile(wave_chunk, 95),
		distance = round(fs*(0.200))
	)

	R_peaks_shifted = R_peaks + wstart
	new_peaks = R_peaks_shifted.reshape(-1, 1)
	valid_mask = np.zeros(shape=(len(new_peaks[:, 0]),1), dtype=int)

	#stack the new peaks and valid mask into a single array
	new_peaks_arr = np.hstack((new_peaks, valid_mask))

	sect_valid, new_peaks_arr = STFT(
		new_peaks_arr, 
		peak_info,
		rolled_med,
		(0, wstart, wend),
		False,
		wave,
		fs
	)

	assert sect_valid == True, "STFT Section validation error"
	assert np.mean(wave[np.where(new_peaks_arr[:,1]==1)[0]]) == -0.16816176470588234, "STFT R peak mean error"
	logger.warning("STFT passed")

