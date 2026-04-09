import utils
import support
import numpy as np
import setup_globals
import scipy.signal as ss
from scipy.stats import entropy
from scipy.signal import welch
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from pathlib import Path
from support import logger, console, log_time, mainspinner

###############################################################################
# 1. Data Structures
###############################################################################
@dataclass
class ECGData:
    """Stores the state and results of the ECG processing pipeline."""
    fs             : float      = None
    wave           : np.ndarray = None
    sect_info      : np.ndarray = field(init=False)
    rolling_med    : np.ndarray = field(init=False)
    interior_peaks : np.ndarray = field(init=False)
    peaks          : np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int32))
    
    def __post_init__(self):
        self.rolling_med = np.zeros_like(self.wave, dtype=np.float32)
        self.interior_peaks = np.zeros((0, 16), dtype=np.int32)

class SignalDataLoader:
    """Handles loading and structuring of the data."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.fs = None
        self.wave = None
        self.window = None
        self.dtypes = None

    def load_signal_data(self):
        """Loads the signal based on file type suffix
        """        
        record = None
        #Load signal data
        file_type = self.file_path.suffix

        #Determine filetype and use appropriate library to load
        try:
            match file_type:
                case "ebm": 
                    from lib_ebm.pyebmreader import ebmreader
                    record, header = ebmreader(
                        filepath = self.file_path,
                        onlyheader = False
                    )
                    self.wave = record[0]
                    self.window = 2
                    self.fs = float(header["frequency"])

                case "ecg": 
                    pass
                case "h12": 
                    pass            

                #BUG - Fix this eventually.  Bad habit to use a blank as a valid value
                case "":
                    from wfdb import rdrecord
                    record = rdrecord(
                        self.file_path / f"{self.file_path._tail[-1]}",
                        sampfrom=0,
                        sampto=None,
                        channels=[0]
                    )
                    self.fs = record.fs
                    self.wave = record.p_signal
                    self.window = 10
                    self.dtypes = setup_globals.SECTION_DTYPES

        except Exception as e:
            logger.critical(f"Unable to load file. Error {e}")

        #Segment the signal
        self.segments = utils.segment_ECG(self.wave, self.fs, windowsize=self.window)
    
    def load_structures(self) -> ECGData:
        """Loading data structures for RAD_ECG

        Returns:
            ECGData (dataclass): Dataclass of ready objects.
        """        
        logger.info(f"Loading data from {self.file_path}")
        rad = ECGData(
            wave = self.wave,
            fs = self.fs,
        )
        rad.sect_info = np.zeros(shape=(self.segments.shape[0]), dtype=self.dtypes)
        rad.sect_info["wave_section"] = np.arange(self.segments.shape[0])
        rad.sect_info["start_point"] = self.segments[:,0]
        rad.sect_info["end_point"] = self.segments[:,1]
        rad.sect_info["valid"] = self.segments[:,2]
        #Don't need segments anymore, so delete it. 
        del self.segments
        return rad

###############################################################################
# 2. Tool Classes
###############################################################################

class CardiacFreqTools:
    """Handles frequency domain evaluations and Signal Quality Indices (SQI)."""
    def __init__(self, fs: float = 1000.0):
        self.fs = fs

    def evaluate_signal(self, signal:np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Evaluates if the signal is physiological or noise using Welch's PSD.
        Calculates In-Band Power Ratio (0.5 - 20 Hz) and normalized Spectral Entropy.
        """
        if len(signal) == 0:
            return 0.0, 1.0, None

        nperseg = min(len(signal), int(self.fs * 2.0))
        freqs, psd = welch(signal, fs=self.fs, nperseg=nperseg)
        
        total_power = np.sum(psd)
        if total_power == 0:
            return 0.0, 1.0, None

        # In-Band Power Ratio (0.5 to 20.0 Hz)
        band_mask = (freqs >= 0.5) & (freqs <= 20.0)
        in_band_power = np.sum(psd[band_mask])
        power_ratio = in_band_power / total_power
        
        # Spectral Entropy
        psd_norm = psd / total_power
        spec_entropy = entropy(psd_norm)
        norm_spec_entropy = spec_entropy / np.log(len(psd_norm))
        
        return power_ratio, norm_spec_entropy, psd_norm

class SignalGUI:
    """Handles all Matplotlib visualizations"""
    def __init__(self, ecg_data: ECGData):
        self.data = ecg_data

    def plot_errors(self, start_idx: int, end_idx: int, error_type: str):
        # Update Error plotting goes here
        logger.info(f"Visualizing error: {error_type} at {start_idx}:{end_idx}")

###############################################################################
# 3. Core Extraction Engine
###############################################################################
class RadECG:
    """Main search class for finding, validating, and extracting ECG information."""
    def __init__(self, data: ECGData, configs:dict, fp:Path, window_size: int = 10):
        self.fp = fp
        self.data = data
        self.fs = data.fs
        self.configs = configs
        self.window_size = window_size
        self.gui = SignalGUI(self.data)
        self.freq_tools = CardiacFreqTools(fs=self.fs)
        self.stack_range = np.arange(10000, self.data.sect_info.shape[0], 10000)
        # Historical trackers
        self.low_counts = 0
        self.sect_counter = 0
        self.iqr_low_thresh = 1.0
    
    @log_time
    def peak_stack_test(self, new_peaks_arr:np.array) -> np.array:
        """Times how long it takes to run the vstack.  Useful for debugging

        Args:
            new_peaks_arr (np.array): new array

        Returns:
            np.array: Stacked array
        """        
        return np.vstack((self.data.peaks, new_peaks_arr)).astype(np.int32)
        
    def run_extraction(self):
        """Iterates through the ECG waveform in overlapping sections."""
        sect_que = deque(self.data.sect_info[['start_point', 'end_point']])
        progbar, job_id = support.mainspinner(console, len(sect_que))
        with progbar:
            while len(sect_que) > 0:
                progbar.update(task_id=job_id, description=f"[green] Extracting Peaks", advance=1)
                curr_section = sect_que.popleft()
                start_p = curr_section[0].item()
                end_p = curr_section[1].item()
                wave_chunk = self.data.wave[start_p:end_p].flatten()

                # Check Signal Quality Index (SQI)
                power_ratio, spec_entropy, _ = self.freq_tools.evaluate_signal(wave_chunk)
                self.data.sect_info[self.sect_counter]["power_ratio"] = power_ratio
                self.data.sect_info[self.sect_counter]["spec_entropy"] = spec_entropy

                # In-Band Power Ratio and Shannon Entropy thresholds
                if power_ratio < 0.85 or spec_entropy > 0.60:
                    logger.warning(f"Section {self.sect_counter} rejected via SQI. Pwr: {power_ratio:.2f}, Ent: {spec_entropy:.2f}")
                    self.data.sect_info["fail_reason"][self.sect_counter] = "SQI_Noise"
                    continue

                # Extract Initial R Peaks
                r_peaks, peak_info = ss.find_peaks(
                    wave_chunk.flatten(), 
                    prominence=np.percentile(wave_chunk, 99),
                    height=np.percentile(wave_chunk, 95),
                    distance=int(self.fs * 0.200)
                )
                #Basic count reality check
                if r_peaks.size < 2 or r_peaks.size > 100:
                    logger.warning(f"Section {self.sect_counter} rejected: Invalid peak count ({r_peaks.size}).")
                    self.data.sect_info["fail_reason"][self.sect_counter] = "no_sig"
                    continue

                # Format Peak Array
                r_peaks_shifted = r_peaks + start_p
                valid_mask = np.ones((len(r_peaks_shifted), 1), dtype=int)
                new_peaks_arr = np.hstack((r_peaks_shifted.reshape(-1, 1), valid_mask))
                
                # Calculate Rolling Median for the chunk
                rolled_med = utils.roll_med(wave_chunk).astype(np.float32)
                
                # Historical Validation
                if len(self.data.peaks) > 10:
                    last_keys = self.get_consecutive_valid_peaks(self.data.peaks)
                    if last_keys is not False:
                        sect_valid, new_peaks_arr = self.historical_validation_check(
                            new_peaks_arr, last_keys, peak_info, rolled_med, self.sect_counter, start_p, end_p
                        )
                        if not sect_valid:
                            self.data.sect_info["fail_reason"][self.sect_counter] = "historical_fail"
                    else:
                        sect_valid = True # Resetting baseline if no recent valid history
                else:
                    sect_valid = True # First few sections are automatically valid if they passed SQI

                # Finalize Section
                if sect_valid:
                    self.data.sect_info[self.sect_counter]["valid"] = 1
                    if self.sect_counter in self.stack_range:
                        self.data.peaks = self.peak_stack_test(new_peaks_arr)
                    else:
                        self.data.peaks = np.vstack((self.data.peaks, new_peaks_arr)).astype(np.int32)
                    
                    # Proceed to PQRST extract and stats
                    # int_peaks = self.extract_pqrst(new_peaks_arr, peak_info, rolled_med, start_p)
                    # self.data.interior_peaks = np.vstack((self.data.interior_peaks, int_peaks))
                
                # Advance section id to next section
                self.data.sect_info["valid"][self.sect_counter] = 1
                self.sect_counter += 1
                logger.info(f'Section counter at {self.section_counter}')

    def get_consecutive_valid_peaks(self, r_peaks: np.ndarray, lookback: int = 1500):
        """Scans back in time to find consecutive validated R peaks."""
        arr = r_peaks[::-1]
        counts = []
        for i in range(arr.shape[0]):
            is_last = i + 1 >= arr.shape[0]
            if arr[i, 1] == 1:
                counts.append(i)
            if is_last or arr[i, 1] == 0:
                if is_last:
                    return False
                else:
                    counts = []
            elif (arr[counts[0], 0] - arr[counts[-1], 0] > lookback):
                return arr[counts][::-1, 0]
        return False

    def historical_validation_check(self, new_peaks_arr, last_keys, peak_info, rolled_med, sect_id, start_idx, end_idx) -> Tuple[bool, np.ndarray]:
        """Rejects whole segments based on historical averages (IQR, Slope, Separation, Height)."""
        sect_valid = True
        
        # Grab historical rolling median
        rolling_med_start = last_keys[0]
        rolling_med_end = last_keys[-1]
        med_arr = self.data.rolling_med[rolling_med_start:rolling_med_end]
        
        # Calculate IQR
        q1 = np.quantile(med_arr, 0.20) if len(med_arr) > 0 else 0
        q3 = np.quantile(med_arr, 0.80) if len(med_arr) > 0 else 0
        iqr = q3 - q1

        # Prevent vanishing gradient for IQR
        if iqr <= self.iqr_low_thresh:
            self.low_counts += 1
            if self.low_counts > 6:
                iqr = 3 * iqr
            elif self.low_counts > 3:
                iqr = iqr + 0.50 * iqr
        else:
            self.iqr_low_thresh = iqr
            self.low_counts = 0

        # Peak Separation Check
        med_diff = np.diff(last_keys)
        last_avg_p_sep = np.mean(med_diff) if len(med_diff) > 0 else 1.0
        lower_bound_sep = last_avg_p_sep * 0.5
        upper_bound_sep = last_avg_p_sep * 2.0
        
        diffs = np.diff(new_peaks_arr[:, 0])
        if np.any((diffs < lower_bound_sep) | (diffs > upper_bound_sep)):
            logger.warning(f"Peak separation violation in section {sect_id}")
            sect_valid = False

        # Peak Height Check
        last_avg_peak_heights = np.mean([self.data.wave[x] for x in last_keys])
        r_roll_diff = self.data.wave[last_keys] - self.data.rolling_med[last_keys]
        lower_bound_ht = np.mean(r_roll_diff) * 0.51
        upper_bound_ht = last_avg_peak_heights * 3.0
        
        if np.any((peak_info['peak_heights'] < lower_bound_ht) | (peak_info['peak_heights'] > upper_bound_ht)):
            logger.warning(f"Peak height violation in section {sect_id}")
            sect_valid = False

        return sect_valid, new_peaks_arr

    def extract_pqrst(self, new_peaks_arr, peak_info, rolled_med, start_p):
        """Placeholder for PQRST geometry extraction."""
        pass

# --- Entry Point ---
def main():
    configs      :dict = setup_globals.load_config()
    fp           :Path = Path.cwd() / configs["data_path"]
    batch_process:bool = configs["batch"]
    selected     :int  = setup_globals.load_choices(fp, batch_process)
    loader = SignalDataLoader(selected)
    loader.load_signal_data()
    ECG = loader.load_structures()
    RAD = RadECG(ECG, configs, fp)
    RAD.run_extraction()

if __name__ == "__main__":
    main()  