import numpy as np
import scipy.signal as ss
from scipy.stats import entropy
from scipy.signal import welch
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from pathlib import Path
from support import logger, console, log_time
import setup_globals

###############################################################################
# 1. Data Structures
###############################################################################
@dataclass
class ECGData:
    """Stores the state and results of the ECG processing pipeline."""
    fs            : float
    wave          : np.ndarray
    rolling_med   : np.ndarray = field(init=False)
    section_info  : np.ndarray = field(init=False)
    peaks         : np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int32))
    interior_peaks: np.ndarray = field(default_factory=lambda: np.zeros((0, 16), dtype=np.int32))
    
    def __post_init__(self):
        self.rolling_med = np.zeros_like(self.wave, dtype=np.float32)

class SignalDataLoader:
    """Handles loading and structuring of the data."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.fs = None
        self.wave = None
        self.dtypes = None

    def load_signal_data(self):
        """Loads the signal based on file type suffix
        """        
        record = None
        #Load signal data
        file_type = self.file_path.suffix

        try:
            match file_type:
                case "ebm": 
                    from lib_ebm.pyebmreader import ebmreader
                    record, header = ebmreader(
                        filepath = self.file_path,
                        onlyheader = False
                    )
                    record = record[0]
                case "ecg": 
                    pass
                case "h12": 
                    pass
                case "hea":
                    from wfdb import rdrecord
                    record = rdrecord(
                        self.file_path,
                        sampfrom=0,
                        sampto=None,
                        channels=[0]
                    )
                    self.fs = record.fs
                    self.wave = record.p_signal

        except Exception as e:
            logger.critical(f"Unable to load file. Error {e}")
      
    def load_detector(self) -> ECGData:
        """Loading logic for RAD_ECG

        Returns:
            ECGData (dataclass): Dataclass of ready objects.
        """        
        logger.info(f"Loading data from {self.file_path}")
        return ECGData(wave=self.wave, fs=self.fs)

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

# --- Entry Point ---
def main():
    configs:dict = setup_globals.load_config()
    batch_process:bool = configs["batch"]
    fp:Path = Path.cwd() / configs["data_path"]
    selected:int = setup_globals.load_choices(fp, batch_process)
    data = SignalDataLoader(selected)
    data.load_signal_data()
    RAD = data.load_detector()
    RAD.run_detection()

if __name__ == "__main__":
    main()