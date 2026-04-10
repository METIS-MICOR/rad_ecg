import utils
import support
import numpy as np
import setup_globals
from pathlib import Path
import scipy.signal as ss
from collections import deque
from scipy.signal import welch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from scipy.stats import entropy, kurtosis
from matplotlib.patches import Rectangle, Arrow
from support import logger, console, log_time, mainspinner, DATE_JSON
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

class SignalLoader:
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
    def evaluate_ecg_sqi(self, signal: np.ndarray) -> Tuple[bool, str, dict]:
        """
        Evaluates ECG signal quality using Kurtosis and targeted Spectral Banding.
        Returns a boolean (is_valid), a string (fail_reason), and a dictionary of the metrics.
        """
        if len(signal) == 0:
            return False, "Empty Signal", {}

        # Statistical SQI: Kurtosis
        # A clean ECG has high kurtosis due to sharp R-peaks. Noise flattens this.
        k_sqi = kurtosis(signal)

        # Spectral SQI: Welch's PSD
        nperseg = min(len(signal), int(self.fs * 2.0))
        freqs, psd = welch(signal, fs=self.fs, nperseg=nperseg)
        total_power = np.sum(psd)

        if total_power == 0:
            return False, "Zero Power", {}

        # QRS Power Ratio: Power strictly in the 5 - 15 Hz band
        qrs_band = (freqs >= 5.0) & (freqs <= 15.0)
        p_sqi = np.sum(psd[qrs_band]) / total_power

        # High-Frequency Noise Ratio: Power above 20 Hz (EMG / Artifacts)
        hf_band = freqs > 20.0
        hf_sqi = np.sum(psd[hf_band]) / total_power

        # Decision Matrix
        is_valid = True
        fail_reason = ""

        # Thresholds (These may need slight tuning based on your specific signal noise floor)
        if k_sqi < 5.0:
            is_valid = False
            fail_reason += "SQI: Low Kurtosis " #(Missing QRS / High Noise)
        if p_sqi < 0.10:
            is_valid = False
            fail_reason += "SQI: Low QRS Power " #(Baseline Wander / Artifact)
        if hf_sqi > 0.45:
            is_valid = False
            fail_reason += "SQI: High EMG Noise "

        metrics = {
            "kurtosis": k_sqi,
            "qrs_pwr_ratio": p_sqi,
            "hf_pwr_ratio": hf_sqi
        }

        return is_valid, fail_reason, metrics

    # def evaluate_signal(self, signal:np.ndarray) -> Tuple[float, float, np.ndarray]:
    #     """Evaluates if the signal is physiological or noise using Welch's PSD.
    #     Calculates In-Band Power Ratio (0.5 - 20 Hz) and normalized Spectral Entropy.

    #     Args:
    #         signal (np.ndarray): _description_

    #     Returns:
    #         Tuple[float, float, np.ndarray]: _description_
    #     """        
    #     if len(signal) == 0:
    #         return 0.0, 1.0, None

    #     nperseg = min(len(signal), int(self.fs * 2.0))
    #     freqs, psd = welch(signal, fs=self.fs, nperseg=nperseg)
        
    #     total_power = np.sum(psd)
    #     if total_power == 0:
    #         return 0.0, 1.0, None

    #     # In-Band Power Ratio (0.1 to 20.0 Hz)
    #     band_mask = (freqs >= 0.1) & (freqs <= 20.0)
    #     in_band_power = np.sum(psd[band_mask])
    #     power_ratio = in_band_power / total_power
        
    #     # Spectral Entropy
    #     psd_norm = psd / total_power
    #     spec_entropy = entropy(psd_norm)
    #     norm_spec_entropy = spec_entropy / np.log(len(psd_norm))
        
    #     return power_ratio, norm_spec_entropy, psd_norm
    #BUG : While this method worked wonderfully for more periodic signals like blood pressure
        #It does not perform well on ECG's due to the sharp nature of the QRS complex. 
        #Additionally the welch's method tends to wash out the interesting frequencies we want

class SignalGUI:
    """Handles all Matplotlib visualizations for debugging and validation."""
    def __init__(self, ecg_data: 'ECGData', plot_fft: bool = False, plot_errors: bool = False):
        self.data = ecg_data
        self.plot_fft = plot_fft
        self.plot_errors = plot_errors

    def plot_fft_section(self, start_idx: int, end_idx: int, new_peaks_arr: np.ndarray, peak_info: dict, sect_id: int, sqi_metrics: dict):
        """Displays the ECG waveform, SQI stats, and window-level spectral plots."""
        if not self.plot_fft:
            return

        wave_chunk = self.data.wave[start_idx:end_idx]
        rolled_med_chunk = self.data.rolling_med[start_idx:end_idx]
        
        fig = plt.figure(figsize=(12, 9))
        grid = plt.GridSpec(2, 2, hspace=0.4, height_ratios=[1.5, 1])
        ax_ecg = fig.add_subplot(grid[0, :2])
        ax_freq = fig.add_subplot(grid[1, :1])
        ax_spec = fig.add_subplot(grid[1, 1:2])

        # ECG Plot
        ax_ecg.plot(range(start_idx, end_idx), wave_chunk, label='Full ECG', color='dodgerblue')
        ax_ecg.plot(range(start_idx, end_idx), rolled_med_chunk, label='Rolling Median', color='orange')
        
        if len(new_peaks_arr) > 0:
            ax_ecg.scatter(new_peaks_arr[:, 0], peak_info.get('peak_heights', []), marker='D', color='red', label='R peaks', zorder=5)

        ax_ecg.set_title(f'Full ECG waveform for section {sect_id} indices {start_idx}:{end_idx}') 
        ax_ecg.set_xlabel("Timesteps")
        ax_ecg.set_ylabel("ECG mV")
        ax_ecg.legend(loc='upper right')

        # Add SQI Metrics Text Block
        k_sqi = sqi_metrics.get("kurtosis", 0)
        p_sqi = sqi_metrics.get("qrs_pwr_ratio", 0)
        hf_sqi = sqi_metrics.get("hf_pwr_ratio", 0)
        
        stat_text = (
            f"Kurtosis (kSQI): {k_sqi:.2f}   |   "
            f"QRS Power (5-15Hz): {p_sqi:.1%}   |   "
            f"HF Noise (>20Hz): {hf_sqi:.1%}"
        )
        
        # Place text at the bottom center of the ECG plot
        ax_ecg.text(
            0.5, 0.05, stat_text, 
            transform=ax_ecg.transAxes, 
            fontsize=12, fontweight='bold', 
            ha='center', va='bottom', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
        )

        # 3. Draw Spectral Plots for the ENTIRE window
        self._draw_freq_and_spec(ax_freq, ax_spec, wave_chunk, start_idx, end_idx)

        # Interactive Closures
        def onSpacebar(event):
            if event.key == " ": 
                timer.stop()
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', onSpacebar)
        
        # Auto-close timer
        timer = fig.canvas.new_timer(interval=3000)
        timer.single_shot = True
        timer.add_callback(plt.close, fig)
        timer.start()
        plt.show()

    def _draw_freq_and_spec(self, ax_freq, ax_spec, wave_chunk, start_idx, end_idx):
        """Calculates and plots Welch PSD and Spectrogram for the full section."""
        chunk = wave_chunk.flatten()
        if len(chunk) == 0: return

        # --- Welch's PSD ---
        # Match the nperseg exactly to the SQI evaluation logic
        nperseg = min(len(chunk), int(self.data.fs * 2.0))
        freqs, psd = welch(chunk, fs=self.data.fs, nperseg=nperseg)
        
        # Base plot
        ax_freq.plot(freqs, psd, color='darkviolet', lw=1.5)
        ax_freq.fill_between(freqs, psd, color='darkviolet', alpha=0.2)
        
        # Highlight QRS Band (5 - 15 Hz)
        qrs_mask = (freqs >= 5.0) & (freqs <= 15.0)
        ax_freq.fill_between(freqs, psd, where=qrs_mask, color='limegreen', alpha=0.5, label='QRS Band (5-15Hz)')
        
        # Highlight HF Band (> 20 Hz)
        hf_mask = (freqs > 20.0)
        ax_freq.fill_between(freqs, psd, where=hf_mask, color='crimson', alpha=0.5, label='HF Noise (>20Hz)')

        ax_freq.set_title('Welch PSD (Full Section)')
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Power / Hz")
        ax_freq.set_xlim(0, 40) # Restrict to physiological range
        ax_freq.legend(loc='upper right')

        # --- Spectrogram ---
        spec_nfft = max(256, int(self.data.fs * 0.5)) 
        
        ax_spec.specgram(
            chunk,
            NFFT=spec_nfft,
            detrend="linear",
            noverlap=int(spec_nfft * 0.5),
            Fs=self.data.fs,
            cmap='magma' 
        )
        ax_spec.set_xlabel("Time (sec)")
        ax_spec.set_ylabel("Frequency (Hz)")
        ax_spec.set_title("Spectrogram")
        ax_spec.set_ylim(0, 40) # Restrict Y-axis to physiological range
    def plot_validation_error(self, error_type: str, start_idx: int, end_idx: int, new_peaks_arr: np.ndarray, peak_info: dict, sect_id: int, **kwargs):
        """Historical validation error plots."""
        if not self.plot_errors:
            return

        wave_chunk = self.data.wave[start_idx:end_idx]
        rolled_chunk = self.data.rolling_med[start_idx:end_idx]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(start_idx, end_idx), wave_chunk, label='ECG')
        ax.plot(range(start_idx, end_idx), rolled_chunk, label='Rolling Median')
        ax.scatter(new_peaks_arr[:, 0], peak_info['peak_heights'], marker='D', color='red', label='R peaks')

        match error_type:
            case "separation":
                bad_sep = kwargs.get("bad_sep", [])
                for x in bad_sep:
                    ax.axvline(x=new_peaks_arr[x, 0], color='goldenrod', linestyle='--')
                    if x + 1 < len(new_peaks_arr):
                        ax.axvline(x=new_peaks_arr[x + 1, 0], color='goldenrod', linestyle='--')
                ax.set_title(f'Bad Peak Separation: idx {start_idx} to {end_idx} (Sect {sect_id})')

            case "height":
                low_peaks = kwargs.get("low_peaks", [])
                high_peaks = kwargs.get("high_peaks", [])
                
                for idx in low_peaks:
                    arrow = Arrow(new_peaks_arr[idx, 0] - 55, peak_info['peak_heights'][idx], 40, 0, width=0.05, color='goldenrod')
                    ax.add_patch(arrow)
                for idx in high_peaks:
                    arrow = Arrow(new_peaks_arr[idx, 0] - 55, peak_info['peak_heights'][idx], 40, 0, width=0.05, color='darkviolet')
                    ax.add_patch(arrow)
                ax.set_title(f'Bad Peak Height: idx {start_idx} to {end_idx} (Sect {sect_id})')

            case "rolling_median":
                outs = kwargs.get("outs", [])
                iqr = kwargs.get("iqr", 1.0)
                
                ax.axhline(y=(np.quantile(rolled_chunk, .80) + 1.5 * iqr), color='magenta', linestyle='--', label='Upper Guardrail')
                ax.axhline(y=(np.quantile(rolled_chunk, .20) - 1.5 * iqr), color='red', linestyle='--', label='Lower Guardrail')
                
                for out_type, p0, p1 in outs:
                    height = np.max(self.data.wave[p0:p1]) if out_type == 'above' else np.min(self.data.wave[p0:p1])
                    rect = Rectangle((p0, 0), p1 - p0, height, facecolor='lightgrey', alpha=0.9)
                    ax.add_patch(rect)
                ax.set_title(f'Bad Rolling Median: idx {start_idx} to {end_idx} (Sect {sect_id})')

            case "slope":
                leftbases = kwargs.get("leftbases", [])
                slopes = kwargs.get("slopes", [])
                upper_bound = kwargs.get("upper_bound", 0)
                lower_bound = kwargs.get("lower_bound", 0)
                
                if leftbases:
                    ax.scatter(leftbases, self.data.wave[leftbases], marker="o", color="green", label="Left Base")
                    _delt = 0.10 * (np.max(wave_chunk) - np.min(wave_chunk))
                    
                    for i, slope in enumerate(slopes):
                        if i < len(leftbases):
                            if slope > upper_bound:
                                arrow = Arrow(leftbases[i], self.data.wave[leftbases[i]] + _delt*2, 0, -_delt, width=40, color="red")
                                ax.add_patch(arrow)
                            elif slope < lower_bound:
                                arrow = Arrow(leftbases[i], self.data.wave[leftbases[i]] - _delt*2, 0, _delt, width=40, color="red")
                                ax.add_patch(arrow)
                ax.set_title(f'Bad Peak Slope: idx {start_idx} to {end_idx} (Sect {sect_id})')

        ax.legend(loc='upper left')

        def onSpacebar(event):
            if event.key == " ": 
                timer.stop()
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', onSpacebar)
        
        timer = fig.canvas.new_timer(interval=3000)
        timer.single_shot = True
        timer.add_callback(plt.close, fig)
        timer.start()
        
        plt.show()
###############################################################################
# 3. Main Extraction Engine
###############################################################################
class RadECG:
    """Main search class for finding, validating, and extracting ECG information."""
    def __init__(self, data: ECGData, configs:dict, fp:Path, window_size: int = 10):
        self.fp = fp
        self.data = data
        self.fs = data.fs
        self.configs = configs
        self.window_size = window_size
        self.gui = SignalGUI(
            self.data, 
            plot_fft=self.configs.get("plot_fft", False), 
            plot_errors=self.configs.get("plot_errors", False)
        )
        self.freq_tools = CardiacFreqTools(fs=self.fs)
        self.stack_range = np.arange(10000, self.data.sect_info.shape[0], 10000)
        # Historical trackers
        self.low_counts = 0
        self.sect_id = 0
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
        progbar, job_id = mainspinner(console, len(sect_que))
        with progbar:
            while len(sect_que) > 0:
                progbar.update(task_id=job_id, description=f"[green] Extracting Peaks", advance=1)
                curr_section = sect_que.popleft()
                start_p = curr_section[0].item()
                end_p = curr_section[1].item()
                wave_chunk = self.data.wave[start_p:end_p].flatten()

                # Check Signal Quality Index (SQI)
                is_valid, fail_reason, sqi_metrics = self.freq_tools.evaluate_ecg_sqi(wave_chunk)
                
                # Store the metrics
                self.data.sect_info[self.sect_id]["kurtosis"] = sqi_metrics.get("kurtosis", 0)
                self.data.sect_info[self.sect_id]["qrs_pwr"] = sqi_metrics.get("qrs_pwr_ratio", 0)
                self.data.sect_info[self.sect_id]["hf_pwr"] = sqi_metrics.get("hf_pwr_ratio", 0)

                # Gate the section
                if not is_valid:
                    logger.warning(f"Section {self.sect_id} rejected: {fail_reason}. kSQI: {sqi_metrics['kurtosis']:.2f}, HF: {sqi_metrics['hf_pwr_ratio']:.2f}")
                    self.data.sect_info["fail_reason"][self.sect_id] = fail_reason
                    self.sect_id += 1
                    continue

                # Extract Initial R Peaks
                r_peaks, peak_info = ss.find_peaks(
                    wave_chunk.flatten(), 
                    prominence=np.percentile(wave_chunk, 99),
                    height=np.percentile(wave_chunk, 95),
                    distance=int(self.fs * 0.200)
                )
                #Basic count reality check (we shouldn't need this anymore)
                if r_peaks.size < 2 or r_peaks.size > 100:
                    logger.warning(f"Section {self.sect_id} rejected: Invalid peak count ({r_peaks.size}).")
                    self.data.sect_info["fail_reason"][self.sect_id] = "no_sig"
                    self.sect_id += 1
                    continue

                # Calculate Rolling Median for the chunk
                rolled_med = utils.roll_med(wave_chunk).astype(np.float32)

                # Format Peak Array
                r_peaks_shifted = r_peaks + start_p
                same_peaks = sorted(list(set(r_peaks_shifted) & set(self.data.peaks[-20:,0])))
                if len(same_peaks) > 0:
                    f_peak = min(same_peaks)
                    keepers = r_peaks_shifted >= f_peak
                    peak_info['peak_heights'] = peak_info['peak_heights'][keepers]
                    peak_info['prominences'] = peak_info['prominences'][keepers]
                    new_peaks = np.array(new_peaks).reshape(-1, 1)
                else:
                    new_peaks = r_peaks_shifted.reshape(-1, 1)
                
                valid_mask = np.ones((len(r_peaks_shifted), 1), dtype=int)
                new_peaks_arr = np.hstack((r_peaks_shifted.reshape(-1, 1), valid_mask))
                
                if self.gui.plot_fft:
                    self.gui.plot_fft_section(start_p, end_p, new_peaks_arr, peak_info, self.sect_id, sqi_metrics)

                # Historical Validation
                if self.sect_id > 10:
                    last_keys = self.get_consecutive_valid_peaks(self.data.peaks)
                    if last_keys is not False:
                        sect_valid, new_peaks_arr = self.historical_validation_check(
                            new_peaks_arr, last_keys, peak_info, rolled_med, self.sect_id, start_p, end_p
                        )
                        if not sect_valid:
                            self.data.sect_info["fail_reason"][self.sect_id] = "historical_fail"
                    else:
                        sect_valid = True # Resetting baseline if no recent valid history
                else:
                    sect_valid = True # First few sections are automatically valid if they passed SQI

                # Finalize Section
                if sect_valid:
                    self.data.sect_info[self.sect_id]["valid"] = 1
                    if self.sect_id in self.stack_range:
                        self.data.peaks = self.peak_stack_test(new_peaks_arr)
                    else:
                        self.data.peaks = np.vstack((self.data.peaks, new_peaks_arr)).astype(np.int32)
                    
                    # Proceed to PQRST extract and stats
                    # int_peaks = self.extract_pqrst(new_peaks_arr, peak_info, rolled_med, start_p)
                    # self.data.interior_peaks = np.vstack((self.data.interior_peaks, int_peaks))
                
                # Advance section id to next section
                self.data.sect_info["valid"][self.sect_id] = 1
                self.sect_id += 1
                logger.debug(f'Section counter at {self.sect_id}')

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

# --- Program Start ---
def main():
    configs      :dict  = setup_globals.load_config()
    fp           :Path  = Path.cwd() / configs["data_path"]
    batch_process:bool  = configs["batch"]
    selected     :int   = setup_globals.load_choices(fp, batch_process)
    loader = SignalLoader(selected)
    loader.load_signal_data()
    ECG = loader.load_structures()
    RAD = RadECG(ECG, configs, fp)
    RAD.run_extraction()
    support.save_results(RAD.data, configs=configs, current_date=DATE_JSON)

if __name__ == "__main__":
    main()