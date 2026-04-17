import utils
import stumpy
import support
import numpy as np
import setup_globals
from numba import cuda
from pathlib import Path
import scipy.signal as ss
from collections import deque
from scipy.signal import welch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from scipy.stats import entropy, kurtosis
from scipy.fft import rfft, rfftfreq, irfft
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

    def calc_hjorth_complexity(self, signal: np.ndarray) -> float:
        """Calculates Hjorth Complexity (Proxy for overall HF static)."""
        dy = np.diff(signal)
        ddy = np.diff(dy)
        
        var_zero = np.var(signal)
        var_d1 = np.var(dy)
        var_d2 = np.var(ddy)
        
        if var_zero == 0 or var_d1 == 0:
            return 0.0
            
        mobility = np.sqrt(var_d1 / var_zero)
        mobility_d1 = np.sqrt(var_d2 / var_d1)
        return mobility_d1 / mobility
    
    def calc_spec_ratio(self, signal: np.ndarray) -> Tuple[float, float]:
        p_sqi:float = 0.0
        hf_sqi:float = 0.0
        # Spectral SQI: Welch's PSD
        nperseg = min(len(signal), int(self.fs * 2.0))
        freqs, psd = welch(signal, fs=self.fs, nperseg=nperseg, detrend="linear")
        total_power = np.sum(psd)

        if total_power == 0:
            return 0

        # QRS Power Ratio: Power strictly in the 5 - 15 Hz band
        qrs_band = (freqs >= 5.0) & (freqs <= 15.0)
        p_sqi = np.sum(psd[qrs_band]) / total_power

        # High-Frequency Noise Ratio: Power 1 - 40 Hz (EMG / Artifacts).  Ignoring LF and ULF freqs
        hf_band = (freqs >= 1.0) & (freqs <= 40.0)
        hf_sqi = np.sum(psd[hf_band]) / total_power

        return p_sqi / hf_sqi

    def pre_peak_sqi(self, wave_chunk: np.ndarray) -> tuple:
        """
        Ultra-fast window-level checks. 
        Runs BEFORE peak extraction to catch dead sensors or pure static.
        """
        if len(wave_chunk) == 0:
            return False, "Empty Chunk", {}

        k_sqi = kurtosis(wave_chunk)
        complexity = self.calc_hjorth_complexity(wave_chunk)
        spectral = self.calc_spec_ratio(wave_chunk)
        metrics = {
            "kurtosis" :np.round(k_sqi, 2),
            "hjorth"   :np.round(complexity, 2),
            "spectral" :np.round(spectral, 2)
        }

        is_valid = True
        fail_reason = ""

        # Gate 1: Is there a heartbeat shape? (Low Kurtosis = flatline or Gaussian noise)
        if k_sqi < 4:
            is_valid = False
            fail_reason += f"Low Kurtosis {k_sqi:.2f} | " #(Missing QRS / Flatline)00
            
        # Gate 2: Is the window drowning in static?
        if complexity > 3.0: 
            is_valid = False
            fail_reason += f"High Hjorth Complexity {complexity:.2f} | " #(Severe HF Static)
        
        # Gate 3: Is the spectral energy mostly in the QRS band (5-15Hz / 0-40Hz)
        if (spectral != None) & (spectral < 0.3):
            is_valid = False
            fail_reason += f"Low QRS Power {spectral:.2f}"
        return is_valid, fail_reason, metrics

    def post_peak_sqi(self, wave_chunk: np.ndarray, r_peaks: np.ndarray) -> tuple:
        """STAGE 2: Granular beat-by-beat checks via STFT and Matrix Profile."""
        valid_mask = np.zeros(len(r_peaks), dtype=int) 

        if len(r_peaks) < 2:
            return False, "Not enough peaks for STFT/MP", {"bad_beat_ratio": 1.0}, valid_mask

        # Matrix Profile Calculation (Chunk-level)
        m = int(self.fs * 0.12)
        m = max(m, 3) # Stumpy requires a minimum window of 3

        try:
            device_id = cuda.list_devices()[0].id
            mp = stumpy.gpu_stump(wave_chunk.astype(np.float64), m=m, device_id=device_id)
        except Exception as e:
            logger.error(f"GPU Stumpy failed, falling back to CPU: {e}")
            mp = stumpy.stump(wave_chunk.astype(np.float64), m=m)

        distances = mp[:, 0]
        med_dist = np.median(distances)
        mad = np.median(np.abs(distances - med_dist))
        mp_threshold = med_dist + (4.0 * max(mad, 1e-6))

        #Beat-by-Beat Evaluation (MP + STFT)
        bad_beats = 0
        total_beats = len(r_peaks) - 1
        logger.info(f"MP Stats: {med_dist:.3f} | MAD: {mad:.3f} | Threshold: {mp_threshold:.3f}")

        for i in range(total_beats):
            p0 = r_peaks[i]
            p1 = r_peaks[i+1]
            # # --- Matrix Profile Discord ---
            search_start = max(0, p0 - m)
            search_end = min(len(distances), p0 + (m // 2))
            peak_mp_dist = np.max(distances[search_start:search_end])
            
            if peak_mp_dist > mp_threshold:
                logger.debug(f"Beat {i} FAILED Matrix Profile: dist {peak_mp_dist:.3f} > thres {mp_threshold:.3f}")
                bad_beats += 1
                continue 
            else:
                # Log passing peaks just to see how far below threshold they are
                logger.debug(f"Beat {i} PASSED Matrix Profile: dist {peak_mp_dist:.3f} <= thres {mp_threshold:.3f}")
            # --- Check 2: STFT HF Noise ---
            samp = wave_chunk[p0:p1]
            # if len(samp) < 4:
            #     bad_beats += 1
            #     continue

            fft_samp = np.abs(np.fft.rfft(samp))
            freq_list = np.fft.rfftfreq(len(samp), d=1/self.fs)
            thres = np.where(freq_list < 18)[0][-1]

            if thres > 0 and thres < len(fft_samp):
                mean_low = fft_samp[0:thres].mean()
                high_freqs = fft_samp[thres:int(len(samp)/2)]
                if np.any(high_freqs > mean_low):
                    outs = np.where(high_freqs > mean_low)[0]
                    if outs.size >= 2:
                        bad_beats += 1
                        continue
            else:
                bad_beats += 1
                continue

            #If it passes both the Matrix Profile and STFT, mark as valid
            valid_mask[i] = 1 

        bad_beat_ratio = bad_beats / total_beats
        metrics = {"bad_beat_ratio": bad_beat_ratio}
        
        is_valid = bad_beat_ratio <= 0.25
        fail_reason = "Bad Beats" if not is_valid else ""

        return is_valid, fail_reason, metrics, valid_mask
    
class SignalGUI:
    """Handles all Matplotlib visualizations for debugging and validation."""
    def __init__(self, ecg_data: 'ECGData', plot_fft: bool = False, plot_errors: bool = False):
        self.data = ecg_data
        self.plot_fft = plot_fft
        self.plot_errors = plot_errors

    def plot_fft_sect(self, start_idx: int, end_idx: int, new_peaks_arr: np.ndarray, peak_info: dict, sect_id: int, sqi_metrics: dict):
        """Displays the ECG waveform, SQI stats, and interactive beat-level spectral plots."""
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

        # Shade R-R intervals based on the valid_mask (column 1 of new_peaks_arr)
        for peak in range(new_peaks_arr.shape[0] - 1):
            band_color = 'red' if new_peaks_arr[peak, 1] == 0 else 'lightgreen'
            rect = Rectangle(
                xy=(new_peaks_arr[peak, 0], 0), 
                width=new_peaks_arr[peak+1, 0] - new_peaks_arr[peak, 0], 
                height=np.max(self.data.wave[new_peaks_arr[peak, 0]:new_peaks_arr[peak+1, 0]]), 
                facecolor=band_color, edgecolor="grey", alpha=0.7
            )
            ax_ecg.add_patch(rect)

        ax_ecg.set_title(f'Full ECG waveform for section {sect_id} indices {start_idx}:{end_idx} (Click R-R to update STFT)') 
        ax_ecg.set_xlabel("Timesteps")
        ax_ecg.set_ylabel("ECG mV")
        ax_ecg.legend(loc='upper right')

        # Add SQI Metrics Text Block
        k_sqi = sqi_metrics["kurtosis"].item()
        hjorth = sqi_metrics["hjorth"].item()
        spectral = sqi_metrics["spectral"].item()
        bad_ratio = sqi_metrics["bad_b_rat"].item()
        
        stat_text = (
            f"Kurtosis: {k_sqi:.2f}   |   "
            f"Hjorth Complexity: {hjorth:.2f}   |   "
            f"QRS Ratio: {spectral:.2f}   |   "
            f"Bad Beat Ratio: {bad_ratio:.1%}"
        )
        
        # Place text at the bottom center of the ECG plot
        ax_ecg.text(
            0.5, 0.05, stat_text, 
            transform=ax_ecg.transAxes, 
            fontsize=12, fontweight='bold', 
            ha='center', va='bottom', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
        )

        # Initialize the spectral plots using the last two peaks
        if new_peaks_arr.shape[0] >= 2:
            p0, p1 = new_peaks_arr[-2, 0], new_peaks_arr[-1, 0]
            self._draw_freq_and_spec(ax_freq, ax_spec, p0, p1, start_idx, end_idx)
        
        # timer = fig.canvas.new_timer(interval=3000)
        # timer.single_shot = True
        # timer_cid = timer.add_callback(plt.close, fig)
        
        # 3. Interactive Closures
        # def onSpacebar(event):
        #     if event.key == " ": 
        #         timer.stop()
        #         timer.remove_callback(timer)
        #         logger.warning('Timer stopped')
        #         plt.close(fig)

        def onClick(event):
            if event.inaxes == ax_ecg:
                rects = [i for i in ax_ecg.patches if isinstance(i, Rectangle)]
                for rect in rects:
                    if rect.contains(event)[0]:
                        p0 = int(rect.get_x())
                        p1 = int(p0 + rect.get_width())
                        ax_freq.cla()
                        ax_spec.cla()
                        self._draw_freq_and_spec(ax_freq, ax_spec, p0, p1, start_idx, end_idx)
                        fig.canvas.draw_idle()
         
        # Auto-close timer
        fig.canvas.mpl_connect("button_press_event", onClick)
        # fig.canvas.mpl_connect('key_press_event', onSpacebar)
        # timer.start()
        plt.show()
        plt.close()

    def _draw_freq_and_spec(self, ax_freq, ax_spec, p0, p1, start_idx, end_idx):
        """Calculates STFT for the clicked R-R interval, and Spectrogram for the section."""
        samp = self.data.wave[p0:p1].flatten()
        if len(samp) == 0: 
            return

        # --- STFT for the specific R-R interval ---
        fft_samp = np.abs(np.fft.rfft(samp))
        freq_list = np.fft.rfftfreq(len(samp), d=1/self.data.fs)
        freqs = fft_samp[0:int(len(samp)/2)]
        
        # Determine index for 18 Hz physiological threshold
        thres = np.where(freq_list < 18)[0][-1]
        
        # Plot the frequencies
        ax_freq.stem(freqs)
        
        # Plot mean line and highlight HF noise violations if they exist
        if thres > 0 and thres < len(fft_samp):
            mean_low = fft_samp[0:thres].mean()
            ax_freq.axhline(y=mean_low, color='dodgerblue', linestyle='--', label='Mean LF Pwr (<18Hz)')
            high_freqs = fft_samp[thres:int(len(samp)/2)]
            outs = np.where(high_freqs > mean_low)[0]
            if outs.size > 0:
                # Scatter red dots exactly where HF noise spikes above the mean
                ax_freq.scatter(freq_list[thres + outs], high_freqs[outs], color='red', zorder=5, label='HF Noise Spikes')

        ax_freq.set_title(f'STFT Spectrum (peaks {p0}:{p1})')
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Magnitude")
        ax_freq.set_xlim(0, 40) 
        ax_freq.legend(loc='upper right')

        # --- Spectrogram (Full Section) ---
        chunk = self.data.wave[start_idx:end_idx].flatten()
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
        ax_spec.set_title("Spectrogram (Full Section)")
        ax_spec.set_ylim(0, 40) # Restrict Y-axis to physiological range

    def plot_validation_error(
            self, 
            error_type   : str, 
            start_idx    : int, 
            end_idx      : int, 
            new_peaks_arr: np.ndarray, 
            peak_info    : dict, 
            sect_id      : int, 
            **kwargs
        ):
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
            #TODO - Case extend
                #Extend cases for Kurtosis, Hijorth, Spectral and maybe matrix profile. 
                #Still not sold how much that works

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
        self.stack_range:range = np.arange(10000, self.data.sect_info.shape[0], 10000)
        # Historical trackers
        self.low_counts:int = 0
        self.sect_id:int = 0
        self.iqr_low_thresh:float = 1.0
        self.is_stable:bool = False
    
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
        # progbar, job_id = mainspinner(console, len(sect_que))
        # with progbar:
        while len(sect_que) > 0:
            # progbar.update(task_id=job_id, description=f"[green] Extracting Peaks", advance=1)
            curr_section = sect_que.popleft()
            start_p = curr_section[0].item()
            end_p = curr_section[1].item()
            wave_chunk = self.data.wave[start_p:end_p].flatten()

            # Check Signal Quality Index (SQI) using lightweight checks
            is_valid, fail_reason, pre_metrics = self.freq_tools.pre_peak_sqi(wave_chunk)
            self.data.sect_info[self.sect_id]["kurtosis"] = pre_metrics.get("kurtosis", 0)
            self.data.sect_info[self.sect_id]["hjorth"] = pre_metrics.get("hjorth", 0)
            self.data.sect_info[self.sect_id]["spectral"] = pre_metrics.get("spec_ratio", 0)

            if not is_valid:
                logger.warning(f"Section {self.sect_id} rejected: {fail_reason}")
                self.data.sect_info["fail_reason"][self.sect_id] = fail_reason
                self.sect_id += 1
                if self.gui.plot_errors:
                    #TODO - Need a basic plot routine in SignalGUI.  Just plots wave / rolling median
                    pass
                continue

            # Calculate Rolling Median for the chunk
            rolled_med = utils.roll_med(wave_chunk).astype(np.float32)
            self.data.rolling_med[start_p:end_p] = rolled_med.reshape(-1, 1)
            #BUG -
                # You might not need to store 

            # Extract Initial R Peaks
            r_peaks, peak_info = ss.find_peaks(
                wave_chunk.flatten(), 
                prominence=np.percentile(wave_chunk, 99),
                height=np.percentile(wave_chunk, 95),
                distance=int(self.fs * 0.200)
            )
            #Basic count check (we shouldn't need this anymore)
            if r_peaks.size < 2 or r_peaks.size > 100:
                logger.warning(f"Section {self.sect_id} rejected: Invalid peak count ({r_peaks.size}).")
                self.data.sect_info["fail_reason"][self.sect_id] += " no_sig"
                self.sect_id += 1
                continue        

            #Check each beat with the matrix profile and STFT. 
            is_valid, fail_reason, post_metrics, val_mask = self.freq_tools.post_peak_sqi(wave_chunk, r_peaks)
            self.data.sect_info[self.sect_id]["bad_b_rat"] = post_metrics.get("bad_beat_ratio", 1.0)

            if not is_valid:
                logger.warning(f"Section {self.sect_id} rejected: {fail_reason}")
                self.data.sect_info["fail_reason"][self.sect_id] += fail_reason
                self.sect_id += 1
                if self.gui.plot_errors:
                    self.gui.plot_validation_error(start_p, end_p, new_peaks_arr, peak_info, self.sect_id, self.data.sect_info[self.sect_id]) 
                continue

            # Format Peak Array
            r_peaks_shifted = r_peaks + start_p
            same_peaks = sorted(list(set(r_peaks_shifted) & set(self.data.peaks[-20:,0])))
            #BUG 
                #Could speed above line up with np.intersect.
            #If there's peak overlap, find the intersection and shift which peaks to evaluate
            if len(same_peaks) > 0:
                #Find the last peak in common. 
                f_peak = max(same_peaks)
                #Index those peaks from the last same peak, to the end of the r_peaks_shifted
                keepers = r_peaks_shifted >= f_peak
                #Stack the mask to create the 1x1 array
                new_peaks_arr = np.hstack((r_peaks_shifted[keepers].reshape(-1, 1), val_mask[keepers].reshape(-1, 1)))
                peak_info['peak_heights'] = peak_info['peak_heights'][keepers]
                peak_info['prominences'] = peak_info['prominences'][keepers]
                #Don't process the last peak. That will get indexed in the next section. 
                self.data.peaks = self.data.peaks[:-1, :]
            else:
                new_peaks_arr = np.hstack((r_peaks_shifted.reshape(-1, 1), val_mask.reshape(-1, 1)))

            if self.gui.plot_fft:
                self.gui.plot_fft_sect(start_p, end_p, new_peaks_arr, peak_info, self.sect_id, self.data.sect_info[self.sect_id])

            # Historical Validation
            #BUG - validation 
                #Need a better way to do this other than count.  
            if self.sect_id > 10:
                last_keys = self.get_consecutive_valid_peaks(self.data.peaks)
                if last_keys is not False:
                    sect_valid, new_peaks_arr = self.historical_validation_check(
                        new_peaks_arr, last_keys, peak_info, 
                        rolled_med, self.sect_id, start_p, end_p
                    )
                    if not sect_valid:
                        self.data.sect_info["fail_reason"][self.sect_id] += " historical_fail"
                else:
                    # Resetting baseline if no recent valid history
                    sect_valid = True 
            else:
                # First few sections are automatically valid if they passed SQI
                sect_valid = True 

            # Finalize Section
            if sect_valid:
                self.data.sect_info[self.sect_id]["valid"] = 1
                if self.sect_id in self.stack_range:
                    self.data.peaks = self.peak_stack_test(new_peaks_arr)
                else:
                    self.data.peaks = np.vstack((self.data.peaks, new_peaks_arr)).astype(np.int32)
                #BUG - Memory
                    # You're going to still run into the vstack problem

                # Proceed to PQRST extract and stats
                # int_peaks = self.extract_pqrst(new_peaks_arr, peak_info, rolled_med, start_p)
                # self.data.interior_peaks = np.vstack((self.data.interior_peaks, int_peaks))
            
            # Advance section id to next section
            del new_peaks_arr
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