import gc
import utils
import stumpy
import support
import numpy as np
import setup_globals
from numba import cuda
from pathlib import Path
import scipy.signal as ss
from collections import deque
from kneed import KneeLocator
from scipy.signal import welch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from scipy.interpolate import interp1d
from dataclasses import dataclass, field, astuple
# from scipy.fft import rfft, rfftfreq, irfft
from numpy.polynomial import polynomial as P
from matplotlib.patches import Rectangle, Arrow
from scipy.stats import entropy, kurtosis, wasserstein_distance
from support import logger, console, log_time, mainspinner, DATE_JSON
###############################################################################
# 1. Data Structures
###############################################################################
@dataclass
class HeartBeat:
    """Dataclass to house beat information"""    
    p_peak   : int = None
    q_peak   : int = None
    r_peak   : int = None
    s_peak   : int = None
    t_peak   : int = None
    valid_qrs: bool = False
    p_peak_a : float = None
    q_peak_a : float = None
    r_peak_a : float = None
    s_peak_a : float = None
    t_peak_a : float = None
    p_onset  : int = None
    q_onset  : int = None
    j_point  : int = None
    t_onset  : int = None
    t_offset : int = None
    u_wave   : bool = False
    PR       : float = None #ms
    QRS      : float = None #ms
    ST       : float = None #ms
    QT       : float = None #ms
    QTc      : float = None #ms
    TpTe     : float = None #ms

    def to_row(self) -> tuple:
        """
        Dumps the dataclass to a tuple for NumPy structured array stacking.
        Safely converts any un-extracted 'None' values to 0 to prevent dtype crashes.
        """
        return tuple(0 if val is None else val for val in astuple(self))

@dataclass
class SectionStat:
    """Dataclass to house section data"""
    HR    : float = np.nan #bpm
    iso   : float = np.nan #mV 
    SDNN  : float = np.nan #ms
    RMSSD : float = np.nan #ms
    PR    : float = np.nan #ms
    QRS   : float = np.nan #ms
    ST    : float = np.nan #ms
    QT    : float = np.nan #ms
    QTc   : float = np.nan #ms
    QTVI  : float = np.nan #s
    TpTe  : float = np.nan #ms

@dataclass
class ECGData:
    """Stores the state and results of the ECG processing pipeline."""
    fs             : float      = None              #Sampling Frequency
    wave           : np.ndarray = None              #ECG Signal
    sect_info      : np.ndarray = field(init=False) #Section Averages
    rolling_med    : np.ndarray = field(init=False) #Rolling Median
    interior_peaks : np.ndarray = field(init=False) #All the other peaks/onsets/offsets
    peaks          : np.ndarray = field(init=False) #R peaks
    # peaks          : np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int32)) # R peaks
    
    def __post_init__(self):
        # Calculate mathematical maximum possible peaks (200ms minimum distance)
        max_peaks = int(len(self.wave) / (self.fs * 0.200)) + 1000 # +1000 for safety
        self.peaks = np.zeros((max_peaks, 2), dtype=np.int32)
        self.rolling_med = np.zeros_like(self.wave, dtype=np.float32)
        self.interior_peaks = np.empty((max_peaks,), dtype=setup_globals.PEAK_DTYPES)

###############################################################################
# 2. Tool Classes
###############################################################################

class CardiacFreqTools:
    """Handles frequency domain evaluations and Signal Quality Indices (SQI)."""
    def __init__(
            self, 
            fs: float = 1000.0, 
            history_size: int = 6,
            freq_lim: float = 15, 
            qrs_lim : float = 0.35
            ):
        self.fs = fs
        self.history_size = history_size
        self.psd_history = deque(maxlen=self.history_size)
        self.mp_med_history = deque(maxlen=self.history_size)
        self.mp_mad_history = deque(maxlen=self.history_size)
        self.freq_lim = freq_lim
        self.qrs_lim = qrs_lim
    #TODO -  adding ENTROPY to your meaures. 
        #Shannon Entropy didnt work as well. 
        #Look at paper Bob sent. 

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
    
    def calc_spec_metrics(self, signal: np.ndarray) -> Tuple[float, float]:
        # Spectral SQI: Welch's PSD
        nperseg = min(len(signal), int(self.fs * 2.0))
        freqs, psd = welch(signal, fs=self.fs, nperseg=nperseg, detrend="linear")
        total_power = np.sum(psd)

        if total_power == 0:
            return 0, 0, False, 0
        
        # QRS Power Ratio: Power strictly in the 5 - 15 Hz band
        qrs_band = (freqs >= 5.0) & (freqs <= self.freq_lim)
        qrs_power = np.sum(psd[qrs_band])
        qrs_sqi = qrs_power / total_power
        
        # High-Frequency Band: Power 1 - 40 Hz (EMG / Artifacts).  Ignoring LF and ULF freqs
        hf_band = (freqs >= 1.0) & (freqs <= 40.0)
        hf_power = np.sum(psd[hf_band])
        hf_sqi = hf_power / total_power
        spec_ratio =  qrs_sqi / hf_sqi
        
        # --- Spectral Shannon Entropy ---
        # Normalize the full PSD into a probability mass function
        psd_norm = psd / total_power
        # Calculate Shannon entropy (base 2 is standard for bits of information)
        spec_entropy = entropy(psd_norm, base=2)

        # --- Metric 2: Wasserstein Distribution Shift ---
        f_target = freqs[qrs_band]
        psd_target = psd[qrs_band]
        w_dist = 0.0
        is_stable = True
        
        if qrs_power > 0:
            # Normalize PSD so it acts as a valid probability distribution (mass = 1)
            normalized_psd = psd_target / qrs_power
            
            if not self.psd_history:
                # First valid chunk seeds the baseline
                self.psd_history.append(normalized_psd)
            else:
                # Get median baseline across recent history
                baseline = np.median(np.vstack(self.psd_history), axis=0)
                
                # Calculate Earth Mover's Distance
                w_dist = wasserstein_distance(
                    u_values=f_target, 
                    v_values=f_target, 
                    u_weights=baseline, 
                    v_weights=normalized_psd
                )
                # Threshold for a large shift in signal composition
                is_stable = w_dist < 3.0 #3.0
                if is_stable:
                    self.psd_history.append(normalized_psd)

        return spec_ratio, w_dist, is_stable, spec_entropy

    def pre_peak_sqi(self, wave_chunk: np.ndarray) -> tuple:
        """
        Ultra-fast window-level checks. 
        Runs BEFORE peak extraction to catch dead sensors or pure static.
        """
        if len(wave_chunk) == 0:
            return False, "Empty Chunk", {}
          
        k_sqi = kurtosis(wave_chunk)
        complexity = self.calc_hjorth_complexity(wave_chunk)
        spectral, wdist, is_stable, spect_ent = self.calc_spec_metrics(wave_chunk)
       #BUG May need to recode this section return from total_power=0
        metrics = {
            "kurtosis" :np.round(k_sqi, 2),
            "hjorth"   :np.round(complexity, 2),
            "spectral" :np.round(spectral, 2), 
            "wdist"    :np.round(wdist, 2),
            "spect_ent":np.round(spect_ent, 2)
        }

        is_valid = True
        fail_reason = ""

        # Gate 1: Is there a heartbeat shape? (Low Kurtosis = flatline or Gaussian noise)
        if k_sqi < 4:
            is_valid = False
            fail_reason += f"Low Kurtosis | " #(Missing QRS / Flatline)
            
        # Gate 2: Is the window drowning in static?
        if complexity > 3.0: 
            is_valid = False
            fail_reason += f"High Hjorth Complexity | " #(Severe HF Static)
        # Gate 3: Is the spectral energy mostly in the QRS band (5-15Hz / 0-40Hz)
        if (spectral != None) & (spectral < self.qrs_lim): #0.4
            is_valid = False
            fail_reason += f"Low QRS Power | "
        # Gate 4: Broadband noise (Spectral Entropy)
        if spect_ent > 6.5:
            is_valid = False
            fail_reason += "High Spec Entropy | "
        # Gate 5: Wasserstein Distribution Shift (Global sensor degradation)
        if not is_stable:
            is_valid = False
            fail_reason += f"Shift in W-Dist | "
        return is_valid, fail_reason, metrics

    def post_peak_sqi(self, wave_chunk: np.ndarray, r_peaks: np.ndarray) -> tuple:
        """STAGE 2: Granular beat-by-beat checks via STFT and Matrix Profile."""
        valid_mask = np.zeros(len(r_peaks), dtype=int) 

        if len(r_peaks) < 4:
            return False, "Not enough peaks for STFT/MP", {"bad_beat_ratio": 1.0}, valid_mask

        # --- Matrix Profile Calculation (Chunk-level) ---
        m = int(self.fs * .30) #.12
        m = max(m, int(np.median(np.diff(r_peaks // 2)))) 

        # try:
        #     device_id = cuda.list_devices()[0].id
        #     mp = stumpy.gpu_stump(wave_chunk.astype(np.float64), m=m, device_id=device_id)
        # except Exception as e:
        #     logger.error(f"GPU Stumpy failed, falling back to CPU: {e}")
        mp = stumpy.stump(wave_chunk.astype(np.float64), m=m)

        distances = mp[:, 0]
        # med_dist = np.median(distances)
        local_med = np.median(distances)
        local_mad = np.median(np.abs(distances - local_med))        
        
        # --- Historical MP Smoothing ---
        if not self.mp_med_history:
            # Seed the history on the first pass
            self.mp_med_history.append(local_med)
            self.mp_mad_history.append(local_mad)
            med_dist = local_med
            mad = local_mad
        else:
            # Use historical median of medians/MADs for extreme stability
            med_dist = np.median(self.mp_med_history)
            mad = np.median(self.mp_mad_history)        
        
        # Prevent vanishing MAD on ultra-clean sections
        # mad = np.median(np.abs(distances - med_dist))
        safe_mad = max(mad, 0.4) 
        mp_threshold = med_dist + (5.5 * safe_mad)
        mp_threshold = max(mp_threshold, 7)

        # Beat-by-Beat Evaluation 
        bad_beats = 0
        total_beats = len(r_peaks) - 1
        
        #Container for rectangle labeling
        reject_reasons = [None] * total_beats
        # Window offsets (100ms)
        offset = int(self.fs * 0.10) 

        for i in range(total_beats):
            p0 = r_peaks[i]
            p1 = r_peaks[i+1]
            # ==========================================================
            # GATE 1: Matrix Profile Discord
            # ==========================================================
            search_start = max(0, p0 - (m // 2))
            search_end = min(len(distances), p0 + (m // 2)) #p1 -
            # peak_mp_dist = np.median(distances[search_start:search_end])
            if search_start < search_end:
                gap_wave = wave_chunk[search_start:search_end + m]
                if np.ptp(gap_wave) < 0.20:
                    peak_mp_dist = 0.0
                else:
                    peak_mp_dist = np.max(distances[search_start:search_end])
            else:
                peak_mp_dist = 0.0

            if peak_mp_dist > mp_threshold:
                logger.info(f"Beat {i} FAILED MP: dist {peak_mp_dist:.3f} > thres {mp_threshold:.3f} (ptp: {np.ptp(gap_wave):.2f})")
                reject_reasons[i] = "MP"
                bad_beats += 1
                continue 
    
            # ==========================================================
            # GATE 2: Local Hjorth (QRS Morphology)
            # ==========================================================
                # Tightly centered on the QRS complex [-100ms to +100ms]
            qrs_samp = wave_chunk[max(0, p0 - offset) : min(len(wave_chunk), p0 + offset)]
            if len(qrs_samp) > 4 and self.calc_hjorth_complexity(qrs_samp) > 4.5:
                logger.info(f"Beat {i} FAILED: High QRS Complexity")
                reject_reasons[i] = "HJH" 
                bad_beats += 1
                continue
            # ==========================================================
            # GATE 3: Inter-Beat STFT (Baseline Stability) 
            # ==========================================================
                # Slice strictly BETWEEN the QRS complexes to evaluate the stregnth of the T and P peak waves in between the R-R
            start_inter = p0 + offset
            end_inter = p1 - offset
            
            # Ensure the gap is at least 200ms (avoids crashing on high heart rates like 180+ BPM)
            if end_inter - start_inter > int(self.fs * 0.20):
                inter_samp = wave_chunk[start_inter:end_inter]
                
                # Apply Hann window 
                window = np.hanning(len(inter_samp))
                fft_inter = np.abs(np.fft.rfft(inter_samp * window))
                freq_inter = np.fft.rfftfreq(len(inter_samp), d=1/self.fs)
                
                total_inter_pwr = np.sum(fft_inter)
                if total_inter_pwr > 0:
                    # In the T-P segment, energy should be very low frequency.
                    # Anything > 15 Hz is baseline noise/instability.
                    hf_noise_mask = freq_inter > 15.0
                    hf_noise_pwr = np.sum(fft_inter[hf_noise_mask])
                    inter_noise_ratio = hf_noise_pwr / total_inter_pwr
                    
                    # If more than 40% of the inter-beat gap is HF noise, the baseline is unstable
                    if inter_noise_ratio > 0.50:
                        logger.info(f"Beat {i} FAILED: Unstable Inter-Beat Baseline (Noise: {inter_noise_ratio:.0%})")
                        reject_reasons[i] = "STFT" 
                        bad_beats += 1
                        continue

            # Passed all checks
            valid_mask[i] = 1

        bad_beat_ratio = bad_beats / total_beats
        metrics = {
            "bad_b_ratio": bad_beat_ratio, 
            "mp_distances": distances,
            "mp_threshold": mp_threshold,
            "rejections" : reject_reasons
        }
        
        is_valid = bad_beat_ratio <= 0.25
        if is_valid:
            self.mp_med_history.append(local_med)
            self.mp_mad_history.append(local_mad)
        fail_reason = f"Bad Beats: {bad_beats}/{total_beats}" if not is_valid else ""

        return is_valid, fail_reason, metrics, valid_mask

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
        # Load signal data
        file_type = self.file_path.suffix

        # Determine filetype and use appropriate library to load
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
                    if self.file_path.is_dir():
                        from wfdb import rdrecord
                        record = rdrecord(
                            self.file_path / f"{self.file_path.name}",
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
            ECGData (dataclass): Dataclass of data objects.
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
    
class SignalGUI:
    """Handles all Matplotlib visualizations for debugging and validation."""
    def __init__(
            self, 
            ecg_data    : ECGData, 
            tools       : CardiacFreqTools,
            plot_section: bool = False, 
            plot_errors : bool = False,
            timeout_ms  : int = 2000
        ):
        self.data = ecg_data
        self.hijorth = tools.calc_hjorth_complexity
        self.plot_section = plot_section
        self.plot_errors = plot_errors
        self.timeout_ms = timeout_ms

    def _apply_timer_and_show(self, fig, timeout:int = None):
        """Helper method to attach an auto-close timer and keypress overrides to a figure."""
        timeout = self.timeout_ms if timeout == None else timeout
        timer = fig.canvas.new_timer(interval=timeout)
        timer.single_shot = True
        
        def onKeyPress(event):
            # If down arrow is pressed, stop the timer and close the figure manually
            if event.key == "down": 
                timer.stop()
                plt.close(fig)
            # If up arrow is pressed, stop the timer to allow for interaction
            elif event.key == "up":
                timer.stop()
                logger.info('Auto-close timer stopped. Plot is now interactive.')

        fig.canvas.mpl_connect('key_press_event', onKeyPress)
        timer.add_callback(plt.close, fig)
        timer.start()
        plt.show()
        plt.close(fig)
        plt.close('all')
        gc.collect()

    def plot_pre_error(
            self, 
            error_type   : str, 
            start_idx    : int, 
            end_idx      : int, 
            sect_id      : int, 
        ):
        """Historical validation error plots."""
        if not self.plot_errors:
            return

        wave_chunk = self.data.wave[start_idx:end_idx]
        rolled_chunk = self.data.rolling_med[start_idx:end_idx]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(start_idx, end_idx), wave_chunk, label='ECG')
        ax.plot(range(start_idx, end_idx), rolled_chunk, label='Rolling Median')
        ax.set_title(f'Section {sect_id} indices {start_idx}:{end_idx}\n{error_type}') 
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("ECG mV")
        ax.legend(loc='upper right')
        self._apply_timer_and_show(fig=fig)
    
    def plot_post_error(
            self, 
            error_type   : str, 
            start_idx    : int, 
            end_idx      : int, 
            r_peaks_abs  : np.ndarray,
            peak_info    : dict,
            sect_id      : int, 
            post_metrics : dict = None,
            val_mask     : np.ndarray = None
        ):
        """Visualizes the ECG against the Matrix Profile when a beat-by-beat failure occurs."""
        if not self.plot_errors:
            return

        wave_chunk = self.data.wave[start_idx:end_idx]
        rolled_chunk = self.data.rolling_med[start_idx:end_idx]
        x_range = np.arange(start_idx, end_idx)
        
        fig = plt.figure(figsize=(12, 6))
        grid = plt.GridSpec(2, 1, hspace=0.3, height_ratios=[2, 1])
        
        ax_ecg = fig.add_subplot(grid[0])
        ax_mp = fig.add_subplot(grid[1], sharex=ax_ecg)

        # --- ECG Subplot ---
        ax_ecg.plot(x_range, wave_chunk, label='ECG', color='dodgerblue')
        ax_ecg.plot(x_range, rolled_chunk, label='Rolling Median', color='orange')
        
        if r_peaks_abs.shape[0] > 0 and 'peak_heights' in peak_info:
            ax_ecg.scatter(r_peaks_abs, peak_info['peak_heights'], marker='D', color='red', label='R peaks', zorder=5)

        ax_ecg.set_title(f'Section {sect_id} indices {start_idx}:{end_idx}\nRejected: {error_type}') 
        ax_ecg.set_ylabel("ECG mV")
        ax_ecg.legend(loc='upper right')

        # --- Matrix Profile Subplot ---
        if post_metrics and "mp_distances" in post_metrics:
            mp_dist = post_metrics["mp_distances"]
            mp_thresh = post_metrics["mp_threshold"]
            
            # Stumpy arrays are shorter by (m-1). Pad with NaNs to align with the ECG indices.
            pad_len = len(wave_chunk) - len(mp_dist)
            padded_mp = np.pad(mp_dist, (0, pad_len), constant_values=np.nan)
            ax_mp.plot(x_range, padded_mp, color='purple', label='Matrix Profile Distance')
            ax_mp.axhline(y=mp_thresh, color='red', linestyle='--', label=f'Threshold ({mp_thresh:.2f})')
            ax_mp.set_ylabel("MP Distance")
            ax_mp.legend(loc='upper right')
        else:
            ax_mp.text(0.5, 0.5, "Matrix Profile Data Unavailable", ha='center', va='center', transform=ax_mp.transAxes)

        # Shade R-R intervals based on the valid_mask (column 1 of new_peaks_arr)
        if val_mask is not None:
            for idx, peak in enumerate(range(r_peaks_abs.shape[0] - 1)):
                band_color = 'red' if val_mask[idx] == 0 else 'lightgreen'
                rect = Rectangle(
                    xy=(r_peaks_abs[peak], 0), 
                    width=r_peaks_abs[peak+1] - r_peaks_abs[peak], 
                    height=np.max(self.data.wave[r_peaks_abs[peak]:r_peaks_abs[peak+1]]), 
                    facecolor=band_color, edgecolor="grey", alpha=0.7
                )
                ax_ecg.add_patch(rect)
        ax_mp.set_xlabel("Timesteps")
        self._apply_timer_and_show(fig=fig)

    def plot_fft_sect(
            self, 
            start_idx: int, 
            end_idx: int, 
            new_peaks_arr: np.ndarray, 
            peak_info: dict, 
            sect_id: int, 
            sqi_metrics: dict,
            post_metrics: dict = None
        ):
        """Displays the ECG waveform, SQI stats, and interactive beat-level spectral plots."""
        if not self.plot_section:
            return

        wave_chunk = self.data.wave[start_idx:end_idx]
        rolled_med_chunk = self.data.rolling_med[start_idx:end_idx]
        x_range = np.arange(start_idx, end_idx, dtype=np.float64)

        fig = plt.figure(figsize=(12, 9))
        grid = plt.GridSpec(2, 2, hspace=0.6, height_ratios=[1.5, 0.75])
        ax_ecg = fig.add_subplot(grid[0, :2])
        ax_freq = fig.add_subplot(grid[1, :1])
        ax_spec = fig.add_subplot(grid[1, 1:2])
        
        ax_mp_overlay = ax_ecg.twinx()
        ax_mp_overlay.set_visible(False)
        
        # State variables for toggles
        mp_line_drawn = False
        pqrst_visible = False
        pqrst_artists = []
        
        # ECG Plot
        ax_ecg.plot(x_range, wave_chunk, label='Full ECG', color='dodgerblue')
        ax_ecg.plot(x_range, rolled_med_chunk, label='Rolling Median', color='orange')
        
        # Initialize R-Peaks
        r_scatter = None
        if len(new_peaks_arr) > 0:
            r_scatter = ax_ecg.scatter(
                new_peaks_arr[:, 0], peak_info.get('peak_heights', []), 
                marker='D', color='red', label='R peaks', zorder=5, visible=False
            )
            pqrst_artists.append(r_scatter)

        # Shade R-R intervals
        reasons = post_metrics.get("rejections", []) if post_metrics else []
        for peak in range(new_peaks_arr.shape[0] - 1):
            is_valid = new_peaks_arr[peak, 1]
            band_color = 'lightgreen' if is_valid else 'red'
            p0_x = new_peaks_arr[peak, 0]
            p1_x = new_peaks_arr[peak+1, 0]
            rect_height = np.max(self.data.wave[p0_x:p1_x])
            rect = Rectangle(
                xy=(p0_x, 0), width=p1_x - p0_x, height=rect_height, 
                facecolor=band_color, edgecolor="grey", alpha=0.7
            )
            ax_ecg.add_patch(rect)
            
            if not is_valid and peak < len(reasons) and isinstance(reasons[peak], str):
                mid_x = p0_x + (p1_x - p0_x) / 2
                mid_y = rect_height / 2
                ax_ecg.text(
                    mid_x, mid_y, reasons[peak], color='black', fontsize=10, 
                    fontweight='bold', ha='center', va='center', 
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2', edgecolor='none')
                )

        ax_ecg.set_title(f'Section {sect_id} | {start_idx}:{end_idx} (Left/Right Arrows to toggle Overlays | Click R-R for STFT)') 
        ax_ecg.set_xlabel("Timesteps")
        ax_ecg.set_ylabel("ECG mV")
        ax_ecg.legend(loc='upper right')

        # Interactive PQRST Overlay (Hidden)
        # Filter interior peaks belonging strictly to this section
        inners = self.data.interior_peaks[
            (self.data.interior_peaks['r_peak'] >= start_idx) & 
            (self.data.interior_peaks['r_peak'] <= end_idx)
        ]
        
        # Helper to plot arrays safely and add to the toggle list
        def plot_inners(col_name: str, color: str, marker: str, size: int = 40):
            valid_x = inners[col_name][inners[col_name] > 0]
            if len(valid_x) > 0:
                valid_y = self.data.wave[valid_x]
                scat = ax_ecg.scatter(
                    valid_x, valid_y, color=color, marker=marker, 
                    s=size, zorder=6, visible=False, label=col_name.capitalize()
                )
                pqrst_artists.append(scat)

        plot_inners('p_peak', 'green', 'o')
        plot_inners('q_peak', 'cyan', 'v')
        plot_inners('s_peak', 'magenta', '^')
        plot_inners('t_peak', 'black', 'o')
        plot_inners('p_onset', 'purple', '|', size=150)
        plot_inners('q_onset', 'darkgoldenrod', '|', size=150)
        plot_inners('j_point', 'dodgerblue', 'o', size=80)
        plot_inners('t_onset', 'teal', '|', size=150)
        plot_inners('t_offset', 'orange', '|', size=150)

        # Stacked SQI & Interval Metrics Text Block
        stat_text = (
            f"Kurtosis: {sqi_metrics['kurtosis']:.2f}   |   "
            f"Hjorth Complexity: {sqi_metrics['hjorth']:.2f}   |   "
            f"Wasserstein Dist: {sqi_metrics['wdist']:.2f}   |   "
            f"QRS Ratio: {sqi_metrics['spectral']:.2f}   |   "
            f"Bad Beat Ratio: {sqi_metrics['bad_b_rat']:.1%}\n"
            f"PR: {sqi_metrics['PR']:.0f}ms   |   "
            f"QRS: {sqi_metrics['QRS']:.0f}ms   |   "
            f"ST: {sqi_metrics['ST']:.0f}ms   |   "
            f"QT: {sqi_metrics['QT']:.0f}ms   |   "
            f"QTc: {sqi_metrics['QTc']:.0f}ms   |   "
            f"QTVI: {sqi_metrics['QTVI']:.2f}   |   "
            f"TpTe: {sqi_metrics['TpTe']:.0f}ms"
        )
        
        ax_ecg.text(
            0.5, -0.3, stat_text, 
            transform=ax_ecg.transAxes, 
            fontsize=11, fontweight='bold', 
            ha='center', va='bottom', zorder=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
        )

        # Spectral Plots
        if new_peaks_arr.shape[0] >= 2:
            p0, p1 = new_peaks_arr[-2, 0], new_peaks_arr[-1, 0]
            self._draw_freq_and_spec(ax_freq, ax_spec, p0, p1, start_idx, end_idx)

        # Event Handlers 
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

        def onKeyPress_local(event):
            nonlocal mp_line_drawn, pqrst_visible
            if event.key == "right":
                if ax_mp_overlay.get_visible():
                    ax_mp_overlay.set_visible(False)
                    ax_mp_overlay.set_axis_off()
                else:
                    if not mp_line_drawn and post_metrics and "mp_distances" in post_metrics:
                        mp_dist = post_metrics["mp_distances"]
                        mp_thresh = post_metrics["mp_threshold"]
                        pad_len = len(wave_chunk) - len(mp_dist)
                        padded_mp = np.pad(mp_dist, (0, pad_len), constant_values=np.nan)
                        ax_mp_overlay.plot(x_range, padded_mp, color='purple', alpha=0.6, linestyle='-', linewidth=2)
                        ax_mp_overlay.axhline(y=mp_thresh, color='red', linestyle='--', label=f'Threshold ({mp_thresh:.2f})')
                        ax_mp_overlay.set_ylabel("Matrix Profile Distance", color='purple')
                        ax_mp_overlay.tick_params(axis='y', labelcolor='purple')
                        mp_line_drawn = True
                    ax_mp_overlay.set_visible(True)
                    ax_mp_overlay.set_axis_on()
                fig.canvas.draw_idle()
                
            elif event.key == "left":
                pqrst_visible = not pqrst_visible
                for artist in pqrst_artists:
                    artist.set_visible(pqrst_visible)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", onClick)
        fig.canvas.mpl_connect('key_press_event', onKeyPress_local)
        self._apply_timer_and_show(fig, timeout=3000)

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
        """Additive historical validation error plots."""
        if not self.plot_errors:
            return

        wave_chunk = self.data.wave[start_idx:end_idx]
        rolled_chunk = self.data.rolling_med[start_idx:end_idx]
        x_range = np.arange(start_idx, end_idx)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(x_range, wave_chunk, label='ECG', color='dodgerblue')
        ax.plot(x_range, rolled_chunk, label='Rolling Median', color='orange')
        ax.scatter(new_peaks_arr[:, 0], peak_info['peak_heights'], marker='D', color='red', label='R peaks', zorder=5)

        for peak in range(new_peaks_arr.shape[0] - 1):
            is_valid = new_peaks_arr[peak, 1]
            band_color = 'lightgreen' if is_valid else 'red'
            p0_x = new_peaks_arr[peak, 0]
            p1_x = new_peaks_arr[peak+1, 0]
            rect_height = np.max(self.data.wave[p0_x:p1_x])
            rect = Rectangle(
                xy=(p0_x, 0), width=p1_x - p0_x, height=rect_height, 
                facecolor=band_color, edgecolor="grey", alpha=0.3 
            )
            ax.add_patch(rect)

        # Peak Separation
        if "bad_sep" in kwargs:
            bad_sep = kwargs["bad_sep"]
            for x in bad_sep:
                ax.axvline(x=new_peaks_arr[x, 0], color='goldenrod', linestyle='--', linewidth=2, label='Bad Separation' if x == bad_sep[0] else "")
                if x + 1 < len(new_peaks_arr):
                    ax.axvline(x=new_peaks_arr[x + 1, 0], color='goldenrod', linestyle='--', linewidth=2)

        # Peak Height
        if "low_peaks" in kwargs or "high_peaks" in kwargs:
            low_peaks = kwargs.get("low_peaks", [])
            high_peaks = kwargs.get("high_peaks", [])
            for idx in low_peaks:
                arrow = Arrow(new_peaks_arr[idx, 0] - 55, peak_info['peak_heights'][idx], 40, 0, width=0.05, color='goldenrod', label='Low Peak' if idx == low_peaks[0] else "")
                ax.add_patch(arrow)
            for idx in high_peaks:
                arrow = Arrow(new_peaks_arr[idx, 0] - 55, peak_info['peak_heights'][idx], 40, 0, width=0.05, color='darkviolet', label='High Peak' if idx == high_peaks[0] else "")
                ax.add_patch(arrow)

        # Rolling Median IQR
        if "outs" in kwargs:
            outs = kwargs["outs"]
            iqr = kwargs.get("iqr", 1.0)
            upper_b = np.quantile(rolled_chunk, .80) + 1.5 * iqr
            lower_b = np.quantile(rolled_chunk, .20) - 1.5 * iqr
            ax.axhline(y=upper_b, color='magenta', linestyle='--', label='Upper Guardrail')
            ax.axhline(y=lower_b, color='red', linestyle='--', label='Lower Guardrail')
            
            for out_type, p0, p1 in outs:
                height = np.max(self.data.wave[p0:p1]) if out_type == 'above' else np.min(self.data.wave[p0:p1])
                rect = Rectangle((p0, 0), p1 - p0, height, facecolor='yellow', alpha=0.5, label='Wandering Baseline' if outs.index((out_type, p0, p1)) == 0 else "")
                ax.add_patch(rect)

        # Peak Slopes
        if "slopes" in kwargs:
            leftbases = kwargs["leftbases"]
            slopes = kwargs["slopes"]
            upper_bound = kwargs["upper_bound"]
            lower_bound = kwargs["lower_bound"]
            ax.scatter(leftbases, self.data.wave[leftbases], marker="o", color="green", label="Left Base", zorder=6)
            _delt = 0.10 * (np.max(wave_chunk) - np.min(wave_chunk))
            
            for i, slope in enumerate(slopes):
                if i < len(leftbases) and i < len(new_peaks_arr):
                    if slope > upper_bound or slope < lower_bound:
                        dy = -_delt if slope > upper_bound else _delt
                        arrow = Arrow(leftbases[i], self.data.wave[leftbases[i]] - dy*2, 0, dy, width=40, color="red")
                        ax.add_patch(arrow)

        ax.set_title(f'{error_type} | Sect {sect_id}')
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("ECG mV")
        
        # Deduplicate legend items
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        self._apply_timer_and_show(fig=fig)

    def _draw_freq_and_spec(self, ax_freq, ax_spec, p0, p1, start_idx, end_idx):
        """Calculates STFT for the isolated inter-beat baseline, and Spectrogram for the section."""
        offset = int(self.data.fs * 0.10) # 100ms offset
        
        # --- 1. Local Hjorth (QRS Morphology) ---
        qrs_start = max(0, p0 - offset)
        qrs_end = min(len(self.data.wave), p0 + offset)
        qrs_samp = self.data.wave[qrs_start:qrs_end].flatten()
        local_hjorth = self.hijorth(qrs_samp) if len(qrs_samp) > 4 else 0.0

        # --- 2. Inter-Beat STFT (Baseline Stability) ---
        start_inter = p0 + offset
        end_inter = p1 - offset
        
        inter_noise_ratio = 0.0
        freq_inter = np.array([])
        fft_inter = np.array([])
        hf_mask = np.array([], dtype=bool)
        
        # Only process if we have a valid >200ms gap between peaks
        if end_inter - start_inter > int(self.data.fs * 0.20):
            inter_samp = self.data.wave[start_inter:end_inter].flatten()
            window = np.hanning(len(inter_samp))
            fft_inter = np.abs(np.fft.rfft(inter_samp * window))
            freq_inter = np.fft.rfftfreq(len(inter_samp), d=1/self.data.fs)
            
            total_inter_pwr = np.sum(fft_inter)
            if total_inter_pwr > 0:
                hf_mask = freq_inter > 15.0
                hf_noise_pwr = np.sum(fft_inter[hf_mask])
                inter_noise_ratio = hf_noise_pwr / total_inter_pwr
        
        # --- 3. Plotting the Inter-Beat Spectrum ---
        if len(freq_inter) > 0:
            # Plot Low-Frequency Physiological Baseline (<= 15 Hz)
            lf_mask = ~hf_mask
            if np.any(lf_mask):
                ax_freq.stem(
                    freq_inter[lf_mask], fft_inter[lf_mask], 
                    basefmt=" ", linefmt='dodgerblue', markerfmt='bo', label='Physio Baseline (<=15Hz)'
                )
            
            # Plot High-Frequency Baseline Noise (> 15 Hz)
            if np.any(hf_mask):
                # Paint red if it violates the 40% noise ratio threshold
                noise_color = 'red' if inter_noise_ratio > 0.50 else 'orange'
                ax_freq.stem(
                    freq_inter[hf_mask], fft_inter[hf_mask], 
                    basefmt=" ", linefmt=noise_color, markerfmt=f'{noise_color}', label='HF Noise (>15Hz)'
                )

        # Draw boundaries and shade background if rejected
        ax_freq.axvline(x=15.0, color='grey', linestyle='--', alpha=0.5)
        
        if inter_noise_ratio > 0.50:
            ax_freq.axvspan(15.0, 50.0, color='red', alpha=0.1, label='Rejected Baseline')
            
        # Embed the exact local metrics into the title
        ax_freq.set_title(f'Inter-Beat STFT | Baseline Noise: {inter_noise_ratio:.0%} | Local Hjorth: {local_hjorth:.2f}')
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Magnitude")
        ax_freq.set_xlim(0, 50) 
        ax_freq.legend(loc='upper right')

        # --- Spectrogram (Full Section) ---
        chunk = self.data.wave[start_idx:end_idx].flatten()
        spec_nfft = max(256, int(self.data.fs * 2)) 
        
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
        ax_spec.set_title("Spectrogram (Full Section | 2s Window)")
        ax_spec.set_ylim(0, 50)

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
        self.freq_tools = CardiacFreqTools(fs=self.fs)
        self.gui = SignalGUI(
            ecg_data = self.data, 
            tools = self.freq_tools,
            plot_section=self.configs.get("plot_section", False), 
            plot_errors=self.configs.get("plot_errors", False),
        )
        self.stack_range:range = np.arange(10000, self.data.sect_info.shape[0], 10000)
        # Historical trackers
        self.low_counts:int = 0
        self.sect_id:int = 0
        self.iqr_low_thresh:float = 1.0
        # self.is_stable:bool = False Only used in CardiacFT class. 
        #Pointers 
        self.p_ptr:int = 0
        self.ip_ptr:int = 0

    @log_time
    def peak_stack_test(self, new_peaks_arr:np.array) -> np.array:
        """Times how long it takes to run the vstack.  Useful for debugging.  
        4/29/26-Changed stacking to pointers for faster runtime. May not need this func anymore

        Args:
            new_peaks_arr (np.array): new array

        Returns:
            np.array: Stacked array
        """        
        return np.vstack((self.data.peaks, new_peaks_arr)).astype(np.int32)
    
    def consecutive_valid_peaks(self, r_peaks: np.ndarray, lookback: int = 1500):
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

    def historical_validation(
            self, 
            new_peaks_arr:np.ndarray, 
            last_keys:list, 
            peak_info:dict,
            start_idx:int, 
            end_idx:int
        ) -> Tuple[bool, np.ndarray, str]:
        """Rejects whole segments based on historical averages (IQR, Slope, Separation, Height)."""
        sect_valid = True
        fail_reason = ""
        plot_kwargs = {}

        # Grab historical rolling median
        rolling_med_start = last_keys[0]
        rolling_med_end = last_keys[-1]
        med_arr = self.data.rolling_med[rolling_med_start:rolling_med_end]
        r_peaks = new_peaks_arr[:, 0]
        # ==========================================================
        # GATE 1: Peak Separation Check
        # ==========================================================
        med_diff = np.diff(last_keys)
        if len(med_diff) > 0:
            last_med_p_sep = np.median(med_diff) if len(med_diff) > 0 else (self.data.fs * 0.8)
            lower_bound_sep = last_med_p_sep * 0.4
            upper_bound_sep = last_med_p_sep * 4.0
            valid_idx = np.where(new_peaks_arr[:, 1] == 1)[0]
            if len(valid_idx) > 1:
                valid_r_peaks = new_peaks_arr[valid_idx, 0]
                diffs = np.diff(valid_r_peaks)
                bad_sep_short = np.where(diffs < lower_bound_sep)[0]
                bad_sep_long = np.where(diffs > upper_bound_sep)[0]
                bad_idxs = []
                #Quick spikes ie - short gaps
                if bad_sep_short.size > 0:
                    for b_idx in bad_sep_short:
                        #Invalidate the peaks
                        orig_1 = valid_idx[b_idx]
                        orig_2 = valid_idx[b_idx + 1]
                        new_peaks_arr[orig_1, 1] = 0
                        new_peaks_arr[orig_2, 1] = 0
                        bad_idxs.extend([orig_1, orig_2])
                    fail_reason += "short_sep | "
                    sect_valid = False
                    logger.warning(f"FAILED:Peak separation violation in section {self.sect_id}")

                if bad_sep_long.size > 0:
                    fail_reason += "long_sep | "
                    sect_valid = False
                    logger.warning(f"FAILED:Peak separation violation in section {self.sect_id}")
                
                if bad_idxs:
                    plot_kwargs["bad_sep"] = list(set(bad_idxs))
                    
        # ==========================================================
        # GATE 2: Peak Height Check (ECG to rolling diff)
        # ==========================================================
        hist_peak_heights = self.data.wave[last_keys] - self.data.rolling_med[last_keys]
        med_heights = np.median(hist_peak_heights)
        lower_bound_ht = med_heights * 0.4
        upper_bound_ht = med_heights * 4.0

        curr_peak_heights = self.data.wave[r_peaks] - self.data.rolling_med[r_peaks]
        low_peaks = np.where(curr_peak_heights < lower_bound_ht)[0]
        high_peaks = np.where(curr_peak_heights > upper_bound_ht)[0]

        if low_peaks.size > 0 or high_peaks.size > 0:
            new_peaks_arr[low_peaks, 1] = 0
            new_peaks_arr[high_peaks, 1] = 0
            fail_reason += "height | "
            sect_valid = False
            logger.warning(f"FAILED:Peak height violation in section {self.sect_id}")
            plot_kwargs["low_peaks"] = low_peaks
            plot_kwargs["high_peaks"] = high_peaks
        # ==========================================================
        # GATE 3: Rolling Median (IQR Wandering Baseline Check)
        # ==========================================================
        q1 = np.quantile(med_arr, 0.20) if len(med_arr) > 0 else 0
        q3 = np.quantile(med_arr, 0.80) if len(med_arr) > 0 else 0
        raw_iqr = max(q3 - q1, 0.05)
        iqr = raw_iqr

        # Prevent vanishing gradient for IQR
        if iqr <= self.iqr_low_thresh + 0.001:
            self.low_counts = min(self.low_counts + 1, 20) # Cap at 20 to prevent OverflowError before min() evaluates
            if self.low_counts > 3: 
                multiplier = min(1.5**(self.low_counts - 3), 15.0)
                iqr *= multiplier
                logger.info(f'Increased IQR {multiplier:.3f}x to {iqr:.4f} for section {self.sect_id}')

            self.iqr_low_thresh = min(self.iqr_low_thresh, raw_iqr)

        else:
            self.iqr_low_thresh = iqr
            self.low_counts = 0
        
        #Grab Current roll median for eval
        curr_rolled_med = self.data.rolling_med[start_idx:end_idx].flatten()

        upper_med_bound = np.quantile(curr_rolled_med, 0.80) + 1.5 * iqr
        lower_med_bound = np.quantile(curr_rolled_med, 0.20) - 1.5 * iqr
        out_above = np.where(curr_rolled_med > upper_med_bound)[0]
        out_below = np.where(curr_rolled_med < lower_med_bound)[0]

        if out_above.size > 0 or out_below.size > 0:
            bad_pandas = 0
            outs = []
            for i in range(len(r_peaks) - 1):
                p0_rel, p1_rel = r_peaks[i] - start_idx, r_peaks[i+1] - start_idx
                samp_section = curr_rolled_med[p0_rel:p1_rel]
                
                if np.any(samp_section > upper_med_bound):
                    outs.append(('above', r_peaks[i], r_peaks[i+1]))
                    new_peaks_arr[i, 1] = 0
                    bad_pandas += 1
                elif np.any(samp_section < lower_med_bound):
                    outs.append(('below', r_peaks[i], r_peaks[i+1]))
                    new_peaks_arr[i, 1] = 0
                    bad_pandas += 1

            #BUG - Consider bumping this down possibly...Previous gates are at 25% sect failure
            if bad_pandas > (round(0.50 * (len(r_peaks) - 1))):
                fail_reason += "roll_med | "
                sect_valid = False
                plot_kwargs["outs"] = outs
                plot_kwargs["iqr"] = iqr
                logger.warning(f"FAILED:Rolling median in section {self.sect_id}")

        # ==========================================================
        # GATE 4: Slope / Morphology Check
        # ==========================================================
        #BUG - Also might not needs this check if we're already doing the matrix
            #profile for morphology checks
            #NOTE: Firing on jagged slopes that may misrepresent slope.
            #Stumpy is also very slow, this could be a low cost replacement.

        lookbacks = r_peaks - int(last_med_p_sep * 0.75)
        leftbases, slopes = [], []

        for lookback, RP in zip(lookbacks, r_peaks):
            # Apply a lightweight Hanning window to kill HF static before derivation
            raw_seg = self.data.wave[max(0, lookback):RP+1].flatten()
            if len(raw_seg) > 7:
                window = np.hanning(7)
                window /= window.sum()
                # mode='same' keeps array length identical to raw_seg
                smoothed_seg = np.convolve(raw_seg, window, mode='same')
            else:
                smoothed_seg = raw_seg
            grad = np.diff(smoothed_seg)
            signchange = np.roll(np.sign(grad), 1) - np.sign(grad)
            np_inflections = np.where((signchange == -2) | (signchange == -1))[0]
            
            if np_inflections.size > 0:
                leftbases.append(max(0, lookback) + np_inflections[-1])
            else:
                leftbases.append(lookback) # Fallback

        if len(leftbases) == len(r_peaks) and len(leftbases) > 2:
            slopes = [np.polyfit(range(x1, x2), self.data.wave[x1:x2], 1)[0].item() if x2 - x1 > 3 else 0 for x1, x2 in zip(leftbases, r_peaks)]
            lower_bound_slope = np.median(slopes) * 0.1
            upper_bound_slope = np.median(slopes) * 5.0
            
            slope_arr = np.array(slopes)
            bad_slopes = np.where((slope_arr < lower_bound_slope) | (slope_arr > upper_bound_slope))[0]
            
            if bad_slopes.size > 0:
                new_peaks_arr[bad_slopes, 1] = 0
                fail_reason += "slope | "
                sect_valid = False
                plot_kwargs["leftbases"] = leftbases
                plot_kwargs["slopes"] = slopes
                plot_kwargs["upper_bound"] = upper_bound_slope
                plot_kwargs["lower_bound"] = lower_bound_slope
                logger.warning(f"FAILED:Slope in section {self.sect_id}")

        if not sect_valid and self.gui.plot_errors:
            fail_reason = fail_reason.strip("|")
            self.gui.plot_validation_error(f"FAILED:Historical {fail_reason}", start_idx, end_idx, new_peaks_arr, peak_info, self.sect_id, **plot_kwargs)

        return sect_valid, new_peaks_arr, fail_reason
        # Add HR stats for that secti

    def estimate_iso(self, r_peaks:list) -> float:
        iso = []
        for idx, r_pe in enumerate(r_peaks[:-1]): 
            try:
                start = r_peaks[idx].t_peak
                end = r_peaks[idx + 1].p_peak
                if start and end:
                    lil_wave = self.data.wave[start:end].flatten()
                    lil_grads = np.gradient(np.gradient(lil_wave))
                    half = lil_grads.shape[0]//2
                    T_off = start + np.argmax(lil_grads[:half])
                    P_on = start + half + np.argmax(lil_grads[half:])
                    iso.append(np.nanmean(self.data.wave[T_off:P_on]))
                else:
                    continue
            except Exception as e:
                logger.warning(f'Iso extraction Error for Rpeak {r_pe.r_peak} {e} ')

        if iso:
            isoelectric = np.round(np.nanmean(iso), 6)
            return isoelectric
        else:
            return None

    def _calc_qtc(self, QT:int, RR:int, formula:str="Bazzett"):
        """Calculates corrected QT in seconds

        Args:
            QT (int): QT interval (seconds)
            RR (int): R to R interval (seconds)
            formula (str, optional): What type of QT correction you want. Defaults to "Bazzett".

        Returns:
            QTc (int): Corrected QT in ms
        """
        if not QT or not RR or RR <= 0:
            return 0
            
        # Clinical formulas require QT in milliseconds, and RR in seconds.
        rr_sec = RR / 1000.0
        match formula:
            case "Bazzett":
            # Bazett Formula: QTc = QT / sqrt(RR)
                QTc = QT / np.sqrt(rr_sec)
            case "Fridericia":
            # Fridericia Formula: QTc = QT / (RR^(1/3))
                QTc = QT / (rr_sec ** (1/3))
            case "Framingham":
            # Framingham Formula: QTc = QT + 0.154 * (1 - RR)
                QTc = QT + 154 * (1 - rr_sec)
        return int(QTc) 

    def _calc_qtvi(self, qt_intervals: list, rr_intervals: list) -> float:
        """
        Calculates QT Variability Index (QTVI) using Berger's formula:
        QTVI = log10[(QTv / QTm^2) / (RRv / RRm^2)]
        """
        # Filter out missing extractions (NaNs/Nones)
        clean_qt = [v for v in qt_intervals if v is not None and not np.isnan(v)]
        clean_rr = [v for v in rr_intervals if v is not None and not np.isnan(v)]
        
        # Variance calculation requires at least 2 valid samples
        if len(clean_qt) < 2 or len(clean_rr) < 2:
            return np.nan
            
        qt_m = np.mean(clean_qt)
        qt_v = np.var(clean_qt, ddof=1) # Sample variance
        
        rr_m = np.mean(clean_rr)
        rr_v = np.var(clean_rr, ddof=1)
        
        # Prevent division by zero or log(0) crashes
        if qt_m == 0 or rr_m == 0 or rr_v == 0 or qt_v == 0:
            return np.nan
            
        qt_norm = qt_v / (qt_m ** 2)
        rr_norm = rr_v / (rr_m ** 2)
        
        return np.round(np.log10(qt_norm / rr_norm), 2).item()

    def _calc_tpte(self, t_peak: int, t_offset: int) -> int:
        """Calculates T-peak to T-end (Tp-Te) interval in milliseconds."""
        if not t_peak or not t_offset:
            return 0
            
        # TpTe is the distance from the peak of the T-wave to the end of the T-wave
        return int(1000 * ((t_offset - t_peak) / self.fs))
        
    def _curve_line_dist(self, point:tuple, coef:tuple)->float:
        """This function calculates the distance from every point
        in our manufactured line, to the curve.  We use this to determine
        where the elbow of a curve is at its maximal.

        Args:
            point (tuple): _description_
            coef (tuple): _description_

        Returns:
            float: _description_
        """			
        d = abs((coef[0]*point[0])-point[1]+coef[1])/np.sqrt((coef[0]*coef[0])+1)
        
        return d

    def _find_p_onset(self, P_peak:int, srch_width:int):
        try:
            slope_end = P_peak + 1
            slope_start = slope_end - int(srch_width*2)
            lil_wave = self.data.wave[slope_start:slope_end].flatten()
            lil_grads = np.gradient(np.gradient(lil_wave))
            P_onset = slope_start + np.argmax(lil_grads).item()
            logger.debug(f'Adding P onset')
            return P_onset
        except Exception as e:
            logger.warning(f'P onset error {e}')

    def _find_q_onset(self, Q_peak:int, P_peak:int):
        try:
            slope_start = Q_peak - int((Q_peak - P_peak)*.70)
            slope_end = Q_peak + 1
            lil_wave = self.data.wave[slope_start:slope_end].flatten()
            lil_grads = np.gradient(np.gradient(lil_wave))
            shoulder = np.where(np.abs(lil_grads) >= np.mean(np.abs(lil_grads)))[0]
            Q_onset = slope_start + shoulder[0].item() + 1
            logger.debug(f'Adding Q onset')
            return Q_onset
        
        except Exception as e:
            logger.warning(f'Q onset error {e}')

    def _find_t_offset(self, T_peak:int, srch_width:int, isoelectric:float):
        if not T_peak: 
            return None
        slope_start = T_peak
        slope_end = T_peak + int(srch_width*2) 

        try:
            lil_wave = self.data.wave[slope_start:slope_end].flatten()
            lil_grads = np.gradient(np.gradient(lil_wave))
            T_offset = slope_start + np.argmax(lil_grads).item()
            logger.debug(f'Adding T offset')
            return T_offset
            
        except Exception as e:
            logger.warning(f'T Offset error {e}')
            logger.debug("secondary T_offset extraction")
            # NOTE backup T_offset extract
                # If the acceleration method fails.  Add in another check to look at
                # the slope after the T peak.  Draw a line down to the isoelectric
                # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7080915

            try:
                m, b = np.polyfit(range(slope_start, slope_end), self.data.wave[slope_start:slope_end], 1)
                x_intercept = -b / m
                isoelectric = self.data.sect_info["isoelectric"] if self.data.sect_info["isoelectric"] is not None else x_intercept
                x_tans = np.linspace(T_peak, T_offset, 100)
                y_tans = m * x_tans + b
                T_cross = np.abs(y_tans - isoelectric)
                T_offset = x_tans[T_cross]
                logger.info(f'Adding T offset backup')
                return T_offset

            except Exception as e:
                logger.warning(f'T Offset backup extraction error = \n{e}')
                return None

    def _find_j_point(self, s_peak: int, t_peak: int, rolled_med: np.ndarray, start_p: int) -> int:
        if not s_peak or not t_peak: return None
        try:
            slope_start = s_peak
            lil_wave = self.data.wave[slope_start:t_peak].flatten()
            med_sect = rolled_med[slope_start - start_p : t_peak - start_p].flatten()
            ecg_less_median = np.where(lil_wave < med_sect)[0]
            
            groups = np.split(ecg_less_median, np.where(np.diff(ecg_less_median) != 1)[0] + 1)
            last_group = groups[0] if len(groups) > 0 and len(groups[0]) > 0 else [0]
            slope_end = slope_start + last_group[-1]

            X = np.arange(slope_start, slope_end)
            y = self.data.wave[slope_start:slope_end].flatten()
            
            if X.shape[0] > 5:
                knee = KneeLocator(X, y, curve="concave", direction="increasing")
                if knee.elbow is not None:
                    return int(knee.elbow) + 1
            return None
        except Exception as e:
            logger.debug(f'J point error: {e}')
            return None

    def _find_t_onset(self, t_peak: int, j_point: int, s_peak: int, samp_min: int, rolled_med: np.ndarray, start_p: int) -> int:
        if not t_peak: 
            return None
        try:
            slope_end = t_peak + 1
            if j_point:
                slope_start = j_point
            else:
                slope_st = samp_min if samp_min else s_peak
                if not slope_st: 
                    return None
                lil_wave = self.data.wave[slope_st:slope_end].flatten()
                med_sect = rolled_med[slope_st - start_p : slope_end - start_p].flatten()
                ecg_less_median = np.where(lil_wave < med_sect)[0]
                groups = np.split(ecg_less_median, np.where(np.diff(ecg_less_median) != 1)[0] + 1)
                first_group = groups[0] if len(groups) > 0 and len(groups[0]) > 0 else [0]
                slope_start = slope_st + first_group[-1]

            lil_wave = self.data.wave[slope_start:slope_end].flatten()
            lil_grads = np.gradient(np.gradient(lil_wave))
            return slope_start + np.argmax(lil_grads).item()
        except Exception as e:
            logger.debug(f'T onset error: {e}')
            return None
    
    def _yabig_meanie(self, values: list, precision: int = 4) -> float:
        """Safely calculates the mean, ignoring Nones/NaNs. Returns np.nan if empty."""
        clean_vals = [v for v in values if v is not None and not np.isnan(v)]
        if not clean_vals:
            return np.nan
        return np.round(np.nanmean(clean_vals), precision).item()
    
    def extract_pqrst(self, new_peaks_arr:np.ndarray, peak_info:dict, rolled_med:np.ndarray, start_p:int):
        """Routine for PQRST geometry extraction."""
        beats: List[HeartBeat] = [HeartBeat(r_peak=int(p)) for p in new_peaks_arr[:, 0]]
        samp_mins = [None] * len(beats)

        # Recover P and Q from the previous section's overlap beat to prevent data loss!
        if self.ip_ptr > 0 and len(beats) > 0:
            old_peak = self.data.interior_peaks[self.ip_ptr]
            if old_peak['r_peak'] == beats[0].r_peak:
                beats[0].p_peak = old_peak['p_peak'] if old_peak['p_peak'] != 0 else None
                beats[0].q_peak = old_peak['q_peak'] if old_peak['q_peak'] != 0 else None
                beats[0].p_peak_a = old_peak['p_peak_a'] if old_peak['p_peak_a'] != 0 else None
                beats[0].q_peak_a = old_peak['q_peak_a'] if old_peak['q_peak_a'] != 0 else None

        #Extract main peaks for PQRST
        for i in range(len(beats)-1):
            beat = beats[i]
            next_beat = beats[i+1]
            peak0 = beat.r_peak
            beat.r_peak_a = np.round(self.data.wave[peak0].item(), 6)
            peak1 = next_beat.r_peak

            #Take the difference of each point
            grad = np.diff(self.data.wave[peak0:peak1+1].flatten())
            #Calculate a sign change for that difference
            signchange = np.roll(np.sign(grad), 1) - np.sign(grad)
            #Locate those changes where flipping from negative
            np_inflections = np.where((signchange == -2) | (signchange == -1))[0]
            #Std deviation check
            std_dev_rng = self.data.wave[peak0:peak1][np_inflections[0]:np_inflections[-1]] if len(np_inflections) > 0 else []
            std_dev_SQ = np.std(std_dev_rng) if len(std_dev_rng) > 0 else 0
            reject_limit = 0.30 * np.mean(peak_info.get('prominences', [1]))
            if std_dev_SQ >= reject_limit or len(np_inflections) == 0:
                #TODO - I should probably fail the beat as well here. 
                #This rejection may not be needed considering the matrix profile/inband tests
                # self.data.peaks[peak0, 1] = 0
                # self.data.peaks[peak1, 1] = 0
                continue

            # MEAS Q peak (next)
            next_beat.q_peak = int(np_inflections[-1]) + peak0
            if next_beat.q_peak:
                next_beat.q_peak_a = np.round(self.data.wave[next_beat.q_peak].item(), 6)
            
            # MEAS S peak
            slope_start = peak0
            slope_end = peak0 + int((peak1 - peak0) // 3)
            lil_wave = self.data.wave[slope_start:slope_end].flatten()
            if len(lil_wave) > 3:
                f = interp1d(np.arange(slope_start, slope_end), lil_wave, kind="cubic")
                x_vals = np.linspace(slope_start, slope_end-1, num=lil_wave.shape[0]*10)
                y_vals = f(x_vals)
                coeffs = np.polyfit((x_vals[0], x_vals[-1]), (y_vals[0], y_vals[-1]), 1)
                p_dist = [self._curve_line_dist(pt, coeffs) for pt in zip(x_vals, y_vals)]
                closest = int(np.round(p_dist.index(max(p_dist)) / 10) + peak0)
                beat.s_peak = closest
            else:
                beat.s_peak = int(np.argmin(lil_wave)) + slope_start
            if beat.s_peak:
                beat.s_peak_a = np.round(self.data.wave[beat.s_peak].item(), 6)
            
            #Figure out the samp min for the T peak
            samp_min = int(np.argmin(self.data.wave[peak0:slope_end]))
            if (self.data.wave[peak0+samp_min].item() < rolled_med[samp_min]) and (samp_min in np_inflections[:6]):
                samp_min = samp_min + peak0
            else:
                samp_min = int(min(np_inflections)) + peak0 if len(np_inflections) > 0 else peak0
            samp_mins[i] = samp_min
            
            # T Peak & (Next) P Peak
            SQ_range = self.data.wave[samp_min:next_beat.q_peak if next_beat.q_peak else peak1].flatten()
            filt_rol_med = rolled_med[samp_min - start_p : (next_beat.q_peak if next_beat.q_peak else peak1) - start_p].flatten()
            
            if len(SQ_range) == len(filt_rol_med) and len(SQ_range) > 0:
                SQ_med_reduced = SQ_range - filt_rol_med
                half_idx = len(SQ_med_reduced) // 2
                
                # MEAS T Peak
                try:
                    RR_first_half = SQ_med_reduced[:half_idx]
                    peak_T_find = ss.find_peaks(RR_first_half, height=np.percentile(SQ_med_reduced, 60))
                    if peak_T_find[0].shape[0] > 0:
                        top_T = peak_T_find[0][np.argmax(peak_T_find[1]['peak_heights'])].item()
                        beat.t_peak = peak0 + (samp_min - peak0) + top_T
                        if beat.t_peak:
                            beat.t_peak_a = np.round(self.data.wave[beat.t_peak].item(), 6)
                except Exception as e:
                    logger.info(f"T peak extraction error for {peak0}. Error message {e}")
                # MEAS P Peak (next)
                try:
                    RR_second_half = SQ_med_reduced[half_idx:]
                    peak_P_find = ss.find_peaks(RR_second_half, height=np.percentile(SQ_med_reduced, 60))
                    if peak_P_find[0].shape[0] > 0:
                        top_P = peak_P_find[0][np.argmax(peak_P_find[1]['peak_heights'])].item() + half_idx
                        next_beat.p_peak = peak0 + (samp_min - peak0) + top_P
                        if next_beat.p_peak:
                            next_beat.p_peak_a = np.round(self.data.wave[next_beat.p_peak].item(), 6)
                except Exception as e:
                    logger.info(f"P peak extraction error for {peak0}. Error message {e}")

        isoelectric = self.estimate_iso(beats)
        self.data.sect_info["isoelectric"][self.sect_id] = isoelectric
        temp_arr = []
        
        # Iterate over beats to ensure no overlap data is lost
        for i, beat in enumerate(beats):
            beat.valid_qrs = utils.valid_QRS(beat)
            if beat.valid_qrs:
                # Provide a fallback width if peaks are missing to prevent crash
                if beat.s_peak and beat.q_peak:
                    srch_width = (beat.s_peak - beat.q_peak) * 2
                else:
                    srch_width = int(self.fs * 0.1)

                if beat.p_peak:
                    beat.p_onset  = self._find_p_onset(beat.p_peak, srch_width)
                if beat.q_peak and beat.p_peak:
                    beat.q_onset  = self._find_q_onset(beat.q_peak, beat.p_peak)
                if beat.t_peak:
                    beat.t_offset = self._find_t_offset(beat.t_peak, srch_width, isoelectric)
                if beat.s_peak and beat.t_peak:
                    beat.j_point  = self._find_j_point(beat.s_peak, beat.t_peak, rolled_med, start_p)
                if beat.t_peak and (beat.j_point or beat.s_peak):
                    samp_m = samp_mins[i] if i < len(samp_mins) else None
                    beat.t_onset  = self._find_t_onset(beat.t_peak, beat.j_point, beat.s_peak, samp_m, rolled_med, start_p)

            # Map Intervals directly to milliseconds
            # MEAS PR
            if beat.q_onset and beat.p_onset:
                beat.PR = int(1000 * ((beat.q_onset - beat.p_onset) / self.fs))
            # MEAS QRS
            # If we have a Jpoint, use that for QRS.  If not, use the S peak
            if beat.q_onset and beat.j_point:
                beat.QRS = int(1000 * ((beat.j_point - beat.q_onset) / self.fs))
            elif beat.q_onset and beat.s_peak:
                beat.QRS = int(1000 * ((beat.s_peak - beat.q_onset) / self.fs))
            # MEAS ST
            if beat.t_onset and beat.j_point:
                beat.ST = int(1000 * ((beat.t_onset - beat.j_point) / self.fs))
            elif beat.t_onset and beat.s_peak:
                beat.ST = int(1000 * ((beat.t_onset - beat.s_peak) / self.fs))
            # MEAS QT
            if beat.t_offset and beat.q_onset:
                beat.QT = int(1000 * ((beat.t_offset - beat.q_onset) / self.fs))
            # MEAS QTc
            # Calculate the RR interval in ms using the current and next R peak
            if beat.QT:
                rr_interval_ms = int(1000 * ((beats[i+1].r_peak - beat.r_peak) / self.fs))
                beat.QTc = self._calc_qtc(QT=beat.QT, RR=rr_interval_ms, formula="Bazzett")
            # MEAS TpTe
            if beat.t_peak and beat.t_offset:
                beat.TpTe = self._calc_tpte(beat.t_peak, beat.t_offset)
                
            temp_arr.append(beat.to_row())

        #Add the peak data to the interior_peaks structured array
        if temp_arr:
            new_interior_peaks = np.array(temp_arr, dtype=setup_globals.PEAK_DTYPES)
            n_ip = len(new_interior_peaks)
            self.data.interior_peaks[self.ip_ptr: self.ip_ptr + n_ip] = new_interior_peaks
            self.ip_ptr += n_ip

            # self.data.interior_peaks = np.concatenate((self.data.interior_peaks, new_interior_peaks))

    def section_stats(self, new_peaks_arr:np.ndarray, start_p:int, end_p:int):
        peak_check = np.any(new_peaks_arr[:-1, 1] == 0)
        if peak_check:
            bad_peaks = np.where(new_peaks_arr[:-1, 1] == 0)[0]
            logger.info(f'Ignored {len(bad_peaks)} invalid peaks for HR calc in section {self.sect_id}')
        
        # Now see if we have the bare minimum for peaks to extract. 
        if new_peaks_arr.size <= 2:
            self.data.sect_info["fail_reason"][self.sect_id] += " no_peaks | "
            logger.warning(f"Not enough peaks to calculate section stats")

        # Initialize the dataclass
        stats = SectionStat()

        # --- Time Domain HR Measures ---
        valid_intervals = []
        #Isolate beats that are valid, adjacent beats
        for i in range(len(new_peaks_arr) - 1):
            if new_peaks_arr[i,1] == 1 and new_peaks_arr[i+1, 1] == 1:
                gap = new_peaks_arr[i+1, 0] - new_peaks_arr[i, 0]
                valid_intervals.append(gap)

        # valid_peaks = new_peaks_arr[new_peaks_arr[:, 1] == 1, 0]
        valid_peaks = np.array(valid_intervals)
        if len(valid_peaks) > 1:
            # RR_diffs = np.diff(valid_peaks)
            RR_diffs_time = np.abs(valid_peaks / self.fs) * 1000 # Format to ms
            HR = 60 / (valid_peaks / self.fs) # BPM
            stats.HR    = self._yabig_meanie(HR.tolist(), 2)
            stats.SDNN  = np.round(np.std(HR), 5)

            if len(valid_peaks) > 1:
                rr_time_diffs = np.abs(np.diff(RR_diffs_time))
                stats.RMSSD = np.round(np.sqrt(np.mean(np.power(rr_time_diffs, 2))), 5)
            else:
                stats.RMSSD = 0.0

        # PQRST Averages
        try:
            # Filter interior peaks that belong to this section
            live_inners = self.data.interior_peaks[:self.ip_ptr]
            inners = live_inners[
                (live_inners["r_peak"] > start_p) & 
                (live_inners["r_peak"] < end_p)
            ]
            
            if len(inners) > 0:
                # Helper to extract a column and convert '0' (missed extraction) to NaN
                def get_clean_col(col_idx):
                    col_data = inners[col_idx].astype(float)
                    col_data[col_data == 0] = np.nan
                    return col_data.tolist()

                # Use _yabig_meanie to safely average the extracted columns
                stats.PR   = self._yabig_meanie(get_clean_col("PR"), 1)
                stats.QRS  = self._yabig_meanie(get_clean_col("QRS"), 1)
                stats.ST   = self._yabig_meanie(get_clean_col("ST"), 1)
                stats.QT   = self._yabig_meanie(get_clean_col("QT"), 1)
                stats.QTc  = self._yabig_meanie(get_clean_col("QTc"), 1)

                # Calc QTVI
                qt_list = get_clean_col("QT")
                rr_list = RR_diffs_time.tolist() if len(valid_intervals) > 0 else []
                stats.QTVI = self._calc_qtvi(qt_list, rr_list)

                # Calc TpTe
                stats.TpTe = self._yabig_meanie(get_clean_col("TpTe"), 1)

        except Exception as e:
            logger.warning(f'averages error in section: {self.sect_id} {e}')

        # Move info back to the sect_info
        self.data.sect_info["HR"][self.sect_id]    = stats.HR if not np.isnan(stats.HR) else 0
        self.data.sect_info["SDNN"][self.sect_id]  = stats.SDNN if not np.isnan(stats.SDNN) else 0
        self.data.sect_info["RMSSD"][self.sect_id] = stats.RMSSD if not np.isnan(stats.RMSSD) else 0
        
        # Update Segment info
        self.data.sect_info["PR"][self.sect_id]  = stats.PR if not np.isnan(stats.PR) else 0
        self.data.sect_info["QRS"][self.sect_id] = stats.QRS if not np.isnan(stats.QRS) else 0
        self.data.sect_info["ST"][self.sect_id]  = stats.ST if not np.isnan(stats.ST) else 0
        self.data.sect_info["QT"][self.sect_id]  = stats.QT if not np.isnan(stats.QT) else 0
        self.data.sect_info["QTc"][self.sect_id] = stats.QTc if not np.isnan(stats.QTc) else 0
        self.data.sect_info["QTVI"][self.sect_id] = stats.QTVI if not np.isnan(stats.QTc) else 0
        self.data.sect_info["TpTe"][self.sect_id] = stats.TpTe if not np.isnan(stats.TpTe) else 0

        # Add the R peaks to the peaks container. Time it every 10k sections
        n_p = len(new_peaks_arr)
        self.data.peaks[self.p_ptr:self.p_ptr + n_p] = new_peaks_arr
        self.p_ptr += n_p

    @log_time
    def run_extraction(self):
        """Iterates through the ECG waveform in overlapping sections."""
        sect_que = deque(self.data.sect_info[['start_point', 'end_point']])
        progbar, job_id = mainspinner(console, len(sect_que))
        with progbar:
            while len(sect_que) > 0:
                progbar.update(task_id=job_id, description=f"[green] searching sect {self.sect_id}/{self.data.sect_info.shape[0]}")
                curr_section = sect_que.popleft()
                start_p = curr_section[0].item()
                end_p = curr_section[1].item()
                wave_chunk = self.data.wave[start_p:end_p].flatten()

                # Calculate Rolling Median for the chunk
                rolled_med = utils.roll_med(wave_chunk).astype(np.float32)
                self.data.rolling_med[start_p:end_p] = rolled_med.reshape(-1, 1)

                # Check Signal Quality Index (SQI) using lightweight checks
                is_valid, fail_reason, pre_metrics = self.freq_tools.pre_peak_sqi(wave_chunk)
                self.data.sect_info["kurtosis"][self.sect_id] = pre_metrics.get("kurtosis", 0)
                self.data.sect_info["hjorth"][self.sect_id] = pre_metrics.get("hjorth", 0)
                self.data.sect_info["spectral"][self.sect_id] = pre_metrics.get("spectral", 0)
                self.data.sect_info["wdist"][self.sect_id] = pre_metrics.get("wdist", 0)
                self.data.sect_info["spec_entropy"][self.sect_id] = pre_metrics.get("spect_ent", 0)

                if not is_valid:
                    logger.warning(f"Section {self.sect_id} rejected: {fail_reason}")
                    self.data.sect_info["fail_reason"][self.sect_id] = fail_reason
                    if self.gui.plot_errors:
                        self.gui.plot_pre_error(fail_reason, start_p, end_p, self.sect_id, pre_metrics)
                    progbar.advance(job_id, advance=1)
                    self.sect_id += 1
                    continue

                # Extract Initial R Peaks
                r_peaks, peak_info = ss.find_peaks(
                    wave_chunk.flatten(), 
                    prominence=np.percentile(wave_chunk, 99), #99
                    height=np.percentile(wave_chunk, 94),     #95
                    distance=int(self.fs * 0.200)
                )
                #Basic count check (we shouldn't need this anymore)
                if r_peaks.size < 4 or r_peaks.size > 100:
                    logger.warning(f"Section {self.sect_id} rejected: Invalid peak count ({r_peaks.size}).")
                    self.data.sect_info["fail_reason"][self.sect_id] += " no_sig | "
                    progbar.advance(job_id, advance=1)
                    self.sect_id += 1
                    continue        

                # Shift peak array to present section start/end indexes.  Check for overlap with history
                r_peaks_shifted = r_peaks + start_p
                recent_peaks = self.data.peaks[max(0, self.p_ptr - 20):self.p_ptr, 0]
                same_peaks = sorted(list(set(r_peaks_shifted) & set(recent_peaks)))

                if len(same_peaks) > 0:
                    #Find the last peak in common. 
                    f_peak = max(same_peaks)
                    #Index those peaks from the last same peak, to the end of the r_peaks_shifted
                    keepers = r_peaks_shifted >= f_peak
                    r_p_shift = r_peaks_shifted[keepers]
                    r_p_new = r_peaks[keepers]
                    peak_info['peak_heights'] = peak_info['peak_heights'][keepers]
                    peak_info['prominences'] = peak_info['prominences'][keepers]
                    
                    # SYNCHRONIZE POINTERS: Step both pointers back by the overlap amount
                    overlap_count = np.sum(recent_peaks >= f_peak)
                    self.p_ptr -= overlap_count
                    self.ip_ptr -= overlap_count
                else:
                    r_p_shift = r_peaks_shifted
                    r_p_new = r_peaks

                # Historical data Validation
                is_stale = False
                lookback = int(self.fs * 10) 
                last_keys = self.consecutive_valid_peaks(r_peaks=self.data.peaks[:self.p_ptr], lookback=lookback)

                if last_keys is not False:
                    #See if the last keys are more than 60 seconds in the past
                    time_since_valid = (start_p - last_keys[-1]) / self.fs
                    if time_since_valid > 60:
                        is_stale = True
                        logger.warning(f"history deadlocked {time_since_valid:.2f}s")
                else:
                    #No history yet
                    is_stale = True
                
                is_turbulent = (
                    pre_metrics.get("hjorth", 0) > 2 or 
                    pre_metrics.get("wdist", 0) > 2
                )

                val_mask = np.ones(len(r_p_new), dtype=int) # Default all valid
                post_metrics = {}

                if is_turbulent or is_stale:
                    if is_stale:
                        logger.info(f"History stale/missing. Running heavy vetting on section {self.sect_id}")
                    
                    #Check each beat with the matrix profile and Welch's STFT. 
                    is_valid, fail_reason, post_metrics, val_mask = self.freq_tools.post_peak_sqi(wave_chunk, r_p_new)
                    self.data.sect_info["bad_b_rat"][self.sect_id]= post_metrics.get("bad_b_ratio", 1.0)

                    if not is_valid:
                        if self.gui.plot_errors:
                            self.gui.plot_post_error(
                                error_type=fail_reason, 
                                start_idx=start_p, 
                                end_idx=end_p, 
                                r_peaks_abs=r_p_shift, 
                                peak_info=peak_info, 
                                sect_id=self.sect_id, 
                                post_metrics=post_metrics,
                                val_mask = val_mask
                            )
                        logger.warning(f"Section {self.sect_id} rejected: {fail_reason}")
                        self.data.sect_info["fail_reason"][self.sect_id] += fail_reason
                        progbar.advance(job_id, advance=1)
                        self.sect_id += 1
                        continue
                else:
                    self.data.sect_info["bad_b_rat"][self.sect_id] = 0

                new_peaks_arr = np.hstack((r_p_shift.reshape(-1, 1), val_mask.reshape(-1, 1)))

                if not is_stale:
                    sect_valid, new_peaks_arr, fail_reason = self.historical_validation(
                        new_peaks_arr, last_keys, peak_info, 
                        start_idx=start_p, end_idx=end_p
                    )
                    if not sect_valid:
                        self.data.sect_info["fail_reason"][self.sect_id] += f" | {fail_reason}"
                else:
                    # If we don't have enough consecutive valids, trust the matrix profile / STFT are doing their jobs
                    sect_valid = True 

                # Finalize Section
                if sect_valid:
                    self.data.sect_info[self.sect_id]["valid"] = 1
                    # Proceed to PQRST extract and stats if section valid
                    self.extract_pqrst(new_peaks_arr, peak_info, rolled_med, start_p)
                    # Generate Section Stats
                    self.section_stats(new_peaks_arr, start_p, end_p)
                    #Update bad_b_ratio
                    #BUG - Fix this tomorrow
                    # self.data.sect_info["bad_b_rat"][self.sect_id] = np.round(new_peaks_arr[new_peaks_arr[:, 1] == 0].shape[0] / new_peaks_arr.shape[0], 2)
                    #BUG - Might need to return new_peaks_arr
                else:
                    self.data.sect_info["valid"][self.sect_id] = 0
                    new_peaks_arr[:, 1] = 0 
                    n_p = len(new_peaks_arr)
                    self.data.peaks[self.p_ptr : self.p_ptr + n_p] = new_peaks_arr
                    self.p_ptr += n_p

                #Plot all section info
                if self.gui.plot_section and sect_valid:
                    self.gui.plot_fft_sect(
                        start_p, end_p, new_peaks_arr, peak_info, 
                        self.sect_id, self.data.sect_info[self.sect_id],
                        post_metrics = post_metrics
                    )
                # Advance section id/progbar to next section
                del new_peaks_arr
                logger.debug(f'Section counter at {self.sect_id}')
                progbar.advance(job_id, advance=1)
                self.sect_id += 1

            #Trim the array's back to their true size
            self.data.peaks = self.data.peaks[:self.p_ptr]
            self.data.interior_peaks = self.data.interior_peaks[:self.ip_ptr]

# --- Program Start ---
def main():
    configs      :dict = setup_globals.load_config()
    fp           :Path = Path.cwd() / configs["data_path"]
    batch_process:bool = configs["batch"]
    selected     :int  = setup_globals.load_choices(fp, batch_process)
    
    if not isinstance(selected, list):
        file_list = [selected]
    else:
        file_list = selected
    for file_path in file_list:
        logger.info(f"Processing file {file_path.stem}")
        loader:SignalLoader = SignalLoader(file_path)
        configs["cam_name"] = file_path.stem
        loader.load_signal_data()
        configs["samp_freq"] = loader.fs
        ECG = loader.load_structures()
        RAD = RadECG(ECG, configs, fp)
        RAD.run_extraction()
        support.save_results(RAD.data, configs=configs, current_date=DATE_JSON)
        gc.collect()

if __name__ == "__main__":
    main()