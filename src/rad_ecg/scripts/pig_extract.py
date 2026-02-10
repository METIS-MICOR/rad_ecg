import os
import json
import numpy as np
import stumpy
from pathlib import Path
from numba import cuda
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Button, TextBox
from scipy.signal import find_peaks, stft, welch, convolve, butter, filtfilt
from scipy.fft import rfft, rfftfreq
from scipy.stats import wasserstein_distance
from rich import print
from rich.tree import Tree
from rich.text import Text
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn
)
from utils import segment_ECG
from setup_globals import walk_directory
from support import logger, console, log_time, NumpyArrayEncoder

# Ignore numba error
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# --- Wavelet / Phase Calculation Logic  ---
class CardiacPhaseTools:
    """Helper class for Phase Variance calculations."""
    def __init__(self, fs=1000, bandwidth_parameter=8.0):
        self.fs = fs
        self.c = bandwidth_parameter

    def complex_morlet_cwt(self, data, center_freq):
        """Performs CWT and returns envelope and phase.

        Args:
            data (np.array): view of the signal
            center_freq (int): Main Frequency to focus on

        Returns:
            envelope, phase: _description_
        """

        w_desired = 2 * np.pi * center_freq
        s = self.c * self.fs / w_desired
        M = int(2 * 4 * s) + 1
        t = np.arange(-M//2 + 1, M//2 + 1)
        norm = 1 / np.sqrt(s)
        wavelet = norm * np.exp(1j * self.c * t / s) * np.exp(-0.5 * (t / s)**2)
        cwt_complex = convolve(data, wavelet, mode='same')
        return np.abs(cwt_complex), np.angle(cwt_complex)

    def compute_continuous_phase_metric(self, signal, window_beats=10) -> np.array:
        """Generates a continuous time-series metric representing phase stability.
        1. Finds Peaks. (scipy)
        2. Segments Signal.
        3. Calculates Phase Variance across a rolling window of beats.

        Args:
            signal (np.array): Signal you want to look at
            window_beats (int, optional): default number of beats. Defaults to 10.

        Returns:
            metric_curve (np.array): Chunked array of the phase variance over time
        """

        # 1. Find Peaks
        #TODO - peak params
            # This could be improved upon - ie parameter adjustments

        peaks, _ = find_peaks(signal, distance=int(self.fs * 0.4), height=np.mean(signal)) 
        if len(peaks) < window_beats:
            return np.zeros_like(signal)

        metric_curve = np.zeros_like(signal, dtype=float)
        
        # We will use a rolling window of beats to calculate stability
        # Beat window size (fixed for alignment)
        beat_win = int(0.6 * self.fs) # 600ms
        pre_peak = int(0.2 * self.fs)
        
        # Center frequency for analysis (High freq usually shows jitter best)
        target_freq = 20 #30 
        
        # Pre-allocate beat segments
        beats = []
        valid_peaks = []
        
        for p in peaks:
            start = p - pre_peak
            end = start + beat_win
            if start >= 0 and end < len(signal):
                beats.append(signal[start:end])
                valid_peaks.append(p)
        
        beats = np.array(beats)
        n_beats = len(beats)
        
        if n_beats < window_beats:
            return metric_curve

        # Iterate through beats with rolling window
        half_w = window_beats // 2
        
        # Progress bar since this can be slow
        with Progress(
            SpinnerColumn(), 
            BarColumn(), 
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("Calculating Phase Variance...", total=n_beats)
            for i in range(n_beats):
                # Define rolling window indices
                start_b = max(0, i - half_w)
                end_b = min(n_beats, i + half_w)
                batch = beats[start_b:end_b]
                
                if len(batch) < 3: 
                    continue
                
                # --- Phase Variance Calculation ---
                # 1. CWT on batch
                phases = []
                envelopes = []
                for b in batch:
                    env, phi = self.complex_morlet_cwt(b, target_freq)
                    phases.append(phi)
                    envelopes.append(env)
                
                phases = np.array(phases)
                
                # 2. Mean Phase
                mean_phase = np.mean(phases, axis=0)
                
                # 3. Variance of deviations
                phase_dev = phases - mean_phase
                
                # Wrap phase differences to [-pi, pi] for correct variance
                phase_dev = (phase_dev + np.pi) % (2 * np.pi) - np.pi
                
                # Variance across the beat (time axis)
                phase_var_curve = np.var(phase_dev, axis=0)
                
                # 4. Collapse to scalar (Mean Variance during QRS complex)
                # We focus on the center 100ms where QRS is
                center_idx = len(phase_var_curve) // 2
                qrs_region = phase_var_curve[center_idx - 50 : center_idx + 50]
                scalar_score = np.mean(qrs_region)
                
                # Fill the metric curve for the duration of this R-R interval
                # (From current peak to next peak)
                current_p = valid_peaks[i]
                next_p = valid_peaks[i+1] if i < n_beats - 1 else len(signal)
                
                metric_curve[current_p : next_p] = scalar_score
                progress.advance(task)

        return metric_curve

# --- Data Loader ---
class SignalDataLoader:
    """Handles loading and structuring the NPZ data."""
    def __init__(self, file_path):
        self.file_path = str(file_path)
        if self.file_path.endswith("npz"):
            self.container = np.load(self.file_path)
            self.files = self.container.files
            self.channels = self._identify_and_sort_channels()
            self.full_data = self._stitch_blocks()
        elif self.file_path.endswith("pkl"):
            self.container = np.load(self.file_path, allow_pickle=True)
            self.full_data = self.container.to_dict(orient="series")
            self.channels = self.container.columns.to_list()
            if "ShockClass" in self.channels:
                self.outcomes = self.full_data.pop("ShockClass")
                self.channels.pop(self.channels.index("ShockClass"))
            else:
                self.outcomes = None
        
    def _identify_and_sort_channels(self):
        """
        Identifies unique channel names from NPZ keys and returns them 
        in a deterministic (alphabetical) order.
        """
        raw_names = set()
        
        for k in self.files:
            # Extract channel name from keys like 'ECG_block_1', 'HR_block_0'
            if '_block_' in k:
                name = k.split('_block_')[0]
                raw_names.add(name)
            else:
                # Catch-all for keys that don't follow the block naming convention
                raw_names.add(k)
        
        # Sort alphabetically to ensure the plot labels consistently map to the data indices
        return sorted(list(raw_names))
    
    def _stitch_blocks(self):
        full_data = {}
        for ch in self.channels:
            # Filter keys for this channel and sort by block index
            ch_blocks = sorted(
                [k for k in self.files if k.startswith(f"{ch}_block_")], 
                key=lambda x: int(x.split('_block_')[-1])
            )
            
            if ch_blocks:
                full_data[ch] = np.concatenate([self.container[b] for b in ch_blocks])
            else:
                # Fallback: if no blocks found, maybe it's a single file entry
                if ch in self.files:
                    full_data[ch] = self.container[ch]
        return full_data

# --- Advanced Viewer ---
class RegimeViewer:
    """
    Interactive viewer for signal, Semantic Segmentation via minimum CAC (Corrected Arc Curve), and Phase Variance.
    Includes frequency analysis and custom navigation.
    """
    def __init__(
        self, 
        signal_data  :np.array, 
        cac_data     :np.array, 
        regime_locs  :np.array, 
        m            :int, 
        sampling_rate:float=1000.0, 
        lead         :str='Carotid (TS420)'
        ):
        """
        Args:
            signal_data (np.array): np.array of the signal data
            cac_data (np.array): np.array of CAC curve data
            regime_locs (np.array): np.array of regime change locations
            m (int): stumpy search window width
            sampling_rate (float): in Hz. Defaults to 1000.0
            lead (str): Signal being analyzed
        """        
        # 1. Data Setup
        self.signal = signal_data
        self.cac = cac_data
        self.regime_locs = regime_locs
        self.m = m
        self.fs = sampling_rate
        self.lead = lead        

        # Calculate Phase Variance Stream
        console.print("[cyan]Pre-computing Phase Variance Stream...[/]")
        self.ptools = CardiacPhaseTools(fs=self.fs)
        self.phase_var_stream = self.ptools.compute_continuous_phase_metric(self.signal)
        
        # 2. State Settings
        self.window_size = 10_000
        self.current_pos = 0
        self.step_size = 20
        self.paused = False
        
        # Frequency State: 0=Off, 1=Stem, 2=Specgram
        self.freq_mode = 0 
        
        # 3. Setup Figure
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click_jump)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.setup_layout()
        self._init_axes_pool()
        
        # 4. Start Animation
        self.ani = FuncAnimation(
            self.fig, self.update_frame, interval=30, blit=False, cache_frame_data=False
        )
        plt.show()

    def setup_layout(self):
        """Define GridSpec layout."""
        self.gs_main = gridspec.GridSpec(1, 2, width_ratios=[10, 1.5], figure=self.fig)
        
        # Main plot area: signal, CAC, Phase Var, Navigator
        self.gs_plots = gridspec.GridSpecFromSubplotSpec(
            4, 1, 
            subplot_spec=self.gs_main[0], 
            height_ratios=[3, 1.5, 1.5, 0.5],
            hspace=0.15
        )
        
        # Side controls
        self.gs_side = gridspec.GridSpecFromSubplotSpec(
            9, 1, subplot_spec=self.gs_main[1], hspace=0.5
        )
        self.setup_controls()

    def setup_controls(self):
        """This will set all the objects you need into their respective axes
        """        
        self.btn_pause = Button(self.fig.add_subplot(self.gs_side[0]), 'Pause/Play')
        self.btn_freq = Button(self.fig.add_subplot(self.gs_side[1]), 'Freq: OFF')
        self.btn_reset = Button(self.fig.add_subplot(self.gs_side[2]), 'Reset Scale')
        self.btn_gif = Button(self.fig.add_subplot(self.gs_side[3]), 'Export GIF')

        ax_speed = self.fig.add_subplot(self.gs_side[4])
        self.txt_speed = TextBox(ax_speed, 'Speed: ', initial=str(self.step_size))
        ax_window = self.fig.add_subplot(self.gs_side[5])
        self.txt_window = TextBox(ax_window, 'Window: ', initial=str(self.window_size))
        
        self.btn_pause.on_clicked(self.toggle_pause)
        self.btn_freq.on_clicked(self.toggle_frequency)
        self.btn_reset.on_clicked(self.manual_rescale)
        self.btn_gif.on_clicked(self.export_gif)
        self.txt_speed.on_submit(self.update_speed)
        self.txt_window.on_submit(self.update_window_size)

    def _init_axes_pool(self):
        """to initlizae the axis.  Plot empty figures for faster filling at animation time
        """        
        # --- Row 1: signal + Frequency ---
        if self.freq_mode == 0:
            self.ax_sig = self.fig.add_subplot(self.gs_plots[0])
            self.ax_freq = None
        else:
            gs_row = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=self.gs_plots[0], wspace=0.1, width_ratios=[0.8, 0.8])
            self.ax_sig = self.fig.add_subplot(gs_row[0])
            self.ax_freq = self.fig.add_subplot(gs_row[1])

        self.line_sig, = self.ax_sig.plot([], [], color='black', lw=1, label=f"lead {self.lead}")
        self.ax_sig.set_ylabel(f"{self.lead}")
        self.ax_sig.legend(loc="upper right")
        
        # --- Row 2: CAC ---
        self.ax_cac = self.fig.add_subplot(self.gs_plots[1], sharex=self.ax_sig)
        self.line_cac, = self.ax_cac.plot([], [], color='dodgerblue', lw=1.5, label="FLUSS CAC")
        self.ax_cac.fill_between([], [], color='dodgerblue', alpha=0.1)
        self.ax_cac.set_ylabel("Arc Curve (0-1)")
        self.ax_cac.set_ylim(0, 1.05)
        self.ax_cac.legend(loc="upper right")

        # --- Row 3: Phase Variance ---
        self.ax_phase = self.fig.add_subplot(self.gs_plots[2], sharex=self.ax_sig)
        self.line_phase, = self.ax_phase.plot([], [], color='purple', lw=1.5, label="Phase Instability")
        self.ax_phase.set_ylabel("Phase Var (rad²)")
        self.ax_phase.legend(loc="lower right")

        # --- Row 4: Navigator ---
        self.ax_nav = self.fig.add_subplot(self.gs_plots[3])
        # Downsample for nav
        ds = max(1, len(self.signal) // 5000)
        self.ax_nav.plot(np.arange(0, len(self.signal), ds), self.signal[::ds], color='gray', alpha=0.5)
        self.nav_cursor = self.ax_nav.axvline(0, color='red', lw=2)
        
        # Mark Regimes
        for loc in self.regime_locs:
            self.ax_nav.axvline(loc, color='blue', alpha=0.3, ymax=0.5)
            # Add markers to main plots if handled in update, but typically static lines need careful management in blit

        self.ax_nav.set_yticks([])
        self.ax_nav.set_xlabel("Timeline (Click to Jump) | Press SPACE to Pause")
        
        # Hide x labels for shared axes
        plt.setp(self.ax_sig.get_xticklabels(), visible=False)
        plt.setp(self.ax_cac.get_xticklabels(), visible=False)

    def rebuild_layout(self):
        self.ani.event_source.stop()
        
        self.ax_sig.remove()
        if self.ax_freq: 
            self.ax_freq.remove()
        self.ax_cac.remove()
        self.ax_phase.remove()
        self.ax_nav.remove()
        self._init_axes_pool()
        self.fig.canvas.draw_idle()
        self.ani.event_source.start()

    def toggle_frequency(self, event):
        self.freq_mode = (self.freq_mode + 1) % 3
        labels = {0: "Freq: OFF", 1: "Freq: STEM", 2: "Freq: SPEC"}
        self.btn_freq.label.set_text(labels[self.freq_mode])
        self.rebuild_layout()

    def update_frame(self, frame):
        if not self.paused:
            self.current_pos += self.step_size
            if self.current_pos + self.window_size > len(self.signal):
                self.current_pos = 0 # Loop

        # Data Slicing
        s = self.current_pos
        e = s + self.window_size
        x_data = np.arange(s, e)

        # 1. Update signal
        view_sig = self.signal[s:e]
        self.line_sig.set_data(x_data, view_sig)
        
        # Auto-scale signal y-axis roughly
        if len(view_sig) > 0:
            # self._apply_scale(ax=self.ax_sig, view_data=view_sig)
            mn, mx = np.min(view_sig), np.max(view_sig)
            self.ax_sig.set_xlim(s, e)
            self.ax_sig.set_ylim(mn - 0.2, mx + 0.2)

        # 2. Update CAC (Corrected Arc Curve)
        view_cac = self.cac[s : min(e, len(self.cac))]
        # Pad if short
        if len(view_cac) < (e-s):
            view_cac = np.pad(view_cac, (0, (e-s)-len(view_cac)), constant_values=1.0)
            
        self.line_cac.set_data(x_data, view_cac)
        # Iterate and remove instead of clearing the ArtistList directly
        for c in list(self.ax_cac.collections):
            c.remove()
            
        self.ax_cac.fill_between(x_data, view_cac, color='dodgerblue', alpha=0.1)
        
        # 3. Update Phase Variance
        view_phase = self.phase_var_stream[s:e]
        self.line_phase.set_data(x_data, view_phase)
        if len(view_phase) > 0:
             self.ax_phase.set_ylim(0, max(np.max(view_phase)*1.1, 0.1))

        # 4. Regime Lines (Vertical Markers)
        # Clear previous vertical lines
        for line in self.ax_sig.lines[1:]: 
            line.remove() # Keep index 0 (signal)
        for line in self.ax_cac.lines[1:]:
            line.remove()
        
        local_regimes = [r for r in self.regime_locs if s <= r < e]
        for r in local_regimes:
            self.ax_sig.axvline(r, color='red', linestyle='--', alpha=0.8)
            self.ax_cac.axvline(r, color='red', linestyle='--', alpha=0.8)

        # 5. Frequency Plot
        if self.freq_mode > 0 and self.ax_freq:
            self.ax_freq.cla()
            # STEM
            if self.freq_mode == 1: 
                yf = np.abs(rfft(view_sig))                 #fft sample
                xf = rfftfreq(len(view_sig), 1 / self.fs)   #frequency list
                half_point = int(len(view_sig)/2)           #Find nyquist freq
                freqs = yf[:half_point]
                freq_l = xf[:half_point]
                self.ax_freq.plot(freq_l, freqs, color='purple', lw=1, label=f"FFT_{self.lead}")
                self.ax_freq.fill_between(freq_l, freqs, color='purple', alpha=0.3)
                self.ax_freq.set_xlim(0, 50)                # Zoom on relevant signal bands
                self.ax_freq.set_title(f"FFT {self.lead}")
            # SPECGRAM  
            elif self.freq_mode == 2: 
                try:
                    self.ax_freq.specgram(view_sig, NFFT=128, Fs=self.fs, noverlap=64, cmap='inferno')
                    self.ax_freq.set_yticks([])
                except Exception as e:
                    logger.error(f"{e}")

        # 6. Nav Cursor
        self.nav_cursor.set_xdata([s])
        
        return []

    def on_click_jump(self, event):
        if event.inaxes == self.ax_nav:
            self.current_pos = int(event.xdata)
            self.current_pos = max(0, min(self.current_pos, len(self.signal) - self.window_size))
            if self.paused:
                self.update_frame(0)
                self.fig.canvas.draw_idle()

    def toggle_pause(self, event=None):
        self.paused = not self.paused

    def _apply_scale(self, ax, view_data):
        if view_data.size > 1:
            v_min, v_max = np.min(view_data), np.max(view_data)
            pad = (v_max - v_min) * 0.1 if v_max != v_min else 0.1
            ax.set_ylim(v_min - pad, v_max + pad)

    def manual_rescale(self, event):
        s = self.current_pos
        e = s + self.window_size
        view = self.signal[s:e]
        if len(view) > 0:
            self._apply_scale(ax=self.ax_sig, view_data=view)
            self.fig.canvas.draw_idle()

    def update_speed(self, text):
        try: 
            self.step_size = int(text)
        except ValueError as v: 
            logger.error(f"{v}")

    def update_window_size(self, text):
        try: 
            self.window_size = int(text)
        except ValueError as v: 
            logger.error(f"{v}")

    def on_key_press(self, event):
        if event.key == ' ': 
            self.toggle_pause()

    def _on_close(self, event):
        self.ani.event_source.stop()

    def export_gif(self, event):
        was_paused = self.paused
        self.paused = True
        f_path = f"export_pos{self.current_pos}.gif"
        logger.info(f"Exporting GIF to {f_path}...")
        writer = PillowWriter(fps=15)
        with writer.saving(self.fig, f_path, dpi=80):
            for _ in range(60):
                self.current_pos += self.step_size
                self.update_frame(0)
                self.fig.canvas.draw()
                writer.grab_frame()
        logger.info("Gif saved :tada:")
        self.paused = was_paused

class PigRAD:
    def __init__(self, npz_path):
        # 1. load data / params
        self.npz_path     :Path = npz_path
        self.fp_save      :Path = Path(npz_path).parent / (Path(npz_path).stem + "_regimes.json")  # For saving regime indices
        self.fp_dos       :Path = Path(npz_path).parent / (Path(npz_path).stem + "_cac.npz")       # For saving Corrected Arc Curve
        self.loader       :SignalDataLoader = SignalDataLoader(str(self.npz_path))
        self.full_data    :dict = self.loader.full_data
        self.channels     :list = self.loader.channels
        self.fs           :float = 1000.0                   #Hz
        self.windowsize   :int = 20                         #size of section window 
        self.lead         :str = self.pick_lead()
        self.sections     :np.array = segment_ECG(self.full_data[self.lead], self.fs, self.windowsize)
        self.sections     :np.array = np.concatenate((self.sections, np.zeros((self.sections.shape[0], 2), dtype=int)), axis=1)
        self.gpu_devices  :list = [device.id for device in cuda.list_devices()]
        self.results      :list = []
        self.view_gui     :bool = True
        self.multi_stump  :bool = False

    def pick_lead(self) -> str:
        """Picks the lead you'd like to analyze

        Raises:
            ValueError: Gotta pick an integer

        Returns:
            lead (str): the lead you picked!
        """

        tree = Tree(f":select channel:", guide_style="bold bright_blue")
        for idx, channel in enumerate(self.channels):
            tree.add(Text(f'{idx}:', 'blue') + Text(f'{channel} ', 'red'))
        print(tree)
        question = "What channel would you like to load?\n"
        file_choice = console.input(f"{question}")
        if file_choice.isnumeric():
            lead_to_load = self.channels[int(file_choice)]
            print(f"lead {lead_to_load} loaded")
            return lead_to_load
        else:
            raise ValueError("Invalid selection")
    
    def bandpass_filter(data, lowcut=0.1, highcut=40.0, fs=1000.0, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    @log_time
    def detect_regime_changes(self, m_override: int = None, n_regimes: int = 5) -> None:
        """Uses STUMPY FLUSS to find semantic boundaries (regime changes).
        Launches interactive RegimeViewer.

        Args:
            m_override (int, optional): window search overide if you'd like to try something else. Defaults to None.
            n_regimes (int, optional): number of regime changes you hope to find.  Being that there are 4 stages of hem shock, we'll shoot for 5 regimes. Defaults to 5.
        """        
        logger.info("Running Semantic Segmentation (FLUSS)...")
        data = self.full_data[self.lead].astype(np.float64)
        
        if m_override:
            m = m_override
        else:
            # Default to ~400ms (one beat)
            m = int(self.fs * 0.4) 
        
        logger.info(f"Using window size m={m}...")

        try:
            if self.multi_stump:
                pass
                #TODO - Multi stump. 
                    #Issue - FLUSS isn't really able to identify major morpohological shifts.  Increase and decrease in heart rate tend to screw things up
                    #Euclidean distance isn't .... quite cutting it.  
                    #Solution
                        #1. Multi Stump
                            # For finding hem shock stages.
                            # Use the other input streams to determine shape change
            
                        #2. Self- Join comparison
                            #Self join of the initial x minutes.  Then compare sections thereafter to that initial loop with stumpy

                        #3. 

            else:
                # Calculate MP and MPI
                if self.gpu_devices:
                    logger.info("using GPU for MP")
                    mp = stumpy.gpu_stump(data, m=m, device_id=self.gpu_devices)
                    mpi = mp[:, 1]
                else:
                    logger.info("using CPU for MP")
                    mp = stumpy.stump(data, m=m)
                    mpi = mp[:, 1]
                    
                # Calculate FLUSS
                logger.info("Calculating FLUSS (Arc Curve)...")
                cac, regime_locs = stumpy.fluss(mpi, L=m, n_regimes=n_regimes, excl_factor=5)
                
                # Normalize CAC length to match data for plotting
                pad_width = len(data) - len(cac)
                if pad_width > 0:
                    cac = np.pad(cac, (0, pad_width), 'constant', constant_values=1.0)
            
                # Save Logic
                self.regime_results = {
                    "m": m,
                    "regime_indices": regime_locs,
                }
                self.results = cac
                self.save_regime_results()
                self.save_cac_results()

            # Launch Interactive Navigator
            # NOTE: Phase Variance calc happens inside RegimeViewer init
            if self.view_gui:
                RegimeViewer(data, cac, regime_locs, m, self.fs, self.lead)
            
            return regime_locs

        except Exception as e:
            logger.error(f"Failed during FLUSS segmentation: {e}")
            import traceback
            traceback.print_exc()
            return []

    def save_regime_results(self):
        """Saves the locations of the regime changes to json
        """
        if not hasattr(self, 'regime_results'): 
            return
        out_name = self.fp_save
        out_path = self.npz_path.parent / out_name
        output_data = {
            "source_file": str(self.npz_path.name),
            "lead": self.lead,
            "m": int(self.regime_results['m']),
            "regime_indices": self.regime_results['regime_indices'].tolist(),
        }
        try:
            with open(out_path, 'w') as f:
                json.dump(output_data, f, indent=2, cls=NumpyArrayEncoder)
                # Log the size
                mb_size = os.path.getsize(out_path) / (1024 * 1024)
                logger.warning(f"Saved {out_path.name} ({mb_size:.2f} MB)")

        except Exception as e:
            logger.error(f"Failed to save regime JSON: {e}")
    
    def save_cac_results(self):
        """Saves the Corrected Arc Curve Results
        """
        if not hasattr(self, 'regime_results'): 
            return
        out_name = self.fp_dos
        out_path = self.npz_path.parent / out_name
        output_path = Path(out_path).with_suffix('.npz')
        np.savez_compressed(output_path, self.results)
        
        # Log the size
        mb_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.warning(f"Saved {output_path.name} ({mb_size:.2f} MB)")

    def run_pipeline(self, n_regimes=5):
        """Checks for existing save files. If found, loads them to save computation time.
        If not found, runs the full STUMPY detection suite.
        """
        # 1. Check if files exist
        if self.fp_save.exists() and self.fp_dos.exists():
            console.print(f"[green]Found saved files for {self.lead}. Loading...[/]")
            
            # Load Regime JSON
            with open(self.fp_save, 'r') as f:
                meta = json.load(f)
                m = meta['m']
                regime_locs = np.array(meta['regime_indices'])
            
            # Load CAC NPZ
            # np.savez_compressed saves unnamed args as arr_0
            container = np.load(self.fp_dos)
            cac = container['arr_0'] 
            
            console.print("[bold green]Data loaded. Launching Viewer...[/]")
            
            if self.view_gui:
                # Launch Viewer
                RegimeViewer(
                    signal_data=self.full_data[self.lead].astype(np.float64),
                    cac_data=cac,
                    regime_locs=regime_locs,
                    m=m,
                    sampling_rate=self.fs,
                    lead=self.lead
                )
        else:
            console.print("[yellow]No saved data found. Running STUMPY algorithms...[/]")
            self.detect_regime_changes(m_override=410, n_regimes=n_regimes)

# --- Entry Point ---
def load_choices(fp:str):
    """Loads whatever file you pick

    Args:
        fp (str): file path

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """    
    try:
        tree = Tree(f":open_file_folder: [link file://{fp}]{fp}", guide_style="bold bright_blue")
        walk_directory(Path(fp), tree)
        print(tree)
    except Exception as e:
        logger.warning(f"{e}")        

    question = "What file would you like to load?\n"
    file_choice = console.input(f"{question}")
    if file_choice.isnumeric():
        files = sorted(f for f in Path(str(fp)).iterdir() if f.is_file())
        return files[int(file_choice)]
    else:
        raise ValueError("Invalid choice")

def main():
    fp = Path.cwd() / "src/rad_ecg/data/datasets/JT"
    selected = load_choices(fp)
    rad = PigRAD(selected)
    rad.run_pipeline(n_regimes=5)

if __name__ == "__main__":
    main()

#Problem statement.  
# We're looking to classify the 4 stages of hemorhagic shock. 
# We'll look for 5 regime changes and hope for the best!  

#Workflow
#File choice and signal loading
#Pigrad initialization kicks off stumpy matrix profile calculation
#Runs FLUSS algorithm to look for semantic shifts in the morphology of the signal
#Loads stumpy CAC () curve results to RegimeViewer matplotlib GUI
#During that load, we calculate the phase variance over time with each beat
#To do that we need to isolate the beats and then set a standard window before and after
#Align all of them and then look for the variance in the aligned beat to run the wavelet over.
#Gives the phase variance a kind of a step curve.

#Good results
#sep-3-24

#NOTES 2-10-26
#FLUSS not performing as well as I'd like. 
#Possible problems. 
    #an m that changes throughout the signal because the signal I want to look at is too long.  
        # Could section the ecg according to the phase labels in BSOS data.  
    
    #It also could be that the  periodicity of the carotid flow is so strong it completely obliterates
    #any smaller signal change.  Which we did see in the freqwuency spectrum as the power for that signal
    #was really large.  

    #Additionally, euclidean distance's might break down in this instance because the change isn't immediate.  
    #It's gradual over time which FLUSS won't be able to see.  s

    #Mortlet Wavelet might not be suitable (meant for ecg's not flow traces)
    #debauchies 4/ symlet 5 and gaussian may be more appropriate



#IDEA 
#What about shooting for a change point detection algorithm.  BOCPD (Bayesian optimized change point detection) might work here.  
#Could isolate the dicrotic notch of the carotid.  The R peak of the ecg.  then create a time based feature of the difference between them. representing depolarization speeds
#Could also use the slopes off the carotid as an indicator of pressure. 
#Proposed outline
#1. Downsample if necessary (not in this case)
#2. apply a zero-phase butterworth bandpass filter (0.5 - 30 Hz) on the carotid and LAD
    #traces in order to remove wander and artifacts. 
#3. dicrotic notch index (DNI)
    # Find the r peak. Use scipy find_peaks 
    # find the systolic peak(SBP) and diastolic trough (DBP)
    # use the second deriv to get the local maxima (aka the dichrotic notch)
    # dni =  (Pnotch - DBP) / (SBP - DBP)
    # Supposedly falls off quickly from hem stages 2 and up.
#4. Pulse wave reflection ratios
    #p1 - percussion wave - initial upstroke by lv ejection
    #p2 - tidal wave - reflection from the upper body and renal
    #p3 - dicrotic wave - reflection from the lower body. 
#5. Systolic + Diastolic Slopes 
    #max slope - max value of the first derivative during the upstroke. 
        #gets greater in class 1.  decreases in following
    #Decay time constant
        #fir an exponential decay func p(t) = P0e^-t/T to the diastolic portion - notch to end diastole
#6. Use AUC for calculating MAP
#7  Calcualte shannon energy maybe?
#8. Calculate diastolic retrograde fraction
    # Don't really understand this one, so will need to come back. 

