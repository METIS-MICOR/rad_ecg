import gc
import json
import utils
import stumpy
import numpy as np
from numba import cuda
from pathlib import Path
from utils import segment_ECG
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, TextBox
from matplotlib.animation import FuncAnimation
from setup_globals import walk_directory
from scipy.stats import wasserstein_distance
from scipy.signal import find_peaks, stft, welch
from support import logger, console, log_time, NumpyArrayEncoder
import warnings
from rich import print
from rich.tree import Tree
from rich.text import Text
from rich.filesize import decimal
from rich.markup import escape
from rich.progress import (
    Progress,
    BarColumn,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn
)

#Ignore numba error
from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

class SignalDataLoader:
    """Handles loading and structuring the NPZ data."""
    def __init__(self, npz_path):
        self.container = np.load(npz_path)
        self.files = self.container.files
        self.channels = self._identify_and_sort_channels()
        self.full_data = self._stitch_blocks()
        
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
                else:
                    full_data[ch] = np.array([])
        return full_data

class RegimeViewer:
    """
    Interactive viewer for FLUSS Regime Segmentation.
    Top: ECG Signal
    Middle: Corrected Arc Curve (CAC)
    Bottom: Global Navigation
    """
    def __init__(self, ecg_data, cac_data, regime_locs, m, sampling_rate=1000):
        # 1. Data Setup
        self.ecg = ecg_data
        self.cac = cac_data
        self.regime_locs = regime_locs
        self.m = m
        self.fs = sampling_rate
        
        # 2. State Settings
        self.window_size = 2000  # Samples to show at once
        self.current_pos = 0
        self.step_size = 20      # Animation speed
        self.paused = False
        
        # 3. Setup Figure
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click_jump)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        self.setup_layout()
        self._init_plots()
        
        # 4. Start Animation
        self.ani = FuncAnimation(
            self.fig, self.update_frame, interval=30, blit=True, cache_frame_data=False
        )
        plt.show()

    def setup_layout(self):
        self.gs_main = gridspec.GridSpec(1, 2, width_ratios=[10, 1], figure=self.fig)
        
        # Plot Area: 3 Rows (ECG, CAC, Nav)
        self.gs_plots = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=self.gs_main[0], height_ratios=[2, 2, 0.5], hspace=0.1
        )
        
        # Side Controls
        self.gs_side = gridspec.GridSpecFromSubplotSpec(
            6, 1, subplot_spec=self.gs_main[1], hspace=0.3
        )

        # Create Axes
        self.ax_ecg = self.fig.add_subplot(self.gs_plots[0])
        self.ax_cac = self.fig.add_subplot(self.gs_plots[1], sharex=self.ax_ecg)
        self.ax_nav = self.fig.add_subplot(self.gs_plots[2])
        
        # Hide x-labels for top plots
        plt.setp(self.ax_ecg.get_xticklabels(), visible=False)
        plt.setp(self.ax_cac.get_xticklabels(), visible=False)

        # Setup Controls
        self.btn_pause = Button(self.fig.add_subplot(self.gs_side[0]), 'Pause/Play')
        self.btn_pause.on_clicked(self.toggle_pause)
        
        ax_speed = self.fig.add_subplot(self.gs_side[1])
        self.txt_speed = TextBox(ax_speed, 'Speed: ', initial=str(self.step_size))
        self.txt_speed.on_submit(self.update_speed)
        
        ax_window = self.fig.add_subplot(self.gs_side[2])
        self.txt_window = TextBox(ax_window, 'Window: ', initial=str(self.window_size))
        self.txt_window.on_submit(self.update_window_size)

    def _init_plots(self):
        # --- ECG Line ---
        self.line_ecg, = self.ax_ecg.plot([], [], color='black', lw=1)
        self.ax_ecg.set_ylabel("ECG (mV)")
        self.regime_lines_ecg = [] # Store vertical lines for regimes
        
        # --- CAC Line ---
        self.line_cac, = self.ax_cac.plot([], [], color='blue', lw=1.5)
        self.ax_cac.set_ylabel("Arc Curve (0-1)")
        self.ax_cac.set_ylim(0, 1.05)
        # We fill under the curve for visual emphasis
        self.poly_cac = self.ax_cac.fill_between([], [], color='blue', alpha=0.1)

        # --- Navigation Bar ---
        # Plot a downsampled version of the whole CAC for context
        ds = max(1, len(self.cac) // 5000)
        self.ax_nav.plot(np.arange(0, len(self.cac), ds), self.cac[::ds], color='gray', alpha=0.5)
        
        # Mark all regime changes on Nav
        for loc in self.regime_locs:
            self.ax_nav.axvline(loc, color='red', alpha=0.5, lw=1)
            
        self.nav_cursor = self.ax_nav.axvline(0, color='dodgerblue', lw=2)
        self.ax_nav.set_yticks([])
        self.ax_nav.set_xlim(0, len(self.cac))
        self.ax_nav.set_xlabel("Click to Jump | Space to Pause")

    def update_frame(self, frame):
        if not self.paused:
            self.current_pos += self.step_size
            if self.current_pos + self.window_size > len(self.ecg):
                self.current_pos = 0 # Loop

        # Data Slicing
        s = self.current_pos
        e = s + self.window_size
        ecg_view = self.ecg[s:e]
        
        # CAC might be shorter by m-1, handle bounds
        cac_len = len(self.cac)
        if s < cac_len:
            cac_view = self.cac[s : min(e, cac_len)]
            # If at the very end, pad for consistent array size
            if len(cac_view) < (e-s):
                pad = np.zeros((e-s) - len(cac_view))
                cac_view = np.concatenate((cac_view, pad))
        else:
            cac_view = np.zeros(self.window_size)

        x_data = np.arange(s, e)
        
        # Update Data
        self.line_ecg.set_data(x_data, ecg_view)
        self.line_cac.set_data(x_data, cac_view)
        
        # Update Fill (PolyCollection is tricky to animate efficiently, clearer to just redraw lines)
        # For 'blit=True', we must return artists. fill_between is hard to blit. 
        # We will skip animating the fill for performance or use a simple line.
        
        # Handle Dynamic Regime Markers (Vertical Lines)
        # Remove old lines
        for line in self.regime_lines_ecg:
            line.remove()
        self.regime_lines_ecg = []
        
        # Find regimes in current window
        local_regimes = [r for r in self.regime_locs if s <= r < e]
        
        artists = [self.line_ecg, self.line_cac, self.nav_cursor]
        
        for r in local_regimes:
            # Draw on ECG
            l1 = self.ax_ecg.axvline(r, color='red', linestyle='--', alpha=0.8)
            # Draw on CAC
            l2 = self.ax_cac.axvline(r, color='red', linestyle='--', alpha=0.8)
            self.regime_lines_ecg.extend([l1, l2])
            artists.extend([l1, l2])

        # Auto Scale ECG
        if len(ecg_view) > 0:
            mn, mx = np.min(ecg_view), np.max(ecg_view)
            self.ax_ecg.set_ylim(mn - 0.1, mx + 0.1)
            self.ax_ecg.set_xlim(s, e)
            self.ax_cac.set_xlim(s, e)

        # Update Nav Cursor
        self.nav_cursor.set_xdata([s])
        
        return artists

    def on_click_jump(self, event):
        if event.inaxes == self.ax_nav:
            self.current_pos = int(event.xdata)
            self.current_pos = max(0, min(self.current_pos, len(self.ecg) - self.window_size))
            if self.paused:
                self.update_frame(0) # Force update if paused
                self.fig.canvas.draw_idle()

    def toggle_pause(self, event=None):
        self.paused = not self.paused

    def update_speed(self, text):
        try: 
            self.step_size = int(text)
        except ValueError: 
            pass

    def update_window_size(self, text):
        try: 
            self.window_size = int(text)
        except ValueError: 
            pass
            
    def on_key_press(self, event):
        if event.key == ' ':
            self.toggle_pause()

    def _on_close(self, event):
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()


class PigRAD:
    def __init__(self, npz_path):
        # 1. load data / params
        self.npz_path   :Path = npz_path
        self.loader     :SignalDataLoader = SignalDataLoader(str(self.npz_path))
        self.full_data  :dict = self.loader.full_data
        self.channels   :list = self.loader.channels
        self.fs         :float = 1000.00 #Hz
        self.windowsize :int = 30  #size of section window 
        self.lead       :str = self.pick_lead()
        self.sections   :np.array = segment_ECG(self.full_data[self.lead], self.fs, self.windowsize)
        self.sections   :np.array = np.concatenate((self.sections, np.zeros((self.sections.shape[0], 2), dtype=int)), axis=1) #Add HR, RMSSD cols
        self.gpu_devices:list = [device.id for device in cuda.list_devices()]     #available cuda devices
        self.results    :list = []     #Stumpy results container
        self.shifts     :list = []     #Track distribution shifts
        self.plot_shifts:bool = True   
        self.make_plots :bool = True
        self.regime_shifts:list = []

    def pick_lead(self):
        tree = Tree(
            f":select channel:",
            guide_style="bold bright_blue",
        )
        for idx, channel in enumerate(self.channels):
            tree.add(Text(f'{idx}:', 'blue') + Text(f'{channel} ', 'red'))
        print(tree)
        question = "What channel would you like to load?\n"
        file_choice = console.input(f"{question}")
        if file_choice.isnumeric():
            lead_to_load = self.channels[int(file_choice)]
            #check output directory exists
            print(f"lead {lead_to_load} loaded")
            return lead_to_load
        
        else:
            raise ValueError("Please restart and select an integer of the file you'd like to import")
    
    @log_time
    def detect_regime_changes(self, m_override: int = None, n_regimes: int = 2):
        """
        Uses STUMPY FLUSS to find semantic boundaries (regime changes).
        Launches interactive RegimeViewer.
        """
        logger.info("Running Semantic Segmentation (FLUSS)...")
        data = self.full_data[self.lead][5700000:].astype(np.float64)
        
        # Determine 'm' (Subsequence Length)
        # If not provided, we estimate it. For ECG morphology changes, 
        # m should cover a full heartbeat (P-QRS-T). ~400ms is a safe standard for pigs/humans.
        if m_override:
            m = m_override
        else:
            m = int(self.fs * 0.4) 
        
        logger.info(f"Using window size m={m}...")

        try:
            # Calculate MP and MPI
            if self.gpu_devices:
                logger.info("using GPU")
                mp = stumpy.gpu_stump(data, m=m, device_id=self.gpu_devices)
                mpi = mp[:, 1]
            else:
                logger.info("using CPU")
                mp = stumpy.stump(data, m=m)
                mpi = mp[:, 1]
                
            # Calculate FLUSS
            cac, regime_locs = stumpy.fluss(mpi, L=m, n_regimes=n_regimes, excl_factor=5)
            
            # Pad CAC to match data length (FLUSS returns len(data) - m + 1)
            # We pad the end with 1.0 (max arc) so arrays align in plotter
            pad_width = len(data) - len(cac)
            if pad_width > 0:
                cac = np.pad(cac, (0, pad_width), 'constant', constant_values=1.0)
            
            # Store results for saving
            self.regime_results = {
                "m": m,
                "regime_indices": regime_locs,
                "cac": cac
            }
            self.save_regime_results()

            # Launch Interactive Navigator 
            RegimeViewer(data, cac, regime_locs, m, self.fs)
            
            return regime_locs

        except Exception as e:
            logger.error(f"Failed during FLUSS segmentation: {e}")
            return []

    def save_regime_results(self):
        """
        Saves the detected regimes to JSON.
        Format:
        {
            "file": "filename",
            "m": 400,
            "regime_indices": [10500, 23000, ...]
            "cac": [] Corrected Arc Curve
        }
        """
        if not hasattr(self, 'regime_results'):
            logger.warning("No regime results to save. Run detect_regime_changes first.")
            return

        out_name = self.npz_path.stem + "_regimes.json"
        out_path = self.npz_path.parent / out_name
        
        # Prepare dictionary for JSON
        output_data = {
            "source_file": str(self.npz_path.name),
            "lead": self.lead,
            "m": int(self.regime_results['m']),
            "regime_indices": self.regime_results['regime_indices'].tolist(),
            "cac": self.regime_results['cac'].tolist()
        }
        
        try:
            with open(out_path, 'w') as f:
                json.dump(output_data, f, indent=2, cls=NumpyArrayEncoder)
            logger.info("regime data saved")

        except Exception as e:
            logger.error(f"Failed to save regime JSON: {e}")

    def _plot_regimes(self, data, cac, regime_locs, m):
        """
        Helper to visualize the Arc Curve aligned with the ECG signal.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})
        
        # Top: ECG Signal
        ax1.plot(data, color='black', linewidth=1, alpha=0.8)
        
        # Highlight the detected regime change boundaries
        for loc in regime_locs:
            ax1.axvline(x=loc, color='red', linestyle='--', linewidth=2, label='Detected Change')
            ax2.axvline(x=loc, color='red', linestyle='--', linewidth=2)

        ax1.set_ylabel("Amplitude (mV)")
        ax1.set_title(f"ECG Signal with Detected Regime Changes (m={m})")
        ax1.legend(loc='upper right')

        # Bottom: Corrected Arc Curve (CAC)
        # Ideally, we pad the CAC to match data length for alignment (it is shorter by m-1)
        # STUMPY CAC is usually len(data) - m + 1
        x_axis = np.arange(len(cac))
        
        ax2.plot(x_axis, cac, color='blue', linewidth=1.5, label='Arc Curve (CAC)')
        ax2.set_ylabel("Arc Curve Value (0-1)")
        ax2.set_xlabel("Time Index")
        ax2.set_title("Semantic Segmentation: Low values indicate morphology change")
        ax2.fill_between(x_axis, 0, cac, color='blue', alpha=0.1)
        ax2.grid(True, alpha=0.3)

        # Mark the regime locations on the curve
        ax2.scatter(regime_locs, cac[regime_locs], color='red', zorder=5)

        plt.tight_layout()
        plt.show()

    def run_search(self):
        """
        Iterates through signal sections, checks for distribution shifts,
        calculates dynamic matrix profiles using GPU-STUMP, and identifies discords.
        """
        # Threshold for Wasserstein distance to consider distributions "similar"
        WD_THRESHOLD = 0.001
        CPU_CUTOFF = 50000
        previous_dist = None

        if self.gpu_devices:
            gpu_indicator = "[bold green]GPU[/]"
        else:
            gpu_indicator = "[bold red]GPU[/]"

        prog = Progress(
            SpinnerColumn(),
            TimeElapsedColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("{task.fields[gpu]}"), 
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        )
        with prog as progress:
            task = progress.add_task("[cyan]Processing Sections...", total=len(self.sections), gpu=gpu_indicator)
            for i, section in enumerate(self.sections):
                start = section[0]
                end = section[1]
                sig_section = self.full_data[self.lead][start:end].flatten().astype(np.float64)

                # 1. Calculate Distribution (STFT -> Magnitude spectrum)
                current_dist = self.STFT(sig_section)
                
                #IDEA - Vector comparison. 
                    #Would a vector comparison of the current distances give us any extra 
                    #ability here to pull out morphology changes?

                # 2. Check Distribution Shift (Skip first section)
                if i > 0 and previous_dist is not None:
                    #Calc earth movers distance between distributions
                    wd = wasserstein_distance(previous_dist, current_dist)
                    if wd > WD_THRESHOLD:
                        # Distribution shift detected - skip extraction or flag
                        logger.warning(f"Section {i}: Major distribution shift detected (WD: {wd:.4f}). Skipping extraction.")
                        self.shifts.append({
                            'section_idx': i,
                            'wd': wd,
                            'prev_dist': previous_dist.copy(),
                            'curr_dist': current_dist.copy()}
                        )
                        previous_dist = current_dist
                        #Mark section invalid
                        self.sections[i, 2] = 0
                        if self.make_plots:
                            self.plot_rpeaks(i = i - 1, pause_duration=2)
                            self.plot_rpeaks(i = i, pause_duration=2)
                        if self.plot_shifts:
                            self.plot_distribution_shifts(pause_duration=2)
                        progress.update(task, advance=1, gpu=gpu_indicator)
                        continue

                # Update previous distribution for next iteration
                previous_dist = current_dist
                
                # 3. Extraction: Find R-Peaks to determine 'm'
                # Distance is ~200ms in samples (0.2 * 1000) to avoid T-wave detection
                peaks, peak_info = find_peaks(
                    sig_section, 
                    height = np.percentile(sig_section, 90),     #90 -> stock
                    prominence = np.percentile(sig_section, 95), #95 -> stock
                    distance = int(self.fs * 0.3)                #300 bpm limit for porcine
                )
                
                if len(peaks) < 4:
                    logger.warning(f"Section {i}: Not enough peaks to calculate motif length 'm'. Skipping.")
                    if self.make_plots:
                        self.plot_rpeaks(i, peaks, peak_info, with_scipy=True)
                    self.sections[i, 2] = 0
                    progress.update(task, advance=1, gpu=gpu_indicator)
                    continue

                # Calculate average R-to-R interval
                r_r_intervals = np.diff(peaks)
                avg_rr = np.mean(r_r_intervals)
                m = int(avg_rr) # Window size for Stumpy

                # Safety check for m.  make sure its within the bounds
                if m < 3 or m >= len(sig_section):
                    logger.warning(f"m {m} is out of the window 3 or {len(sig_section)}")
                    progress.update(task, advance=1, gpu=gpu_indicator)
                    continue

                # Store heart rate
                RR_diffs = np.diff(peaks)
                RR_diffs_time = np.abs(np.diff((RR_diffs / self.fs) * 1000)) #Formats to time domain in milliseconds
                HR = np.round((60 / (RR_diffs / self.fs)), 2) #Formatted for BPM
                Avg_HR = int(np.mean(HR))
                RMSSD = np.round(np.sqrt(np.mean(np.power(RR_diffs_time, 2))), 5)
                self.sections[i, 2] = Avg_HR
                self.sections[i, 3] = RMSSD

                # 4. Matrix Profile via GPU Stump.  stumpy.gpu_stump returns the Matrix Profile (MP) and Matrix Profile Index (MPI)
                try:
                    use_gpu = self.gpu_devices and len(sig_section) > CPU_CUTOFF
                    if use_gpu:
                        gpu_indicator = "[bold green]GPU[/]"
                        mp = stumpy.gpu_stump(sig_section, m=m, device_id=self.gpu_devices)
                    else:
                        gpu_indicator = "[bold red]GPU[/]"
                        mp = stumpy.stump(sig_section, m=m)

                    # 5. Identify Major Discord (Anomaly)
                    # The discord is the subsequence with the largest Nearest Neighbor Distance (max value in MP)
                    discord_idx = np.argsort(mp[:, 0])[-1]
                    discord_dist = mp[discord_idx, 0]
                    
                    # Store result
                    self.results.append({
                        'section_idx': i,
                        'hr':int(Avg_HR),
                        'rmssd':round(RMSSD, 5),
                        'm': m,
                        'discord_index': discord_idx,
                        'discord_score': discord_dist,
                        'wasserstein_metric': wd if i > 0 else 0.0
                    })
                    
                except Exception as e:
                    logger.error(f"GPU Stump failed on section {i}: {e}")

                if i % 100 == 0:
                    #Every 100 sections pop in a print
                    if self.make_plots:
                        self.plot_rpeaks(i, peaks, peak_info, with_scipy=True)
                    logger.info(f"section {i} Current HR: {Avg_HR:.0f}")

                #Memory cleanup
                del sig_section
                if 'mp' in locals(): 
                    del mp
                #Take out the garbage every 50 loops
                if i % 50 == 0:
                    plt.close('all')
                    gc.collect()
                    logger.debug("garbage collected")
                # Mark section as valid and advance progbar
                self.sections[i, 2] = 1
                progress.update(task, advance=1, gpu=gpu_indicator)
        
        # Summary Output
        if self.results:
            console.print(f"[bold green]Processing Complete.[/] Analyzed {len(self.results)} valid sections.")
            # Simple list of top 3 discords found across all sections
            sorted_discords = sorted(self.results, key=lambda x: x['discord_score'], reverse=True)
            console.print("[bold]Top 10 Global Discords found:[/]")
            for d in sorted_discords[:10]:
                console.print(f"Section {d['section_idx']} | Score: {d['discord_score']:.2f} | m: {d['m']}")
        else:
            console.print("[bold red]No valid sections processed.[/]")

    def STFT(self, signal_section):
        """
        Runs an STFT over the section and returns a normalized probability distribution 
        (Power Spectral Density) for comparison via Wasserstein distance.
        """
        # Compute STFT
        f, t, Zxx = stft(signal_section, fs=self.fs, nperseg=256)
        
        # Calculate Magnitude Spectrum
        magnitude = np.abs(Zxx)
        
        # Collapse time axis to get a frequency distribution (PSD-like) for the whole section as a 1d vector
        freq_dist = np.sum(magnitude, axis=1)
        
        # Normalize to sum to 1 to treat as a probability distribution for Wasserstein
        if np.sum(freq_dist) > 0:
            freq_dist = freq_dist / np.sum(freq_dist)
        
        return freq_dist

    def plot_discords(self, top_n:int=5, pause_duration:int=10):
        """
        Iterates through the top N discords and displays them in a Matplotlib window
        with the discord highlighted by a gray patch.
        """
        if not self.results:
            logger.warning("No results available to plot.")
            return

        # Sort results to get the top discords
        sorted_discords = sorted(self.results, key=lambda x: x['discord_score'], reverse=True)
        console.print(f"[bold yellow]Starting playback of top {len(sorted_discords)} discords...[/]")
        plt.ion() # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(14, 6))
        for i, res in enumerate(sorted_discords):
            try:
                # 1. Retrieve the data for this section
                sec_idx = res['section_idx']
                start_idx = self.sections[sec_idx][0]
                end_idx = self.sections[sec_idx][1]
                data = self.full_data[self.lead][start_idx:end_idx]
                
                # 2. Clear and Plot Signal
                ax.clear()
                ax.plot(range(start_idx, end_idx), data, color='black', linewidth=1, label='ECG Signal')
                ax.set_xlim(start_idx, end_idx)
                
                # 3. Highlight the Discord
                discord_start = res['discord_index'] + start_idx
                m = res['m']
                # Create a gray rectangle patch
                # Height is based on min/max of the data to cover the vertical area
                y_min, y_max = np.min(data), np.max(data)
                height = y_max - y_min
                rect = patches.Rectangle(
                    (discord_start, y_min), 
                    m, 
                    height, 
                    linewidth=1, 
                    edgecolor='red', 
                    facecolor='gray', 
                    alpha=0.5, 
                    label='Discord Motif'
                )
                ax.add_patch(rect)
                # 4. Decoration
                ax.set_title(f"Discord Rank #{i+1} | Section {sec_idx} | Score: {res['discord_score']:.2f} | M: {m}")
                ax.set_xlabel("Time index")
                ax.set_ylabel("Amplitude (mV)")
                ax.legend(loc='upper right')
                ax.set_xticks(ax.get_xticks(), labels = utils.label_formatter(ax.get_xticks()) , rotation=-30)
                # 5. Render and Pause
                plt.draw()
                console.print(f"Displaying Discord #{i+1} (Section {sec_idx})...")
                plt.pause(pause_duration)

            except Exception as e:
                logger.error(f"Error plotting discord {i}: {e}")

        plt.ioff() # Turn off interactive mode
        plt.close(fig)
        console.print("[bold green]Playback complete.[/]")

    def plot_distribution_shifts(self, pause_duration=3):
        """
        Replays the distribution shifts detected during processing.
        Plots the Previous (Baseline) vs Current (Shifted) Frequency Distribution.
        """
        shifts = self.shifts[-2:]
        if not shifts:
            return
        
        logger.info("Showing Distribution Shifts")

        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        for event in shifts:
            try:
                ax.clear()
                # Plot Previous Distribution (Baseline for this comparison)
                ax.plot(event['prev_dist'], label='Previous Section', color='blue', alpha=0.6, linewidth=2)
                # Plot Current Distribution (The Shift)
                ax.plot(event['curr_dist'], label='Current Section (Shifted)', color='red', alpha=0.8, linestyle='--')
                # Fill intersection/difference for visual effect
                ax.fill_between(range(len(event['prev_dist'])), event['prev_dist'], event['curr_dist'], color='gray', alpha=0.2)
                ax.set_title(f"Distribution Shift Detected @ Section {event['section_idx']} | Wasserstein Dist: {event['wd']:.4f}")
                ax.set_xlabel("Frequency Bins")
                ax.set_ylabel("Normalized Power/Probability")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.draw()
                plt.pause(pause_duration)

            except Exception as e:
                logger.error(f"Error plotting shift: {e}")
            
        plt.ioff()
        plt.close(fig)

    def plot_rpeaks(self, i:int, peaks:np.array=None, peak_info:dict=None, with_scipy:bool=False, pause_duration:int=3):
        """
        Plots the scipy.find_peaks R peak search
        """

        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            #Plot the main wave
            start = self.sections[i, 0]
            end = self.sections[i, 1]
            ax.plot(range(start, end), self.full_data[self.lead][start:end], label='Waveform', color='blue', alpha=0.6, linewidth=2)
            if with_scipy:
                ax.scatter(peaks + start, peak_info['peak_heights'], marker='D', color='red', label='R peaks')
                ax.set_title(f"Section {i} with R peaks")
            else:
                ax.set_title(f"Section {i}")
            ax.set_xlabel("Time index")
            ax.set_ylabel("Amplitude (mV)")
            ax.set_xticks(ax.get_xticks(), labels = utils.label_formatter(ax.get_xticks()) , rotation=-30)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.draw()
            plt.pause(pause_duration)

        except Exception as e:
            logger.error(f"Error plotting R peaks: {e}")

        plt.close()

    def load_results(self, json_path):
        """
        Loads analysis results from a JSON file.
        """
        try:
            with open(json_path, 'r') as f:
                self.results = json.load(f)
            console.print(f"[bold green]Successfully loaded {len(self.results)} entries from results file.[/]")
            return True
        except Exception as e:
            logger.error(f"Failed to load results from {json_path}: {e}")
            return False
        
    def save_results(self):
        """
        Saves the analysis results to a JSON file in the same directory as the source file.
        """
        if not self.results:
            logger.warning("No results to save.")
            return

        # Generate output filename: original_name + _results.json
        out_name = self.npz_path.stem + "_results.json"
        out_path = self.npz_path.parent / out_name
        
        try:
            with open(out_path, 'w') as f:
                json.dump(self.results, f, cls=NumpyArrayEncoder, indent=2)
            console.print(f"[bold green]Results successfully saved to:[/]\n[link file://{out_path}]{out_path}[/link]")
        except Exception as e:
            logger.error(f"Failed to save results to JSON: {e}")

def load_choices(fp:str):
    try:
        tree = Tree(
            f":open_file_folder: [link file://{fp}]{fp}",
            guide_style="bold bright_blue",
        )
        walk_directory(Path(fp), tree)
        print(tree)
    
    except IndexError:
        logger.info("[b]Usage:[/] python tree.py <DIRECTORY>")

    except Exception as e:
        logger.warning(f"{e}")        

    question = "What file would you like to load?\n"
    file_choice = console.input(f"{question}")
    if file_choice.isnumeric():
        files = sorted(f for f in Path(str(fp)).iterdir() if f.is_file())
        file_to_load = files[int(file_choice)]
        #check output directory exists
        return file_to_load
    else:
        raise ValueError("Please restart and select an integer of the file you'd like to import")

def main():
    #target data folder goes here.
    fp = Path.cwd() / "src/rad_ecg/data/datasets/sharc_fem/converted"
    
    #Check file existence, load mini detection scheme.  
    if not fp.exists():
        logger.warning(f"Warning: File {fp} not found.")
    else:
        test()
        selected = load_choices(fp)
        fp_save = Path(selected).parent / (Path(selected).stem + "_regimes.json")
        rad = PigRAD(selected)
        if fp_save.exists():
            rad.load_results(fp_save)
        else:
            regimes = rad.detect_regime_changes(n_regimes=3)
            # rad.save_results()
        # rad.plot_discords(top_n=10)

if __name__ == "__main__":
    main()

#This script will be for extraction of heart rhythms within porcine data. 
#Gameplan is as follows. 
#1. Rewrite the peak detect but with just the STFT on the front end. 
#2. Use annotated guided vectors to look at the EKG and ABP at the point of exanguation. 
#3. Have a visual scrolling result popup after runtime. 
    #Confirming both the point of which ABP gets in the 30 to 40 range. 
    #Immediately firing off a discord search before and after the moment to look for irregularities. 
    #Definition of irregularities

#Steps
#1. Load numpy arrays
#2. Choose ECG lead
#3. Choose ABP lead
#4. Run section division of signal into sections (20 second sections)
#5. Begin iterating and extraction. 
#6. Run STFT to test for lower power freq signal.
    #6b.  Also need logic to turn STFT on and off.  Something simpler than previous
#7. Use Wasserbein distribution test for low power majority vote (log reg / SVM?)


#Notes. 
#in the pig ecg there is an R and Rprime peak.  Which... Apparnetly is the whole QRS?
#T and P peaks show up as inverted u waves.  
#There's no jpoint to distinguish between stages. 

#FLUSS
#Same idea but over the whole signal now.  Might not be able to fit all of that into memory
