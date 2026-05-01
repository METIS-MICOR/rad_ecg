import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, RadioButtons, SpanSelector
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
import scipy.signal as ss
from scipy.fft import rfft, rfftfreq
import stumpy
from pathlib import Path

# Import your existing architecture
import utils
import setup_globals
from support import logger, console
from peak_detect_v4 import SignalLoader, ECGData  

class AnimatedECGViewer:
    def __init__(self, ecg_data: ECGData):
        self.data = ecg_data
        self.wave = ecg_data.wave
        self.fs = ecg_data.fs
        
        # Calculate median section length to manage window widths smoothly
        self.sect_length = int(np.median(self.data.sect_info['end_point'] - self.data.sect_info['start_point']))
        self.current_pos = self.data.sect_info['start_point'][0]
        
        self.is_playing = False
        self.anim = None
        self.current_span = None
        self.transient_artists = []
        
        # UI State
        self.view_mode = 'Base Figure'
        self.show_roll_med = False
        self.show_interiors = False
        self.show_validity = False
        
        self._setup_figure()
        self._setup_ui_elements()
        self.update_main_plot()

    def _setup_figure(self):
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title('MiCOR V4 - Animated ECG Viewer')
        
        # Keyboard Hotkeys
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # [left, bottom, width, height]
        self.ax_main   = self.fig.add_axes([0.05, 0.20, 0.75, 0.75])
        self.ax_slider = self.fig.add_axes([0.05, 0.05, 0.75, 0.05])
        
        # Right Panel Elements
        self.ax_play   = self.fig.add_axes([0.83, 0.90, 0.06, 0.05])
        self.ax_gif    = self.fig.add_axes([0.90, 0.90, 0.06, 0.05])
        
        self.ax_speed  = self.fig.add_axes([0.83, 0.83, 0.13, 0.02])
        self.ax_width  = self.fig.add_axes([0.83, 0.78, 0.13, 0.02])
        
        self.ax_radio  = self.fig.add_axes([0.83, 0.50, 0.14, 0.25])
        self.ax_stats  = self.fig.add_axes([0.83, 0.05, 0.14, 0.43])
        self.ax_stats.axis('off')

        # Initialize persistent line objects for high-speed rendering
        self.line_ecg, = self.ax_main.plot([], [], color='dodgerblue', label='ECG')
        self.line_med, = self.ax_main.plot([], [], color='orange', label='Roll Med')
        self.line_med.set_visible(False)

    def _setup_ui_elements(self):
        # 1. Bottom Navigation Bar (Broken BarH based on 'valid' flag)
        valid_mask = self.data.sect_info['valid'] == 1
        valid_idx = np.where(valid_mask)[0]
        
        if len(valid_idx) > 0:
            breaks = np.where(np.diff(valid_idx) != 1)[0] + 1
            groups = np.split(valid_idx, breaks)
            bar_ranges = [(g[0], len(g)) for g in groups]
            self.ax_slider.broken_barh(bar_ranges, (0, 1), facecolors='dodgerblue', alpha=0.5)
            
        self.ax_slider.set_ylim(0, 1)
        self.ax_slider.set_xlim(0, len(valid_mask))
        self.ax_slider.set_yticks([])

        first_sect = valid_idx[0] if len(valid_idx) > 0 else 0
        self.slider = Slider(
            self.ax_slider, 'Section', 0, len(valid_mask)-1, 
            valinit=first_sect, valstep=1, color='green'
        )
        self.slider.on_changed(self._slider_changed)

        # 2. Control Buttons
        self.btn_playpause = Button(self.ax_play, '▶ PLAY', color='lightgreen')
        self.btn_playpause.on_clicked(self.toggle_play)
        
        self.btn_gif = Button(self.ax_gif, 'GIF', color='gold')
        self.btn_gif.on_clicked(self.export_gif)

        # 3. Speed & Width Sliders
        self.speed_slider = Slider(self.ax_speed, 'FPS ', 1, 30, valinit=10, valstep=1, color='orange')
        self.speed_slider.on_changed(self.update_speed)
        self.anim_step = int(self.fs * 0.05 * self.speed_slider.val)
        
        self.width_slider = Slider(self.ax_width, 'Width', 1, 10, valinit=1, valstep=1, color='purple')
        self.width_slider.on_changed(lambda val: self.update_main_plot())

        # 4. Radio Options
        options = ('Base Figure', 'Roll Median', 'Add Inter', 'Show R Valid', 'Freq (Stem)', 'Stumpy Search')
        self.radio = RadioButtons(self.ax_radio, options)
        self.radio.on_clicked(self.radio_action)

    def on_key_press(self, event):
        """Keyboard shortcuts for navigation and playback."""
        if event.key == ' ':
            self.toggle_play()
        elif event.key == 'right':
            self.step_forward()
        elif event.key == 'left':
            self.step_backward()
        elif event.key == 'down':
            self.jump_next_invalid()
        elif event.key == 'up':
            self.jump_prev_invalid()

    def update_speed(self, val):
        self.anim_step = int(self.fs * 0.05 * val)

    def _slider_changed(self, val):
        """Triggers when the user manually drags the slider."""
        self.current_pos = self.data.sect_info['start_point'][int(val)]
        self.update_main_plot(from_anim=True)

    def step_forward(self):
        self.current_pos = min(self.current_pos + self.sect_length, len(self.wave) - self.sect_length)
        self.update_main_plot()

    def step_backward(self):
        self.current_pos = max(0, self.current_pos - self.sect_length)
        self.update_main_plot()

    def jump_next_invalid(self):
        curr_sect = int(self.slider.val)
        invalid_idx = np.where(self.data.sect_info['valid'][curr_sect+1:] == 0)[0]
        if len(invalid_idx) > 0:
            next_sect = invalid_idx[0] + curr_sect + 1
            self.current_pos = self.data.sect_info['start_point'][next_sect]
            self.update_main_plot()

    def jump_prev_invalid(self):
        curr_sect = int(self.slider.val)
        invalid_idx = np.where(self.data.sect_info['valid'][:curr_sect] == 0)[0]
        if len(invalid_idx) > 0:
            prev_sect = invalid_idx[-1]
            self.current_pos = self.data.sect_info['start_point'][prev_sect]
            self.update_main_plot()

    def toggle_play(self, event=None):
        if self.is_playing:
            self.is_playing = False
            self.btn_playpause.label.set_text('▶ PLAY')
            self.btn_playpause.color = 'lightgreen'
            if self.anim: self.anim.event_source.stop()
        else:
            self.is_playing = True
            self.btn_playpause.label.set_text('⏸ PAUSE')
            self.btn_playpause.color = 'lightcoral'
            if not self.anim:
                self.anim = FuncAnimation(self.fig, self._animate_step, interval=50, blit=False, cache_frame_data=False)
            self.anim.event_source.start()
        self.fig.canvas.draw_idle()

    def _animate_step(self, frame):
        if self.current_pos >= len(self.wave) - self.sect_length:
            self.toggle_play()
            return
        self.current_pos = min(self.current_pos + self.anim_step, len(self.wave) - self.sect_length)
        self.update_main_plot(from_anim=True)

    def update_stats(self, sect_id: int):
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        sect_data = self.data.sect_info[sect_id]
        
        # Word wrap the failure reasons nicely
        fail_str = sect_data['fail_reason'].strip()
        if fail_str:
            reasons = [r.strip() for r in fail_str.split('|') if r.strip()]
            formatted_reason = ("|\n          ").join(reasons)
        else:
            formatted_reason = "None"
        
        stat_text = (
            f"--- SECTION {sect_id} ---\n\n"
            f"Valid:    {'Yes' if sect_data['valid'] == 1 else 'NO'}\n"
            f"Reason:   {formatted_reason}\n\n"
            f"--- SQI ---\n"
            f"Kurtosis: {sect_data['kurtosis']:.2f}\n"
            f"Hjorth:   {sect_data['hjorth']:.2f}\n"
            f"W-Dist:   {sect_data['wdist']:.2f}\n"
            f"Spec Rat: {sect_data['spectral']:.2f}\n"
            f"Bad Beat: {sect_data['bad_b_rat']:.1%}\n\n"
            f"--- VITALS ---\n"
            f"HR:       {sect_data['HR']:.0f} bpm\n"
            f"SDNN:     {sect_data['SDNN']:.0f} ms\n"
            f"RMSSD:    {sect_data['RMSSD']:.0f} ms\n\n"
            f"--- INTERVALS ---\n"
            f"PR:       {sect_data['PR']:.0f} ms\n"
            f"QRS:      {sect_data['QRS']:.0f} ms\n"
            f"QTc:      {sect_data['QTc']:.0f} ms\n"
            f"QTVI:     {sect_data['QTVI']:.2f}"
        )
        
        self.ax_stats.text(
            0.05, 0.95, stat_text, transform=self.ax_stats.transAxes,
            fontsize=11, family='monospace', va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8)
        )

    def _draw_frequency(self, start_w, end_w, sect_id):
        """Draws the dynamic frequency spectrum for the current animated window."""
        self.ax_main.clear()
        samp = self.wave[start_w:end_w].flatten()
        
        if len(samp) > 2:
            fft_samp = np.abs(rfft(samp))
            freq_list = rfftfreq(len(samp), d=1/self.fs)
            
            # Create masks for the physiological bands
            qrs_mask = (freq_list >= 5.0) & (freq_list <= 15.0)
            hf_mask = (freq_list > 15.0) & (freq_list <= 40.0)
            base_mask = ~qrs_mask & ~hf_mask
            
            # Plot stems with distinct colors for the bands
            self.ax_main.stem(freq_list[base_mask], fft_samp[base_mask], linefmt="b-", markerfmt=" ", basefmt=" ")
            self.ax_main.stem(freq_list[qrs_mask], fft_samp[qrs_mask], linefmt="g-", markerfmt=" ", basefmt=" ")
            self.ax_main.stem(freq_list[hf_mask], fft_samp[hf_mask], linefmt="C1-", markerfmt=" ", basefmt=" ")
            
            # Shade the backgrounds
            self.ax_main.axvspan(5.0, 15.0, color='lightgreen', alpha=0.2, label='QRS Band (5-15Hz)')
            self.ax_main.axvspan(15.0, 40.0, color='orange', alpha=0.1) # Unlabeled high frequency noise band
            
            spectral_ratio = self.data.sect_info['spectral'][sect_id]
            title_str = f"Frequency Spectrum | Section {sect_id} | Spectral Pwr (5-15Hz): {spectral_ratio:.1%}"
            self.ax_main.set_title(title_str)
            self.ax_main.set_xlim(0, 50)
            self.ax_main.set_xlabel("Frequency (Hz)")
            self.ax_main.legend(loc='upper right')

    def update_main_plot(self, from_anim=False):
        if self.current_span:
            self.current_span.set_active(False)

        # Base bounds on current continuous sample position
        start_w = int(self.current_pos)
        width_offset = int(self.width_slider.val) - 1  # <-- FIXED: Defined width_offset here!
        width_samples = int(self.width_slider.val * self.sect_length)
        end_w = min(start_w + width_samples, len(self.wave))
        
        # Figure out which section we are primarily floating over
        sect_id = np.searchsorted(self.data.sect_info['start_point'], start_w, side='right') - 1
        sect_id = max(0, min(sect_id, len(self.data.sect_info) - 1))
        is_valid = self.data.sect_info['valid'][sect_id]

        # Silently update the slider position to match the scroll
        if not from_anim:
            self.slider.eventson = False
            self.slider.set_val(sect_id)
            self.slider.eventson = True

        # If we are in Frequency Mode, route to the FFT plotter
        if self.view_mode == 'Freq (Stem)':
            self._draw_frequency(start_w, end_w, sect_id)
            self.update_stats(sect_id)
            self.fig.canvas.draw_idle()
            return

        # Clean up old transient artists (Memory Management)
        for artist in self.transient_artists:
            try: artist.remove()
            except: pass
        self.transient_artists.clear()

        # Update fast persistent lines
        x_range = np.arange(start_w, end_w)
        wave_chunk = self.wave[start_w:end_w]
        
        # Guard against empty axes resets
        if not self.ax_main.lines:
            self.line_ecg, = self.ax_main.plot([], [], color='dodgerblue', label='ECG')
            self.line_med, = self.ax_main.plot([], [], color='orange', label='Roll Med')
            
        self.line_ecg.set_data(x_range, wave_chunk)
        
        if self.show_roll_med:
            rolled = utils.roll_med(wave_chunk)
            self.line_med.set_data(x_range, rolled)
            self.line_med.set_visible(True)
        else:
            self.line_med.set_visible(False)

        # Dynamic View Scaling
        self.ax_main.set_xlim(start_w, end_w)
        if len(wave_chunk) > 1:
            v_min, v_max = np.min(wave_chunk), np.max(wave_chunk)
            pad = (v_max - v_min) * 0.15 if v_max != v_min else 0.1
            self.ax_main.set_ylim(v_min - pad, v_max + pad)

        self.ax_main.set_facecolor('mistyrose' if is_valid == 0 else 'white')

        # Transients: R-Peaks
        r_peaks = self.data.peaks[(self.data.peaks[:, 0] >= start_w) & (self.data.peaks[:, 0] <= end_w)]
        if len(r_peaks) > 0:
            scat_r = self.ax_main.scatter(r_peaks[:, 0], self.wave[r_peaks[:, 0]], color='red', marker='D', zorder=5, label='R Peak')
            self.transient_artists.append(scat_r)
            
            if self.show_validity:
                for peak in r_peaks:
                    color = 'lightgreen' if peak[1] == 1 else 'red'
                    rect = Rectangle(
                        xy=(peak[0] - 10, v_min), width=int(self.fs*0.1), height=v_max-v_min,
                        facecolor=color, alpha=0.3
                    )
                    self.ax_main.add_patch(rect)
                    self.transient_artists.append(rect)

        # Transients: Interior Peaks
        if self.show_interiors:
            inners = self.data.interior_peaks[
                (self.data.interior_peaks['r_peak'] >= start_w) & 
                (self.data.interior_peaks['r_peak'] <= end_w)
            ]
            
            def plot_inners(col_name: str, color: str, marker: str, size: int = 40):
                valid_x = inners[col_name][inners[col_name] > 0]
                if len(valid_x) > 0:
                    valid_y = self.wave[valid_x]
                    scat = self.ax_main.scatter(
                        valid_x, valid_y, color=color, marker=marker, 
                        s=size, zorder=6, label=col_name.capitalize()
                    )
                    self.transient_artists.append(scat)

            plot_inners('p_peak', 'green', 'o')
            plot_inners('q_peak', 'cyan', 'v')
            plot_inners('s_peak', 'magenta', '^')
            plot_inners('t_peak', 'black', 'o')
            plot_inners('p_onset', 'purple', '|', size=150)
            plot_inners('q_onset', 'darkgoldenrod', '|', size=150)
            plot_inners('j_point', 'dodgerblue', 'o', size=80)
            plot_inners('t_onset', 'teal', '|', size=150)
            plot_inners('t_offset', 'orange', '|', size=150)

        title_str = f"ECG Signal - Section {sect_id}"
        if width_offset > 0:
            end_sect = min(sect_id + width_offset, len(self.data.sect_info) - 1)
            title_str += f" to {end_sect}"
        self.ax_main.set_title(title_str + f" ({start_w}:{end_w})")
        
        # Deduplicate legend safely using existing artists
        handles, labels = self.ax_main.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        leg = self.ax_main.legend(by_label.values(), by_label.keys(), loc='upper right')
        self.transient_artists.append(leg) 
        
        self.update_stats(sect_id)
        self.fig.canvas.draw_idle()

    def radio_action(self, label):
        self.view_mode = label
        self.show_roll_med = False
        self.show_interiors = False
        self.show_validity = False
        
        # Clear out the FFT plot if returning to a base wave visualization
        if label in ['Base Figure', 'Roll Median', 'Add Inter', 'Show R Valid']:
            self.ax_main.clear()
            self.line_ecg, = self.ax_main.plot([], [], color='dodgerblue', label='ECG')
            self.line_med, = self.ax_main.plot([], [], color='orange', label='Roll Med')
            self.line_med.set_visible(False)
        
        if label == 'Roll Median':
            self.show_roll_med = True
        elif label == 'Add Inter':
            self.show_interiors = True
        elif label == 'Show R Valid':
            self.show_validity = True

        if label == 'Stumpy Search':
            if self.is_playing:
                self.toggle_play()
            self.activate_stumpy()
        else:
            self.update_main_plot()

    def activate_stumpy(self):
        self.update_main_plot()
        self.ax_main.set_title("Stumpy Search: Highlight a wave segment to search history!")
        
        def onselect(xmin, xmax):
            xmin, xmax = int(xmin), int(xmax)
            if xmax - xmin < 10: return
            
            Q_s = self.wave[xmin:xmax].flatten()
            
            # 10 Minutes forward and back (600 seconds)
            search_start = max(0, xmin - int(self.fs * 600)) 
            search_end = min(len(self.wave), xmax + int(self.fs * 600))
            T_s = self.wave[search_start:search_end].flatten()
            
            matches = stumpy.match(Q_s, T_s, max_distance=lambda D: max(np.mean(D) - 4 * np.std(D), np.min(D)))
            
            self.ax_main.clear()
            self.ax_main.set_title(f"Found {len(matches)} matches in surrounding 20 minutes (±10m)")
            Q_z = stumpy.core.z_norm(Q_s)
            
            for dist, idx in matches:
                match_z = stumpy.core.z_norm(T_s[idx:idx+len(Q_s)])
                self.ax_main.plot(match_z, color='dodgerblue', alpha=0.5)
            self.ax_main.plot(Q_z, color='red', linewidth=2, label="Query")
            self.ax_main.legend()
            self.fig.canvas.draw_idle()

        self.current_span = SpanSelector(
            self.ax_main, onselect, 'horizontal', 
            props=dict(alpha=0.3, facecolor='red'), interactive=True
        )

    def export_gif(self, event):
        if self.is_playing:
            self.toggle_play()
            
        logger.info("Generating GIF... Please wait.")
        self.ax_main.set_title("Exporting GIF... UI Locked")
        self.fig.canvas.draw()
        
        start_pos = self.current_pos
        frames = 50 # 50 frames of animation
        
        def animate(i):
            self.current_pos = min(start_pos + (i * self.anim_step), len(self.wave) - self.sect_length)
            self.update_main_plot(from_anim=True)
            return self.ax_main,
            
        anim = FuncAnimation(self.fig, animate, frames=frames, interval=1000/self.speed_slider.val, blit=False)
        writer = PillowWriter(fps=int(self.speed_slider.val))
        
        filename = "ecg_playback.gif"
        anim.save(filename, writer=writer)
        logger.info(f"GIF saved successfully to {filename}")
        
        self.current_pos = start_pos
        self.update_main_plot()
        self.fig.canvas.draw_idle()

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    configs = setup_globals.load_config()
    fp = Path.cwd() / configs["data_path"]
    
    selected = setup_globals.load_choices(fp, batch_process=False)
    if isinstance(selected, list): 
        selected = selected[0] 
        
    cam_name = selected.stem
    configs["cam_name"] = cam_name
    
    logger.info(f"Loading raw signal for {cam_name}...")
    loader = SignalLoader(selected)
    loader.load_signal_data()
    
    ECG = loader.load_structures() 
    
    save_dir = Path(configs["save_path"]) / cam_name
    if not save_dir.exists():
        logger.error(f"No output directory found at {save_dir}. Has this file been processed?")
        return
        
    npz_files = list(save_dir.glob("*.npz"))
    if not npz_files:
        logger.error(f"No .npz result files found in {save_dir}.")
        return
        
    npz_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_npz = npz_files[0]
    
    logger.info(f"Loading mathematical results from {latest_npz.name}...")
    loaded_data = np.load(latest_npz)
    
    ECG.peaks = loaded_data['peaks']
    ECG.interior_peaks = loaded_data['interior_peaks']
    ECG.sect_info = loaded_data['section_info']
    
    logger.info("Launching Animated GUI...")
    viewer = AnimatedECGViewer(ecg_data=ECG)
    plt.show()

if __name__ == "__main__":
    main()