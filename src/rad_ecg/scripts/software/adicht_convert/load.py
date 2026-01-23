import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Button, TextBox

class SignalDataLoader:
    """Handles loading and structuring the NPZ data."""
    def __init__(self, npz_path):
        self.container = np.load(npz_path)
        self.files = self.container.files
        self.channels = self._identify_and_sort_channels()
        self.full_data = self._stitch_blocks()

    def _identify_and_sort_channels(self):
        raw_names = sorted(list(set([k.split('_block_')[0] for k in self.files])))
        order = [
            'ECG', 'HR', 'MAP', 'Temperature', 'LV_Magnitude', 'LV_Phase', 
            'LV_Pressure', 'LV_Volume', 'DBP', 'SBP', 
            'Distal_Aortic_Pressure', 'Proximal_Aortic_Pressure', 
            'Venous_Pressure', 'Expired__CO2'
        ]
        if all(item in raw_names for item in order) and len(raw_names) == len(order):
            return order
        return raw_names

    def _stitch_blocks(self):
        full_data = {}
        for ch in self.channels:
            ch_blocks = sorted(
                [k for k in self.files if k.startswith(f"{ch}_block_")], 
                key=lambda x: int(x.split('_block_')[-1])
            )
            if ch_blocks:
                full_data[ch] = np.concatenate([self.container[b] for b in ch_blocks])
            else:
                full_data[ch] = np.array([])
        return full_data

class LabChartNavigator:
    def __init__(self, npz_path):
        # 1. Load Data
        self.loader = SignalDataLoader(npz_path)
        self.full_data = self.loader.full_data
        self.channels = self.loader.channels
        
        # 2. State Settings
        self.streams_per_page = 4
        self.total_pages = int(np.ceil(len(self.channels) / self.streams_per_page))
        self.current_page = 0
        self.window_size = 2000
        self.current_pos = 0
        self.step_size = 20
        self.paused = False
        self.alert_timers = {} # Dictionary to store active timers for alerts
        
        # 3. Setup Figure and Events
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click_jump)
        
        # --- NEW: Connect Spacebar Event ---
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        self.setup_layout()
        
        # 4. Initialize Plot Objects (Pooling)
        self.plot_lines = []   
        self.alert_texts = [] 
        self._init_axes_pool()
        
        # 5. Load Initial Data
        self.load_page_content()
        
        # 6. Start Animation
        self.ani = FuncAnimation(
            self.fig, self.update_frame, interval=30, blit=True, cache_frame_data=False
        )
        plt.show()

    def setup_layout(self):
        self.gs_main = gridspec.GridSpec(1, 2, width_ratios=[10, 1.5], figure=self.fig)
        self.gs_plots = gridspec.GridSpecFromSubplotSpec(
            self.streams_per_page + 1, 1, 
            subplot_spec=self.gs_main[0], 
            height_ratios=[3] * self.streams_per_page + [1]
        )
        self.gs_side = gridspec.GridSpecFromSubplotSpec(
            9, 1, subplot_spec=self.gs_main[1], hspace=0.5
        )
        self.setup_controls()

    def setup_controls(self):
        self.btn_pause = Button(self.fig.add_subplot(self.gs_side[0]), 'Pause/Play')
        self.btn_next = Button(self.fig.add_subplot(self.gs_side[1]), 'Next Page')
        self.btn_prev = Button(self.fig.add_subplot(self.gs_side[2]), 'Prev Page')
        self.btn_reset_scale = Button(self.fig.add_subplot(self.gs_side[3]), 'Reset Scale')
        self.btn_gif = Button(self.fig.add_subplot(self.gs_side[4]), 'Export GIF')
        
        ax_speed = self.fig.add_subplot(self.gs_side[5])
        self.txt_speed = TextBox(ax_speed, 'Speed: ', initial=str(self.step_size))
        
        self.btn_pause.on_clicked(self.toggle_pause)
        self.btn_next.on_clicked(self.next_page)
        self.btn_prev.on_clicked(self.prev_page)
        self.btn_reset_scale.on_clicked(self.manual_rescale)
        self.btn_gif.on_clicked(self.export_gif)
        self.txt_speed.on_submit(self.update_speed)

    def _init_axes_pool(self):
        self.axes_pool = []
        for i in range(self.streams_per_page):
            ax = self.fig.add_subplot(self.gs_plots[i])
            line, = ax.plot([], [], lw=1, color='dodgerblue')
            
            # Alert text is hidden by default
            alert = ax.text(0.98, 0.1, "RESCALED", transform=ax.transAxes, 
                            color='red', fontsize=8, ha='right', va='top', 
                            fontweight='bold', visible=False)
            
            self.axes_pool.append(ax)
            self.plot_lines.append(line)
            self.alert_texts.append(alert)
        
        self.nav_ax = self.fig.add_subplot(self.gs_plots[-1])
        self.nav_trace, = self.nav_ax.plot([], [], color='black', alpha=0.3, lw=0.5)
        self.nav_cursor = self.nav_ax.axvline(0, color='red', lw=2)
        self.nav_ax.set_yticks([])
        self.nav_ax.set_xlabel("Timeline (Click to Jump) | Press SPACE to Pause", fontsize=8)

    def load_page_content(self):
        start = self.current_page * self.streams_per_page
        end = start + self.streams_per_page
        current_channels = self.channels[start:end]
        
        self.active_data_map = [] 
        
        for i in range(self.streams_per_page):
            ax = self.axes_pool[i]
            line = self.plot_lines[i]
            alert = self.alert_texts[i]
            
            # Reset any lingering alert timers when changing pages
            if alert in self.alert_timers:
                self.alert_timers[alert].stop()
            
            if i < len(current_channels):
                ch_name = current_channels[i]
                data = self.full_data[ch_name]
                ax.set_visible(True)
                line.set_label(ch_name)
                ax.legend(loc='upper right', fontsize='small')
                ax.set_xlim(0, self.window_size)
                
                if data.size > 0:
                    init_view = data[self.current_pos : self.current_pos + self.window_size]
                    self._apply_scale(ax, init_view)
                
                alert.set_visible(False)
                self.active_data_map.append((line, data, ax, alert))
            else:
                ax.set_visible(False)

        if current_channels:
            ref_data = self.full_data[current_channels[0]]
            ds_step = max(1, len(ref_data) // 5000) 
            self.nav_trace.set_data(np.arange(0, len(ref_data), ds_step), ref_data[::ds_step])
            self.nav_ax.set_xlim(0, len(ref_data))
            self.nav_ax.set_visible(True)
        else:
            self.nav_ax.set_visible(False)
            
        self.fig.canvas.draw_idle()

    # --- NEW: Alert Handling Logic ---
    def trigger_alert(self, alert_obj):
        """Shows alert and sets a timer to hide it."""
        alert_obj.set_visible(True)
        
        # 1. Stop existing timer for this specific alert if it exists
        if alert_obj in self.alert_timers:
            self.alert_timers[alert_obj].stop()
        
        # 2. Create new timer
        timer = self.fig.canvas.new_timer(interval=3000)
        timer.single_shot = True
        timer.add_callback(self._hide_alert, alert_obj)
        self.alert_timers[alert_obj] = timer
        timer.start()

    def _hide_alert(self, alert_obj):
        """Callback to hide the alert."""
        alert_obj.set_visible(False)
        # We don't need a manual draw here; the next update_frame will handle the visual removal
        # because the artist will no longer be returned to the blit manager.

    # --- NEW: Spacebar Logic ---
    def on_key_press(self, event):
        if event.key == ' ':
            self.toggle_pause()

    def _apply_scale(self, ax, view_data):
        if view_data.size > 1:
            v_min, v_max = np.min(view_data), np.max(view_data)
            pad = (v_max - v_min) * 0.1 if v_max != v_min else 0.1
            ax.set_ylim(v_min - pad, v_max + pad)

    def update_frame(self, frame):
        if not self.active_data_map: return []
        
        if not self.paused:
            self.current_pos += self.step_size
            max_len = self.active_data_map[0][1].size
            if self.current_pos + self.window_size > max_len:
                self.current_pos = 0
                for _, _, _, alert in self.active_data_map:
                    alert.set_visible(False)

        updated_artists = []
        needs_redraw = False

        for line, data, ax, alert in self.active_data_map:
            if data.size == 0: continue
            
            end_pos = self.current_pos + self.window_size
            view = data[self.current_pos : end_pos]
            
            line.set_data(np.arange(len(view)), view)
            updated_artists.append(line)
            
            # --- Auto-Scale Check ---
            if view.size > 0:
                v_min, v_max = np.min(view), np.max(view)
                y_min, y_max = ax.get_ylim()
                
                # Rescale if data clips out of view
                if v_min < y_min or v_max > y_max:
                    self._apply_scale(ax, view)
                    self.trigger_alert(alert) # Trigger the timed alert
                    needs_redraw = True
            
            # Important: Always return the alert if it is currently visible
            # This ensures it stays on screen during the timer duration
            if alert.get_visible():
                updated_artists.append(alert)

        self.nav_cursor.set_xdata([self.current_pos])
        updated_artists.append(self.nav_cursor)

        if needs_redraw:
            self.fig.canvas.draw_idle()
            
        return updated_artists

    def manual_rescale(self, event):
        for line, data, ax, alert in self.active_data_map:
            view = data[self.current_pos : self.current_pos + self.window_size]
            self._apply_scale(ax, view)
            alert.set_visible(False)
        self.fig.canvas.draw_idle()

    def next_page(self, event):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.load_page_content()

    def prev_page(self, event):
        if self.current_page > 0:
            self.current_page -= 1
            self.load_page_content()

    def on_click_jump(self, event):
        if event.inaxes == self.nav_ax:
            self.current_pos = int(event.xdata)
            if self.active_data_map:
                max_len = self.active_data_map[0][1].size
                self.current_pos = max(0, min(self.current_pos, max_len - self.window_size))
            
            if self.paused:
                self.update_frame(0)
                self.fig.canvas.draw_idle()

    def toggle_pause(self, event=None):
        self.paused = not self.paused

    def update_speed(self, text):
        try: self.step_size = int(text)
        except ValueError: self.txt_speed.set_val(str(self.step_size))

    def _on_close(self, event):
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
        # Clean up timers
        for timer in self.alert_timers.values():
            timer.stop()

    def export_gif(self, event):
        was_paused = self.paused
        self.paused = True
        f_path = f"export_pg{self.current_page+1}_pos{self.current_pos}.gif"
        print(f"Exporting GIF to {f_path}...")
        writer = PillowWriter(fps=15)
        with writer.saving(self.fig, f_path, dpi=80):
            for _ in range(40):
                self.current_pos += self.step_size
                self.update_frame(0)
                self.fig.canvas.draw()
                writer.grab_frame()
        print("Done.")
        self.paused = was_paused

if __name__ == "__main__":
    target = Path.cwd() / "src/rad_ecg/data/datasets/sharc_fem/converted/SHARC2_47132_6Hr_June-2-25.npz"
    if not target.exists():
        print(f"Warning: File {target} not found.")
    else:
        viewer = LabChartNavigator(str(target))