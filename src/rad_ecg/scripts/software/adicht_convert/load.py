import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Button, TextBox

class LabChartNavigator:
    def __init__(self, npz_path):
        self.container = np.load(npz_path)
        
        # 1. Identify all unique channel names
        all_keys = self.container.files
        raw_names = sorted(list(set([k.split('_block_')[0] for k in all_keys])))
        
        # 2. Apply Custom Ordering
        order = [
            'ECG','HR','MAP','Temperature','LV_Magnitude','LV_Phase','LV_Pressure', 'LV_Volume',
            'DBP','SBP', 'Distal_Aortic_Pressure', 'Proximal_Aortic_Pressure', 'Venous_Pressure','Expired__CO2'
        ]
        
        # Use custom order if all those channels exist, otherwise use alpha order
        if all(item in raw_names for item in order) and len(raw_names) == len(order):
            self.channels = order
        else:
            self.channels = raw_names
        
        # 3. Stitch all blocks together per channel
        self.full_data = {}
        for ch in self.channels:
            ch_blocks = sorted([k for k in all_keys if k.startswith(f"{ch}_block_")], 
                               key=lambda x: int(x.split('_block_')[-1]))
            if ch_blocks:
                self.full_data[ch] = np.concatenate([self.container[b] for b in ch_blocks])
            else:
                self.full_data[ch] = np.array([])

        # 4. State Initialization
        self.streams_per_page = 4
        self.total_pages = int(np.ceil(len(self.channels) / self.streams_per_page))
        self.lines = []
        self.axes = []
        self.current_page = 0
        self.window_size = 500
        self.current_pos = 0
        self.paused = True 
        self.step_size = 5
        
        # 5. Figure and UI setup
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click_jump)
        
        self.setup_ui_layout()
        self.load_page()
        
        # Use cache_frame_data=False to reduce memory overhead
        self.ani = FuncAnimation(self.fig, self.update_frame, interval=20, blit=True, cache_frame_data=False)
        self.paused = False 
        plt.show()

    def _on_close(self, event):
        """Safely stops animation to prevent _resize_id errors"""
        if hasattr(self, 'ani') and self.ani.event_source:
            self.ani.event_source.stop()
        # Do not call plt.close(self.fig) here to avoid recursion/Tcl errors

    def load_page(self):
        if hasattr(self, 'ani'): self.ani.pause()
        
        for ax in self.axes: ax.remove()
        if hasattr(self, 'nav_ax'): self.nav_ax.remove()
        self.axes.clear()
        self.lines.clear() 
        
        start = self.current_page * self.streams_per_page
        end = start + self.streams_per_page
        page_channels = self.channels[start:end]
        
        n = len(page_channels)
        self.gs_plots = gridspec.GridSpecFromSubplotSpec(n + 1, 1, subplot_spec=self.gs_main[0], height_ratios=[3]*n + [1])
        
        for i, ch in enumerate(page_channels):
            ax = self.fig.add_subplot(self.gs_plots[i])
            data = self.full_data[ch]
            line, = ax.plot([], [], lw=1, color='dodgerblue', label=ch)
            
            if data.size > 0:
                ax.set_ylim(np.min(data) - 0.1, np.max(data) + 0.1)
                ax.set_xlim(0, self.window_size)
            
            ax.legend(loc='upper right', fontsize='small')
            self.axes.append(ax)
            self.lines.append((line, data))

        self.nav_ax = self.fig.add_subplot(self.gs_plots[-1])
        if self.lines:
            self.nav_ax.plot(self.lines[0][1], color='black', alpha=0.3, lw=0.5)
        
        self.nav_line = self.nav_ax.axvline(self.current_pos, color='red', lw=2)
        self.nav_ax.set_yticks([])
        self.nav_ax.set_xlabel("Click to Jump to Section", fontsize=8, color='gray')
        
        self.fig.canvas.draw()
        if hasattr(self, 'ani') and not self.paused: 
            self.ani.resume()

    def update_frame(self, frame):
        if not self.lines: return []
        
        if not self.paused:
            self.current_pos += self.step_size
            total_len = self.lines[0][1].size
            if self.current_pos + self.window_size > total_len:
                self.current_pos = 0

        updated = []
        for line, data in self.lines:
            if data.size > 0:
                view = data[self.current_pos : self.current_pos + self.window_size]
                line.set_data(np.arange(len(view)), view)
                updated.append(line)
        
        self.nav_line.set_xdata([self.current_pos])
        updated.append(self.nav_line)
        return updated

    def on_click_jump(self, event):
        if event.inaxes == self.nav_ax:
            self.current_pos = int(event.xdata)
            total_len = self.lines[0][1].size
            self.current_pos = max(0, min(self.current_pos, total_len - self.window_size))
            
            if self.paused:
                self.update_frame(0)
                self.fig.canvas.draw()

    def setup_ui_layout(self):
        self.gs_main = gridspec.GridSpec(1, 2, width_ratios=[10, 1.5], figure=self.fig)
        self.gs_side = gridspec.GridSpecFromSubplotSpec(9, 1, subplot_spec=self.gs_main[1], hspace=0.5)
        
        self.btn_pause = Button(self.fig.add_subplot(self.gs_side[0]), 'Pause/Play')
        self.btn_next_pg = Button(self.fig.add_subplot(self.gs_side[1]), 'Next Page')
        self.btn_prev_pg = Button(self.fig.add_subplot(self.gs_side[2]), 'Prev Page')
        self.btn_autoscale = Button(self.fig.add_subplot(self.gs_side[3]), 'Auto-Scale')
        self.btn_gif = Button(self.fig.add_subplot(self.gs_side[4]), 'Export GIF')
        
        ax_speed = self.fig.add_subplot(self.gs_side[5])
        self.txt_speed = TextBox(ax_speed, 'Speed: ', initial=str(self.step_size))
        
        self.btn_pause.on_clicked(self.toggle_pause)
        self.btn_next_pg.on_clicked(self.next_page)
        self.btn_prev_pg.on_clicked(self.prev_page)
        self.btn_autoscale.on_clicked(self.autoscale_visible_axes)
        self.btn_gif.on_clicked(self.export_gif)
        self.txt_speed.on_submit(self.update_speed)

    def autoscale_visible_axes(self, event):
        for i, (line, data) in enumerate(self.lines):
            if data.size > 0:
                end_pos = min(self.current_pos + self.window_size, data.size)
                view = data[self.current_pos : end_pos]
                if view.size > 1:
                    v_min, v_max = np.min(view), np.max(view)
                    pad = (v_max - v_min) * 0.1 if v_max != v_min else 0.1
                    self.axes[i].set_ylim(v_min - pad, v_max + pad)
        self.fig.canvas.draw()

    def toggle_pause(self, event=None):
        self.paused = not self.paused
        if self.paused: self.ani.pause()
        else: self.ani.resume()

    def update_speed(self, text):
        try: self.step_size = max(1, int(text))
        except ValueError: self.txt_speed.set_val(str(self.step_size))

    def next_page(self, event):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.load_page()

    def prev_page(self, event):
        if self.current_page > 0:
            self.current_page -= 1
            self.load_page()

    def export_gif(self, event):
        self.ani.pause()
        f_path = f"export_pg{self.current_page+1}_pos{self.current_pos}.gif"
        writer = PillowWriter(fps=15)
        with writer.saving(self.fig, f_path, dpi=80):
            curr = self.current_pos
            for _ in range(30):
                self.current_pos += self.step_size
                self.update_frame(0)
                self.fig.canvas.draw()
                writer.grab_frame()
            self.current_pos = curr
        if not self.paused: self.ani.resume()

if __name__ == "__main__":
    target = Path.cwd() / "src/rad_ecg/data/datasets/sharc_fem/converted/EPICS01P03.npz"
    if target.exists():
        viewer = LabChartNavigator(str(target))