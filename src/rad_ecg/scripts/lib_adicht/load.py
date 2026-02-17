import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Button, TextBox
from scipy.fft import rfft, rfftfreq
from rich.filesize import decimal
from rich import print as pprint
from rich.tree import Tree
from rich.text import Text
from rich.table import Table
from rich.theme import Theme
from rich.markup import escape

class SignalDataLoader:
    """Handles loading and structuring the NPZ data."""
    def __init__(self, file_path):
        if file_path.endswith("npz"):
            self.container = np.load(file_path)
            self.files = self.container.files
            self.channels = self._identify_and_sort_channels()
            self.full_data = self._stitch_blocks()

        elif file_path.endswith("pkl"):
            self.container = np.load(file_path, allow_pickle=True)
            self.full_data = self.container.to_dict(orient="series")
            self.channels = self.container.columns.to_list()
            self.full_data.pop("Time")
            self.channels.pop(self.channels.index("Time"))
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
                else:
                    full_data[ch] = np.array([])
        return full_data

class LabChartNavigator:
    def __init__(self, npz_path):
        # 1. Load Data
        self.load_path = npz_path
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
        self.alert_timers = {} 
        
        # Frequency State: 0=Off, 1=Stem, 2=Specgram
        self.freq_mode = 0 
        self.sampling_rate = 1000 
        
        # 3. Setup Figure and Events
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click_jump)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        self.setup_layout()
        
        # 4. Initialize Plot Objects
        self.use_blit = True 
        self._init_axes_pool()
        
        # 5. Load Initial Data
        self.load_page_content()
        
        # 6. Start Animation
        self.start_animation()
        plt.show()

    def start_animation(self):
        if hasattr(self, 'ani') and self.ani.event_source:
            self.ani.event_source.stop()
        
        self.ani = FuncAnimation(
            self.fig, self.update_frame, interval=30, blit=self.use_blit, cache_frame_data=False
        )

    def setup_layout(self):
        """Define GridSpec layout."""
        self.gs_main = gridspec.GridSpec(1, 2, width_ratios=[10, 1.5], figure=self.fig)
        
        # Main plot area
        self.gs_plots = gridspec.GridSpecFromSubplotSpec(
            self.streams_per_page + 1, 1, 
            subplot_spec=self.gs_main[0], 
            height_ratios=[3] * self.streams_per_page + [1]
        )
        
        # Side controls
        self.gs_side = gridspec.GridSpecFromSubplotSpec(
            11, 1, subplot_spec=self.gs_main[1], hspace=0.5
        )
        self.setup_controls()

    def setup_controls(self):
        self.btn_pause = Button(self.fig.add_subplot(self.gs_side[0]), 'Pause/Play')
        self.btn_next = Button(self.fig.add_subplot(self.gs_side[1]), 'Next Page')
        self.btn_prev = Button(self.fig.add_subplot(self.gs_side[2]), 'Prev Page')
        self.btn_reset_scale = Button(self.fig.add_subplot(self.gs_side[3]), 'Reset Scale')
        self.btn_gif = Button(self.fig.add_subplot(self.gs_side[4]), 'Export GIF')
        self.btn_freq = Button(self.fig.add_subplot(self.gs_side[5]), 'Freq Mode: OFF')

        ax_speed = self.fig.add_subplot(self.gs_side[6])
        self.txt_speed = TextBox(ax_speed, 'Speed: ', initial=str(self.step_size))
        
        ax_window = self.fig.add_subplot(self.gs_side[7])
        self.txt_window = TextBox(ax_window, 'Window: ', initial=str(self.window_size))
        
        self.btn_pause.on_clicked(self.toggle_pause)
        self.btn_next.on_clicked(self.next_page)
        self.btn_prev.on_clicked(self.prev_page)
        self.btn_reset_scale.on_clicked(self.manual_rescale)
        self.btn_gif.on_clicked(self.export_gif)
        self.btn_freq.on_clicked(self.toggle_frequency)
        
        self.txt_speed.on_submit(self.update_speed)
        self.txt_window.on_submit(self.update_window_size)

    def _init_axes_pool(self):
        self.axes_pool = []
        self.plot_lines = []
        self.alert_texts = []
        self.freq_axes_pool = [] 
        
        for i in range(self.streams_per_page):
            if self.freq_mode == 0:
                ax = self.fig.add_subplot(self.gs_plots[i])
                if i == 0:
                    ax.set_title(f"{Path(self.load_path).name}")
                ax_freq = None
            else:
                gs_row = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=self.gs_plots[i], wspace=0.2)
                ax = self.fig.add_subplot(gs_row[0])
                ax_freq = self.fig.add_subplot(gs_row[1])

            # Setup Signal Line
            line, = ax.plot([], [], lw=1, color='dodgerblue')
            alert = ax.text(0.98, 0.1, "RESCALED", transform=ax.transAxes, 
                            color='red', fontsize=8, ha='right', va='top', 
                            fontweight='bold', visible=False)

            self.axes_pool.append(ax)
            self.plot_lines.append(line)
            self.alert_texts.append(alert)
            self.freq_axes_pool.append(ax_freq)
        
        # Navigator
        self.nav_ax = self.fig.add_subplot(self.gs_plots[-1])
        self.nav_trace, = self.nav_ax.plot([], [], color='black', alpha=0.3, lw=0.5)
        self.nav_cursor = self.nav_ax.axvline(0, color='red', lw=2)
        self.nav_ax.set_yticks([])
        self.nav_ax.set_xlabel("Timeline (Click to Jump) | Press SPACE to Pause", fontsize=8)

    def rebuild_layout(self):
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
        
        for ax in self.axes_pool:
            ax.remove()
        for axf in self.freq_axes_pool:
            if axf is not None: 
                axf.remove()
        self.nav_ax.remove()
        
        self._init_axes_pool()
        self.load_page_content()
        
        self.use_blit = (self.freq_mode == 0)
        self.start_animation()
        self.fig.canvas.draw_idle()

    def toggle_frequency(self, event):
        self.freq_mode = (self.freq_mode + 1) % 3
        labels = {0: "Freq Mode: OFF", 1: "Freq Mode: STEM", 2: "Freq Mode: SPEC"}
        self.btn_freq.label.set_text(labels[self.freq_mode])
        self.rebuild_layout()

    def load_page_content(self):
        start = self.current_page * self.streams_per_page
        end = start + self.streams_per_page
        current_channels = self.channels[start:end]
        self.active_data_map = [] 
        for i in range(self.streams_per_page):
            if self.full_data[current_channels[i]].dtype.name != "float64" :
                continue
            ax = self.axes_pool[i]
            line = self.plot_lines[i]
            alert = self.alert_texts[i]
            ax_freq = self.freq_axes_pool[i]
            
            if alert in self.alert_timers:
                self.alert_timers[alert].stop()
            
            if i < len(current_channels):
                ch_name = current_channels[i]
                data = self.full_data[ch_name]
                
                ax.set_visible(True)
                if ax_freq: 
                    ax_freq.set_visible(True)
                
                line.set_label(ch_name)
                ax.legend(loc='upper right', fontsize='small')
                ax.set_xlim(0, self.window_size)
                
                if data.size > 0:
                    init_view = data[self.current_pos : self.current_pos + self.window_size]
                    self._apply_scale(ax, init_view)
                
                alert.set_visible(False)
                self.active_data_map.append((line, data, ax, alert, ax_freq, ch_name))
            else:
                ax.set_visible(False)
                if ax_freq: 
                    ax_freq.set_visible(False)

        if current_channels:
            ref_data = self.full_data[current_channels[0]]
            ds_step = max(1, len(ref_data) // 5000) 
            self.nav_trace.set_data(np.arange(0, len(ref_data), ds_step), ref_data[::ds_step])
            self.nav_ax.set_xlim(0, len(ref_data))
            self.nav_ax.set_visible(True)
        else:
            self.nav_ax.set_visible(False)
        
        self.fig.canvas.draw_idle()

    def update_frame(self, frame):
        if not self.active_data_map: 
            return []
        
        if not self.paused:
            self.current_pos += self.step_size
            max_len = self.active_data_map[0][1].size
            if self.current_pos + self.window_size > max_len:
                self.current_pos = 0
                for _, _, _, alert, _, _ in self.active_data_map:
                    alert.set_visible(False)

        updated_artists = []
        needs_redraw = False

        for line, data, ax, alert, ax_freq, ch_name in self.active_data_map:
            if data.size == 0:
                continue
            
            end_pos = self.current_pos + self.window_size
            view = data[self.current_pos : end_pos]
            
            # 1. Update Signal
            line.set_data(np.arange(len(view)), view)
            updated_artists.append(line)
            
            # 2. Auto-Scale
            if view.size > 0:
                v_min, v_max = np.min(view), np.max(view)
                y_min, y_max = ax.get_ylim()
                if v_min < y_min or v_max > y_max:
                    self._apply_scale(ax, view)
                    self.trigger_alert(alert)
                    needs_redraw = True
            
            if alert.get_visible():
                updated_artists.append(alert)

            # 3. Frequency Update
            if self.freq_mode > 0 and ax_freq is not None and view.size > 100:
                ax_freq.cla() 
                
                if self.freq_mode == 1: # STEM
                    fft_samp = np.abs(rfft(view))
                    freq_list = rfftfreq(len(view), d=1/self.sampling_rate)
                    
                    half_point = int(len(view)/2)
                    freqs = fft_samp[:half_point]
                    freq_l = freq_list[:half_point]
                    
                    ax_freq.plot(freq_l, freqs, color='purple', lw=1, label=f"FFT: {ch_name}")
                    ax_freq.fill_between(freq_l, freqs, color='purple', alpha=0.3)
                    ax_freq.legend(loc='upper right', fontsize='small')
                    ax_freq.set_xlim(0, 50) 
                    
                elif self.freq_mode == 2: # SPECGRAM
                    nfft = min(256, len(view))
                    try:
                        ax_freq.specgram(
                            view, NFFT=nfft, Fs=self.sampling_rate, noverlap=nfft//2, cmap='inferno'
                        )
                        # Add a fake line just to get a legend entry
                        ax_freq.plot([], [], color='black', label=f"Spec: {ch_name}")
                        ax_freq.legend(loc='upper right', fontsize='small')
                        ax_freq.set_yticks([])
                    except:
                        pass

        self.nav_cursor.set_xdata([self.current_pos])
        updated_artists.append(self.nav_cursor)

        if needs_redraw:
            pass 
            
        return updated_artists

    # --- Helpers ---
    def trigger_alert(self, alert_obj):
        alert_obj.set_visible(True)
        if alert_obj in self.alert_timers:
            self.alert_timers[alert_obj].stop()
        timer = self.fig.canvas.new_timer(interval=3000)
        timer.single_shot = True
        timer.add_callback(self._hide_alert, alert_obj)
        self.alert_timers[alert_obj] = timer
        timer.start()

    def _hide_alert(self, alert_obj):
        alert_obj.set_visible(False)

    def on_key_press(self, event):
        if event.key == ' ':
            self.toggle_pause()

    def _apply_scale(self, ax, view_data):
        if view_data.size > 1:
            v_min, v_max = np.min(view_data), np.max(view_data)
            pad = (v_max - v_min) * 0.1 if v_max != v_min else 0.1
            ax.set_ylim(v_min - pad, v_max + pad)

    def manual_rescale(self, event):
        for line, data, ax, alert, _, _ in self.active_data_map:
            view = data[self.current_pos : self.current_pos + self.window_size]
            self._apply_scale(ax, view)
            alert.set_visible(False)
        self.fig.canvas.draw_idle()

    def next_page(self, event):
        if self.current_page < self.total_pages - 1:
            self._safe_page_update(1)

    def prev_page(self, event):
        if self.current_page > 0:
            self._safe_page_update(-1)

    def _safe_page_update(self, direction):
        # 1. Stop the animation to prevent the loop from overwriting our changes
        if hasattr(self, 'ani') and self.ani.event_source:
            self.ani.event_source.stop()

        # 2. Update State
        self.current_page += direction
        self.load_page_content()

        # 3. Force a full draw to update the "Background" (Legends, Axis Labels)
        # This refreshes the blit cache
        self.fig.canvas.draw()
        
        # 4. Restart animation
        if hasattr(self, 'ani') and self.ani.event_source:
            self.ani.event_source.start()

    def manual_rescale(self, event):
        # Apply the same logic here to ensure axis ticks update immediately
        if hasattr(self, 'ani') and self.ani.event_source:
            self.ani.event_source.stop()

        for line, data, ax, alert, _, _ in self.active_data_map:
            view = data[self.current_pos : self.current_pos + self.window_size]
            self._apply_scale(ax, view)
            alert.set_visible(False)
        
        self.fig.canvas.draw()
        
        if hasattr(self, 'ani') and self.ani.event_source:
            self.ani.event_source.start()
    
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
        try: 
            self.step_size = int(text)
        except ValueError: 
            self.txt_speed.set_val(str(self.step_size))

    def update_window_size(self, text):
        try:
            new_size = int(text)
            if new_size > 10:
                self.window_size = new_size
                for _, _, ax, _, _, _ in self.active_data_map:
                    ax.set_xlim(0, self.window_size)
                self.manual_rescale(None)
        except ValueError:
            self.txt_window.set_val(str(self.window_size))

    def _on_close(self, event):
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
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

#FUNCTION Walk Directory
def walk_directory(directory: Path, tree: Tree, files:bool = False) -> None:
    """Build a Tree with directory contents.
    Source Code: https://github.com/Textualize/rich/blob/master/examples/tree.py

    """
    # Sort dirs first then by filename
    paths = sorted(
        Path(directory).iterdir(),
        key=lambda path: (path.is_file(), path.name.lower()),
    )
    idx = 0
    for path in paths:
        # Remove hidden files
        if path.name.startswith("."):
            continue
        if path.is_dir():
            style = "dim" if path.name.startswith("__") else ""
            file_size = getfoldersize(path)
            branch = tree.add(
                f"[bold green]{idx} [/bold green][bold magenta]:open_file_folder: [link file://{path}]{escape(path.name)}[/bold magenta] [bold blue]{file_size}[/bold blue]",
                style=style,
                guide_style=style,
            )
            walk_directory(path, branch)
        else:
            text_filename = Text(path.name, "green")
            text_filename.highlight_regex(r"\..*$", "bold red")
            text_filename.stylize(f"link file://{path}")
            file_size = path.stat().st_size
            text_filename.append(f" ({decimal(file_size)})", "blue")
            tree.add(Text(f'{idx} ', "blue") + text_filename) # + Text(icon)
        idx += 1

def getfoldersize(folder:Path):
    fsize = 0
    for root, dirs, files in os.walk(folder):
        for f in files:
            fp = os.path.join(folder,f)
            fsize += os.stat(fp).st_size

    return sizeofobject(fsize)

def sizeofobject(folder)->str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(folder) < 1024:
            return f"{folder:4.1f} {unit}"
        folder /= 1024.0
    return f"{folder:.1f} PB"
          
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
        pprint(tree)
    except Exception as e:
        print(f"{e}")        

    question = "What file would you like to load?\n"
    file_choice = input(f"{question}")
    if file_choice.isnumeric():
        files = sorted(f for f in Path(str(fp)).iterdir() if f.is_file())
        return files[int(file_choice)]
    else:
        raise ValueError("Invalid choice")
    
if __name__ == "__main__":
    base = Path.cwd() / "src/rad_ecg/data/datasets/JT"
    # targets = sorted(base.iterdir())
    # for target in targets:
    target = load_choices(base)
    if target.exists() and target.is_file():
        try:
            viewer = LabChartNavigator(str(target))
        except Exception as e:
            print(f"{e}")
    
    else:
        print(f"Warning: File {target} not found.")
    print(f"{target.name} closed")