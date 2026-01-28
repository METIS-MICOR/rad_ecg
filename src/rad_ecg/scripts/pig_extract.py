import json
import stumpy
import numpy as np
from numba import cuda
from pathlib import Path
from utils import segment_ECG
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from setup_globals import walk_directory
from scipy.stats import wasserstein_distance
from scipy.signal import find_peaks, stft, welch
from support import logger, console, log_time, NumpyArrayEncoder
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

class MiniRAD():
    def __init__(self, npz_path):
        # 1. load data / params
        self.npz_path = npz_path
        self.loader = SignalDataLoader(str(self.npz_path))
        self.full_data = self.loader.full_data
        self.channels = self.loader.channels
        self.fs = 1000.00 #Hz
        self.windowsize = 20
        self.lead = self.pick_lead()
        self.sections = segment_ECG(self.full_data[self.lead], self.fs, self.windowsize)
        self.gpu_devices = []
        self.results = []

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
        
    def run_stump(self):
        """
        Iterates through signal sections, checks for distribution shifts,
        calculates dynamic matrix profiles using GPU-STUMP, and identifies discords.
        """
        # Threshold for Wasserstein distance to consider distributions "similar"
        WD_THRESHOLD = 0.05 
        previous_dist = None

        try:
            self.gpu_devices = [device.id for device in cuda.list_devices()]
        except Exception:
            self.gpu_devices = []

        if self.gpu_devices:
            gpu_indicator = "[bold green]GPU[/]"
        else:
            gpu_indicator = "[bold red]GPU[/]"

        prog = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn(gpu_indicator), 
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        )
        with prog as progress:
            task = progress.add_task("[cyan]Processing Sections...", total=len(self.sections))
            for i, section in enumerate(self.sections):
                start = section[0]
                end = section[1]
                sig_section = self.full_data[self.lead][start:end].flatten().astype(np.float64)

                # 1. Calculate Distribution (STFT -> PSD)
                current_dist = self.STFT(sig_section)
                
                # 2. Check Distribution Shift (Skip first section)
                if i > 0 and previous_dist is not None:
                    wd = wasserstein_distance(previous_dist, current_dist)
                    
                    if wd > WD_THRESHOLD:
                        # Distribution shift detected - skip extraction or flag
                        logger.warning(f"Section {i}: Major distribution shift detected (WD: {wd:.4f}). Skipping extraction.")
                        previous_dist = current_dist
                        progress.advance(task)
                        self.sections[i, 2] = 0
                        continue
                
                # Update previous distribution for next iteration
                previous_dist = current_dist
                
                # 3. Extraction: Find R-Peaks to determine 'm'
                # Distance is ~200ms in samples (0.2 * 1000) to avoid T-wave detection
                peaks, _ = find_peaks(
                    sig_section, 
                    height = np.percentile(sig_section, 90),     #90 -> stock
                    prominence = np.percentile(sig_section, 95), #95 -> stock
                    distance = int(self.fs * 0.2) 
                )
                
                if len(peaks) < 2:
                    logger.warning(f"Section {i}: Not enough peaks to calculate motif length 'm'. Skipping.")
                    progress.advance(task)
                    continue

                # Calculate average R-to-R interval
                r_r_intervals = np.diff(peaks)
                avg_rr = np.mean(r_r_intervals)
                m = int(avg_rr) # Window size for Stumpy
                
                # Safety check for m
                if m < 3 or m >= len(sig_section):
                    logger.warning(f"m {m} is out of the window 3 or {len(sig_section)}")
                    progress.advance(task)
                    continue

                # 4. Matrix Profile via GPU Stump.  stumpy.gpu_stump returns the Matrix Profile (MP) and Matrix Profile Index (MPI)
                try:
                    if self.gpu_devices:
                        mp = stumpy.gpu_stump(sig_section, m=m, device_id=self.gpu_devices)
                    else:
                        mp = stumpy.stump(sig_section, m=m)
                    
                    # 5. Identify Major Discord (Anomaly)
                    # The discord is the subsequence with the largest Nearest Neighbor Distance (max value in MP)
                    discord_idx = np.argsort(mp[:, 0])[-1]
                    discord_dist = mp[discord_idx, 0]
                    
                    # Store result
                    self.results.append({
                        'section_idx': i,
                        'm': m,
                        'discord_index': discord_idx,
                        'discord_score': discord_dist,
                        'wasserstein_metric': wd if i > 0 else 0.0
                    })
                    
                except Exception as e:
                    logger.error(f"GPU Stump failed on section {i}: {e}")
                if i // 10 == 0:
                    #Every 10 sections pop in a print
                    logger.info(f"section {i}")
                progress.advance(task)
        
        # Summary Output
        if self.results:
            console.print(f"[bold green]Processing Complete.[/] Analyzed {len(self.results)} valid sections.")
            # Simple list of top 3 discords found across all sections
            sorted_discords = sorted(self.results, key=lambda x: x['discord_score'], reverse=True)
            console.print("[bold]Top 3 Global Discords found:[/]")
            for d in sorted_discords[:3]:
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
        
        # Collapse over time to get a frequency distribution (PSD-like) for the whole section
        # We sum over the time axis
        freq_dist = np.sum(magnitude, axis=1)
        
        # Normalize to sum to 1 to treat as a probability distribution for Wasserstein
        if np.sum(freq_dist) > 0:
            freq_dist = freq_dist / np.sum(freq_dist)
        
        return freq_dist
    
    def plot_discords(self, top_n=5, pause_duration=4):
        """
        Iterates through the top N discords and displays them in a Matplotlib window
        with the discord highlighted by a gray patch.
        """
        if not self.results:
            console.print("[red]No results available to plot.[/]")
            return

        # Sort results to get the top discords
        sorted_discords = sorted(self.results, key=lambda x: x['discord_score'], reverse=True)[:top_n]
        
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
                ax.plot(data, color='black', linewidth=1, label='ECG Signal')
                ax.set_xlim(0, len(data))
                
                # 3. Highlight the Discord
                discord_start = res['discord_index']
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
                ax.set_title(f"Discord Rank #{i+1} | Section {sec_idx} | Score: {res['discord_score']:.2f}")
                ax.set_xlabel("Samples")
                ax.set_ylabel("Amplitude")
                ax.legend(loc='upper right')
                
                # 5. Render and Pause
                plt.draw()
                console.print(f"Displaying Discord #{i+1} (Section {sec_idx})...")
                plt.pause(pause_duration)
                
            except Exception as e:
                logger.error(f"Error plotting discord {i}: {e}")

        plt.ioff() # Turn off interactive mode
        plt.close()
        console.print("[bold green]Playback complete.[/]")
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
                json.dump(self.results, f, cls=NumpyArrayEncoder, indent=4)
            console.print(f"[bold green]Results successfully saved to:[/]\n[link file://{out_path}]{out_path}[/link]")
        except Exception as e:
            logger.error(f"Failed to save results to JSON: {e}")

def load_choices(fp:str):
    try:
        tree = Tree(
            f":open_file_folder: [link file://{fp}]{fp}",
            guide_style="bold bright_blue",
        )
        files = walk_directory(Path(fp), tree)
        print(tree)
    
    except IndexError:
        logger.info("[b]Usage:[/] python tree.py <DIRECTORY>")

    except Exception as e:
        logger.warning(f"{e}")        

    question = "What file would you like to load?\n"
    file_choice = console.input(f"{question}")
    if file_choice.isnumeric():
        file_to_load = files[int(file_choice) - 1]
        #check output directory exists
        return file_to_load
    else:
        raise ValueError("Please restart and select an integer of the file you'd like to import")

@log_time
def main():
    #target data folder goes here.
    fp = Path.cwd() / "src/rad_ecg/data/datasets/sharc_fem/converted"
    
    #Check file existence, load mini detection scheme.  
    if not fp.exists():
        logger.warning(f"Warning: File {fp} not found.")
    else:
        selected = load_choices(fp)
        fp_save = Path(selected).parent / (Path(selected).stem + "_results.json")
        rad = MiniRAD(selected)
        if fp_save.exists():
            rad.load_results(fp_save)
        else:
            rad.run_stump()
            rad.save_results()
        rad.plot_discords(top_n=8)

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