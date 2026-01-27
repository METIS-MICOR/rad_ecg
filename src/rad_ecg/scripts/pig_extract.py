from scipy.signal import find_peaks
from pathlib import Path, PurePath
from numba import cuda
import numpy as np
from support import logger, console, log_time
from setup_globals import walk_directory
from utils import segment_ECG
from rich import print
from rich.tree import Tree
from rich.text import Text
from rich.filesize import decimal
from rich.markup import escape

DTYPES = []

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

class miniRAD():
    def __init__(self, npz_path):
        # 1. Load Data
        self.loader = SignalDataLoader(npz_path)
        self.full_data = self.loader.full_data
        self.channels = self.loader.channels
        self.dtypes = DTYPES
        # 2. Choose Stream
        
        # 3. section signal
        self.sections = self.section_ecgs()

    def section_ecgs(self):
        pass

    def run_stump(self):
        pass
        #Use rich progbar as a measure for the stump routine. 
        #then have it launch mpl

    def STFT(self):
        #For this, lets update the distibution matching to Wasserbein
        pass

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
        
def main():
    #target data folder goes here.
    fp = Path.cwd() / "src/rad_ecg/data/datasets/sharc_fem/converted" #converted/SHARC2_60653_4Hr_Aug-19-25.npz"
      
    #Check file existence, load mini detection scheme.  
    if not fp.exists():
        logger.warning(f"Warning: File {fp} not found.")
    else:
        selected = load_choices(fp)
        rad = miniRAD(str(selected))
        rad.run_stump()
        #rad.plot_discords()

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