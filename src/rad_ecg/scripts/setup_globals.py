import utils #from rad_ecg.scripts 
import numpy as np
import logging
import json
import os
import subprocess
from google.cloud import storage
from support import logger, console
from pathlib import PurePath, Path
from rich import print as pprint
from rich.tree import Tree
from rich.text import Text
from rich.filesize import decimal
from rich.markup import escape

################################# Global Imports ############################################
#Section Metrics
SECTION_DTYPES = [
    ('wave_section', 'i4'),    # 0 
    ('start_point' , 'i4'),    # 1 
    ('end_point'   , 'i4'),    # 2 
    ('valid'       , 'i4'),    # 3 
    ('fail_reason' , str, 16), # 4 
    ('isoelectric' , 'f4'),    # 5 
    ('kurtosis'    , 'f4'),    # 6 
    ('hjorth'      , 'f4'),    # 7 
    ('spectral'    , 'f4'),    # 8 
    ('bad_b_rat'   , 'f4'),    # 9
    ('wdist'       , 'f4'),    # 10
    # ('power_ratio' , 'f4'),    # 11
    ('spec_entropy', 'f4'),    # 11
    ('HR'          , 'f4'),    # 11
    ('SDNN'        , 'f4'),    # 12
    ('RMSSD'       , 'f4'),    # 13
    ('NN50'        , 'f4'),    # 14
    ('PNN50'       , 'f4'),    # 15
    ('PR'          , 'f4'),    # 16
    ('QRS'         , 'f4'),    # 17
    ('ST'          , 'f4'),    # 18
    ('QT'          , 'f4'),    # 19
    ('QTc'         , 'f4'),    # 20
    ('QTVI'        , 'f4'),    # 21
    ('TpTe'        , 'f4'),    # 22
]

#Interior Peaks
PEAK_DTYPES = [
    ('p_peak'      , 'i4'),   # 0
    ('q_peak'      , 'i4'),   # 1
    ('r_peak'      , 'i4'),   # 2
    ('s_peak'      , 'i4'),   # 3
    ('t_peak'      , 'i4'),   # 4
    ('valid_qrs'   , 'bool'), # 5
    ('p_peak_a'    , 'f4'),   # 6
    ('q_peak_a'    , 'f4'),   # 7
    ('r_peak_a'    , 'f4'),   # 8
    ('s_peak_a'    , 'f4'),   # 9
    ('t_peak_a'    , 'f4'),   # 10
    ('p_onset'     , 'i4'),   # 11
    ('q_onset'     , 'i4'),   # 12
    ('j_point'     , 'i4'),   # 13
    ('t_onset'     , 'i4'),   # 14
    ('t_offset'    , 'i4'),   # 15
    ('u_wave'      , 'bool'), # 16
    ('PR'          , 'i4'),   # 17
    ('QRS'         , 'i4'),   # 18
    ('ST'          , 'i4'),   # 19
    ('QT'          , 'i4'),   # 20
    ('QTc'         , 'i4'),   # 21
    ('TpTe'        , 'f4'),   # 22
]

################################# Custom INIT / Loading functions ############################################
#FUNCTION Load Config
def load_config()->json:
    """Load global variable configs

    Returns:
        config_data: Loads configuation data
    """
    with open("./src/rad_ecg/config.json", "r") as f:
        config_data = json.loads(f.read())
    return config_data

#FUNCTION Load Signal Data
def load_signal_data(file_path:str):
    record = None
    header = None
    #Load signal data
    if "." in file_path:
        file_type = file_path[file_path.rindex(".") + 1:].lower()
    else:
        file_type = "hea"
    
    try:
        match file_type:
            case "ebm": 
                from lib_ebm.pyebmreader import ebmreader
                record, header = ebmreader(
                    filepath = file_path,
                    onlyheader = False
                )
                record = record[0]
            case "ecg": 
                pass
            case "h12": 
                pass
            case "hea":
                from wfdb import rdrecord
                # cam = file_path.split("/")[-1]
                # file_path = file_path[:file_path.rindex(cam)]
                record = rdrecord(
                    file_path,
                    sampfrom=0,
                    sampto=None,
                    channels=[0]
                )
        return record, header

    except Exception as e:
        logger.critical(f"Unable to load file. Error {e}")

#FUNCTION Load Chart Data
def load_chart_data(configs:dict, datafile:Path, logger:logging):
    inputdirs = os.listdir(configs["data_path"])
    if datafile.name in inputdirs:
        idx = inputdirs.index(datafile.name)
        input_path = PurePath(Path(configs["data_path"]), Path(datafile.name), Path(inputdirs[idx]))
        record, header = load_signal_data(str(input_path))
    else: 
        logger.warning(f"Input data for {datafile.name} not found")
        logger.warning("Make sure base waveform data is stored in the data/inputdata folder")
        exit()

    #ECG data
    wave = record.p_signal
    
    #Frequency
    fs = record.fs
    
    #folder return 
    folderp = os.listdir(PurePath(Path(configs["save_path"], Path(datafile.name))))
    return wave, fs, folderp

################################# Size Funcs ############################################
def sizeofobject(folder)->str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(folder) < 1024:
            return f"{folder:4.1f} {unit}"
        folder /= 1024.0
    return f"{folder:.1f} PB"

def getfoldersize(folder:Path):
    fsize = 0
    for root, dirs, files in os.walk(folder):
        for f in files:
            fp = os.path.join(folder,f)
            fsize += os.stat(fp).st_size

    return sizeofobject(fsize)


################################# TUI Funcs ############################################
#FUNCTION Launch TUI
def launch_tui(configs:dict):
    try:
        if configs["slider"] | configs["run_anomalyd"]:
            directory = PurePath(Path.cwd(), Path("./src/rad_ecg/data/output"))
        else:
            directory = PurePath(Path.cwd(), Path(configs["data_path"]))

    except IndexError:
        logger.info("[b]Usage:[/] python tree.py <DIRECTORY>")
    else:
        tree = Tree(
            f":open_file_folder: [link file://{directory}]{directory}",
            guide_style="bold bright_blue",
        )
        walk_directory(Path(directory), tree)
        pprint(tree)
        
    question = "What file would you like to load?\n"
    file_choice = console.input(f"{question}")
    if file_choice.isnumeric():
        files = sorted(
            Path(directory).iterdir(),
            key=lambda path: (path.is_file(), path.name.lower()),
        )
        file_to_load = files[int(file_choice)]
        #check output directory exists
        return file_to_load
    else:
        raise ValueError("Please restart and select an integer of the file you'd like to import")

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

#FUNCTION Walk Directory
def load_choices(fp:str, batch_process:bool=False):
    """Loads whatever file you pick

    Args:
        fp (str): file path
        batch_process (bool): whether we're loading all the files

    Raises:
        ValueError: If you don't give it a numeric selection in single select, it errors

    Returns:
        files (list|str): File(s) we want to load
    """    
    try:
        tree = Tree(f":open_file_folder: [link file://{fp}]{fp}", guide_style="bold bright_blue")
        walk_directory(Path(fp), tree)
        pprint(tree)
    except Exception as e:
        logger.warning(f"{e}")
    
    # Build the Tree
    paths = sorted(
        Path(fp).iterdir(),
        key=lambda path: (path.is_file(), path.name.lower())
    )
    # Filter out hidden files just like the Tree does
    valid_paths = [p for p in paths if not p.name.startswith(".")]
    
    if not batch_process:
        question = "What file would you like to load?\n"
        file_choice = console.input(f"{question}")
        if file_choice.isnumeric():
            choice_idx = int(file_choice)
            if 0 <= choice_idx < len(valid_paths):
                # If we are looking at the root inputdata folder, return the folder.
                # If we are inside a specific folder, return the file. 
                return valid_paths[choice_idx]
            else:
                raise IndexError(f"Choice {choice_idx} is out of bounds for the available files.")
        else:
            raise ValueError("Invalid choice. Please enter a number.")
    else:
        # --- BATCH PROCESS LOGIC ---
        # If the folder is full of subdirectories, assume the folders are the targets
        dirs = [p for p in valid_paths if p.is_dir()]
        if dirs:
            return dirs
            
        # If the folder has files, filter smartly
        files = [p for p in valid_paths if p.is_file()]
        
        # If WFDB files exist, only return the .hea headers so it doesn't double-count the .dat files
        heas = [p for p in files if p.suffix.lower() == '.hea']
        if heas:
            return heas
            
        # Otherwise, return all valid files (e.g., .ebm files)
        return files

################################# GCP Client Funcs ############################################
# gcsfuse does all this now.  So easy to use!!! 
# def authenticate_with_gcs(credentials_path:str):
#     """Set up Google Cloud authentication

#     Args:
#         credentials_path (_type_): _description_
#     """ 
#     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# def test_endpoint(test_sp:str):
#     try:
#         command = ["gsutil", "ls", f"gs://{test_sp}"]
#         runcommand = subprocess.run(command, capture_output=True, text=True, check=True)
#         return True
#     except subprocess.CalledProcessError as e:
#         if "One or more URLs matched no objects" in e.stderr:
#             return False
#         else:
#             raise e
        
# def create_endpoint(test_sp:str):
#     try:
#         #TODO - check this the next time you run a full cam
#         create_command = ["gsutil", "touch", f"gs://{test_sp}/test.txt"]
#         subprocess.run(create_command, capture_output=True, text=True, check=True)

#         # Optionally remove the dummy file:
#         # remove_command = ["gsutil", "rm", f"gs://{test_sp}/test.txt"]
#         # subprocess.run(remove_command, capture_output=True, text=True, check=True)

#         return True

#     except subprocess.CalledProcessError as e:
#         if "One or more URLs matched no objects" in e.stderr:
#             return False
#         else:
#             return e
