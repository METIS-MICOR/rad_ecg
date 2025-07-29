import utils #from rad_ecg.scripts 
import numpy as np
import wfdb
import logging
import json
import os
import subprocess
from google.cloud import storage
from support import logger, console
from pathlib import PurePath, Path
from rich import print
from rich.tree import Tree
from rich.markup import escape
from pathlib import Path, PurePath

################################# Custom INIT / Loading functions ############################################
#FUNCTION Custom init
def init(source:str):
    """Custom init

    Args:
        source (str): Where the init was called from (test/main)

    Returns:
        tuple (ecg_data, wave, fs): Returns the data containers to run software. 
            - ecg_data = dict of various measures
            - wave = EKG to analyze in question
            - fs = Sampling Frequency of the wave
            - configs = global configuration settings
    """     
    #Load config variables
    global configs
    configs = load_config()
    datafile = launch_tui(configs)
    ecg_data, wave, fs = load_structures(source, datafile)
    
    return ecg_data, wave, fs, configs

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
def load_signal_data(hea_path:str):
    #Load signal data 
    record = wfdb.rdrecord(
        hea_path,
        sampfrom=0,
        sampto=None,
        channels=[0]
    )
    return record

#FUNCTION Load Chart Data
def load_chart_data(configs:dict, datafile:Path, logger:logging):
    inputdirs = os.listdir(configs["data_path"])
    if datafile.name in inputdirs:
        idx = inputdirs.index(datafile.name)
        input_path = PurePath(Path(configs["data_path"]), Path(datafile.name), Path(inputdirs[idx]))
        record = load_signal_data(input_path)
    else: 
        logger.warning(f"Input data for {datafile.name} not found")
        logger.warning("Make sure base waveform data is stored in the data/input folder")
        exit()

    #ECG data
    wave = record.p_signal
    
    #Frequency
    fs = record.fs
    
    #folder return 
    folderp = os.listdir(PurePath(Path(configs["save_path"], Path(datafile.name))))
    return wave, fs, folderp

#FUNCTION Load Structures
def load_structures(source:str, datafile:Path):
    if source == "test":
        #Set paths and global variables
        fpath = "./src/rad_ecg/data/sample/scipy_sample.csv"
        fs = 360
        wave = np.loadtxt(
            fpath,
            dtype=np.float64
        )
        wave = wave.reshape(-1, 1)
        windowsi = 9

    elif source == "__main__":
        #Load all possibles in the input dir or gcp
        #Check output folder for existence
        test_sp = os.path.join(configs["save_path"], datafile.name)
        if os.path.exists(test_sp):
            logger.critical(f"{datafile.name} output folder already exists")
            logger.critical("Do you want to overwrite results?")
            overwrite = input("(y/n)?")
            if overwrite.lower() == "n":
                exit()
        else:
            os.mkdir(test_sp)
            logger.info(f"folder created @ {test_sp} ")

        if configs["gcp_bucket"]:
            #Test for endpoint in gcp bucket
            test_sp = os.path.join(configs["bucket_name"], "results", datafile.name)
            passed = test_endpoint(test_sp)
            if passed:
                logger.warning(f"{datafile.name} path exists in gcp")
                logger.warning("Do you want to overwrite results?")
                overwrite = input("(y/n)?")
                if overwrite.lower() == "n":
                    logger.warning("Shutting down program")
                    exit()
            else:
                created = create_endpoint(test_sp)
                if created:
                    logger.info(f"folder created @ {test_sp}")
                else:
                    logger.warning(f"Error {created}")
                    exit()

        configs["cam"] = os.path.join(datafile, datafile.name)
        configs["cam_name"] = datafile.name
        record = load_signal_data(configs["cam"])
        
        #ECG data
        wave = record.p_signal
        
        #Frequency
        fs = record.fs
        configs["samp_freq"] = fs
        
        #Size of timing segment window
        windowsi = 10


    else:
        logger.CRITICAL("New runtime environment detected outside of normal operating params.\nPlease rerun with appropriate configuration")
        exit()

    #Divide waveform into even segments (Leave off the last 1000 or so, usually unreliable)
    wave_sections = utils.segment_ECG(wave, fs, windowsize=windowsi)[:5000]
    #BUG - Getting some errors in the start recently.  lastkeys[- not being estimated on line 1359]
    #Setting mixed datatypes (structured array) for ecg_data['section_info']
    wave_sect_dtype = [
        ('wave_section', 'i4'),
        ('start_point' , 'i4'),
        ('end_point'   , 'i4'),
        ('valid'       , 'i4'),
        ('fail_reason' , str, 16),
        ('Avg_HR'      , 'f4'), 
        ('SDNN'        , 'f4') ,
        ('min_HR_diff' , 'f4'), 
        ('max_HR_diff' , 'f4'), 
        ('RMSSD'       , 'f4'),
        ('NN50'        , 'f4'),
        ('PNN50'       , 'f4'), 
        ('isoelectric' , 'f4')
    ]

    #Base data container keys
    ecg_data = {
        'peaks': np.zeros(shape=(0, 2), dtype=np.int32),
        'rolling_med': np.zeros(shape=(wave.shape[0]), dtype=np.float32),
        'section_info': np.zeros(shape=(wave_sections.shape[0]), dtype=wave_sect_dtype),
        'interior_peaks': np.zeros(shape=(0, 16), dtype=np.int32)
    }

    ecg_data['section_info']['wave_section'] = np.arange(0, wave_sections.shape[0], 1)
    ecg_data['section_info']['start_point'] = wave_sections[:,0]
    ecg_data['section_info']['end_point'] = wave_sections[:,1]
    ecg_data['section_info']['valid'] = wave_sections[:,2]

    del wave_sections
    
    return ecg_data, wave, fs

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

################################# GCP Client Funcs ############################################

def authenticate_with_gcs(credentials_path:str):
    """Set up Google Cloud authentication

    Args:
        credentials_path (_type_): _description_
    """ 
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

def test_endpoint(test_sp:str):
    try:
        command = ["gsutil", "ls", f"gs://{test_sp}"]
        runcommand = subprocess.run(command, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        if "One or more URLs matched no objects" in e.stderr:
            return False
        else:
            raise e
        
def create_endpoint(test_sp:str):
    try:
        #TODO - check this the next time you run a full cam
        create_command = ["gsutil", "touch", f"gs://{test_sp}/test.txt"]
        subprocess.run(create_command, capture_output=True, text=True, check=True)

        # Optionally remove the dummy file:
        # remove_command = ["gsutil", "rm", f"gs://{test_sp}/test.txt"]
        # subprocess.run(remove_command, capture_output=True, text=True, check=True)

        return True

    except subprocess.CalledProcessError as e:
        if "One or more URLs matched no objects" in e.stderr:
            return False
        else:
            return e


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
        files = walk_directory(Path(directory), tree)
        print(tree)
        
    question ="What file would you like to load?\n"
    file_choice = console.input(f"{question}")
    if file_choice.isnumeric():
        file_to_load = files[int(file_choice) - 1]
        #check output directory exists
        return file_to_load
    else:
        raise ValueError("Please restart and select an integer of the file you'd like to import")
    
#FUNCTION Walk Directory
def walk_directory(directory: Path, tree: Tree) -> None:
    """Build a Tree with directory contents.
    Source Code: https://github.com/Textualize/rich/blob/master/examples/tree.py

    """
    # Sort dirs first then by filename
    paths = sorted(
        Path(directory).iterdir(),
        key=lambda path: (path.is_file(), path.name.lower()),
    )
    idx = 1
    for path in paths:
        # Remove hidden files
        if path.name.startswith("."):
            continue
        # Just list the CAM folders
        if path.is_dir():
            style = "dim" if path.name.startswith("__") else ""
            file_size = getfoldersize(path)
            branch = tree.add(
                f"[bold green]{idx} [/bold green][bold magenta]:open_file_folder: [link file://{path}]{escape(path.name)}[/bold magenta] [bold blue]{file_size}[/bold blue]",
                style=style,
                guide_style=style,
            )
            
            # walk_directory(path, branch)
        # else:
        #     text_filename = Text(path.name, "green")
        #     text_filename.highlight_regex(r"\..*$", "bold red")
        #     text_filename.stylize(f"link file://{path}")
        #     file_size = path.stat().st_size
        #     text_filename.append(f" ({decimal(file_size)})", "blue")
        #     if path.suffix == "py":
        #         icon = "🐍 "
        #     elif path.suffix == ".hea":
        #         icon = "🤯  "
        #     elif path.suffix == ".dat":
        #         icon = "🔫 "
        #     elif path.suffix == ".mib":
        #         icon = "👽 "
        #     elif path.suffix == ".zip":
        #         icon = "🤐 "
        #     else:
        #         icon = "📄 "
        #     tree.add(Text(f'{idx} ', "blue") + Text(icon) + text_filename)
        
        idx += 1    
    return paths

