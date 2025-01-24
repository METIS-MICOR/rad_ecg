import utils #from rad_ecg.scripts 
import numpy as np
import wfdb
import logging
import json
import os
from os.path import exists
from google.cloud import storage
from support import logger, console
from pathlib import PurePath, Path
from rich import print
from rich.logging import RichHandler
from rich.table import Table
from rich.console import Console
from rich.tree import Tree
from rich.filesize import decimal
from rich.markup import escape
from rich.text import Text
from rich.table import Table
from pathlib import Path, PurePath

################################# Custom INIT / Loading functions ############################################
#FUNCTION Custom init
def init(source:str, logger:logging):
    """Custom init

    Args:
        source (str): Where the init was called from (test/main)
        logger (logging): For all things logging

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
    ecg_data, wave, fs = load_structures(source, logger)
    
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

################################# GCS Client Funcs ############################################

def authenticate_with_gcs(credentials_path:str):
    """Set up Google Cloud authentication

    Args:
        credentials_path (_type_): _description_
    """ 
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

################################ Downloading Funcs ############################################
#FUNCTION download individual ecg from gcs
def download_ecg_from_gcs(bucket_name:str, save_path:str, logger:logging):
    """Download function for GCS.  It will download all files within a bucket to the inputdata
    folder for ingestion.  Note: Downloading via this method is slower, sometimes its easier
    to copy the data over to the inputdata directory.

    Args:
        bucket_name (str): GCS bucket target
        save_path (str): full save path on VM
        logger (logging): To log things of course

    Returns:
        filenames (list): List of the header files found within the target data directory
    """

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Download all related files for the record
    blobs = list(bucket.list_blobs())
    gcp_folders = {}
    file_names = []
    unprocessed = []

    #Make a dict of the folders and their contents
    for blob in blobs:
        folder = os.path.dirname(blob.name)
        if folder not in gcp_folders:
            gcp_folders[folder] = []
        if not blob.name.endswith("/"):
            gcp_folders[folder].append(blob)
        if blob.name.startswith("unprocessed"):
            unprocessed.append(blob.name)
    
    #Generate list of all unprocessed cams
    unprocessed = list(set([name.split("/")[1] for name in unprocessed]))
    unprocessed = sorted([x for x in unprocessed if len(x) > 1])
    logger.warning("please select a cam to download")
    for idx, file in enumerate(unprocessed):
        logger.warning(f"{idx} : {file}")
    selection = input("Please select the index of desired CAM")
    file_selected = unprocessed[int(selection)]
    keys = list(gcp_folders.keys())
    for key in keys:
        if file_selected not in key:
            gcp_folders.pop(key)
    
    input_path = save_path.replace("output" , "inputdata")

    #TODO Add if secondary confirm if the cam has already been processed.  aka do folders exist and 
    #Process each folder
    for folder, files in gcp_folders.items():
        # if its a results folder, skip it
        if "results" in folder:
            logger.info("Skipping results folder")
            continue
        # If its the unprocessed root folder skip
        if folder == "unprocessed":
            continue
        #If there aren't any files in the folder, 
        if not files:
            logger.info(f"Skipping empty folder {folder}")
            continue

        logger.info(f"Processing folder {folder}")
        #Process each file
        for blob in files:
            #Grab the name
            cam_name = blob.name[blob.name.rindex("/")+1:]       
            cam_f = cam_name.split(".")[0]

            #Generate filename save path
            input_file_p = "/".join([input_path, cam_f, cam_name])
            input_fold = os.path.join(input_path + "/" + cam_f + "/")
            output_fold = os.path.join(save_path + "/" + cam_f + "/")

            #Check if input folder exists.  Create if not
            if os.path.exists(input_fold):
                logger.info(f"{cam_name} Input folder already exists")
            else:
                os.mkdir(input_fold)
                logger.info(f"folder created @ {input_fold} ")

            #Check to see if output folder exists. Create if not
            if os.path.exists(output_fold):
                logger.info(f"{cam_name} input folder already exists")
            else:
                os.mkdir(output_fold)
                logger.info(f"folder created @ {output_fold} ")

            #Check to see if input file exists. Download if not
            # source_f = "/".join([configs["data_path"], cam_f, cam_name])
            if os.path.exists(input_file_p):
                logger.info(f"file {input_file_p} exists already")
            else:
                logger.info(f"Downloading {blob.name} to\n{input_file_p}")
                blob.download_to_filename(input_file_p)
                logger.info("ECG download complete.")
            
            if blob.name.endswith(".hea"):
                file_names.append(input_file_p)
            

    return file_names

#FUNCTION Choose CAM
def choose_cam(configs:dict, logger:logging)->list:
    """Choosing your cam to evaluate.  Logic is as follows.  If the inputdata directory on the install / or git clone is empty.  
    Check the configs path setting.  If that value is None, exit program as no data can be found.

    Args:
        logger (logging): Logger

    Raises:
        ValueError: If neither directory has any header files, exit the program. 

    Returns:
        head_files (list): Returns a list 
    """
    #Check if the inputdata directory, load config_dir, and GCP flag
    empty_inputdir = not os.listdir("./src/rad_ecg/data/inputdata")
    config_dir = configs["data_path"]
    gcp = configs["gcp_bucket"]

    #If data is in the local inputdata folder with no call to GCP
    if not empty_inputdir and not gcp:
        head_files = get_records('inputdata')
    #If there is a config directory target and gcp bucket is true
    elif gcp:
        head_files = download_ecg_from_gcs(configs["bucket_name"], configs["save_path"], logger)
    # If the config directory target is valid
    elif config_dir:
        head_files = get_records(config_dir)
    else:
        logger.critical("No data found in inputdata dir or configs data_path")
        exit()
        
    #Inquire which file you'd like to run
    logger.warning("Please select the index of the CAM you would like to import.\nie: 1, 2, 3...")
    if not gcp:
        header_chosen = input("Please choose a CAM")
    else:
        header_chosen = "0"

    if not header_chosen.isnumeric():
        logger.critical(f'Incorrect file entered, program terminating')
        exit()
    elif int(header_chosen) not in range(len(head_files)):
        logger.critical(f'File not in selection range.  program terminating')
        exit()

    else:
        header_chosen = int(header_chosen)
        head = head_files[header_chosen]
        name = head.split(".")[-2].split("/")[-1]
        if name in configs["save_path"]:
            logger.warning(f'CAM {name} chosen')
        else:
            logger.critical(f"CAM {name} file not in output dir")
            exit()

    return head_files, header_chosen

#FUNCTION Load Header Files
def get_records(folder:str)->list:
    """Pulls the file info out of the data directory for file paths

    Args:
        p (str): [path to root data directory]

    Returns:
        dat_files (list): [List of dat names]
        mib_files (list): [List of mib names]
        head_files (list): [List of header names]
    """

    # There are 3 files for each record
    #.dat = ECG data
    #.hea = header file (file info)
    #.mib = annotation file (beat annotations)
    
    #Get base directory
    # p = os.path.normpath(os.getcwd() + os.sep + os.pardir)

    #First check that the path isn't the standard inputdata directory
    if folder != "inputdata":
        base_dir = folder
    else:
        p = os.getcwd()
        base_dir = os.path.join(p, "src" , "rad_ecg", "data", folder)
    
    dat_files, mib_files, head_files = [], [], []
    for subject in os.scandir(base_dir):
        if subject.is_dir():
            for file_idx in os.scandir(subject.path):
                if file_idx.name.endswith('.dat') and file_idx.is_file():
                    dat_files.append(file_idx.path)

                elif file_idx.name.endswith('.mib') and file_idx.is_file():
                    mib_files.append(file_idx.path)

                elif file_idx.name.endswith('.hea') and file_idx.is_file():
                    head_files.append(file_idx.path)
    
    #leaving these here in case needed later #dat_files, mib_files
    return head_files

#FUNCTION Load Chart Data
def load_chartdata(configs:dict, datafile:Path, logger:logging):
    # head_files, header_chosen = choose_cam(configs, logger)
    inputdirs = os.listdir(configs["data_path"])
    if datafile.name in inputdirs:
        idx = inputdirs.index(datafile.name)
        input_path = configs["data_path"] + "//" + datafile.name + "//" + inputdirs[idx]
        record = load_signal_data(input_path)
    else:
        logger.warning(f"Input data for {datafile.name} not found")
        logger.warning("Make sure base waveform data is stored correctly")
        exit()

    #ECG data
    wave = record.p_signal
    
    #Frequency
    fs = record.fs

    return wave, fs, datafile.name, os.listdir(f"{configs['save_path']}\{datafile.name}")

#FUNCTION Walk Directory
def walk_directory(directory: Path, tree: Tree) -> None:
    """Recursively build a Tree with directory contents.
    Pulled from here. https://github.com/Textualize/rich/blob/master/examples/tree.py

    """
    # Sort dirs first then by filename
    paths = sorted(
        Path(directory).iterdir(),
        key=lambda path: (path.is_file(), path.name.lower()),
    )
    idx = 1
    for path in paths:
        #Normally this routine would enumerate and list individual files and
        # folders recursively.  Instead i just want it to list the folders to
        # select that way.  Not by files as most ecgs in the wfdb format come in
        # 3 files. 
        # Remove hidden files
        if path.name.startswith("."):
            continue
        if path.is_dir():
            style = "dim" if path.name.startswith("__") else ""
            branch = tree.add(
                f"[bold green]{idx} [/bold green][bold magenta]:open_file_folder: [link file://{path}]{escape(path.name)}",
                style=style,
                guide_style=style,
            )
            
            # walk_directory(path, branch)
        else:
            text_filename = Text(path.name, "green")
            text_filename.highlight_regex(r"\..*$", "bold red")
            text_filename.stylize(f"link file://{path}")
            file_size = path.stat().st_size
            text_filename.append(f" ({decimal(file_size)})", "blue")
            if path.suffix == "py":
                icon = "🐍 "
            elif path.suffix == ".hea":
                icon = "🤯  "
            elif path.suffix == ".dat":
                icon = "🔫 "
            elif path.suffix == ".mib":
                icon = "👽 "
            elif path.suffix == ".zip":
                icon = "🤐 "
            else:
                icon = "📄 "
            tree.add(Text(f'{idx} ', "blue") + Text(icon) + text_filename)
        
        idx += 1    
    return paths

def launch_tui(configs:dict):
    try:
        if configs["slider"]:
            directory = PurePath(Path.cwd(), Path("./src/rad_ecg/data/output"))

        elif configs["gcp_bucket"]:
            #TODO - Need a way to read the gcp output bucket here. 
            directory = PurePath(Path.cwd(), Path(configs["bucket_name"]))
        else:
            directory = PurePath(Path.cwd(), Path("./src/rad_ecg/data/input"))

    except IndexError:
        logger.info("[b]Usage:[/] python tree.py <DIRECTORY>")
    else:
        tree = Tree(
            f":open_file_folder: [link file://{directory}]{directory}",
            guide_style="bold bright_blue",
        )
        files = walk_directory(Path(directory), tree)
        #?Try a console print here too
        print(tree)
        
    question ="What file would you like to load?\n"
    file_choice = console.input(f"{question}")
    if file_choice.isnumeric():
        file_to_load = files[int(file_choice) - 1]
        return file_to_load
    else:
        raise ValueError("Please restart and select an integer of the file you'd like to import")

#FUNCTION Load Structures
def load_structures(source:str, logger:logging):
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
        #Load             
        head_files, header_chosen = choose_cam(configs, logger)
        configs["cam"] = head_files[header_chosen]
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
    wave_sections = utils.segment_ECG(wave, fs, windowsize=windowsi)[:-1000]
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
        ('PNN50'       , 'f4')
    ]

    #Base data container keys
    ecg_data = {
        'peaks': np.zeros(shape=(0, 2), dtype=np.int32),
        'rolling_med': np.zeros(shape=(wave.shape[0]), dtype=np.float32),
        'section_info': np.zeros(shape=(wave_sections.shape[0]), dtype=wave_sect_dtype),
        'interior_peaks': np.zeros(shape=(0, 15), dtype=np.int32)
    }

    ecg_data['section_info']['wave_section'] = np.arange(0, wave_sections.shape[0], 1)
    ecg_data['section_info']['start_point'] = wave_sections[:,0]
    ecg_data['section_info']['end_point'] = wave_sections[:,1]
    ecg_data['section_info']['valid'] = wave_sections[:,2]

    del wave_sections
    
    return ecg_data, wave, fs
