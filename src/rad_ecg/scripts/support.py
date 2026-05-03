#################################  main libraries #######################################
import os
import time
import json
import shutil
import logging
import datetime
import subprocess
import numpy as np
import multiprocessing
from pathlib import Path
from collections import Counter

#################################  rich imports #######################################
from rich.progress import (
    Progress,
    BarColumn,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn)
from rich.console import Console
from rich.logging import RichHandler
from rich.terminal_theme import TerminalTheme
from lib_ebm.pyebmreader import ebmreader

################################# Logger functions ####################################
#FUNCTION Logging Futures
def get_file_handler(log_dir:Path)->logging.FileHandler:
    """Assigns the saved file logger format and location to be saved

    Args:
        log_dir (Path): Path to where you want the log saved

    Returns:
        filehandler(handler): This will handle the logger's format and file management
    """	
    log_format = "%(asctime)s|%(levelname)-8s|%(lineno)-4d|%(funcName)-23s|%(message)-100s|" 
                 #f"%(asctime)s - [%(levelname)s] - (%(funcName)s(%(lineno)d)) - %(message)s"
    # current_date = time.strftime("%m_%d_%Y")
    file_handler = logging.FileHandler(log_dir)
    file_handler.setFormatter(logging.Formatter(log_format, "%m-%d-%Y %H:%M:%S"))
    return file_handler

def get_rich_handler(console:Console)-> RichHandler:
    """Assigns the rich format that prints out to your terminal

    Args:
        console (Console): Reference to your terminal

    Returns:
        rh(RichHandler): This will format your terminal output
    """
    rich_format = "|%(funcName)-23s|%(message)s"
    rh = RichHandler(console=console)
    rh.setFormatter(logging.Formatter(rich_format))
    return rh

def get_logger(console:Console, log_dir:Path)->logging.Logger:
    """Loads logger instance.  When given a path and access to the terminal output.  The logger will save a log of all records, as well as print it out to your terminal. Propogate set to False assigns all captured log messages to both handlers.

    Args:
        log_dir (Path): Path you want the logs saved
        console (Console): Reference to your terminal

    Returns:
        logger: Returns custom logger object.  Info level reporting with a file handler and rich handler to properly terminal print
    """	
    #Load logger and set basic level
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #Load file handler for how to format the log file.
    file_handler = get_file_handler(log_dir)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    #Load rich handler for how to display the log in the console
    rich_handler = get_rich_handler(console)
    rich_handler.setLevel(logging.CRITICAL)
    logger.addHandler(rich_handler)
    logger.propagate = False
    return logger

#FUNCTION timer
################################# Timing Funcs ####################################
def log_time(fn):
    """Decorator timing function.  Accepts any function and returns a logging
    statement with the amount of time it took to run. DJ, I use this code everywhere still.  Thank you bud!

    Args:
        fn (function): Input function you want to time
    """	
    def inner(*args, **kwargs):
        tnow = time.time()
        out = fn(*args, **kwargs)
        te = time.time()
        took = round(te - tnow, 2)
        if took <= 60:
            logger.info(f"{fn.__name__} ran in {took:.3f}s")
        elif took <= 3600:
            logger.info(f"{fn.__name__} ran in {(took)/60:.3f}m")		
        else:
            logger.info(f"{fn.__name__} ran in {(took)/3600:.3f}h")
        return out
    return inner

#FUNCTION get time
def get_time():
    """Function for getting current time

    Returns:
        t_adjusted (str): String of current time
    """
    current_t_s = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    current_t = datetime.datetime.strptime(current_t_s, "%m-%d-%Y-%H-%M-%S")
    return current_t

########################## Global Variables to return ##########################################
DATE_JSON = get_time().strftime("%m-%d-%Y_%H-%M-%S")
console = Console(color_system="auto", stderr=True, record=True)
# logger = get_logger(console, log_dir=f"src/rad_ecg/data/logs/{DATE_JSON}.log") 
export_theme = TerminalTheme(
    (0, 0, 0),         # Background: Pure Black
    (255, 255, 255),   # Foreground: White text
    # The standard ANSI palette (black, red, green, yellow, blue, magenta, cyan, white)
    [
        (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
        (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192)
    ],
    # The 'bright' ANSI palette
    [
        (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0),
        (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255)
    ]
)

#####
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.propagate = False

# Prevent duplicate handlers if support.py is imported multiple times
if not logger.handlers:
    if multiprocessing.current_process().name == "MainProcess":
        # MAIN PROCESS: Console ONLY. Restore your custom CRITICAL level!
        rich_handler = get_rich_handler(console)
        rich_handler.setLevel(logging.CRITICAL) 
        logger.addHandler(rich_handler)
    else:
        # WORKER PROCESS: Console ONLY. Workers keep their default INFO level.
        logger.addHandler(get_rich_handler(console))

# Ensure DATE_JSON is globally available
if "DATE_JSON" not in locals():
    DATE_JSON = get_time().strftime("%m-%d-%Y_%H-%M-%S")
#######
#old log load
# # Determine if we are in the Main Process or a Worker Process
# if multiprocessing.current_process().name == "MainProcess":
#     # MAIN PROCESS: Calculate time and create the log file
#     logger = get_logger(console, log_dir=f"src/rad_ecg/data/logs/{DATE_JSON}.log")
# else:
#     # WORKER PROCESS: Do NOT create a file. Just attach the console handler.
#     # This prevents workers from generating new log files with new timestamps.
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#     logger.addHandler(get_rich_handler(console))
    
#     # If you need DATE_JSON defined to avoid NameErrors in workers (though unlikely used there):
#     DATE_JSON = "WORKER_PROCESS"

################################# Rich Spinner Control ####################################
#FUNCTION Progress bar
def mainspinner(console:Console, totalstops:int):
    """Load a rich Progress bar for alerting you to the progress of the algorithm

    Args:
        console (Console): reference to the terminal
        totalstops (int): Amount of categories searched

    Returns:
        prog_bar (Progress): Progress bar for tracking overall progress
        jobtask (int): mainjob id for ecg extraction
    """
    prog_bar = Progress(
        TextColumn("{task.description}"),
        SpinnerColumn("dots"),
        BarColumn(),
        TextColumn("*"),
        "time elapsed:",
        TextColumn("*"),
        TimeElapsedColumn(),
        TextColumn("*"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        transient=True,
        console=console,
        refresh_per_second=6,
        # redirect_stdout=False
    )
    jobtask = prog_bar.add_task("[green]Detecting peaks", total=totalstops + 1)
    return prog_bar, jobtask

def add_spin_subt(prog:Progress, msg:str, howmanysleeps:int):
    """Adds a secondary job to the main progress bar

    Args:
        prog (Progress): Main progress bar
        msg (str): Message to update secondary progress bar
        howmanysleeps (int): How long to let the timer sleep
    """
    #Add secondary task to progbar
    liljob = prog.add_task(f"[magenta]{msg}", total = howmanysleeps)
    #Run job for random sleeps
    for _ in range(howmanysleeps):
        time.sleep(1)
        prog.update(liljob, advance=1)
    #Hide secondary progress bar
    prog.update(liljob, visible=False)

################################# Saving Funcs ####################################
#CLASS Numpy encoder
class NumpyArrayEncoder(json.JSONEncoder):
    """Custom numpy JSON Encoder.  Takes in any type from an array and formats it to something that can be JSON serialized. Source Code found here. https://pynative.com/python-serialize-numpy-ndarray-into-json/
    
    Args:
        json (object): Json serialized format
    """	
    def default(self, obj):
        match obj:
            case np.integer():
                return int(obj)
            case np.floating():
                return float(obj)
            case np.ndarray():
                return obj.tolist()
            case dict():
                return obj.__dict__
            case datetime.datetime():
                return datetime.datetime.strftime(obj, "%m-%d-%Y_%H-%M-%S")
            case _:
                return super(NumpyArrayEncoder, self).default(obj)

def save_results(ecg_data, configs: dict, current_date: str):
    """Saves all arrays into a single compressed NPZ file and uploads to GCP.(╯°□°）╯︵ ┻━┻

    Args:
        ecg_data (ECGData): main data container
        configs (dict): configuration dict
        current_date (str): current date
        tobucket (bool, optional): Whether to send the data to a GCP bucket. Defaults to False.

    Raises:
        e: _description_
        e: _description_
        e: _description_
    """  
    logger.info("Saving results to compressed NPZ ...")
    camname = configs["cam_name"]
    configs["last_run"] = current_date
    
    # Safely construct the directory path and ensure it exists
    save_dir = Path(configs["save_path"]) / camname
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the single .npz file path
    file_name = f"{camname}_{current_date}_results.npz"
    file_path = save_dir / file_name 

    try:
        # Check for FUSE mount
        if "/mnt/" in str(file_path):
            temp_dir = Path("/tmp/rad_ecg_saves") / camname
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_file_path = temp_dir / file_name
            
            # Save locally
            logger.info(f"Saving temporary local file: {temp_file_path}")
            try:
                np.savez_compressed(
                    temp_file_path,
                    peaks=ecg_data.peaks,
                    interior_peaks=ecg_data.interior_peaks,
                    section_info=ecg_data.sect_info,
                    configs=configs
                )
                logger.warning(f"Saved all arrays to {temp_file_path}")
                logger.info(f"Transferring to GCS Bucket: {file_path}")
                shutil.copy2(temp_file_path, file_path)
                
                # 3. Clean up the local temp file to save disk space
                os.remove(temp_file_path)
                logger.info(f"Results successfully saved to {file_path}")
                
            except FileNotFoundError as e:
                logger.warning(f"FileNotFound (Is gsutil installed/in PATH?):\n{e}")
                raise e
            except shutil.ExecError as e:
                logger.warning(f"GCP Upload Failed with error {e}")
                raise e
        else:
            np.savez_compressed(
                file_path,
                peaks=ecg_data.peaks,
                interior_peaks=ecg_data.interior_peaks,
                section_info=ecg_data.sect_info,
                configs=configs
            )              
    except Exception as e:
        logger.warning(f"A general error has occurred during saving: {e}")
        raise e

    # Save configs
    save_configs(configs, "./src/rad_ecg/config.json")
    logger.info("Configs updated and saved")

    # Log Final Stats
    logger.critical(f"Wave section counts: {np.unique(ecg_data.sect_info['valid'], return_counts=True)}")
    fail_counts = Counter(ecg_data.sect_info['fail_reason'])
    logger.critical(f"Fail reasons found: {list(fail_counts.items())}")
    logger.critical(f"Runtime configuration: {list(configs.items())}")
    
    # if tobucket:
        # transfer_logfile(logger, configs, camname, current_date)

#FUNCTION Transfer Logfile
def transfer_logfile(logger:logging, configs:dict, cam:str, current_date:str):
    local_path  = configs["log_path"]
    bucket_name = configs["bucket_name"]
    destination_gcp = f'gs://{bucket_name}/results/{cam}/{current_date}.log'
    
    # Reverted back to the highly-stable gsutil command!
    gsutil_command = ['gsutil', 'cp', local_path, destination_gcp]
    
    try:
        # We keep capture_output=True so we can read the terminal if it fails
        result = subprocess.run(gsutil_command, check=True, capture_output=True, text=True)
        logger.warning(f"logfile successfully saved to {bucket_name} on GCP via gsutil")
        
    except FileNotFoundError as e:
        logger.warning(f"FileNotFound (Is gsutil installed/in PATH?):\n{e}")
        raise e
    except subprocess.CalledProcessError as e:
        logger.warning(f"gsutil transfer failed!\nTerminal Error: {e.stderr}")
        raise e
    except Exception as e:
        logger.warning(f"Exception:\n{e}\nType:{type(e)}")
        raise e

#FUNCTION Save Configs
def save_configs(configs:dict, spath:str):
    """This function saves the configs dictionary to a JSON file. 

    Args:
        jsond (dict): Main dictionary container
    """    
    out_json = json.dumps(configs, indent=2, cls=NumpyArrayEncoder)
    with open(spath, "w") as out_f:
        out_f.write(out_json)

################################# Email Funcs ####################################
#FUNCTION Send Email
def send_run_email(run_time:str):
    """Function for sending an email.  Inputs the model runtime into the
    docstrings via decorator for easy formatting of the HTML body of an email.

    Args:
        url (str): [url of the listing]

    Returns:
        [None]: [Just sends the email.  Doesn't return anything]
    """	
    import smtplib, ssl
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    def inputdata(run_time:str):
        """Formats the runtime into an HTML format

        Args:
            run_time (str): Time it took to run the function

        Returns:
            html (str): HTML formatted response with run time
        """	
        html="""
            <html>
                <body>
                    <p>Your ECG is done!<br>
                    Your ECG was processed and """ +str(run_time)+ """<br>
                    Thank you!
                    </p>
                </body>
            </html>
            """
        return html

    with open('./src/rad_ecg/secret/login.txt') as login_file:
        login = login_file.read().splitlines()
        sender_email = login[0].split(':')[1]
        password = login[1].split(':')[1]
        receiver_email = login[2].split(':')[1]
        
    # Establish a secure session with gmail's outgoing SMTP server using your gmail account
    smtp_server = "smtp.gmail.com"
    port = 465

    message = MIMEMultipart("alternative")
    message["Subject"] = "Model is Finished!"
    message["From"] = sender_email
    message["To"] = receiver_email

    html = inputdata(run_time)

    attachment = MIMEText(html, "html")
    message.attach(attachment)
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)		
        server.sendmail(sender_email, receiver_email, message.as_string())

# FUNCTION send notification email
def send_email(log_path:str):
# Save ECG data to file and send confirmation email that the run is done. 
    with open(log_path, 'r') as f:
        line = f.readlines()[-1:][0]
        peak_search_runtime = line.split("|")[4].strip("")

    send_run_email(peak_search_runtime)
    logger.warning("Runtime email sent")
    logger.warning(f"{peak_search_runtime}")
