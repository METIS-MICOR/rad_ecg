import datetime
import numpy as np
import time
import json
from os.path import exists
import logging
from pathlib import Path
#Progress bar fun
from rich.progress import (
    Progress,
    BarColumn,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn
)
from rich.logging import RichHandler
from rich.align import Align
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Console
import subprocess
from collections import Counter

################################# Logger functions ####################################

def get_file_handler(log_dir:Path)->logging.FileHandler:
    """Assigns the saved file logger format and location to be saved

    Args:
        log_dir (Path): Path to where you want the log saved

    Returns:
        filehandler(handler): This will handle the logger's format and file management
    """	
    log_format = "%(asctime)s|%(levelname)-8s|%(lineno)-4d|%(funcName)-23s|%(message)s|" 
                 #f"%(asctime)s - [%(levelname)s] - (%(funcName)s(%(lineno)d)) - %(message)s"
    current_date = time.strftime("%m_%d_%Y")
    log_file = log_dir / f"{current_date}.log"
    file_handler = logging.FileHandler(log_file)
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

def get_logger(log_dir:Path, console:Console)->logging.Logger:
    """Loads logger instance.  When given a path and access to the terminal output.  The logger will save a log of all records, as well as print it out to your terminal. Propogate set to False assigns all captured log messages to both handlers.

    Args:
        log_dir (Path): Path you want the logs saved
        console (Console): Reference to your terminal

    Returns:
        logger: Returns custom logger object.  Info level reporting with a file handler and rich handler to properly terminal print
    """	
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(get_file_handler(log_dir))
    logger.addHandler(get_rich_handler(console))
    logger.propagate = False
    return logger

################################# Saving Funcs ####################################

#FUNCTION save results
def save_results(ecg_data:dict, configs:dict, logger:logging, current_date:datetime, tobucket:bool=False):
    #Because structured arrays will do(ecg_data['section_info']) have mixed dtypes. You
    #have to feed the types back to the save routine when you save it.
    #(╯°□°）╯︵ ┻━┻

    #Export the CSV files
    logger.info("Savings CSV's")
    #Eventually need a folder existence check here.  If it doesn't, create it. 
    cam = configs["cam"].split(".")[-2].split("/")[-1]
    configs["last_run"] = current_date
    for x in ["peaks", "interior_peaks", "section_info"]:
        file_path = "/".join([configs["save_path"], cam, current_date]) + "_" + x + ".csv"
        if x == "section_info":
            save_format = '%i, '*4 + '%s, ' + '%.2f, '*7
        else:
            save_format = '%i, '*ecg_data[x].shape[1]

        np.savetxt(
            fname = file_path,
            X = ecg_data[x], 
            fmt = save_format,
            delimiter=',',
        )
        logger.warning(f"Saved {x} to {file_path}")
        
        if tobucket:
            gcp_path  = file_path[file_path.index(f"{current_date}"):]
            bucket_name = configs["bucket_name"]
            destination_gcp = f'gs://{bucket_name}/results/{cam}/{gcp_path}'
            gsutil_command = ['gsutil', 'cp', file_path, destination_gcp]
            try:
                subprocess.run(gsutil_command, check=True)
                logger.warning(f"{cam} successfully saved to {bucket_name} on GCP")
            #Trapping FNF error specifically
            except FileNotFoundError as e:
                logger.warning(f"FileNotFound:\n{e}")
                raise e
            except Exception as e:
                logger.warning(f"Exception:\n{e}\nType:{type(e)}")
                raise e

    #Save configs
    save_configs(configs, "./src/rad_ecg/config.json")
    logger.info("Configs updated and saved")

    # logger.warning(f'Size of rolling median as {ecg_data["rolling_med"].dtype} {sys.getsizeof(ecg_data["rolling_med"])*.000_001:.2f} MB')
    logger.critical(f"Wave section counts{np.unique(ecg_data['section_info']['valid'], return_counts=True)}")
    fail_counts = Counter(ecg_data['section_info']['fail_reason'])
    logger.critical(f"Fail reasons found:{list(fail_counts.items())}")
    logger.critical(f"Runtime configuration {list(configs.items())}")
    if tobucket:
        transfer_logfile(logger, configs, cam, current_date)

def transfer_logfile(logger:logging, configs:dict, cam:str, current_date:datetime):
    local_path  = configs["log_path"]
    bucket_name = configs["bucket_name"]
    destination_gcp = f'gs://{bucket_name}/results/{cam}/{current_date}.log'
    gsutil_command = ['gsutil', 'cp', local_path, destination_gcp]
    try:
        subprocess.run(gsutil_command, check=True)
        logger.warning(f"logfile successfully saved to {bucket_name} on GCP")
    #Trapping FNF error specifically
    except FileNotFoundError as e:
        logger.warning(f"FileNotFound:\n{e}")
        raise e
    except Exception as e:
        logger.warning(f"Exception:\n{e}\nType:{type(e)}")
        raise e


def save_configs(configs:dict, spath:str):
    """This function saves the configs dictionary to a JSON file. 

    Args:
        jsond (dict): Main dictionary container
    """    
    out_json = json.dumps(configs, indent=2, cls=NumpyArrayEncoder)
    with open(spath, "w") as out_f:
        out_f.write(out_json)

#CLASS Numpy encoder
class NumpyArrayEncoder(json.JSONEncoder):
    """Custom numpy JSON Encoder.  Takes in any type from an array and formats it to something that can be JSON serialized.
    Source Code found here.  https://pynative.com/python-serialize-numpy-ndarray-into-json/
    Args:
        json (object): Json serialized format
    """	
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, str):
            return str(obj)
        elif isinstance(obj, datetime.datetime):
            return datetime.datetime.strftime(obj, "%m-%d-%Y_%H-%M-%S")
        else:
            return super(NumpyArrayEncoder, self).default(obj)
        
################################# Email Funcs ####################################
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

