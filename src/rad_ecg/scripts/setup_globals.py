import utils#from rad_ecg.scripts 
import numpy as np
import wfdb
import logging
import json
from os.path import exists

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
    with open("config.json", "r") as f:
        config_data = json.loads(f.read())
    return config_data

#FUNCTION Load Signal Data
def load_signal_data(head_file:str):
    #Load signal data 
    record = wfdb.rdrecord(
        head_file.strip('.hea'),
        sampfrom=0,
        sampto=None,
        channels=[0]
    )
    return record

#FUNCTION Choose CAM
def choose_cam(logger:logging):

    #TODO - Need to update this loader
    head_files = utils.get_records('inputdata')

    #Inquire which file you'd like to run
    logger.critical("Which CAM would you like to import?")
    for idx, head in enumerate(head_files):
        name = head.split(".")[0].split("\\")[-1]
        logger.critical(f'file {idx}:\t{name}')
    header_chosen = input("Please choose a file number ")
    if not header_chosen.isnumeric():
        logger.critical(f'Incorrect file entered, program terminating')
        exit()
    elif int(header_chosen) not in range(len(head_files)):
        logger.critical(f'File not in selection range.  program terminating')
        exit()

    else:
        header_chosen = int(header_chosen)
        name = head_files[header_chosen].split(".")[0].split("\\")[-1]
        logger.critical(f'CAM {name} chosen')

    return head_files, header_chosen

#FUNCTION Load Chart Data
def load_chartdata(logger:logging):
    head_files, header_chosen = choose_cam(logger)
    record = load_signal_data(head_files[header_chosen])

    #ECG data
    wave = record.p_signal
    
    #Frequency
    fs = record.fs

    return wave, fs


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
        head_files, header_chosen = choose_cam(logger)
        record = load_signal_data(head_files[header_chosen])

        #ECG data
        wave = record.p_signal
        
        #Frequency
        fs = record.fs

        #Size of timing segment window
        windowsi = 10
    else:
        logger.CRITICAL("New runtime environment detected outside of normal operating params.\nPlease rerun with appropriate configuration")
        exit()

    #Divide waveform into even segments (Leave off the last 1000 or so, usually unreliable)
    wave_sections = utils.segment_ECG(wave, fs, windowsize=windowsi)[:-1000]

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
