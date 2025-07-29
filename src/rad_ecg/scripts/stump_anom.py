import numpy as np
import stumpy
from collections import deque
from stumpy import gpu_stump
import setup_globals
from numba import cuda
import support
from support import logger

def run_stumpy_discord(ecg_data:dict, wave:np.array):
    if cuda.is_available():
        logger.debug("Algorithm running on GPU")
        all_gpu_devices = [device.id for device in cuda.list_devices()]
        if len(all_gpu_devices) > 1:
            device_id = all_gpu_devices
        else:
            device_id = 0
        stump_func = stumpy.gpu_stump
    else:
        logger.debug("Algorithm running on CPU")
        device_id = None
        stump_func = stumpy.stump
    
    sect_que = deque(ecg_data['section_info'][['start_point', 'end_point']])
    sect_track = 0
    
    while len(sect_que) > 0:
        section = sect_que.popleft()
        start_p = section[0]
        end_p = section[1]
        sect_valid = ecg_data["section_info"]["valid"][sect_track]
        match_delta = 5
        match_count = 0
        #Only compare valid sections. No R peak reference for failed area's.
        if sect_valid:
            peak_info = ecg_data['interior_peaks'][(ecg_data['interior_peaks'][:, 2] >= start_p) & (ecg_data['interior_peaks'][:, 2] <= end_p), :]
            valid_QRS = peak_info[:, 5]
            #NOTE maybe change to mean
            m = round(np.median(np.diff(peak_info[:, 2])))
            try:
                TA = wave[start_p:end_p].flatten()
                if device_id is not None:
                    mp = stump_func(
                        T_A = TA,
                        m = m, 
                        device_id = device_id
                    )
                else:
                    mp = stump_func(
                        T_A = TA,
                        m = m,
                    )
            
            except Exception as e:
                logger.info(f'stumpy extraction error in section {sect_track}\n{e}')
                mp = None

            if mp is not None:
                #Find the peaks that were invalid
                invalid_r_peaks = peak_info[np.where(valid_QRS == 0)[0], 2]

                #Perform scipy peak search again on the MP?
                discord_idx = np.argsort(mp[:, 0])[:len(invalid_r_peaks)] + start_p

                for discord in discord_idx:
                    discord_range = range(discord - match_delta, discord + match_delta)
                    if any(invalid_r_peaks) in discord_range:
                        match_count += 1
        
        sect_track += 1

def main():
    global configs
    configs = setup_globals.load_config()
    configs["slider"], configs["run_anomalyd"] = True, True
    datafile = setup_globals.launch_tui(configs)
    wave, fs, outputf = setup_globals.load_chart_data(configs, datafile, logger)
    wave_sect_dtype = [
        ('wave_section', 'i4'),
        ('start_point', 'i4'),
        ('end_point', 'i4'),
        ('valid', 'i4'),
        ('fail_reason', str, 16),
        ('Avg_HR', 'f4'), 
        ('SDNN', 'f4'),
        ('min_HR_diff', 'f4'), 
        ('max_HR_diff', 'f4'), 
        ('RMSSD', 'f4'),
        ('NN50', 'f4'),
        ('PNN50', 'f4')
    ]

    for fname in outputf:
        if fname.endswith("_section_info.csv"):
            fpath = datafile._str + "\\" + fname.split("_section_info")[0]
            break
    # NOTE: #Consider storing the mp results in here. 
    global ecg_data
    ecg_data = {
        "peaks": np.genfromtxt(fpath+"_peaks.csv", delimiter=",", dtype=np.int32, usecols=(0, 1)),
        "section_info": np.genfromtxt(fpath+"_section_info.csv", delimiter=",", dtype=wave_sect_dtype),
        "interior_peaks": np.genfromtxt(fpath+"_interior_peaks.csv", delimiter=",", dtype=np.int32, usecols=(range(16)), filling_values=0)
    } 

    run_stumpy_discord(ecg_data, wave)
    
if __name__ == "__main__":
    main()

#Game plan
    # First we make a deque of the sections. 
    # Run anomaly detection on each section. 
    # Pull back the indexes of the main discords.  
    # Check if they're around the same index as the stored invalid QRS (interior_peaks col 5)
    # use the amount of invalid peaks and pull back those top x indexes of discords.  
    # See if all the indexes align.  
