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
        stump_func = stumpy.gpu_stump()
    else:
        logger.debug("Algorithm running on CPU")
        device_id = None
        stump_func = stumpy.stump()

    sect_que = deque(ecg_data['section_info'][['start_point', 'end_point']])
    sect_track = 0
    while len(sect_que) > 0:
        section = sect_que.popleft()
        start_p = section[0]
        end_p = section[1]
        peak_info = ecg_data['interior_peaks'][(ecg_data['interior_peaks'][:, 2] >= start_p) & (ecg_data['interior_peaks'][:, 2] <= end_p), :]
        valid_QRS = peak_info[:, 5]
        m = np.median(np.diff(peak_info[:, 2]))
        try:
            if device_id is not None:
                mp = stump_func(
                    T_A = wave[start_p:end_p], 
                    m = m, 
                    device_id = device_id
                )
            else:
                mp = stump_func(
                    T_A = wave[start_p:end_p], 
                    m = m,
                )
        
        except Exception as e:
            logger.info(f'stumpy extraction error in section {sect_track}\n{e}')
            mp = None

        if mp:
            pass
        sect_track += 1

def main():
    global configs
    configs = setup_globals.load_config()
    configs["slider"], configs["run_anomalyd"] = True, True
    datafile = setup_globals.launch_tui(configs)
    global wave, fs
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
    # fpath = f"./src/rad_ecg/data/output/{datafile.name}/{run}"  
    # lfpath = f"./src/rad_ecg/data/logs/{run}"

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
    # Pulling back the actual R peak indexes.  Maybe pull back the QRS valids too?
    # 