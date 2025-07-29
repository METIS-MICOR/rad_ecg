import stumpy
import support
import numpy as np
import setup_globals
from numba import cuda
from collections import deque
from support import logger, console

def group_numbers(arr, delta:int=10):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    if arr.size == 0:
        return []

    # Sort the array to make grouping easier
    sorted_arr = np.sort(arr)
    groups = []
    current_group = []

    for num in sorted_arr:
        if not current_group:
            current_group.append(num)
        else:
            if abs(num - current_group[-1]) <= delta:
                current_group.append(num)
            else:
                groups.append(current_group)
                current_group = [num]

    if current_group:
        groups.append(current_group)

    return [round(np.median(x)) for x in groups]

def run_stumpy_discord(ecg_data:dict, wave:np.array):    
    if cuda.is_available():
        logger.info("Algorithm running on GPU")
        all_gpu_devices = [device.id for device in cuda.list_devices()]
        if len(all_gpu_devices) > 1:
            device_id = all_gpu_devices
        else:
            device_id = 0
        stump_func = stumpy.gpu_stump
    else:
        logger.info("Algorithm running on CPU")
        device_id = None
        stump_func = stumpy.stump
    
    sect_que = deque(ecg_data['section_info'][['start_point', 'end_point']])
    match_count = 0
    sect_track = 0
    progbar, job_id = support.mainspinner(console, len(sect_que))
    with progbar:
        while len(sect_que) > 0:
            progbar.update(task_id=job_id, description=f"[green] Stumpy Anomalies", advance=1)
            section = sect_que.popleft()
            start_p = section[0]
            end_p = section[1]
            sect_valid = ecg_data["section_info"]["valid"][sect_track]
            #Only compare valid sections. No R peak reference for failed area's.
            if sect_valid:
                peak_info = ecg_data['interior_peaks'][(ecg_data['interior_peaks'][:, 2] >= start_p) & (ecg_data['interior_peaks'][:, 2] <= end_p), :]
                valid_QRS = peak_info[:, 5]
                #NOTE maybe change to mean
                m = round(np.median(np.diff(peak_info[:, 2])))
                match_delta = m // 10
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
                    if discord_idx.shape[0] > 0:
                        discords = group_numbers(discord_idx, match_delta)
                        for discord in discords:
                            discord_range = range(discord - match_delta, discord + match_delta)
                            for r_peak in invalid_r_peaks:
                                if discord_range.start <= r_peak <= discord_range.stop:
                                    match_count += 1
                                    logger.critical(f"match found.  count={match_count}")
            sect_track += 1
        return match_count
    
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
    match = run_stumpy_discord(ecg_data, wave)
    print(match)

if __name__ == "__main__":
    main()

#Game plan
    # First we make a deque of the sections. 
    # Run anomaly detection on each section. 
    # Pull back the indexes of the main discords.  
    # Check if they're around the same index as the stored invalid QRS (interior_peaks col 5)
    # use the amount of invalid peaks and pull back those top x indexes of discords.  
    # See if all the indexes align.  
