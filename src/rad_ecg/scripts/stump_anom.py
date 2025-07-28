import numpy as np
import stumpy
from collections import deque
from stumpy import gpu_stump
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
    while len(sect_que) > 0:
        section = sect_que.popleft()
        start_p = section[0]
        end_p = section[1]
        Rpeaks = ecg_data['peaks'][(ecg_data['peaks'][:, 0] >= start_p) & (ecg_data['peaks'][:, 0] <= end_p), :]
        m = np.median(np.diff(Rpeaks))
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
            logger.info(f'stumpy extraction error in {start_p}:{end_p}\n{e}')

#Game plan
#reset up rich progress bar. 
    # (means I might have to pull the logger as an input)

# First we make a deque of the sections. 
# Run anomaly detection on each section. 
