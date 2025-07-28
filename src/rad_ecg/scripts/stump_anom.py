import numpy as np
import stumpy
from stumpy import gpu_stump
from numba import cuda

def run_stumpy_anomaly():
    pass

def main():
    all_gpu_devices = [device.id for device in cuda.list_devices()]

if __name__ == "__main__":
    main()
#Game plan
#reset up rich progress bar. 
    # (means I might have to pull the logger as an input)

# First we make a deque of the sections. 
# Run anomaly detection on each section. 
