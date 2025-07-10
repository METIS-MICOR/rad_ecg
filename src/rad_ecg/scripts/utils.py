import numpy as np
from scipy.signal import savgol_filter

#Dev note:Functions are organized most to least important

#FUNCTION Segment ECG
def segment_ECG(
    wave:np.array, 
    fs:float,
    windowsize:int = 10, 
    min_size:int = 5,
    overlap:float = 0.20
    ):
    """Process for segmenting the ECG's for analysis.  
    Modified from Paul Vangents Heartpy software
    https://github.com/paulvangentcom/heartrate_analysis_python/blob/0005e98618d8fc3378c03ab0a434b5d9012b1221/heartpy/peakdetection.py#L21


    Args:
        wave (np.array): ECG wave
        fs (float): Sample Rate
        windowsize (int, optional): Size of the window to eval in question. Defaults to 10.
        min_size (int, optional): Minimum size for last window. Defaults to 5.
        overlap (float, optional): Percentage window overlap. Defaults to 0.20.

    Returns:
        slices (np.array): array of sections.  (start, end, valid)
    """
    ln = len(wave)
    window = windowsize * fs
    stepsize = (1 - overlap) * window
    start = 0
    end = window

    slices = []
    while end < len(wave):
        slices.append((start, end))
        start += stepsize
        end += stepsize

    if min_size == -1:
        slices[-1] = (slices[-1][0], len(wave))
    elif (ln - start) / fs >= min_size:
        slices.append((start, ln))

    #add another column for valid wave or not.  
    #Start with 0's (invalid section) and change to 1 when wave is valid
    slices = np.array(slices, dtype=np.int32)
    slices = np.hstack((slices, np.zeros((slices.shape[0], 1), dtype=np.int32)))
    return slices

#FUNCTION Rolling Median
# @log_time
def roll_med(wave_data:np.array)->np.array:
    """Calculates a rolling median of the HR Signal.  Uses a 40 timestep window. (or 5 milliseconds)
    Rolling median calculation developed by David Josephs
    Args:
        ecg_data [np.array]: [Signal data for which to calc median]

    Returns:
        smoothed_ecg (np.array): [Smoothed rolling median of wave chunk]
    """	
        #TODO  Need a better way to make the windowsize dynamic to the signal.  Calc
        #the current ratio and set it for future signal analysis.  Currently its
        #about a 1/4 of the sampling rate
    
    winsize = 40 #about .2 sec
    smoothed_ecg = np.zeros_like(wave_data)	
    for i in range(len(wave_data)):
        lhs = max(0, i - winsize//2)
        rhs = max(i + winsize//2 + 1, winsize - lhs + 1)
        if rhs >= len(wave_data):
            lhs -= rhs - len(wave_data)
            rhs = len(wave_data) - 1
        smoothed_ecg[i] = np.nanmedian(wave_data[lhs:rhs])
    return smoothed_ecg


#FUNCTION Section Finder
def section_finder(start_p:int, wave:np.array, fs:float):
    """Quick section finder for debugging. 
    Bad sections are stored as a boolean.  When inspecting the log for fail points, 
    it will report them with the peak indices, This function will 
    help you find what section it is in to graph it quickly.
    
    Args:
        start_p (int): point in question

    Returns:
        i (int): wave section where indices are located.
    """	
    wave_sections = segment_ECG(wave, fs)
    # Find the section that start_p would be in range of
    for i in range(len(wave_sections)):
        if start_p >= wave_sections[i, 0] and start_p <= wave_sections[i, 1]:
            return i

#FUNCTION Add chart labels
def add_cht_labels(x:np.array, y:np.array, plt, label:str):
    """[Add's a label for each type of peak]
 
    Args:
        x (np.array): [idx's of peak data]
        y (np.array): [peak data]
        plt ([chart]): [Chart to add label to]
        label (str, optional): [Title of the chart.  Key to the dict of where its label should be shifted]. Defaults to "".
    """
    #Base offsets for each peak
    label_dict = {
        "P":(0, 7),
        "Q":(0, -15),
        "R":(0, 10),
        "S":(6, 5),
        "T":(0, 5)
    }
    for x, y in zip(x,y):
        label = f'{label[0]}' #:{y:.2f}
        plt.annotate(
            label,
            (x,y),
            textcoords="offset points",
            xytext=label_dict[label[0]],
            ha='center')

#FUNCTION Label Formatter
def label_formatter(x_ticks:list)->list:
    """[Formats the x tick labels for millions and thousands]

    Args:
        x_ticks (list): [labels on the x-axis]

    Returns:
        ret (list): [formatted x-tick labels]
    """	
    ret = []
    for x in x_ticks:
        if x >= 1_000_000:
            ret.append(f'{int(x):_d} M')
        elif x >= 1_000:
            ret.append(f'{int(x):_d} K')
        else: 
            ret.append(f'{int(x):_d}')
    return ret

#FUNCTION Valid QRS
def valid_QRS(temp_arr:np.array, temp_counter:int)->int:
    """Checks for valid PQRST peaks in ECG data by testing membership by set comparison

    Args:
        temp_arr (np.array): [ECG data in question]
        temp_counter (int): [index of current peak]

    Returns:
        bool: Whether or not it has all PQST peaks for a valid_peak_type
    """	

    if np.any(temp_arr[temp_counter, :5] == 0):
        return 0
    else:
        return 1

#FUNCTION Load Log results
def load_log_results(file_name:str)->list:
    """Returns a list of all the entries in a log file

    Args:
        file_fold (str): filename for log

    Returns:
        list: List of all the log entries
    """	
    with open(f"{file_name}.log", "r") as f:
        data = f.read().splitlines()
        return data

#FUNCTION Time Convert
def time_convert(fs:int, start_idx:int, end_idx:int, wave:np.array)->float:
    """Converts timesteps into seconds. 

    Returns:
        float: [time in seconds]
    """	
    
    time_length = (end_idx - start_idx) / fs 
    if time_length < 60:
        delt = 's'
    elif time_length < 3600:
        time_length = time_length / 60
        delt = 'm'
    else:
        time_length = time_length / 3600
        delt = 'h'
    
    return (time_length, delt)

#FUNCTION Signal to Noise
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

#FUNCTION Savitzky-Golay
def smooth_signal(y:np.array, window_length:int=15, polyorder:int=3):
    """Applies a Savitzky-Golay filter for smoothing."""
    # window_length must be odd and polyorder < window_length
    if window_length % 2 == 0:
        window_length += 1 # Ensure odd
    if polyorder >= window_length:
        polyorder = window_length - 1 # Ensure polyorder < window_length
        if polyorder < 1: polyorder = 1 # Minimum polyorder of 1 for basic smoothing

    # Handle edge case where window_length is too large for the data
    if len(y) < window_length:
        # Fallback to a simpler smoothing or just return original if not enough data
        if len(y) > 3: # Can still do a basic polyfit if at least 3 points
            window_length = len(y) if len(y) % 2 != 0 else len(y) - 1
            if window_length < 3: window_length = 3 # Minimum for savgol
            polyorder = min(polyorder, window_length - 1)
        else:
            return y # Cannot apply SG filter meaningfully

    return savgol_filter(y, window_length, polyorder)

def calc_rmse(y_true, y_predicted)->float:
    """Calculates the Root Mean Squared Error between true and predicted values."""
    residuals = y_true - y_predicted
    return np.sqrt(np.mean(residuals**2))
