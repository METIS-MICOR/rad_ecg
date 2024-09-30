#NOTE Custom Imports
import utils        #from rad_ecg.scripts # 
import support      #from rad_ecg.scripts # 
import setup_globals#from rad_ecg.scripts # 

#NOTE Main library imports
import scipy.signal as ss
from scipy.fft import rfft, rfftfreq, irfft
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
from matplotlib.colors import rgb2hex
import time
import os
import sys
import logging
# from pathlib import Path
from collections import deque
from rich.logging import RichHandler
from rich.console import Console
from time import strftime

current_date = strftime("%m-%d-%Y_%H-%M-%S")
FORMAT = "%(asctime)s|%(levelname)-8s|%(lineno)-4d|%(funcName)-23s|%(message)s|" 
FORMAT_RICH = "|%(funcName)-23s|%(message)s"
console = Console(color_system="truecolor")
rh = RichHandler(level = logging.WARNING, console=console)
rh.setFormatter(logging.Formatter(FORMAT_RICH))
log_path = f"./src/rad_ecg/data/logs/{current_date}.log"

#Set up basic config for logger
logging.basicConfig(
    level=logging.INFO, 
    format=FORMAT,
    datefmt="[%X]",
    handlers=[
        rh,
        logging.FileHandler(log_path, mode="w")
    ]
)

logger = logging.getLogger(__name__) 

#FUNCTION log time
def log_time(fn):
    """Decorator timing function.  Accepts any function and returns a logging
    statement with the amount of time it took to run.

    Args:
        fn (function): Input function you want to time
    """	
    def inner(*args, **kwargs):
        tnow = time.time()
        out = fn(*args, **kwargs)
        te = time.time()
        took = round(te - tnow, 2)
        if took <= 60:
            logger.warning(f"{fn.__name__} ran in {took:.2f}s")
        elif took <= 3600:
            logger.warning(f"{fn.__name__} ran in {(took)/60:.2f}m")		
        else:
            logger.warning(f"{fn.__name__} ran in {(took)/3600:.2f}h")
        return out
    return inner

# @log_time
#FUNCTION consecutive valid peaks
def consecutive_valid_peaks(R_peaks:np.array, lookback:int=3500):
    """Historical Data search function.  Scans back in time until it finds the lookback amount of continuous
    validated R peaks.  

    Args:
        R_peaks (np.array): Array of the R peaks that have already been found
        lookback (int): How long you want to lookback to find a consecutive chunk of validated R peaks
    Returns:
        last_keys : The last keys where all R peaks were valid for 20 seconds. 
    """
    arr = R_peaks[::-1].copy()
    counts = []
    for i in range(arr.shape[0]):
        is_last = i + 1 >= arr.shape[0]
        if arr[i, 1] == 1:
            counts.append(i)
        if is_last or arr[i, 1] == 0:
            if is_last:
                logger.critical(f'Unable to find valid peak window ')
                return False
            else:
                counts = []
        elif (arr[counts[0], 0] - arr[counts[-1], 0] > lookback):
            logger.info(f'QRS lookback range {arr[counts[-1], 0]} to {arr[counts[0], 0]} at length {len(counts)}')
            return arr[counts][::-1, 0]

# FUNCTION STFT
def STFT(
    new_peaks_arr:np.array,
    peak_info:np.array, 
    rolled_med:np.array, 
    st_fn:tuple, 
    plot_fft:bool=False, 
    *args
):
    """Takes in the new peaks found by scipy find_peaks.  Performs a STFT on
    each of the Rpeak to Rpeak sections to look for high frequency noise. If the
    STFT comes back with mostly low frequency data, the routine marks the peak
    valid in the ecg_data['peaks'] container.

    Rejects individual R_R sections

    Args:
        new_peaks_arr (np.array): Array of new peaks to be checked
        peak_info (np.array): Peak height and prominence information
        rolled_med (np.array): Rolling median of the new peaks array
        st_fn (tuple): section, start and finish point.  
        plot_fft (bool, optional): Whether to plot FFT. Defaults to False.

    Returns:
        T/F, new_peaks_arr (boolean, np.array): Returns the boolean of whether 
        the wave chunks is valid.  As well as the new peaks array with the
        updated peak validity.
    """	
    #Set globals 
    global ecg_data, wave, fs

    bad_sect_counter = 0
    Rpeak_deque = deque(new_peaks_arr[:, 0])
    currsect = st_fn[0]
    start_point = st_fn[1]
    end_point = st_fn[2]

    if len(args) != 0:
        wave = args[0]
        fs = args[1]

    #new_peaks_arr[:, 1]
    #0 = invalid peak
    #1 = Valid peak
    #validation mask is set to 1 to start.  Sets to zero when finds
    #high freq data 

    #Quick check to make sure we have enough peaks to analyze
    if new_peaks_arr.size < 4:
        logger.warning(f'Not enough peaks found for STFT')
        new_peaks_arr[:, 1] = 0
        return False, new_peaks_arr

    while len(Rpeak_deque) > 1:
        p0 = Rpeak_deque.popleft()
        p1 = Rpeak_deque[0] + 1
        samp = wave[p0:p1]
        fft_samp = np.abs(rfft(samp))
        freq_list = np.fft.rfftfreq(len(samp), d=1/fs) #fs is sampling rate
        freqs = fft_samp[0:int(len(samp)/2)]
        # thres=15
        #TODO - Retest at 12 Hz to satisfy Pan Tompkins Comparison
        thres = np.where(freq_list < 18)[0][-1]
        outs = np.where(fft_samp[thres:int(len(samp)/2)] > fft_samp[0:thres].mean())[0]

        if outs.size >= 2:
            bad_sect_counter += 1
            new_peaks_arr[np.where(new_peaks_arr[:, 0]==p0)[0], 1] = 0
        else:
            new_peaks_arr[np.where(new_peaks_arr[:, 0]==p0)[0], 1] = 1
            
    if plot_fft:
        ##################### FULL ECG ######################
        fig = plt.figure(figsize=(10, 9))
        grid = plt.GridSpec(2, 2, hspace=0.7, height_ratios=[1.5, 1])
        ax_ecg = fig.add_subplot(grid[0, :2])
        ax_freq = fig.add_subplot(grid[1, :1])
        ax_spec = fig.add_subplot(grid[1, 1:2])
        ax_ecg.plot(range(start_point, end_point), wave[start_point:end_point])
        ax_ecg.plot(range(start_point, end_point), rolled_med.flatten())
        ax_ecg.scatter(new_peaks_arr[:, 0], peak_info['peak_heights'], marker='D', color='red')
        for peak in range(new_peaks_arr.shape[0] - 1):
            if new_peaks_arr[peak, 1]==0:
                band_color = 'red'
            else:
                band_color = 'lightgreen'
            rect = Rectangle(
                xy=(new_peaks_arr[peak, 0], 0), 
                width=new_peaks_arr[peak+1, 0]-new_peaks_arr[peak, 0], 
                height=np.max(wave[new_peaks_arr[peak, 0]:new_peaks_arr[peak+1, 0]]), 
                facecolor=band_color,
                edgecolor="grey",
                alpha=0.7)
            ax_ecg.add_patch(rect)

        ax_ecg.set_title(f'Full ECG waveform for section {currsect} indices {start_point}:{end_point}') 
        ax_ecg.set_xlabel("Timesteps")
        ax_ecg.set_ylabel("ECG mV")			
        ax_ecg.legend(['Full ECG', 'Rolling Median', 'R peaks'])
        ax_ecg.set_xticks(ax_ecg.get_xticks(), labels = utils.label_formatter(ax_ecg.get_xticks()) , rotation=-30)

        #Frequency stem plot
        #Initially graphs the last R to R range in the section. 
        ##################### FFT ######################
        p0 = new_peaks_arr[-2, 0]
        p1 = new_peaks_arr[-1, 0]
        samp = wave[p0:p1]
        fft_samp = np.abs(rfft(samp))
        freq_list = np.fft.rfftfreq(len(samp), d=1/fs) #fs is sampling rate
        freqs = fft_samp[0:int(len(samp)/2)]
        # thres = 15
        thres = np.where(freq_list < 18)[0][-1]
        outs = np.where(fft_samp[thres:int(len(samp)/2)] > fft_samp[0:thres].mean())[0]
        ax_freq.stem(freqs)
        ax_freq.axhline(y=fft_samp[0:thres].mean(), color='dodgerblue', linestyle='--')
        ax_freq.set_title(f'FFT spectrum peaks {p0}:{p1}')
        ax_freq.set_xlabel("Freq (Hz)")
        ax_freq.set_ylabel("Power")
        ax_freq.legend([f'first {thres} freq mean', 'Frequencies in Hz'])
        ax_freq.scatter(outs+thres, fft_samp[thres:int(len(samp)/2)][outs], marker='o', color='red', s=80)

        #arrow patch
        mid = p0 + (p1 - p0)//2 

        ##################### Spectogram ######################
        ax_spec.specgram(
                        wave[start_point:end_point].flatten(),
                        NFFT= int(np.mean(np.diff(new_peaks_arr[:, 0]))),
                        detrend="linear",
                        noverlap = 10,
                        Fs=fs)

        ax_spec.set_xlabel("Time (sec)")
        ax_spec.set_ylabel("Freq, Hz")
        ax_spec.set_title(f'Spectogram for peaks {new_peaks_arr[0,0]}:{new_peaks_arr[-1,0]}')

        def onSpacebar(event):
            """When scanning ECG's, hit the spacebar if keep the chart from closing. 

            Args:
                event (_type_): accepts the key event.  In this case its looking for the spacebar.
            """	
            if event.key == " ": 
                timer_error.stop()
                timer_error.remove_callback(timer_cid)
                logger.warning(f'Timer stopped')

        def onClick(event):
            def get_rects():
                rects = [i for i in ax_ecg.patches if isinstance(i, Rectangle)]
                return rects
        
            def clear_freq_cht():
                #clear all the data
                ax_freq.cla()

            def redraw_freq(p0:int, p1:int):
                logger.warning(f'redrawing freq')
                samp = wave[p0:p1]
                fft_samp = np.abs(rfft(samp))
                freq_list = np.fft.rfftfreq(len(samp), d=1/fs) #fs is sampling rate
                freqs = fft_samp[0:int(len(samp)/2)]
                # thres = 15
                thres = np.where(freq_list < 18)[0][-1]
                outs = np.where(fft_samp[thres:int(len(samp)/2)] > fft_samp[0:thres].mean())[0]
                ax_freq.stem(freqs)
                ax_freq.axhline(y=fft_samp[0:thres].mean(), color='dodgerblue', linestyle='--')
                ax_freq.set_title(f'FFT spectrum peaks {p0}:{p1}')
                ax_freq.set_xlabel("Freq (Hz)")
                ax_freq.set_ylabel("Power")
                ax_freq.legend([f'first {thres} freq mean', 'Frequencies in Hz'])
                ax_freq.scatter(outs+thres, fft_samp[thres:int(len(samp)/2)][outs], marker='o', color='red', s=80)
                
            def redraw_spec(p0:int, p1:int):
                logger.warning(f'redrawing spec')
                ax_spec.specgram(
                        wave[start_point:end_point].flatten(),
                        NFFT= int(np.mean(np.diff(new_peaks_arr[:, 0]))),
                        noverlap = 10,
                        Fs=fs)
                ax_spec.set_xlabel("Time (sec)")
                ax_spec.set_ylabel("Freq, Hz")
                ax_spec.set_title(f'Spectogram for peaks {new_peaks_arr[0,0]}:{new_peaks_arr[-1,0]}')

            if event.inaxes == ax_ecg:
                rect_locs = get_rects()
                for x, rect in enumerate(rect_locs):
                    cont, ind = rect.contains(event)
                    if cont:
                        p0 = rect_locs[x]._x0
                        p1 = p0 + rect_locs[x]._width
                        clear_freq_cht()
                        redraw_freq(p0, p1)
                        #? Need redraw spect as well?
                        fig.canvas.draw_idle()

        a = 3000
        b = 450 

        fig.canvas.manager.window.wm_geometry("+%d+%d" % (a, b))
        click_control = fig.canvas.mpl_connect("button_press_event", onClick)
        spacejam = fig.canvas.mpl_connect('key_press_event', onSpacebar)
        timer_error = fig.canvas.new_timer(interval = 3000)
        timer_error.single_shot = True
        timer_cid = timer_error.add_callback(plt.close, fig)
        timer_error.start()
        plt.show()
        plt.close()

    #If more than 25% of the R to R FFT's are bad, mark the section rejected.
    if bad_sect_counter >= (round(0.25 * new_peaks_arr.shape[0])):
        logger.warning(f'Found {bad_sect_counter} bad sections out of {new_peaks_arr[:, 0].shape[0]} in section:{currsect}')
        return False, new_peaks_arr
    else:
        return True, new_peaks_arr

#FUNCTION section stats
def section_stats(new_peaks_arr:np.array, section_counter:int)->tuple:
    """This function calculates the time domain stats for a given section. 

    Args:
        new_peaks_arr (np.array): Peaks for evaluation
        section_counter (int):Tracking what section we're in
    Returns:
        (tuple): Tuple of the HR stats for that section.
    """
    #First look and see if there's any peaks that are invalid (ie = 0). 
    #(Ignore the last peak as it's most likely invalid)
    global ecg_data, fs
    peak_check = np.any(new_peaks_arr[:-1, 1] == 0)

    if peak_check:
        ecg_data['section_info'][section_counter]['fail_reason'] = "inv_peak"
        bad_peaks = np.where(new_peaks_arr[:,1] == 0)[0]
        logger.info(f'Failed to extract HR due to invalid peaks {new_peaks_arr[bad_peaks, 0]}')

    #Now see if we have the bare minimum for peaks to extract. 
    elif new_peaks_arr.size <= 2:
        ecg_data['section_info'][section_counter]['fail_reason'] = "no_peaks"
        logger.warning(f'Not enough peaks to calculate section stats')
        
    else:
        #MEAS Section Measures 
        RR_diffs = np.diff(new_peaks_arr[:,0])
        RR_diffs_time = np.abs(np.diff((RR_diffs / fs) * 1000)) #Formats to time domain in milliseconds
        HR = np.round((60 / (RR_diffs / fs)), 2) #Formatted for BPM
        Avg_HR = np.round(np.mean(HR), 2)
        min_HR  = np.min(HR)
        max_HR  = np.max(HR)
        SDNN = np.round(np.std(HR), 5)
        RMSSD = np.round(np.sqrt(np.mean(np.power(RR_diffs_time, 2))), 5)

        try:
            NN50 = np.where(RR_diffs_time > 50)[0].shape[0]
            PNN50 = np.round((NN50 / RR_diffs.shape[0]) * 100, 2)
        except Exception as e:
            logger.warning(f'Unable to find NN50 {e}')
            NN50 = np.nan
            PNN50 = np.nan

        return (Avg_HR, SDNN, min_HR, max_HR, RMSSD, NN50, PNN50)

#FUNCTION peak validation check
# @log_time
def peak_validation_check(
    new_peaks_arr:np.array,
    last_keys:list,
    peak_info:dict,
    rolled_med:np.array,
    st_fn:tuple, 
    low_counts:int,
    IQR_low_thresh:float,
    plot_errors:bool=False, 
):
    """Rejects whole segments based on historical averages. 

    Args:
        new_peaks_arr (np.array): Current peaks to be checked.  
        last_keys (list): Rpeaks of the last valid 30 seconds of wave
        peak_info (dict): Current peak heights/prominences
        rolled_med (np.array): Current peaks rolled median
        st_fn: Tuple of the section_counter, start_p and end_p
        low_counts: Number of times IQR has hit a low point
        IQR_low_thresh: The lowest IQR seen recently
        plot_errors (bool, optional): Whether to plot errors we find. Defaults to False.

    Returns:
        sect_valid:  Boolean of if section is valid
    """	
    #new_peaks_arr[:, 1]
    #0 = invalid peak
    #1 = Valid peak
    #Valid masks are 0 from the start so no need to update them if bad
    #Might have to reverse that though.  As i can 't assign it to zero once
    # its already zero and keep tracking it. 

    def look_back_time_format(lookback:int)->tuple:
            if lookback < 60:
                delt = 's'
            elif lookback < 3600:
                lookback = lookback / 60
                delt = 'm'
            else:
                lookback = lookback / 3600
                delt = 'h'
            return (lookback, delt)
    
    def onSpacebar(event):
        """When scanning ECG's, hit the spacebar if keep the chart from closing. 

        Args:
            event (_type_): accepts the key event.  In this case its looking for the spacebar.
        """	
        if event.key == " ": 
            timer_error.stop()
            timer_error.remove_callback(timer_cid)
            logger.warning(f'Timer stopped')
    
    global ecg_data, wave
    
    #Get current section
    cur_sect = st_fn[0]
    fail_reas = ""
    #Get section start / finish
    start_idx = st_fn[1]
    end_idx = st_fn[2]

    #Start with the section as True.  If any gate fails, turn it to false. 
    sect_valid = True

    #empty array for historical data on last_keys
    med_diff = []
    med_arr = np.zeros((0, 1), dtype=np.float32)

    rolling_med_start = last_keys[0]
    rolling_med_end = last_keys[-1]
    med_arr = ecg_data['rolling_med'][rolling_med_start:rolling_med_end]
    
    #Get the peak differences of the last keys.
    med_diff = np.diff(last_keys)

    # #Get the 20/80 quartiles and IQR
    Q1 = np.quantile(med_arr, .20)
    Q3 = np.quantile(med_arr, .80)
    
    IQR = Q3 - Q1

    # Test to see if IQR is lower than IQR_low_thresh This is to prevent hitting
    # a vanishing gradient for IQR.  
    #BUG - IQR Threshold malfunctioning
        #9-28-24: While watching a run I noticed the lowcount thresholds are getting caught in the global minima's again. 
        #I need some other way to set the threshold width of a normal recent QRS.  Will investigate other methods on Monday but
        #wanted to make a note of the behavior I saw. (Counts were in the 500's so its getting stuck until it loses a signal)
    
    if IQR == IQR_low_thresh:
        low_counts += 1
        if low_counts > 6:
            IQR = 3*IQR
            logger.warning(f'Bumped up IQR 3x to {IQR:.4f} for section {cur_sect} low_count at {low_counts}')
        elif low_counts > 3: 
            IQR = IQR + .50*IQR
            logger.info(f'Bumped up IQR 0.5x to {IQR:.4f} for section {cur_sect} low_count at {low_counts}')

    elif IQR <= IQR_low_thresh:	
        IQR_low_thresh = IQR
    
    else:
        logger.info(f'IQR reset for section {cur_sect}')
        IQR_low_thresh = 1
        low_counts = 0

    logger.info(f'IQR used for section {cur_sect} is {IQR:.5f}')

    samp_roll_med = rolled_med
    ##!Removed and replaced with below
    #Get the outliers outside of the IQR in for rolling med
    # out_above = np.where(samp_roll_med > (Q3 + 1.5*IQR))[0]
    # out_below = np.where(samp_roll_med < (Q1 - 1.5*IQR))[0]
    out_above = np.where(samp_roll_med > (np.quantile(samp_roll_med, .80) + 1.5*IQR))[0]
    out_below = np.where(samp_roll_med < (np.quantile(samp_roll_med, .20) - 1.5*IQR))[0]

    #Peak separation/height variables
    last_avg_p_sep = np.mean(med_diff)
    last_avg_peak_heights = np.mean([wave[x][0] for x in last_keys])

    #NOTE Slope Check
        #TODO update method based on historical data
        #Process.
            #1. Grab the R peaks. 	
            #2. Find 75% of the distance of the Rpeak diffs.
            #3. Calculate all the sign changes from negative to positive in between each point.  
            #4. The last one should be the Q peak.  (basically repeating the Q peak extraction)
            #5. Use those leftbases(Q peaks) to calculate the slope to the R peak
            #6. Mark any erroneous slopes and invalidate section. 
            
    RPeaks = new_peaks_arr[:, 0]
    lookbacks = RPeaks - int(last_avg_p_sep * 0.75) #This needs to be somewhat wider to ensure a sign change
    leftbases = []
    #Loop through the lookback positions and R peaks to build the left bases to build the slopes
    for lookback, RP in zip(lookbacks,RPeaks):
        #First we go through and find the difference between each point.
        #(viewpoint is from the lookback to the R peak)
        grad = np.diff(wave[lookback:RP+1].flatten())

        #Isolate the sign change of each gradient
        asign = np.sign(grad)
        
        #roll/shift the indices by 1, then subtract  off the sign change to
        #isolate when a wave is shifting from positive to negative or vice
        #versa. 
        signchange = np.roll(np.array(asign), 1) - asign

        #Filter for changes from - -> +  and from - -> 0
        np_inflections = np.where((signchange == -2) | (signchange == -1))[0]

        #Checking to make sure we have data
        if np_inflections.size > 0:
                leftbases.append(lookback + np_inflections[-1])
        else:
            logging.warning(f"Left base missed on R peak {RP}")

    if len(leftbases) == len(RPeaks):
        slopes = [np.polyfit(range(x1, x2), wave[x1:x2], 1)[0].item() for x1, x2 in zip(leftbases, RPeaks)]
        lower_bound = np.mean(slopes) * 0.30 #started at .51 
        upper_bound = np.mean(slopes) * 3
        peak_slope_check = np.any((slopes < lower_bound)|(slopes > upper_bound))
    else:
        logging.critical(f"Uneven lengths of leftbases in sect {cur_sect}")
        fail_reas = "slope"
        peak_slope_check = False
        sect_valid = False

    if peak_slope_check:
        if plot_errors:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(12,6))
            plt.plot(range(start_idx, end_idx), wave[start_idx:end_idx], label = 'ECG')
            plt.plot(range(start_idx, end_idx), rolled_med, label = 'Rolling Median')
            plt.scatter(new_peaks_arr[:, 0], peak_info['peak_heights'], marker='D', color='red', label='R peaks')
            plt.scatter(leftbases, wave[leftbases], marker="o", color="green", label="left base")
            high_slopes = np.where(slopes > upper_bound)[0]
            low_slopes = np.where(slopes < lower_bound)[0]
            _smax = np.max(wave[start_idx:end_idx])
            _smin = np.min(wave[start_idx:end_idx])
            _delt = 0.10 * (_smax - _smin)

            if high_slopes.size > 0:
                for highslo in high_slopes:
                    arrow = Arrow(
                        x=leftbases[highslo],
                        y=wave[leftbases[highslo]] + _delt*2,
                        dx = 0,
                        dy = -1 * _delt,
                        width = 40,
                        color="red"
                    )
                    ax.add_patch(arrow)
            if low_slopes.size > 0:
                for lowslo in low_slopes:
                    arrow = Arrow(
                        x=leftbases[lowslo],
                        y=wave[leftbases[lowslo]] - _delt*2,
                        dx = 0,
                        dy = _delt,
                        width = 40,
                        color="red"
                    )
                    ax.add_patch(arrow)
            plt.title(f'Bad peak slope for idx {start_idx:_d} to {end_idx:_d} in sect {cur_sect}')
            plt.legend(loc="upper left")
            ax.set_xticks(ax.get_xticks(), labels = utils.label_formatter(ax.get_xticks()) , rotation=-30)
            
            a = 3000
            b = 450 
            fig.canvas.manager.window.wm_geometry("+%d+%d" % (a, b))
            timer_error = fig.canvas.new_timer(interval = 3000)
            timer_error.single_shot = True
            timer_cid = timer_error.add_callback(plt.close, fig)
            spacejam = fig.canvas.mpl_connect('key_press_event', onSpacebar)
            timer_error.start()
            plt.show()
            plt.close()

    #NOTE Rolling Median Check
    #If either outabove or outbelow has values, proceed with wave check.
    if out_above.size > 0 or out_below.size > 0:
        del out_above, out_below

        #Que up some peaks
        peak_que = deque(new_peaks_arr[:, 0])

        #Counter for bad sections. 
        bad_pandas = 0
        outs = []
        
        while len(peak_que) > 1:
            p0 = peak_que.popleft() 
            p1 = peak_que[0]
            samp_section = samp_roll_med[p0 - start_idx:p1 - start_idx]
            out_above = np.where(samp_section > (np.quantile(samp_roll_med, .80) + 1.5*IQR))[0]
            out_below = np.where(samp_section < (np.quantile(samp_roll_med, .20) - 1.5*IQR))[0]
            
            if out_above.size > 0 or out_below.size > 0:
                if out_above.size > 0:
                    outs.append(('above', p0,  p1))
                    
                if out_below.size > 0: 
                    outs.append(('below', p0,  p1))

                new_peaks_arr[np.where(new_peaks_arr[:, 0] == p0)[0], 1] = 0
                bad_pandas += 1
        
        #If the number of bad wave sections (bad pandas) is greater than 50% of of the Rpeaks, reject section
        if bad_pandas > (round(0.50 * (new_peaks_arr.shape[0]-1))):
            logger.warning(f'Bad Wave segment roll_med in section:{cur_sect}')
            logger.warning(f'Number of bad peaks: {bad_pandas} out of {new_peaks_arr.shape[0]}')
            
            #Log how far back the historical search went
            lookback_time = (new_peaks_arr[0, 0] - last_keys[0] ) / fs 
            lookback_time, delt = look_back_time_format(lookback_time)
            logger.critical(f'QRS lookback was {lookback_time:.2f}{delt} starting at R_peak {last_keys[0]:_d}')

            if len(fail_reas) > 0:
                fail_reas = fail_reas + '|roll'
            else:
                fail_reas = 'roll'

            if plot_errors:
                fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(12,6))
                plt.plot(range(start_idx, end_idx), wave[start_idx:end_idx], label = 'ECG')
                plt.plot(range(start_idx, end_idx), samp_roll_med, label='Rolling Median shifted')
                plt.scatter(new_peaks_arr[:, 0], peak_info['peak_heights'], marker='D', color='red', label='R peaks')
                plt.axhline(y=(np.quantile(samp_roll_med, .80)+1.5*IQR), color='magenta', linestyle='--', label='upper past roll med')
                plt.axhline(y=(np.quantile(samp_roll_med, .20)-1.5*IQR), color='red', linestyle='--', label='lower past roll med')
                plt.axhline(y=samp_roll_med.mean(), color = 'green', linestyle='--', label='rolling median mean (shifted)')
                ax.set_xticks(ax.get_xticks(), labels = utils.label_formatter(ax.get_xticks()) , rotation=-30)

                plt.legend()
                for x in outs:
                    if x[0] == 'above':
                        rect = Rectangle(
                            (x[1], 0), 
                            x[2]-x[1], 
                            np.max(wave[x[1]:x[2]]),
                            facecolor='lightgrey', 
                            alpha=0.9)
                    elif x[0] == 'below':
                        rect = Rectangle(
                            (x[1], 0), 
                            x[2]-x[1], 
                            np.min(wave[x[1]:x[2]]), 
                            facecolor='lightgrey', 
                            alpha=0.9)
                    ax.add_patch(rect)

                plt.title(f'Bad rolling median for idx {start_idx:_d} to {end_idx:_d} in sect {cur_sect}')
                a = 3000
                b = 450 
                fig.canvas.manager.window.wm_geometry("+%d+%d" % (a, b))
                timer_error = fig.canvas.new_timer(interval = 3000)
                timer_error.single_shot = True
                timer_cid = timer_error.add_callback(plt.close, fig)
                spacejam = fig.canvas.mpl_connect('key_press_event', onSpacebar)
                timer_error.start()
                plt.show()
                plt.close()
            sect_valid = False

    Rpeak_roll_diff = wave[last_keys][:,0] - ecg_data['rolling_med'][last_keys]
    # lower_bound = Rpeak_roll_diff.mean() - np.std(Rpeak_roll_diff)*3
    lower_bound = np.mean(Rpeak_roll_diff) * 0.51 
    upper_bound = last_avg_peak_heights * 3 #moved down from 4 on 6-27-23.  Not sure why it was that high
    
    peak_height_check = np.any((peak_info['peak_heights'] < lower_bound)|(peak_info['peak_heights'] > upper_bound))
    if peak_height_check:
        # plot_errors = True
        logger.warning(f'Bad Wave segment peak_height in {start_idx:_d} to {end_idx:_d}')
        # low_peaks = np.where((peak_info['peak_heights'] < lower_bound))[0]
        # high_peaks = np.where((peak_info['peak_heights'] > upper_bound))[0]
        low_peaks = np.where((peak_info['peak_heights'] < lower_bound))[0]
        high_peaks = np.where((peak_info['peak_heights'] > upper_bound))[0]
            #LPT = Last valid peaks - Difference between R peak and rolling median.  
            #HPT = Last valid peaks - Average of R peak heights.  Set to 4x to be able to climb out of minimal area's. 
        if low_peaks.size > 0:
            logger.warning(f'peak height for {new_peaks_arr[low_peaks, 0]} less than threshold of 51% of LPT:{lower_bound:.2f} in section {cur_sect}')
            arrow_color = 'goldenrod'
            new_peaks_arr[low_peaks, 1] = 0

        if high_peaks.size > 0:
            logger.warning(f'peak height for {new_peaks_arr[low_peaks, 0]} greater than threshold of 4x of HPT {upper_bound:.2f} in section {cur_sect}')
            arrow_color = 'darkviolet'
            new_peaks_arr[high_peaks, 1] = 0

        #Log how far back the historical search went
        lookback_time = (new_peaks_arr[0, 0] - last_keys[0] ) / fs 
        lookback_time, delt = look_back_time_format(lookback_time)
        logger.critical(f'QRS lookback was {lookback_time:.2f}{delt} starting at R_peak {last_keys[0]:_d}')
        #Encode fail reason
        if len(fail_reas) > 0:
            fail_reas = fail_reas + '|height'
        else:
            fail_reas = 'height'

        if plot_errors:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(12,6))
            plt.plot(range(start_idx, end_idx), wave[start_idx:end_idx], label = 'ECG')
            plt.plot(range(start_idx, end_idx), rolled_med, label = 'Rolling Median')
            plt.scatter(new_peaks_arr[:, 0], peak_info['peak_heights'], marker='D', color='red', label='R peaks')
            for x in (low_peaks, high_peaks):
                if x.size > 0:
                    for y in x:
                        arrow = Arrow(
                            x=new_peaks_arr[y, 0] - 55,
                            y=peak_info['peak_heights'][y],
                            dx = 40,
                            dy = 0,
                            width = 0.05,
                            color=arrow_color
                        )
                        ax.add_patch(arrow)
            plt.legend()
            plt.title(f'Bad peak height for idx {start_idx:_d} to {end_idx:_d} in sect {cur_sect}')
            ax.set_xticks(ax.get_xticks(), labels = utils.label_formatter(ax.get_xticks()) , rotation=-30)
            a = 3000
            b = 450 
            fig.canvas.manager.window.wm_geometry("+%d+%d" % (a, b))
            timer_error = fig.canvas.new_timer(interval = 3000)
            timer_error.single_shot = True
            timer_cid = timer_error.add_callback(plt.close, fig)
            spacejam = fig.canvas.mpl_connect('key_press_event', onSpacebar)
            timer_error.start()
            plt.show()
            plt.close()
            # plot_errors = False
        sect_valid = False

    lower_bound = last_avg_p_sep * 0.5
    upper_bound = last_avg_p_sep * 2  #stock 1.5
    diff = np.diff(new_peaks_arr[:, 0])
    peak_sep_check = np.any((diff < lower_bound) | (diff > upper_bound))

    if peak_sep_check:
        bad_sep = np.where((diff < lower_bound) | (diff > upper_bound))
        #Set those peaks to invalid
        new_peaks_arr[bad_sep, 1] = 0
        #Need to subract by one to reference the first peak
        lower_v = np.where(diff < lower_bound)[0]
        upper_v = np.where(diff > upper_bound)[0]
        logger.info(f'Last avg peak separation is {last_avg_p_sep:.2f} starting at at {last_keys[0]:_d}')
        if lower_v.size > 0:
            logger.warning(f'peak_sep {diff[lower_v]} for peaks {new_peaks_arr[lower_v - 1, 0]} under low bound of {lower_bound:.2f} in section {cur_sect}')
        if upper_v.size > 0:
            logger.warning(f'peak_sep {diff[upper_v]} for peaks {new_peaks_arr[upper_v - 1, 0]} over upper bound of {upper_bound:.2f} in section {cur_sect}')
                
        
        #Log how far back the historical search went from the current position in time.
        lookback_time = (new_peaks_arr[0, 0] - last_keys[0] ) / fs 
        lookback_time, delt = look_back_time_format(lookback_time)
        logger.critical(f'QRS lookback was {lookback_time:.2f}{delt} starting at R_peak {last_keys[0]:_d}')

        #Encode fail reason
        if len(fail_reas) > 0:
            fail_reas = fail_reas + '|sep'
        else:
            fail_reas = 'sep'
        
        if plot_errors:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(12,6))
            plt.plot(range(start_idx, end_idx), wave[start_idx:end_idx], label = 'ECG')
            plt.plot(range(start_idx, end_idx), rolled_med, label='Rolling Median')

            plt.scatter(new_peaks_arr[:, 0], peak_info['peak_heights'], marker='D', color='red', label='R peaks')
            for x in bad_sep[0]:
                plt.axvline(x=new_peaks_arr[x, 0].item(), color='goldenrod', linestyle='--')
                plt.axvline(x=new_peaks_arr[x + 1, 0].item(), color='goldenrod', linestyle='--')
            
            plt.legend()
            plt.title(f'Bad peak sep for idx {start_idx:_d} to {end_idx:_d} in sect {cur_sect}')
            ax.set_xticks(ax.get_xticks(), labels = utils.label_formatter(ax.get_xticks()) , rotation=-30)
            a = 3000
            b = 450
            fig.canvas.manager.window.wm_geometry("+%d+%d" % (a, b))
            timer_error = fig.canvas.new_timer(interval = 3000)
            timer_error.single_shot = True
            timer_cid = timer_error.add_callback(plt.close, fig)
            spacejam = fig.canvas.mpl_connect('key_press_event', onSpacebar)
            timer_error.start()
            plt.show()
            plt.close()

        sect_valid = False

    #Add failure reason to section_info array
    if len(fail_reas) > 0:
        ecg_data['section_info'][cur_sect]['fail_reason'] = fail_reas
    
    return sect_valid, new_peaks_arr, low_counts, IQR_low_thresh

#FUNCTION extract PQRST
def extract_PQRST(
    st_fn:tuple, 
    new_peaks_arr:np.array, 
    peak_info:np.array,
    rolled_med:np.array,
    )->np.array:
    """This function extract's the interior peaks of an ECG signal. 

    Args:
        st_fn (tuple): _description_
        new_peaks_arr (np.array): _description_
        peak_info (np.array): _description_
        rolled_med (np.array): _description_

    Returns:
        np.array: _description_
    """
    def grouper(arr):
        """Mini function for splitting and grouping arrays where the
        differences between values are not equal to 1, split the array at that
        point and return a array of those one step arrays

        Args:
            arr (Section of the ecg): <-----

        Returns:
            array of np.arrays:

        """		
        return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)
    
    #Set Globals
    global ecg_data, wave
    peak_que = deque(new_peaks_arr[:, 0])
    temp_arr = np.zeros(shape=(new_peaks_arr.shape[0], 15), dtype=np.int32)
    temp_counter = 0
    samp_min_dict = {x:int for x in new_peaks_arr[:, 0]}

    if ecg_data['interior_peaks'].shape[0] == 0:
        pass
    elif ecg_data['interior_peaks'][-1, 2] in new_peaks_arr[:, 0]:
        #Load the values from the last interior peak into the temp array
        temp_arr[temp_counter] = ecg_data['interior_peaks'][-1]
        #remove the last row
        ecg_data['interior_peaks'] = ecg_data['interior_peaks'][:-1]

    while len(peak_que) > 1:
        #move in pairs of peaks through the deque
        peak0 = peak_que.popleft()
        peak1 = peak_que[0]

        #Assign the R peaks to the temp array.
        temp_arr[temp_counter, 2]  = peak0
        temp_arr[temp_counter + 1, 2] = peak1

        #First we go through and find the difference between each point.
        grad = np.diff(wave[peak0:peak1+1].flatten())

        #Isolate the sign change of each gradient
        asign = np.sign(grad)
        
        #roll/shift the indices by 1, then subtract  off the sign change to
        #isolate when a wave is shifting from positive to negative or vice
        #versa. 
        signchange = np.roll(np.array(asign), 1) - asign

        #Filter for changes from - -> +  and from - -> 0
        np_inflections = np.where((signchange == -2) | (signchange == -1))[0]
        
        #Filter for changes from + -> -  and from + -> 0
        # pn_inflections = np.where((signchange == 2) | (signchange == 1))[0]

        #Now look at the std deviation from peak0's S peak to peak1's Q peak
            #The variability of the inner RR range minus the huge slopes of the
            #R peaks because we're indexing it from the first sign change from
            #negative to positive on each side.  

            #If that std deviation of that range is greater than 30% of the avg
            #prominence's found within the wave. This serves as an indicator of
            #an abrupt change in signal variability. It will then continue onto
            #the next peak section without extraction of PQST or other metrics
            
            #Prominence is the difference from the highest peak, to lowest
            #valley surrounding a peak. 
        std_dev_rng = wave[peak0:peak1][np_inflections[0]:np_inflections[-1]]
        std_dev_SQ = np.std(std_dev_rng)
        prominences = peak_info['prominences']
        avg_prom = np.mean(prominences)
        threshold = 0.30
        
        reject_limit = threshold * avg_prom

        if std_dev_SQ < reject_limit:
            logger.info(f'peak {peak0} and peak {peak1} S => Q std dev: {std_dev_SQ:.3f} under a 30% threshold of {reject_limit:.3f}')
            logger.info(f'{peak0}:{peak1} std is {std_dev_SQ:.3f} and under threshold of {reject_limit:.3f}')
        else:
            logger.info(f'Skipping Peak {peak0} - {peak1}')
            logger.info(f'{peak0}:{peak1} std is {std_dev_SQ:.3f} and over a threshold of {reject_limit:.3f}')
            temp_counter += 1
            continue

        #MEAS Q peak
        logger.info("adding Q peak")
        temp_arr[temp_counter + 1, 1] = np_inflections[-1] + peak0

        #MEAS S peak
        #Grab left peak
        slope_start = peak0
        #Select first third of R to R distance
        slope_end = peak0 + int((peak1  - peak0)//3) # np_inflections[0] + 1

        #subset that portion of the wave
        lil_wave = wave[slope_start:slope_end].flatten()

        #Cubic splining routine for upsampling. 
        y_ut = lil_wave
        x_ut = np.arange(slope_start, slope_end)
        #Cubic spline Interp values - function for fitting
        f = interp1d(x_ut, y_ut, kind='cubic') 
        x_vals = np.linspace(slope_start, slope_end - 1, num=x_ut.shape[0]*10) #upsampled x_values
        y_vals = f(x_vals) #cubic splines
        #line coefficients from first point to last point.
        coeffs = np.polyfit((x_vals[0], x_vals[-1]), (y_vals[0], y_vals[-1]), 1) #first/last point in x_vals
        y_plot = coeffs[0]*x_vals + coeffs[1]

        def curve_line_dist(point:tuple, coef:tuple)->float:
            """This function calculates the distance from every point
            in our manufactured line, to the curve.  We use this to determine
            where the elbow of a curve is at its maximal.

            Args:
                point (tuple): _description_
                coef (tuple): _description_

            Returns:
                float: _description_
            """			
            d = abs((coef[0]*point[0])-point[1]+coef[1])/np.sqrt((coef[0]*coef[0])+1)
            
            return d

        p_dist = []

        #Iterate through the cubic spline points
        for points in zip(x_vals, y_vals):
            #Find the shortest distance to the line (perpendicular)
            p_dist.append(curve_line_dist(points, coeffs))

        #Find the elbow
        max_dist = max(p_dist)
        #Index it
        max_dist_idx = p_dist.index(max_dist)
        #Find the nearest point on our curve.
        closest = int(np.round(max_dist_idx / 10) + peak0)
        #Store S peak
        temp_arr[temp_counter, 3] = closest

        #Analyzes the samp minimum of the S peak. 
        #This is used when calculating T onsets as I need to know
        #where the true minimum is to evaluate the J point. 
        samp_min = np.argmin(wave[peak0:peak0 + (peak1-peak0)//3])

        #If the sample min is not in the first 5 minimums of that transition, 
        #Let me know.  Could be a sign of signal instability. 
        if (wave[peak0+samp_min].item() < rolled_med[samp_min]) & (samp_min in np_inflections[:6]): 
            samp_min = samp_min + peak0
            logger.info(f'Samp min for peak {peak0:_d}:{peak1:_d} in first 7')
        else: 
            samp_min = min(np_inflections) + peak0
            logger.info(f"Samp min farther out than expected between {peak0:_d}:{peak1:_d}")
            #TODO - Think about keeping the old method (samp min) for a
            # reference point for the T peak.  
        
        samp_min_dict[peak0] = samp_min

        #Filter the range from sampmin to Q in between the R peaks
        SQ_range = wave[samp_min:temp_arr[temp_counter + 1, 1]]

        #Do the same for the rolling median. 
        filt_rol_med = rolled_med[samp_min-st_fn[1]:temp_arr[temp_counter + 1, 1] - st_fn[1]]

        #Subtract rolling median from wave to flatten it.
        SQ_med_reduced = SQ_range - filt_rol_med


        #MEAS T Peak 
        try:
            RR_first_half = SQ_med_reduced[:(SQ_med_reduced.shape[0]//2)]
            peak_T_find = ss.find_peaks(RR_first_half.flatten(), height=np.percentile(SQ_med_reduced, 60))
            top_T = peak_T_find[0][np.argpartition(peak_T_find[1]['peak_heights'], -1)[-1:]]
            temp_arr[temp_counter, 4] = peak0 + (samp_min - peak0) + top_T[0]
            logger.info("adding T peak")

        except Exception as e:
            logger.warning(f"T peak find error for {peak0}. Error message {e}")
            temp_arr[temp_counter, 4] = 0

        #MEAS P Peak 
        try:
            RR_second_half = SQ_med_reduced[(SQ_med_reduced.shape[0]//2):]
            peak_P_find = ss.find_peaks(RR_second_half.flatten(), height=np.percentile(SQ_med_reduced, 60))
            top_P = peak_P_find[0][np.argpartition(peak_P_find[1]['peak_heights'], -1)[-1:]] + RR_first_half.shape[0]
            #Adds the P peak to the next R peaks data.  (as its the P of the next peaks PQRST)
            temp_arr[temp_counter+1, 0] = peak0 + (samp_min - peak0) + top_P[0]
            logger.info("adding P peak")

        except Exception as e:
            logger.warning(f"P peak find error at {peak1}", )
            temp_arr[temp_counter + 1, 0] = 0

        #Final Check to ensure valid PQRST for peak0 before proceeding to interval extraction
        temp_arr[temp_counter, 5] = utils.valid_QRS(temp_arr, temp_counter)
        if temp_arr[temp_counter, 5] == 0:
            peak_dict = {
                 0:'P',
                 1:'Q', 
                 2:'R', 
                 3:'S',
                 4:'T',
            }
            missing_peak = np.where(temp_arr[temp_counter, :5]==0)[0]
            missing_peaks = [peak_dict[x] for x in missing_peak]
            logger.warning(f"Missing peak for {missing_peaks} in section {st_fn[0]}")
            
        #Advance temp_arr counter
        temp_counter += 1
        logger.info(f'finished interior peak extraction between peaks {peak0} and {peak1}')
    
    
    #NOTE Segment Data  Extraction
    #The earlier iteration was looping between each R_peak to get its
    #consitutient peak values. This iteration moves on each individual peak.  
    peak_que = deque(new_peaks_arr[:, 0])
    temp_counter = 0
    while len(peak_que) > 0:
        
        R_peak = peak_que.popleft()
        #Get Q Shoulder
        #Early terminate if not all valid PQRST present.
        if temp_arr[temp_counter, 5]==0:
            logger.info(f'Cannot process segment data for R peak {R_peak}')
            temp_counter += 1
            continue

        #Get all the surrounding peaks for each R peak
        P_peak = temp_arr[temp_counter, 0].item()
        Q_peak = temp_arr[temp_counter, 1].item()
        S_peak = temp_arr[temp_counter, 3].item()
        T_peak = temp_arr[temp_counter, 4].item()
        
        #Setup shoulder containers. 
        P_onset, Q_onset, T_onset, T_offset = [], [], [], []
        
        #Get the width of the QRS for later. 
        srch_width = (S_peak - Q_peak)

        #MEAS Q_onset
        slope_start = Q_peak - int((Q_peak - P_peak)*.70)
        slope_end = Q_peak + 1

        try:
            lil_wave = wave[slope_start:slope_end].flatten()
            lil_grads = np.gradient(np.gradient(lil_wave))
            shoulder = np.where(np.abs(lil_grads) >= np.mean(np.abs(lil_grads)))[0]
            Q_onset = slope_start + shoulder[0] + 1
            temp_arr[temp_counter, 12] = Q_onset
            logger.info(f'Adding Q onset')
        except Exception as e:
            logger.warning(f'Q onset extraction Error = \n{e} for Rpeak {R_peak:_d}')

        #MEAS T onset
        slope_start = samp_min_dict[R_peak]
        slope_end = T_peak + 1
        try:
            lil_wave = wave[slope_start:slope_end].flatten()
            med_sect = rolled_med[slope_start-st_fn[1]:slope_end-st_fn[1]].flatten()
            ecg_greater_med = np.where(lil_wave < med_sect)[0]
            groups = grouper(ecg_greater_med)
            first_group = groups[0]
            T_onset = slope_start + first_group[-1]
            temp_arr[temp_counter, 13] = T_onset
            logger.info('Adding T onset')

        except Exception as e:
            logger.warning(f'T onset extraction Error = \n{e} for Rpeak {R_peak:_d}')

        #MEAS QRS Complex
        #Add the QRS time in ms if both the onsets exist.
        if Q_onset and T_onset:
            temp_arr[temp_counter, 8] = int(1000*((T_onset - Q_onset)/fs))

        #PR Interval
        slope_start = P_peak - int(srch_width)
        slope_end = P_peak + 1
        try:
            lil_wave = wave[slope_start:slope_end].flatten()
            lil_grads = np.gradient(np.gradient(lil_wave))
            P_onset = slope_start + np.argmax(lil_grads)
            temp_arr[temp_counter, 11] = P_onset
            logger.info(f'Adding P onset')
        except Exception as e:
            logger.warning(f'P Onset extraction Error = \n{e} for Rpeak {R_peak:_d}')
        
        #MEAS PR Interval
        if Q_onset and P_onset:
            # Add PR interval in ms
            temp_arr[temp_counter, 7] = int(1000*((Q_onset - P_onset)/fs))
        
        #MEAS ST Segment
        #ST segments are suppressed in this case as the higher heart rate obliterates them. 

        slope_start = T_peak 
        slope_end = T_peak + int(srch_width*1.25)

        try:
            lil_wave = wave[slope_start:slope_end].flatten()
            lil_grads = np.gradient(np.gradient(lil_wave))
            T_offset = slope_start + np.argmax(lil_grads)
            temp_arr[temp_counter, 14] = T_offset
            logger.info(f'Adding T offset')
        except Exception as e:
            logger.warning(f'T Offset extraction Error = \n{e} for Rpeak {R_peak:_d}')
        
        #MEAS QT Interval
        if Q_onset and T_offset:
            #Add QT interval.  
            temp_arr[temp_counter, 10] = int(1000*((T_offset - Q_onset)/fs))

        # Shift the counter
        temp_counter += 1

    return temp_arr


#FUNCTION main peak search
@log_time
def main_peak_search(
    plot_fft:bool,
    plot_errors:bool,
    *args
    ):
    """Detects R Peaks in the ECG wave.  Inputs peak positions into ecg_data dictionary.

    Args:
        plot_fft (bool): boolean of whether or not to graph FFT
        plot_errors (bool): boolean of whether or not to plot errors
    Additional Args.
        This is to comply with the current test suite configuation.  If tests are run, the sample ECG is loaded.  If the main file is being run, the selected CAM will be analyzed.
        ecg_data (dict): Dictionary of peaks and their info
        wave (np.array): Waveform to be analyzed
        fs (float): Sampling rate of the signal

    Returns:
        ecg_data (dict): dictionary with full dataset on waveform
    """
    if len(args) != 0:
        global ecg_data, wave, fs
        ecg_data = args[0][0]
        wave = args[0][1]
        fs = args[0][2]
        

    #section tracking + invalid section tracking
    section_counter, invalid_sect_counter = 0, 0
    #Whether the wave is found
    found_wave = False
    #Whether dynamic STFT is in a countdown
    stft_loop_on = False
    stft_count = 0
    #Sample ranges to test the array stacking to ensure we're not getting slowdowns there. 
    #Round down to the nearest 100k
    stack_range = [x for x in range(0, int(np.round(np.floor(ecg_data["section_info"].shape[0]), -2)), 5_000)]
    #Stacking test for peak addition	
    
    @log_time
    def peak_stack_test(new_peaks_arr:np.array):
        return np.vstack((ecg_data['peaks'], new_peaks_arr)).astype(np.int32)

    global IQR_low_thresh
    #Set IQR threshold and low count tracker
    IQR_low_thresh = 1
    low_counts = 0

    #Load up a deque of start and end sections ( made by segment ECG)
    sect_que = deque(ecg_data['section_info'][['start_point', 'end_point']])

    while len(sect_que) > 0:
        curr_section = sect_que.popleft()
        start_p = curr_section[0]
        end_p = curr_section[1]

        wave_chunk = wave[start_p:end_p]
        
        #Calculated the  rolling median of the wave chunk.
        rolled_med = utils.roll_med(wave_chunk).astype(np.float32)

        #Grab the overlap between the current section and the previous section. 
        if section_counter == 0:
            shift = 0
        else:
            shift = ecg_data['section_info'][section_counter-1]['end_point'] - ecg_data['section_info'][section_counter]['start_point']

        #Add the rolling median with the overlap removed. 
        ecg_data['rolling_med'][start_p + shift:end_p] = rolled_med[shift:].flatten()

        #Run R peak search with scipy
        R_peaks, peak_info = ss.find_peaks(
            wave_chunk.flatten(), 
            prominence = np.percentile(wave_chunk, 99), #99 -> stock
            height = np.percentile(wave_chunk, 95),     #95 -> stock
            distance = round(fs*(0.200)) #Can't have a heart rate faster than 200ms
        )  

        #Set the section validity to False
        sect_valid = False

        #If the first wave section hasn't been found.
        if not found_wave:
            #Look for early signal reject. Early signal rejection is if the
            #signal is complete garbage. ie - has too many or too little peaks.
            if R_peaks.size < 4 or R_peaks.size > 60:
                logger.warning(f'Num of peaks error for section {section_counter}\nR_peaks_val.size < 4 or > 60')
                ecg_data['section_info'][section_counter]['valid'] = 0
                ecg_data['section_info'][section_counter]['fail_reason'] = "no_sig"
                
            else:
                #Shift the R peaks to align with the start point. 
                R_peaks_shifted = R_peaks + start_p
                
                # reshape it into a 1D array so you can stack it. 
                new_peaks = R_peaks_shifted.reshape(-1, 1)

                #make an empty array of zeros to hold the validity of the R peak
                valid_mask = np.zeros(shape=(len(new_peaks[:, 0]),1), dtype=int)

                #stack the new peaks and valid mask into a single array
                new_peaks_arr = np.hstack((new_peaks, valid_mask))
                
                #Validate the section with a STFT
                sect_valid, new_peaks_arr = STFT(
                    new_peaks_arr, 
                    peak_info, 
                    rolled_med, 
                    (section_counter, start_p, end_p), 
                    plot_fft
                )

                # If you've found the wave and have sufficient num of peaks
                if sect_valid and R_peaks.size > 10:
                    found_wave = True
                    start_sect = section_counter #BUG Possible bug 
                    ecg_data['section_info'][section_counter]['valid'] = 1
                    logger.critical(f'Wave found at {start_p}:{end_p} in section {start_sect}')

                    #Add the current  R peaks to the "peaks" and "interior_peaks" data container. 
                    ecg_data['peaks'] = np.vstack((ecg_data['peaks'], new_peaks_arr)).astype(np.int32)

                    #Make temp container for interior peaks
                    int_peaks = np.zeros(shape=(new_peaks.shape[0], 15), dtype=np.int32)
                    #Add the R peaks to the interior_peaks container. 
                    int_peaks[:, 2] = new_peaks_arr[:, 0]
                    ecg_data['interior_peaks'] = np.vstack((ecg_data['interior_peaks'], int_peaks))
                    
                    #TODO extract PQRST and section stats?  
                        #Not sure i can do that with no historical data

            #In either case advance the section counter forward and keep looking
            #for the first sign of a signal
            section_counter += 1
            continue

        else:
            #WAVE FOUND BELOW
            #Shift the start point to match the wave indices
            R_peaks_shifted = R_peaks + start_p

            #Compare the new peaks, to the last 20 in ecg_data['peaks']  
            #Does a set intersection to find the common peaks 
            #between the two sets.
            same_peaks = sorted(list(set(R_peaks_shifted) & set(ecg_data['peaks'][-20:,0]))) #+start_p
            
            if len(same_peaks) > 0:
                last_peak = max(same_peaks)
                new_peaks = list(set(R_peaks_shifted) - set(same_peaks))
                new_peaks.append(last_peak)
                new_peaks = sorted(new_peaks)
                
                #Need to pop off the last R Peak as it will always be zero (last in the peak loop)
                ecg_data['peaks'] = ecg_data['peaks'][:-1,:]  
                peak_info['peak_heights'] = peak_info['peak_heights'][len(same_peaks)-1:]
                peak_info['prominences'] = peak_info['prominences'][len(same_peaks)-1:]
                new_peaks = np.array(new_peaks).reshape(-1, 1)
            else:
                new_peaks = R_peaks_shifted.reshape(-1, 1)

            #Set the valid mask to ones for the R peaks. 
            #We use ones because the historical validation check will seek to invalidate sections. 
            #I know it seems backwards but it works better this way. 
            valid_mask = np.ones(shape=(len(new_peaks[:, 0]),1), dtype=int)

            #concat the peaks with the valid_mask of zeros
            new_peaks_arr = np.hstack((new_peaks, valid_mask))
            
            #Making sure we have enough historical data to scan backwards in time. 
            #Make sure the section counter is at least 10 ahead of the start_sect
            if section_counter < start_sect + 10:
                sect_valid, new_peaks_arr = STFT(
                    new_peaks_arr, 
                    peak_info, 
                    rolled_med, 
                    (section_counter, start_p, end_p), 
                    plot_fft
                )
                logger.info(f'Building up time for historical data Section:{section_counter}')			
            
            #Still need a quick peak count check. Found 1 edge case that got through
            #and messed up a 2 hour section. 
            elif new_peaks_arr.shape[0] < 4:
                sect_valid = False
                fail_reas = "Not enough peaks"
                ecg_data['section_info'][section_counter]['fail_reason'] = fail_reas
                logger.critical(f'Peak Validation fail sect:{section_counter} idx:{start_p}->{end_p} Reason: {fail_reas}')
                new_peaks_arr[:, 1] = 0

            elif stft_loop_on:
                logger.warning(f'STFT cooldown loop.  Section: {section_counter} Counter at : {stft_count}')
                sect_valid, new_peaks_arr = STFT(
                    new_peaks_arr, 
                    peak_info, 
                    rolled_med, 
                    (section_counter, start_p, end_p),
                    plot_fft
                )

                stft_count -= 1
                #If cooldown is finished, resume historical peak averages
                if stft_count == 0:
                    stft_loop_on = False

                #Make sure to mark the section as invalid due to FFT. 
                if not sect_valid:
                    ecg_data['section_info'][section_counter]['fail_reason'] = "FFT"
                    logger.critical(f'Peak Validation fail sect:{section_counter} idx:{start_p}->{end_p} Reason: FFT')		

            #Checking our bad section counter. More than 10 and we switch back to STFT.  
            elif invalid_sect_counter > 10:
                stft_loop_on = True
                stft_count = 5
                logger.critical(f'Signal lost in section {section_counter} Switching to STFT')
                sect_valid, new_peaks_arr = STFT(
                    new_peaks_arr, 
                    peak_info, 
                    rolled_med, 
                    (section_counter, start_p, end_p), 
                    plot_fft
                )

                if not sect_valid:
                    ecg_data['section_info'][section_counter]['fail_reason'] = "FFT"
                    logger.critical(f'Peak Validation fail sect:{section_counter} idx:{start_p}->{end_p} Reason: FFT')
            else:
                #Set the section validity for Peak Validation
                PV_sect_valid = False
                #Grab the last consecutive peaks that are marked as valid
                last_keys = consecutive_valid_peaks(ecg_data['peaks'])
                #Run Peak validation check based oh historical avgs
                PV_sect_valid, new_peaks_arr, low_counts, IQR_low_thresh = peak_validation_check(new_peaks_arr, last_keys, peak_info, rolled_med, (section_counter, start_p, end_p), low_counts, IQR_low_thresh, plot_errors)

                if not PV_sect_valid: 
                    fail_reas = ecg_data['section_info'][section_counter]['fail_reason']
                    logger.critical(f'Peak Validation fail sect:{section_counter} idx:{start_p}->{end_p} Reason: {fail_reas}')
                    sect_valid = False
                else:
                    sect_valid = True

            if sect_valid:
                #Mark section as good.  Reset invalid sect counter
                ecg_data['section_info'][section_counter]['valid'] = 1

                #If we're in a valid section, reduce the invalid sect count by one. 
                #Ensure the invalid_sect counter is positive to prevent runaways
                if invalid_sect_counter > 0:
                    invalid_sect_counter -= 1
                
                #Add HR stats for that section
                sect_stats = section_stats(new_peaks_arr, section_counter)
                if sect_stats:
                    ecg_data['section_info'][section_counter]['Avg_HR'] = sect_stats[0]
                    ecg_data['section_info'][section_counter]['SDNN'] = sect_stats[1]
                    ecg_data['section_info'][section_counter]['min_HR_diff'] = sect_stats[2]
                    ecg_data['section_info'][section_counter]['max_HR_diff'] = sect_stats[3]
                    ecg_data['section_info'][section_counter]['RMSSD'] = sect_stats[4]
                    ecg_data['section_info'][section_counter]['NN50'] = sect_stats[5]
                    ecg_data['section_info'][section_counter]['PNN50'] = sect_stats[6]

                # Pull out interior peaks and segment data for QRS, PR, QT, etc
                int_peaks = extract_PQRST((section_counter, start_p, end_p), new_peaks_arr, peak_info, rolled_med)
            
                # Stack the interior peak data into data container
                ecg_data['interior_peaks'] = np.vstack((ecg_data['interior_peaks'], int_peaks))

            else:
                #Mark section as bad
                ecg_data['section_info'][section_counter]['valid'] = 0
                # Limit the amount of invalid sections it can grow too. This
                # will cause the detector to use the STFT for 10 sections before
                # it resumes peak validation techniques (if it grows to that
                # high).  Giving it time to regenerate recent historical
                # averages to then compare to the current section.

                if invalid_sect_counter < 15:
                    invalid_sect_counter += 1

            logger.info(f'Invalid count {invalid_sect_counter} in section {section_counter}')

            if section_counter in stack_range: 
                ecg_data['peaks'] = peak_stack_test(new_peaks_arr)
            else:
                ecg_data['peaks'] = np.vstack((ecg_data['peaks'], new_peaks_arr)).astype(np.int32)
            

            #Advance section tracker to next section
            section_counter += 1
            logger.info(f'Section counter at {section_counter}')

    return ecg_data

#Save ECG data to file and send confirmation email that the run is done. 
#FUNCTION send notification email
def send_email(log_path:str):
    if os.getcwd().endswith('scripts'):
        pass
    else:
        with open(log_path, 'r') as f:
            line = f.readlines()[-1:][0]
            peak_search_runtime = line.split("|")[4].strip("")

        support.send_run_email(peak_search_runtime)
        logger.warning("Runtime email sent")
        logger.warning(f"{peak_search_runtime}")

#NOTE START PROGRAM
def main():
    #TODO - Add overall prog to log output in terminal
    #TODO - Nest logger functions and declare as global. maybe

    #Load data 
    ecg_data, wave, fs, configs = setup_globals.init(__name__, logger)
    
    #Run peak search extraction
    ecg_data = main_peak_search(
        configs["plot_fft"],
        configs["plot_errors"],
        (ecg_data, wave, fs)
    )
    #Save logs, results, send update email
    # send_email(log_path)
    use_bucket = configs.get("gcp_bucket")
    has_bucket_name = len(configs.get("bucket_name")) > 0
    configs["log_path"] = log_path
    if use_bucket & has_bucket_name :
        support.save_results(ecg_data, configs, logger, current_date, True)
    else:
        support.save_results(ecg_data, configs, logger, current_date)

    logger.info("Woo hoo!\nECG Analysis Complete")

if __name__ == "__main__":
    main()


#IDEA?  - What if you have a function similar to slider
    #To inspect the ECG before its run.  Would be ideal if 
    #it ran in the cloud for plotting but i'm not sure 
    #how that renders locally
    