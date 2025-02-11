import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from os.path import exists
from support import log_time, logger
from dataclasses import dataclass, fields
import scipy.signal as ss
import numpy as np

@dataclass
class PeakInfo():
    P_peak:int
    Q_peak:int
    R_peak:int
    S_peak:int
    T_peak:int
    P_onset:int
    P_offset:int
    T_onset:int
    T_offset:int
    peaks_p:dict # dict of p_peak finds
    peaks_r:dict # dict of r_peak finds
    peaks_t:dict # dict of t_peak finds
    r_peaks:list # list of r peak indexes
    isoelectric:float
    rr_means:float
    PR:float
    QRS:float
    ST:float
    QT:float
    fs:float
    plot_search:bool=False
    Q_onset:int=np.nan
    J_point:int=np.nan
    # uwave:bool=False

################################ Calculations ############################
#FUNCTION -> estimate_iso
def estimate_iso(wave:np.array, data:PeakInfo) -> float:
    """This function does a few things.  First it estimates the isoelectric line by looking at the range of the wave up to the P_onset.  
    It also assigns the Q peak and calculates the Q onset

    Args:
        wave (np.array): template to be searched
        data (PeakInfo): _description_

    Returns:
        float: _description_
    """    

    #Isolate P peaks at lower threshold.  Start by grabbing the Q peak.  Search for the P peaks at a lower threshold
    data.Q_peak = data.peaks_r[1]["left_bases"].item()
    data.peaks_p = pull_other_peak(wave[:data.Q_peak].flatten(), data)
    #Sometimes findpeaks returns more peaks than you expect for the P peak. (i think due to the mix of pos / negative numbers)
    #this will ensure only the tallest peak get pulled back for P
    if data.peaks_p[0].shape[0] > 1:
        tallest = np.argsort(data.peaks_p[1]["peak_heights"])[::-1][0]
        data.P_peak = data.peaks_p[0][tallest].item()
        data.P_onset = data.peaks_p[1]["left_bases"][tallest].item()
        #BUG - P_offset misbehaving.  Probably needs to switch to the acceleration method here with limited range. 
            #Mainly due to the multiple sign changes in between the P peak and Q peak.
        data.P_offset = data.peaks_p[1]["right_bases"][tallest].item()

    else:
        data.P_peak = data.peaks_p[0][0].item()
        data.P_onset = data.peaks_p[1]["left_bases"][0].item()
        data.P_offset = data.peaks_p[1]["right_bases"][0].item()

    data.isoelectric = np.nanmean(wave[:data.P_onset])

    #Grab Q onset
    slope_start = data.Q_peak - int((data.Q_peak - data.P_peak)*.30)
    slope_end = data.Q_peak + 1

    try:
        lil_wave = wave[slope_start:slope_end].flatten()
        lil_grads = np.gradient(np.gradient(lil_wave))
        shoulder = np.where(np.abs(lil_grads) >= np.mean(np.abs(lil_grads)))[0]
        data.Q_onset = slope_start + shoulder[0] + 1
    except Exception as e:
        logger.warning(f'Q onset extraction Error = \n{e}')

    #Because we always graph to check
    #TODO - Check this during final review
    # fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(12,6))
    # iso_area = wave[:data.P_onset]
    #TODO - Add vertical lines of the max min of the iso area.  At the T peak height
    # ax.plot(range(R_peak), wave[:R_peak], label = 'ECG to R')
    # ax.plot(range(data.P_onset), iso_area, label='searcharea')
    # ax.vline(data.)
    # ax.scatter(R_peak, data.peaks_r[1]['peak_heights'][0], marker='D', color='red', label='R peak')
    # plt.show()
    # plt.close()
    
    return data

#FUNCTION -> pull_R_peak
def pull_R_peak(wave:np.array)-> list:
    """R peak isolation through scipy find peaks.

    Args:
        wave (np.array): Input template

    Returns:
        list: return from ss.find_peaks.  [0] is the peak index, [1] is the peak information
    """
    r_peaks = ss.find_peaks(
        wave.flatten(), 
        prominence = np.percentile(wave, 99), #99 -> stock
        height = np.percentile(wave, 97),     #95 -> stock
    )
    return r_peaks

#FUNCTION -> pull_other_peak
def pull_other_peak(wavesect:np.array, data:PeakInfo)-> list:
    """T or P peak isolation through scipy find peaks.

    Args:
        wave (np.array): Input template

    Returns:
        list: return from ss.find_peaks.  [0] is the peak index, [1] is the peak information
    """
    other_peaks = ss.find_peaks(
        wavesect.flatten(), 
        prominence = np.percentile(wavesect, 60), 
        height = np.percentile(wavesect, 40),
    )

    # fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(12,6))
    # ax.plot(range(wavesect.shape[0]), wavesect, label = 'search section')
    # ax.scatter(other_peaks[0], other_peaks[1]["peak_heights"], marker=' D', color='red', label='possibles')
    # # ax.scatter(R_peak, data.peaks_r[1]['peak_heights'][0], marker='D', color='red', label='R peak')
    # plt.show()
    # plt.close()

    return other_peaks

#FUNCTION -> T_onset
def calc_T_onset(wave:np.array, srch_width:int, T_info: tuple) -> int:
    """_summary_

    Args:
        wave (np.array): _description_
        srch_width (int): _description_
        T_info (tuple): _description_

    Returns:
        int: _description_
    """    
    slope_start = T_info[0] - srch_width
    slope_end = T_info[0] + 1
    
    try:
        gradients = np.gradient(np.gradient(wave[slope_start:slope_end]))
        T_onset = slope_start + np.argmax(gradients)
        return T_onset.item()
    
    except Exception as e:
        logger.warning(f'T Onset extraction error = \n{e}')

#FUNCTION -> T_offset
def calc_T_offset(wave:np.array, srch_width:int, T_info: tuple) -> int:
    """_summary_

    Args:
        wave (np.array): _description_
        srch_width (int): _description_
        T_info (tuple): _description_

    Returns:
        int: _description_
    """    
    slope_start = T_info[0] 
    slope_end = T_info[0] + int(srch_width * 0.75)

    try:
        gradients = np.gradient(np.gradient(wave[slope_start:slope_end]))
        T_offset = slope_start + np.argmax(gradients)
        return T_offset.item()
    
    except Exception as e:
        logger.warning(f'T Offset extraction Error = \n{e}')
        #Backup offset method.
        #(Dev in separate function but will come back and update when finished)
        # try:
        #     gradient = np.gradient(wave[slope_start:slope_end])

        # except Exception as e:
        #     logger.warning(f'T Onset backup extraction error = \n{e}')

#FUNCTION -> uwave?
def u_wave_present() -> bool:
    pass
    #Ways to detect U wave. 
    #1.Klein Method. 
        #Fit a gaussian (scipy.optimize) to that section post T peak. (Good idea honestly)
        
    #2.Signchanges
        #possibly look at sign changes in the post T peak area

#FUNCTION -> T_regress
def calc_T_regress(wave:np.array, T_peak:int, T_offset:int) -> int:
    """_summary_

    Args:
        wave (np.array): _description_
        T_peak (int): _description_
        T_offset (int): _description_

    Returns:
        int: _description_
    """    
    slope_start = T_peak
    slope_end =  T_offset
    
    try:
        m, b = np.polyfit(range(slope_start, slope_end), wave[slope_start:slope_end], 1)
        x_intercept = -b / m

        return  x_intercept.item(), m.item(), b.item()

    except Exception as e:
        logger.warning(f'T regression extraction error = \n{e}')
    
    #Gameplan, implement NIH standard method of taking the slope off the T peak and extending it to the isoelectric line 
    #to estimate the offset position.  (Should be slightly longer than the acceleration method)
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7080915

#FUNCTION -> J_point
def calc_J_point(wave:np.array, data:PeakInfo, threshold:float=0.005):
    """Calculates the J point of the ECG by focusing on the S peak to T onset range.  Extracting the point of greatest acceleration change which should signify a shoulder.

    Args:
        wave (np.array): Template to be examined
        data (PeakInfo): dataclass container

    Returns:
        J_point(int): Location of the J point if findable
    """    
    #Grab Speak and T peak... or T onset
    slope_start = data.S_peak + 1
    slope_end = slope_start + data.QRS*2 #int((data.T_onset - slope_start)*.40)

    try:
        lil_wave = wave[slope_start:slope_end].flatten()
        lil_grads = np.gradient(lil_wave)
        # shoulder = np.where(np.abs(lil_grads) <= np.mean(np.abs(lil_grads)))[0]
        # np.argmax(lil_grads < threshold)
        # shoulder[0] + 1  
        # np.argmax(lil_grads)
        # J_point = slope_start + shoulder[0]
        J_point = slope_start + np.argmin(lil_grads > threshold)
        logger.info(f"J point at {J_point}")
    except Exception as e:
        logger.warning(f'J point extraction Error = \n{e}')
        J_point = np.nan

    return J_point 

################################ Plotting / Exporting ############################

#FUNCTION -> plot_results
def plot_results(
    wave:np.array, 
    wave_sub:np.array,
    idx:int,
    diff:int,
    data:PeakInfo,
    t_bases:tuple,
    r_bases:tuple,
    ):
    """_summary_

    Args:
        wave (np.array): _description_
        idx (int): Template index of analysis
        diff (int): _description_
        data (PeakInfoTemplateindexofanalysis): _description_
        t_bases (tuple): _description_
        r_bases (tuple): _description_

    """ 
    r_lb = r_bases[0]
    r_rb = r_bases[1]
    # t_lb = t_bases[0]
    # t_rb = t_bases[1]

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 8))
    ax.plot(range(wave.shape[0]), wave, label = "Template wave", color="black")
    # ax.plot(range(diff, diff + wave_sub.shape[0]), wave_sub, label = "Search area", color="dodgerblue")
    ax.axhline(y=data.isoelectric, label="Isolelectric", linestyle="-.")
    

    #Add segment rectangles with appropriate colors / heights / and annotations.  
        #Possibly use a bracket
    #QRS
    label_height = data.isoelectric + wave[data.R_peak]*0.5 
    bigfonts = 16
    bwidth = data.J_point - data.Q_onset
    center = (data.Q_onset + 0.12*bwidth, label_height)
    rect = Rectangle(
        xy=(data.Q_onset, data.isoelectric), 
        width=bwidth,  
        height=data.isoelectric + wave[data.R_peak], 
        facecolor="tomato",
        edgecolor="grey",
        alpha=0.2)
    ax.add_patch(rect)
    ax.annotate(
        text=f"QRS",
        xy=center,
        fontsize = bigfonts,
        xycoords="data",
        color="black"
    )
    #PR
    bwidth = data.Q_onset - data.P_onset
    center = (data.P_onset + 0.25*bwidth, label_height)
    rect = Rectangle(
        xy=(data.P_onset, data.isoelectric), 
        width=bwidth, 
        height=data.isoelectric + wave[data.R_peak], 
        facecolor="dodgerblue",
        edgecolor="grey",
        alpha=0.2)
    ax.add_patch(rect)
    ax.annotate(
        text=f"PR",
        xy=center,
        fontsize = bigfonts,
        xycoords="data",
        color="black"
    )
    #ST
    bwidth = data.T_offset - data.J_point
    center = (data.J_point + 0.25*bwidth, label_height)
    rect = Rectangle(
        xy=(data.J_point, data.isoelectric), 
        width=bwidth, 
        height=data.isoelectric + wave[data.R_peak],
        facecolor="lightgreen",
        edgecolor="grey",
        alpha=0.2)
    ax.add_patch(rect)
    ax.annotate(
        text=f"ST",
        xy=center,
        fontsize = bigfonts,
        xycoords="data",
        color="black"
    )

    #QT
    bwidth = data.T_offset - data.Q_onset
    center = (data.Q_onset + 0.5*bwidth, -0.5*label_height)
    rect = Rectangle(
        xy=(data.Q_onset, data.isoelectric), 
        width=bwidth, 
        height=-wave[data.R_peak]*0.5, 
        facecolor="violet",
        edgecolor="grey",
        alpha=0.2)
    ax.add_patch(rect)
    ax.annotate(
        text=f"QT",
        xy=center,
        fontsize = bigfonts,
        xycoords="data",
        color="black"
    )
    #Annotate values bottom right of legend
    metrics = [
        ("QRS", data.QRS),
        ("PR", data.PR),
        ("ST", data.ST),
        ("QT", data.QT)
    ]
    stat_string = ""
    for col, val in metrics:
        fcol = col.ljust(3)
        stat_string += f"{fcol} => {val:3.1f} ms\n"
    
    # stat_string = f"{'QRS:':5}{data.QRS:3.1f} ms\n" + f"{'PR:':7}{data.PR:3.1f} ms\n" + f"{'ST:':7}{data.ST:3.1f} ms\n" + f"{'QT:':7}{data.QT:3.1f} ms\n" 
    # stat_string = f"QRS:    {data.QRS:3.1f} ms\n" + f"PR:    {data.PR:3.1f} ms\n" + f"ST:    {data.ST:3.1f} ms\n" + f"QT:    {data.QT:3.1f} ms\n" 
    ax.annotate(
        text=stat_string[:-1],
        xy=(0.76, 0.1),
        fontsize = bigfonts,
        xycoords="axes fraction",
        color="black",
        xytext=(0, 0),
        textcoords="offset points",
        alpha=0.8,
        multialignment="left",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey") 
    )

    #P peak
    ax.scatter(data.P_peak, wave[data.P_peak], s=50, label="P peak", color="magenta")
    ax.scatter(data.P_onset, wave[data.P_onset], s=40, label="P left base", color="magenta", marker="<")
    # ax.scatter(data.P_offset, wave[data.P_offset], s=40, label="R right base", color="magenta", marker=">")

    #Rpeak and LR base
    ax.scatter(data.R_peak, wave[data.R_peak], s=50, label="R peak", color="red")
    ax.scatter(r_lb, wave[r_lb], s=40, label="R left base/Q peak", color="green", marker="<")
    ax.scatter(r_rb, wave[r_rb], s=40, label="R right base/S peak", color="goldenrod", marker=">")

    #Q_onset
    ax.scatter(data.Q_onset, wave[data.Q_onset], s=50, label="Q onset", color="green")

    # #Tpeak and LR base
    ax.scatter(data.T_peak, wave[data.T_peak], s=50, label="T peak", color="orange")
    # ax.scatter(diff + t_lb, wave_sub[t_lb], s=40, label="T left base", color="orange", marker="<")
    # ax.scatter(diff + t_rb, wave_sub[t_rb], s=40, label="T right base", color="orange", marker=">")

    #TOnset
    ax.scatter(data.T_onset, wave[data.T_onset], s=60, label="T onset", color="orange", marker="*")
    #TOffset
    ax.scatter(data.T_offset, wave[data.T_offset], s=60, label="T offset", color="orange", marker="*")
    #Jpoint
    ax.scatter(data.J_point, wave[data.J_point], s=60, label="J point", color="dodgerblue", marker="p")

    #Backup T Offset calculation.
    Tbackup, m, b = calc_T_regress(wave, data.T_peak, data.T_offset)
    
    #Tbackup
    ax.axvline(Tbackup, color="blue", linestyle="--", label="X-Intercept")
    #generate 100 samples along the tangent line
    x_tans = np.linspace(data.T_peak - 10, data.T_offset + 30, 100)
    y_tans = m * x_tans + b
    ax.plot(x_tans, y_tans, color="orange", linestyle="--", label="T peak tangent line")
    Tcross = np.abs(y_tans - data.isoelectric).argmin()
    ax.scatter(x_tans[Tcross], data.isoelectric, s=50, label="Isoelectric crossing point", color="red", marker="*")
    ax.set_xlabel("Time (idx)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(f"Template {idx} peak results")
    plt.legend(loc="upper right")
    # plt.savefig(f"./data/output/{idx}.png", format="png")
    plt.show()
    plt.close()

#FUNCTION -> export dataclass
def dump_dataclass(data:PeakInfo):
    order = ["P_onset", "P_peak", "P_offset", "R_onset", "Q_peak", 
             "R_peak", "S_peak", "R_offset", "T_peak", "T_offset"]
    ordered_peaks = []
    for location in order:
        if location in fields(data):
            value = getattr(data, location)
            if value:
                ordered_peaks.append(value)
            else:
                ordered_peaks.append(np.nan)
        else:
            raise ValueError(f"{location} not in order")
        
    return ordered_peaks

################################ Base Functions ############################
#FUNCTION -> calc_assets
def calc_assets(wave:np.array, data:PeakInfo)-> list:
    """Extraction function for pulling peaks out of an ECG template

    Args:
        wave (np.array): template to be analyzed
        data (PeakInfo): dataclass with all pertinent peak information

    Returns:
        list: formatted PeakInfo filled with peak locations
    """   
    u_wave = False

    #Grab S peak
    #TODO - S peak update
        # Add logic here that uses acceleration method to validate S peak location off the R peak.  Some cases exist where the S peak depression might be reduced. 
        # Consider upsampling technique for consistent elbow extraction.  
    data.S_peak = data.peaks_r[1]["right_bases"].item()
    wave_sub = wave[data.S_peak:]

    #Calc the diff of the two frames in which the peaks were extracted.
    # aka. All the tpeak info is pulled from the speak forward in the template.
    diff =  wave.shape[0] - wave_sub.shape[0]

    #Grab Isoelectric line. 
    data = estimate_iso(wave, data)

    #Search for T
    data.peaks_t = pull_other_peak(wave_sub.flatten(), data)

    #Grab the indexes of the left and right bases for T and R peaks
    t_idx = data.peaks_t[0][0].item()
    t_lb = data.peaks_t[1]["left_bases"][0].item()
    t_rb = data.peaks_t[1]["right_bases"][0].item()
    r_lb = data.peaks_r[1]["left_bases"][0].item()
    r_rb = data.peaks_r[1]["right_bases"][0].item()

    # Tonset and Toffset
    #Use twice the QRS as a search width for isolating onset/offset
    data.QRS = r_rb - r_lb
    srch_width = data.QRS*2
    
    data.T_onset = calc_T_onset(wave_sub, srch_width, (t_idx, t_lb, t_rb)) + diff
    data.T_offset = calc_T_offset(wave_sub, srch_width, (t_idx, t_lb, t_rb)) + diff
    data.T_peak = t_idx + diff

    #Grab J point
    data.J_point = calc_J_point(wave, data)
    data.PR = round(float(1000*((data.Q_onset - data.P_onset)/data.fs)), 2)
    data.QT = round(float(1000*((data.T_offset - data.Q_onset)/data.fs)), 2)
    data.QRS = round(float(1000*((data.J_point - data.Q_onset)/data.fs)), 2)
    data.ST = round(float(1000*((data.T_offset - data.J_point)/data.fs)), 2)

    #TODO - Eventually implement U wave search here. 
    # uwave = u_wave_present(T_peak, Toffset)

    if data.plot_search:
        plot_results(
            wave, 
            wave_sub,
            diff,
            data,
            (t_lb, t_rb),
            (r_lb, r_rb),
        )
        
    return data

#FUNCTION -> calc confidences
def calc_confidences(data:PeakInfo):
    pass

#FUNCTION -> run_extract
def run_template_extract(
    input_signal:np.array,         #Entire ECG
    sampling_frequency:float,      #Samp freq
    template_signal:np.array,      #Averaged signal
    template_annotations:np.array, #List of R peaks from neurokit
    template_rr:float,             #RR mean
    tracking_points:list,          #List of template peaks/offsets
    plot_steps:bool                #Whether to plot graph
    ):

    #Create object to fill in.
    data = PeakInfo

    #Load template data - Original
    wave = template_signal
    data.r_peaks = template_annotations
    data.rr_means = template_rr
    data.fs = sampling_frequency
    data.plot_search = False
    
    #Isolate R-peak
    data.peaks_r = pull_R_peak(wave)
    
    #If there is one R peak, calculate the other peaks in the template.
    if data.peaks_r[0].shape[0] == 1:
        data.R_peak = data.peaks_r[0].item()
        data = calc_assets(wave, data)
        #Offload findings back to the template
        tracking_points = dump_dataclass(data)
        confidences = calc_confidence(data)
        #BUG - Need a way to calculate the confidences. 
        return tracking_points, confidences
    
    else:
        raise ValueError("Too many R peaks discovered.\nChange parameters and run again")
    

    


#FUNCTION -> main
################################# Main Function ####################################    
# @log_time
# def main():
#     #Path Setup
#     run_date = datetime.now().strftime("%m-%d-%YT%H:%M:%S")
#     fp = PurePath(Path.cwd(), Path(f"./data/original/adaptive_switched_v2.pickle"))

#     # Load S&P Data
#     if exists(fp):
#         templates = support.pickle_load(fp)
#         logger.info("Pickle data loaded")
#     else:
#         templates = []
#         logger.warning("No historical data found")

#     #Run extraction
#     tp = run_template_extract(templates, PeakInfo)
    
#     #If data found, save it, otherwise log an error
#     if tp:
#         support.pickle_data(tp, "results")
#     else:
#         logger.warning("Retrieval malfunction.  Check logs")

# if __name__ == '__main__':
#     main()
            
#TODO - Update main.py to Paolo iteration loop
    # This main function needs a singular call to update one template at a time and not a list of templates. 
        #Preferred inputs are in secret/inputs-for-paolo
    # For delineation confidence,
    #  we're looking at average std deviation of a segment measurment.  Easy
    #  enough to pull from detector as its just the std dev for the template
    #  section in the interior_peaks object.
