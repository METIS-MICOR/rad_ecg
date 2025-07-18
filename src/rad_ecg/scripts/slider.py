# import matplotlib
# matplotlib.use('TkAgg')
import scipy.signal as ss
from scipy.fft import rfft, rfftfreq, irfft
import numpy as np
import stumpy 
import pandas as pd
from rich.table import Table
from collections import Counter
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox, SpanSelector
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import utils # from rad_ecg.scripts
import setup_globals # from rad_ecg.scripts 
from support import logger, console, log_time
import support

PEAKDICT = {
    0:('P', 'green'),
    1:('Q', 'cyan'),
    2:('R', 'red'),
    3:('S', 'magenta'),
    4:('T', 'black')
}
PEAKDICT_EXT = {
    11:('P_on', 'purple'),
    12:('Q_on', 'darkgoldenrod'), 
    13:('T_on', 'teal'), 
    14:('T_off', 'orange'),
    15:('J_poi', 'dodgerblue')
}
LABELDICT = {
    "P":(0, 7),
    "Q":(-11, -4),
    "R":(10, -4),
    "S":(-5, 15),
    "T":(0, 5),
    "P_on":(0, -10),
    "Q_on":(-11, -4),
    "T_on":(10, 0),
    "T_off":(10, 10),
    "J_poi":(-10, 15),
}

#FUNCTION Make rich table
def make_rich_table(failures:dict) -> Table:
    error_table = Table(
        expand=False,
        show_header=True,
        header_style="bold",
        title=f"[red][b]Errors Counts[/b]",
        title_justify = "center",
        highlight=True,
    )
    error_table.add_column(f"Reason", justify="left") 
    error_table.add_column(f"Count", justify="center")
    t_sorts = sorted(failures.items(), key= lambda x:len(x[0]), reverse=True)
    for reason in t_sorts:
        if reason[0] == " ":
            fail = "Valid"
        else:
            fail = reason[0].strip()
            
        error_table.add_row(fail, str(reason[1]))
    error_table.add_section()

    error_table.add_row("% Valid", f"{(failures.get(' ') / sum(failures.values())):.1%}")
    error_table.add_row("Total Sections", str(sum(failures.values())))
    
    return error_table

def load_graph_objects(datafile:str, outputf:str):
    #Ugh, i have to rewrite all of this.  
    #FUNCTION Add chart labels
    def add_cht_labels(x:np.array, y:np.array, plt, label:str):
        """[Add's a label for each type of peak]
    
        Args:
            x (np.array): [idx's of peak data]
            y (np.array): [peak data]
            plt ([chart]): [Chart to add label to]
            label (str, optional): [Title of the chart.  Key to the dict of where its label should be shifted]. Defaults to "".
        """
        for x, y in zip(x,y):
            label = f'{label[0]}' #:{y:.2f}
            plt.annotate(
                label,
                (x,y),
                textcoords="offset points",
                xytext=LABELDICT[label[0]],
                ha='center')
    
    #FUNCTION Valid Grouper
    def valid_grouper(arr):
        sections = np.split(arr, np.where(np.diff(arr) != 0)[0] + 1)
        sect_lengths = [(x[0], len(x)) for x in sections]
        start_sects = np.where(np.diff(arr)!=0)[0] + 1
        start_sects = np.insert(start_sects, 0, 0)
        sect_tups = list(zip(start_sects, sect_lengths))
        sect_filt = list(filter(lambda x: x[1][0] == 1, sect_tups))
        sect_filt = [(x[0], x[1][1]) for x in sect_filt]
        return sect_filt
    
    def reformat_axis():
        #Format axis.  Remove mainplot if its there, or remove the other axis so we can rewdraw what we need
        existlist = [(idx,axis._label) for idx, axis in enumerate(fig.get_axes()) if axis._label != ""]
        labels = list(map(lambda x: x[1]=="mainplot", existlist))
        if any(labels):
            ax_rem = existlist[labels.index(True)][0]
            fig.axes[ax_rem].remove()

        elif check_axis("overlays"):
            remove_axis(["ecg_small", "overlays"])

        elif check_axis("freq_list"):
            remove_axis(["ecg_small", "freq_list"])

        elif check_axis("stumpy"):
            remove_axis(["ecg_small", "stumpy", "dist_locs"])

    def update_main():
        global ax_ecg
        ax_ecg.clear()
        sect = sect_slider.val
        start_w = ecg_data['section_info'][sect]['start_point']
        end_w = ecg_data['section_info'][sect]['end_point']
        valid = ecg_data['section_info'][sect]['valid']

        ax_ecg.set_ylim(np.min(wave[start_w:end_w])-0.2, np.max(wave[start_w:end_w])+0.2)
        ax_ecg.set_xlim(start_w, end_w)
        ax_ecg.set_ylabel('Voltage (mV)')
        ax_ecg.set_xlabel('ECG index')
        ticks_loc = ax_ecg.get_xticks().tolist()
        labels = utils.label_formatter(ticks_loc)
        ax_ecg.set_xticks(ticks_loc)
        ax_ecg.set_xticklabels(labels, rotation=-20)

        ax_ecg.plot(range(start_w, end_w), wave[start_w:end_w], color='dodgerblue', label='mainECG')
        if valid==0:
            ax_ecg.set_facecolor('gainsboro')
            fails = np.char.split(ecg_data["section_info"][sect]["fail_reason"].strip(), sep="|")
            ax_ecg.annotate(f'Failure reason', xy=(1.01, 0.70), xycoords='axes fraction', fontsize=10)
            ax_ecg.annotate(f'{fails}', xy=(1.01, 0.65), xycoords='axes fraction', fontsize=10)
        else:
            ax_ecg.set_facecolor('white')
        Rpeaks = ecg_data['peaks'][(ecg_data['peaks'][:, 0] >= start_w) & (ecg_data['peaks'][:, 0] <= end_w), :]
        inners = ecg_data['interior_peaks'][(ecg_data['interior_peaks'][:, 2] >= start_w) & (ecg_data['interior_peaks'][:, 2] <= end_w), :]
        if Rpeaks.size > 0:
            ax_ecg.scatter(Rpeaks[:, 0], wave[Rpeaks[:, 0]], label='R', color='red')
            #add patches for whether a peak is valid or not. 

        if inners.size > 0:
            [ax_ecg.scatter(inners[:, x], wave[inners[:, x]], label=PEAKDICT[x][0], color=PEAKDICT[x][1], alpha=0.8) for x in PEAKDICT.keys() if x !=2]  
            [add_cht_labels(inners[:, key], wave[inners[:, key].flatten()], ax_ecg, val[0]) for key, val in PEAKDICT.items() if key !=2]
        
        if ax_ecg._label == "mainplot":
            ax_ecg.annotate(f'Avgs ', xy=(1.01, 0.99), xycoords='axes fraction', fontsize=10) #| Counts
            ax_ecg.annotate(f'HR:        {ecg_data["section_info"][sect]["Avg_HR"]:.0f} bpm', xy=(1.01, 0.94), xycoords='axes fraction', fontsize=10)
            if len(np.nonzero(inners[:, 8])[0]) > 0:
                QRS = np.mean(inners[np.nonzero(inners[:, 8])[0],8])
                ax_ecg.annotate(f'QRS:      {QRS:.0f} ms', xy=(1.01, 0.89), xycoords='axes fraction', fontsize=10) # | {QRS[1]:.0f}
            if len(np.nonzero(inners[:, 7])[0]) > 0:
                PR = np.mean(inners[np.nonzero(inners[:, 7])[0],7])
                ax_ecg.annotate(f'PR:         {PR:.0f} ms', xy=(1.01, 0.84), xycoords='axes fraction', fontsize=10)
            if len(np.nonzero(inners[:, 10])[0]) > 0:
                QT = np.mean(inners[np.nonzero(inners[:, 10])[0],10])
                ax_ecg.annotate(f'QT:         {QT:.0f} ms', xy=(1.01, 0.79), xycoords='axes fraction', fontsize=10)
            ax_ecg.annotate(f'RMSSD:  {ecg_data["section_info"][sect]["RMSSD"]:.0f} ms', xy=(1.01, 0.74), xycoords='axes fraction', fontsize=10)
        elif ax_ecg._label == "small_ecg":
            pass
            #TODO Think of way to summarize table data on smaller chart.  Or maybe just leave off. 

        ax_ecg.set_title(f'ECG for idx {start_w:_d}:{end_w:_d} in sect {sect} ')
        ax_ecg.legend(loc="upper left")

    # FUNCTION Create Overlay plot
    def overlay_r_peaks(main_p:bool=False):
        reformat_axis()
        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, gs[0, :2])
        global ax_ecg
        ax_ecg = fig.add_subplot(inner_grid[0, :1], label = "ecg_small")
        ax_over = fig.add_subplot(inner_grid[0, 1:2], label = "overlays")
        update_main()
        sect = sect_slider.val
        start_w = ecg_data['section_info'][sect]['start_point']
        end_w = ecg_data['section_info'][sect]['end_point']
        inners = ecg_data['interior_peaks'][(ecg_data['interior_peaks'][:, 2] >= start_w) & (ecg_data['interior_peaks'][:, 2] <= end_w), :]
        R_peaks = inners[np.nonzero(inners[:, 2])[0], 2]
        RR_diffs = int(np.mean(np.diff(R_peaks))//2)
        if main_p:
            P_peak = inners[np.nonzero(inners[:, 0])[0], 0]
            Q_peak = inners[np.nonzero(inners[:, 1])[0], 1]
            S_peak = inners[np.nonzero(inners[:, 3])[0], 3]
            T_peak = inners[np.nonzero(inners[:, 4])[0], 4]

            for idx, Rpeak in enumerate(R_peaks):
                ax_over.plot(wave[Rpeak-RR_diffs:Rpeak+RR_diffs], label=f'peak_{idx}', color='dodgerblue', alpha=.5)
                ax_over.scatter((P_peak[idx] - Rpeak) + RR_diffs , wave[P_peak[idx]], label='P Peak', s = 60, color=PEAKDICT[0][1])
                ax_over.scatter((Q_peak[idx] - Rpeak) + RR_diffs , wave[Q_peak[idx]], label='Q Peak', s = 60, color=PEAKDICT[1][1])
                ax_over.scatter((R_peaks[idx] - Rpeak) + RR_diffs, wave[R_peaks[idx]], label='R Peak', s = 60, color=PEAKDICT[2][1])
                ax_over.scatter((S_peak[idx] - Rpeak) + RR_diffs , wave[S_peak[idx]], label='S Peak', s = 60, color=PEAKDICT[3][1])
                ax_over.scatter((T_peak[idx] - Rpeak) + RR_diffs , wave[T_peak[idx]], label='T Peak', s = 60, color=PEAKDICT[4][1])

            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='P Peak', markerfacecolor=PEAKDICT[0][1], markersize=15),
                Line2D([0], [0], marker='o', color='w', label='Q Peak', markerfacecolor=PEAKDICT[1][1], markersize=15),
                Line2D([0], [0], marker='o', color='w', label='R Peak', markerfacecolor=PEAKDICT[2][1], markersize=15),
                Line2D([0], [0], marker='o', color='w', label='S Peak', markerfacecolor=PEAKDICT[3][1], markersize=15),
                Line2D([0], [0], marker='o', color='w', label='T Peak', markerfacecolor=PEAKDICT[4][1], markersize=15),
            ]

        else:
            P_onset = inners[np.nonzero(inners[:, 11])[0], 11]
            Q_onset = inners[np.nonzero(inners[:, 12])[0], 12]
            T_onset = inners[np.nonzero(inners[:, 13])[0], 13]
            T_offset = inners[np.nonzero(inners[:, 14])[0], 14]
            J_point = inners[np.nonzero(inners[:, 15])[0], 15]

            for idx, Rpeak in enumerate(R_peaks):
                ax_over.plot(wave[Rpeak-RR_diffs:Rpeak+RR_diffs], label=f'peak_{idx}', color='dodgerblue', alpha=.5)
                ax_over.scatter((P_onset[idx] - Rpeak) + RR_diffs , wave[P_onset[idx]], label='P Onset', s = 60, color=PEAKDICT_EXT[11][1])
                ax_over.scatter((Q_onset[idx] - Rpeak) + RR_diffs , wave[Q_onset[idx]], label='Q Onset', s = 60, color=PEAKDICT_EXT[12][1])
                ax_over.scatter((T_onset[idx] - Rpeak) + RR_diffs , wave[T_onset[idx]], label='T Onset', s = 60, color=PEAKDICT_EXT[13][1])
                ax_over.scatter((T_offset[idx] - Rpeak) + RR_diffs , wave[T_offset[idx]], label='T Offset', s = 60, color=PEAKDICT_EXT[14][1])
                ax_over.scatter((J_point[idx] - Rpeak) + RR_diffs , wave[J_point[idx]], label='J point', s = 60, color=PEAKDICT_EXT[15][1])
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='P Onset', markerfacecolor=PEAKDICT_EXT[11][1], markersize=15),
                Line2D([0], [0], marker='o', color='w', label='Q Onset', markerfacecolor=PEAKDICT_EXT[12][1], markersize=15),
                Line2D([0], [0], marker='o', color='w', label='T Onset', markerfacecolor=PEAKDICT_EXT[13][1], markersize=15),
                Line2D([0], [0], marker='o', color='w', label='T Offset', markerfacecolor=PEAKDICT_EXT[14][1], markersize=15),
                Line2D([0], [0], marker='o', color='w', label='J point', markerfacecolor=PEAKDICT_EXT[15][1], markersize=15)
            ]
        ax_over.set_ylabel('Voltage (mV)')
        ax_over.set_xlabel('ECG index')
        ax_over.legend(handles=legend_elements, loc='upper left')
        ax_over.set_title(f'Overlayed QRS Complexes for section {sect} ', size=14)

    #FUNCTION Frequency
    def frequencytown():
        global fs
        reformat_axis()

        #Set the inner grid to the first row space
        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, gs[0, :2])
        global ax_ecg
        ax_ecg = fig.add_subplot(inner_grid[0, :1], label = "ecg_small")
        ax_freq = fig.add_subplot(inner_grid[0, 1:2], label = "freq_list")
        
        update_main()

        #Plot the frequencies
        sect = sect_slider.val
        #Clear plot and colorbar
        ax_freq.cla()
        kindofgraph = radio.value_selected
        start_w = ecg_data['section_info'][sect]['start_point']
        end_w = ecg_data['section_info'][sect]['end_point']
        inners = ecg_data['interior_peaks'][(ecg_data['interior_peaks'][:, 2] >= start_w) & (ecg_data['interior_peaks'][:, 2] <= end_w), :]
        R_peaks = inners[np.nonzero(inners[:, 2])[0], 2]
        RR_diffs = int(np.mean(np.diff(R_peaks))//2)
        samp = wave[start_w:end_w].flatten()
        fft_samp = np.abs(rfft(samp))
        freq_list = rfftfreq(len(samp), d=1/fs) #samp_freq is sampling rate
        freqs = fft_samp[0:int(len(samp)/2)]
        freq_l = freq_list[:int(len(samp)//2)]
        freqs_idx, peak_power = ss.find_peaks(freqs, height=freqs.mean()//10, distance=10)
        combined = list(zip(freq_list[freqs_idx], peak_power["peak_heights"]))
        sorted_p = sorted(combined, key=lambda x:x[1], reverse=True)[:10]

        if "Stem" in kindofgraph:
            ax_freq.stem(freq_l, freqs, "b", markerfmt=" ", basefmt="-b")
            for power in sorted_p:
                ax_freq.annotate(
                    text=f"{power[0]:.1f}Hz",
                    xy = (power[0]+0.3,(power[1]+power[1]*.02)),
                    color="black", 
                    weight="bold", fontsize=7, 
                    ha="center", va="center"
            )
            if sorted_p:
                ax_freq.set_xlim(0, max(list(map(lambda x:x[0], sorted_p)))*1.5)
                ax_freq.set_ylim(0, sorted_p[0][1]*1.2)
            else:
                ax_freq.set_xlim(0, 50)

            ax_freq.set_xlabel("Freq (Hz)")
            ax_freq.set_ylabel("Frequency Power")
            ax_freq.set_title(f"Top 10 frequencies found in sect {sect:_d}", size=12)

        if "Spec" in kindofgraph:
            remove_colorbar()
            _, specfreqs, _, _ = ax_freq.specgram(
                wave[start_w:end_w].flatten(),
                NFFT= int(np.mean(RR_diffs)),   
                cmap="rainbow",
                detrend="linear",
                noverlap = 10,
                Fs=fs
            )
            ax_freq.set_xlabel(f"Time (sec)\nBinned in {np.mean(RR_diffs/fs):0.1f} sec intervals")
            ax_freq.set_ylabel(f"")
            ax_freq.set_yticks([]) 
            ax_freq.set_title(f'Spectrogram for peaks {R_peaks[0]:_d}:{R_peaks[-1]:_d}')
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(0, np.max(specfreqs)), 
                cmap='rainbow'), 
                ax=ax_freq, 
                pad=0.03,
                orientation='vertical', 
                label=f'Frequency Range (Hz)',
                location='left')
            logger.info("")
    #FUNCTION Wavesearch
    @log_time
    def wavesearch():
        #FUNCTION Stumpy Fast Pattern Matching
        @log_time
        def stumpysearch(query:range, srchwidth:range, ax_stump:plt.axis):
            #BUG Search area too large.  
            #I can't do a direct wave search with stumpy.  
            #So how about lets search an hour before and an hour behind maybe? That could be useful

            Q_s = wave[list(query)].flatten()
            T_s = wave[list(srchwidth)].flatten()
            matches_improved_max_distance = stumpy.match(
                Q = Q_s,
                T = T_s,
                max_distance=lambda D: max(np.mean(D) - 5 * np.std(D), np.min(D))
            )

            # Since MASS computes z-normalized Euclidean distances, we should z-normalize our subsequences before plotting
            Q_z_norm = stumpy.core.z_norm(Q_s)
            ax_stump.set_title(f'{matches_improved_max_distance.shape[0]} Query Matches', fontsize='18')
            ax_stump.set_xlabel('Time')
            ax_stump.set_ylabel('ECG (mv)')
            for match_distance, match_idx in matches_improved_max_distance:
                match_z_norm = stumpy.core.z_norm(T_s[match_idx:match_idx+len(Q_s)])
                ax_stump.plot(match_z_norm, lw=2)

            ax_stump.plot(Q_z_norm, lw=3, color="black", label="Selected Query, Q_s")

            return matches_improved_max_distance
        @log_time
        def draw_selection(region_x:range, region_y:np.array):
            #reformat the axis on entry
            reformat_axis()

            #redraw the top row how we want it. 
            left_grid = gridspec.GridSpecFromSubplotSpec(1, 1, gs[0, :1])
            right_grid = gridspec.GridSpecFromSubplotSpec(2, 1, gs[0, 1:2], height_ratios=[5, 1], hspace=0.2)
            global ax_ecg
            ax_ecg = fig.add_subplot(left_grid[:, :], label = "ecg_small")
            ax_stump = fig.add_subplot(right_grid[:1, :1], label = "stumpy")
            ax_dist = fig.add_subplot(right_grid[1:2, :1], label = "dist_locs")
            
            #update ax_ecg
            update_main()
            #Add the rect patch
            rect = Rectangle((min(region_x), region_y.min()), (max(region_x) - min(region_x)), (np.abs(region_y.min())+region_y.max()), facecolor='lightgrey')
            ax_ecg.add_patch(rect)
            rect.set_zorder(-1)

            #segment the total wave
            segments = utils.segment_ECG(wave, fs)
            sect = sect_slider.val
            sdelta = 1000
            st_sect = sect - sdelta
            fn_sect = sect + sdelta
            searchwidth = segments[st_sect:fn_sect]
            srange = range(searchwidth[0, 0],searchwidth[-1, 1])
            #Run stumpy match algorithm to look for the wave in the 
            #a delta forward and back of its current position
            matchlocs = stumpysearch(region_x, srange, ax_stump)
            matchcounts = {}
            for idx, rag in enumerate(searchwidth):
                start, stop, _ = rag 
                count = 0
                for match in matchlocs[:, 1]:
                    if start <= match + srange[0] <= stop:
                        count += 1
                if count > 0:
                    matchcounts[idx + st_sect] = count
            
            #Plot the distribution bar below
            counts = [matchcounts[key] for key in matchcounts.keys()]
            x_vals = [key for key in matchcounts.keys()]
            ax_dist.bar(x_vals, counts, align="center", label = "Section Counts")
            ax_dist.set_xlim(x_vals[0], x_vals[-1])
            ax_dist.set_ylim(0, max(counts)*1.2)
            ax_dist.set_xlabel("ECG Sections")
            ax_dist.set_ylabel("Counts")
            ax_dist.legend(loc='upper right')

        def onselect_func(xmin, xmax):
            def _confirm_select(event):
                if event.key == "enter":
                    xmin, xmax = span.extents
                    xmin = int(np.floor(xmin))
                    xmax = int(np.ceil(xmax))
                    region_x = range(xmin, xmax)
                    region_y = wave[xmin:xmax]

                    #early terminate if you accidentally click
                    if xmin==xmax or len(region_x) <= 10:
                        raise ValueError(f'Please select a larger range than {len(region_x)}')

                    span.disconnect_events()
                    fig.canvas.mpl_disconnect(cid)
                    draw_selection(region_x, region_y)

            cid = fig.canvas.mpl_connect("key_press_event", _confirm_select)

        global span
        span = SpanSelector(
            ax_ecg, 
            onselect_func, 
            direction ='horizontal',
            props = dict(alpha=0.5, facecolor='red'),
            useblit = False,
            interactive = True,
            button = 1
        )
        remove_colorbar()

    def remove_colorbar():
        for artist in fig.axes:
            if artist._label == "<colorbar>":
                artist.remove()

    #FUNCTION Remove Axis
    def remove_axis(remove_vars:list):
        if isinstance(remove_vars, list):
            for var in remove_vars:
                # Generate a tuple of index and axis label for all axis in the fig.
                existlist = [(idx,axis._label) for idx, axis in enumerate(fig.get_axes()) if axis._label != ""]
                ax_rem = list(map(lambda x:x[1], existlist)).index(var)
                fig.axes[existlist[ax_rem][0]].remove()

        elif isinstance(remove_vars, str):
            existlist = [(idx,axis._label) for idx, axis in enumerate(fig.get_axes()) if axis._label != ""]
            ax_rem = list(map(lambda x:x[1], existlist)).index(remove_vars)
            fig.axes[existlist[ax_rem][0]].remove()
    
    #FUNCTION Check Axis
    def check_axis(valcheck:str):
        existlist = [axis._label for axis in fig.get_axes() if axis._label != ""]
        if valcheck in existlist:
            return True
        else:
            return False

    # FUNCTION Update plot
    def update_plot(val):
        # If the command is not to change to freq
        global ax_ecg
        command = radio.value_selected
        #If you chose anything except frequency or stumpy, clear main axis and redraw it in its original form
        if command in ["Base Figure", "Roll Median", "Add Inter", "Hide Leg", "Show R Valid"]:
            if check_axis("mainplot"):
                ax_ecg.cla()
            else:
                ax_ecg = fig.add_subplot(gs[0, :2], label="mainplot")
            update_main()

        if configs["freq"]:
            # logger.info(f'{check_axis("mainplot")}')
            frequencytown()

        elif configs["overlay"]:
            if command == "Overlay Main":
                overlay_r_peaks(True)
            elif command == "Overlay Inner":
                overlay_r_peaks()

        elif configs["stump"]:
            wavesearch()

        fig.canvas.draw_idle()

    #FUNCTION Radio Button Actions
    def radiob_action(val):
        sect = sect_slider.val
        start_w = ecg_data['section_info'][sect]['start_point']
        end_w = ecg_data['section_info'][sect]['end_point']

        if val in ["Base Figure", "Roll Median", "Add Inter", "Hide Leg", "Show R Valid"]:
            #When selecting various functions.
            #This is to make sure you remove the appropriate axis' before redrawing the main chart
            if configs["freq"] and check_axis("freq_list"):
                remove_axis(["freq_list", "ecg_small"])
                configs["freq"] = False
                update_plot(val)
                
            if configs["overlay"] and check_axis("overlays"):
                remove_axis(["overlays", "ecg_small"])
                configs["overlay"] = False
                update_plot(val)
                
            if configs["stump"] and check_axis("stumpy"):
                remove_axis(["stumpy", "dist_locs", "ecg_small"])
                configs["stump"] = False
                update_plot(val)

        if val == 'Roll Median':	
            ax_ecg.plot(range(start_w, end_w), utils.roll_med(wave[start_w:end_w]), color='orange', label='Rolling Median')
            ax_ecg.legend(loc='upper left')

        elif val == 'Add Inter':
            inners = ecg_data['interior_peaks'][(ecg_data['interior_peaks'][:, 2] >= start_w) & (ecg_data['interior_peaks'][:, 2] <= end_w), :]
            for key, val in PEAKDICT_EXT.items():
                if inners[np.nonzero(inners[:, key])[0], key].size > 0:
                    ax_ecg.scatter(inners[:, key], wave[inners[:, key]], label=val[0], color=val[1], alpha=0.8)
            ax_ecg.set_title(f'All interior peaks for section {sect} ', size=14)
            ax_ecg.legend(loc='upper left')

        elif val == 'Hide Leg':
            ax_ecg.get_legend().remove()

        elif val == 'Show R Valid':
            Rpeaks = ecg_data['peaks'][(ecg_data['peaks'][:, 0] >= start_w) & (ecg_data['peaks'][:, 0] <= end_w), :]
            for peak in range(Rpeaks.shape[0]):
                if Rpeaks[peak, 1] == 0:
                    band_color = 'red'
                else:
                    band_color = 'lightgreen'
                rect = Rectangle(
                    xy=((Rpeaks[peak, 0] - 10), (wave[Rpeaks[peak,0]] + wave[Rpeaks[peak,0]]*(0.05)).item()), 
                    width=np.mean(np.diff(Rpeaks[:, 0])) // 6, 
                    height=wave[Rpeaks[peak, 0]].item() / 10,
                facecolor=band_color,
                edgecolor="grey",
                alpha=0.7)
                ax_ecg.add_patch(rect)

        elif 'Frequency' in val:
            configs["freq"] = True
            frequencytown()

        elif val == 'Overlay Main':
            configs["overlay"] = True
            overlay_r_peaks(True)

        elif val == 'Overlay Inner':
            configs["overlay"] = True
            overlay_r_peaks()

        elif val == 'Stumpy Search':
            configs["stump"] = True
            wavesearch()

        fig.canvas.draw_idle()    

    #FUNCTION Slide forward invalid
    def move_slider_forward(vl):
        #TODO add docstrings
        curr_sect = sect_slider.val
        next_sect = np.where(ecg_data['section_info']['valid'][curr_sect+1:]==0)[0][0] + curr_sect+1
        sect_slider.set_val(next_sect)
    
    #FUNCTION Slide forward one
    def move_slider_sing_forward(vl):
        #TODO add docstrings
        curr_sect = sect_slider.val
        sect_slider.set_val(curr_sect+1)

    #FUNCTION Slide back invalid
    def move_slider_back(vl):
        #TODO add docstrings
        curr_sect = sect_slider.val
        prev_sect = np.where(ecg_data['section_info']['valid'][:curr_sect]==0)[0][-1]
        sect_slider.set_val(prev_sect)

    #FUNCTION Slide back one
    def move_slider_sing_back(vl):
        #TODO add docstrings
        curr_sect = sect_slider.val
        sect_slider.set_val(curr_sect-1)

    #FUNCTION Slide to section
    def jump_slider_to_sect(v1):
        #TODO add docstrings
        jump_num = int(jump_sect_text.text)
        sect_slider.set_val(jump_num)

    
    #Setting mixed datatypes (structured array) for ecg_data['section_info']
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
        "interior_peaks": np.genfromtxt(fpath+"_interior_peaks.csv", delimiter=",", dtype=np.int32, usecols=(range(16)))
    }

    #Draw main plot inititally and set params
    valid_sect = ecg_data['section_info']['valid']
    global ax_ecg, gs, fig
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[6, 1])
    plt.subplots_adjust(hspace=0.40)
    ax_ecg = fig.add_subplot(gs[0, :2], label="mainplot")
    first_sect = np.where(ecg_data['section_info']['valid']!=0)[0][0]
    start_w = ecg_data['section_info'][first_sect]['start_point']
    end_w = ecg_data['section_info'][first_sect]['end_point']
    ax_ecg.plot(range(start_w, end_w), wave[start_w:end_w], color='dodgerblue', label='mainECG')
    ax_ecg.set_xlim(start_w, end_w)
    ax_ecg.set_ylabel('Voltage (mV)')
    ax_ecg.set_xlabel('ECG index')
    ax_ecg.legend(loc='upper left')

    #brokebarH plot for the background of the slider. 
    ax_section = fig.add_subplot(gs[1, :2])
    ax_section.broken_barh(valid_grouper(valid_sect), (0,1), facecolors=('tab:blue'))
    ax_section.set_ylim(0, 1)
    ax_section.set_xlim(0, valid_sect.shape[0])

    sect_slider = Slider(ax_section, 
        label='Sections',
        valmin=first_sect, 
        valmax=len(valid_sect), 
        valinit=first_sect, 
        valstep=1
    )

    #Invalid step axes placeholders
    axnext = plt.axes([0.595, 0.01, 0.15, 0.050])
    axprev = plt.axes([0.44, 0.01, 0.15, 0.050])

    #Singlestep axes placholders
    ax_sing_next = plt.axes([0.28, 0.01, 0.15, 0.050])
    ax_sing_prev = plt.axes([0.125, 0.01, 0.15, 0.050])

    #Jump section axes placeholders
    ax_jump_textb = plt.axes([0.88, 0.01, 0.06, 0.050])

    #Add axis container for radio buttons
    ax_radio = plt.axes([0.905, .33, 0.09, 0.32])

    #Button for Invalid Step process
    next_button = Button(axnext, label='Next Invalid Section')
    prev_button = Button(axprev, label='Previous Invalid Section')

    #Button for single step process
    sing_next_button = Button(ax_sing_next, label='Single Step Forward')
    sing_prev_button = Button(ax_sing_prev, label='Single Step Backward')

    #TextBox for section jump
    jump_sect_text = TextBox(ax_jump_textb, 
        label='Jump to Section',
        textalignment="center", 
        hovercolor='green'
    )

    #Radio buttons
    radio = RadioButtons(ax_radio, ('Base Figure', 'Roll Median', 'Add Inter', 'Hide Leg', 'Show R Valid', 'Overlay Main', 'Overlay Inner', 'Frequency-Stem', 'Frequency-Spec', 'Stumpy Search'))

    #Set actions for GUI items. 
    sect_slider.on_changed(update_plot)
    next_button.on_clicked(move_slider_forward)
    prev_button.on_clicked(move_slider_back)
    sing_next_button.on_clicked(move_slider_sing_forward)
    sing_prev_button.on_clicked(move_slider_sing_back)
    radio.on_clicked(radiob_action)
    jump_sect_text.on_submit(jump_slider_to_sect)

    #Make a custom legend. 
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=val[0], markerfacecolor=val[1], markersize=10) for val in PEAKDICT.values()
        ]
    ax_ecg.legend(handles=legend_elements, loc='upper left')
    plt.show()

    #Print failures table
    failures = Counter(ecg_data['section_info']['fail_reason'])
    table = make_rich_table(failures)
    console.log(table)

def summarize_run():
    RMSSD = ecg_data['section_info'][['wave_section', 'RMSSD']]
    logger.info(f"{sorted(RMSSD, key= lambda x:x[1], reverse=True)[:20]}")

def main():
    global configs
    configs = setup_globals.load_config()
    configs["freq"], configs["stump"] =  False, False
    configs["slider"], configs["overlay"] = True, False

    datafile = setup_globals.launch_tui(configs)
    global wave, fs
    wave, fs, outputf = setup_globals.load_chart_data(configs, datafile, logger)
    graph = load_graph_objects(datafile, outputf)
 
if __name__ == "__main__":
    main()

    #IDEA - Larger section clustering of smaller groups.  Or motif shifts
        #Could run it in the slider.py file.
    #IDEA - Or have a draggable band that switches your viewpoint to a histogram of the width of the band.
        #Like stumpy search function
