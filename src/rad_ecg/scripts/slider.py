import scipy.signal as ss
from scipy.fft import rfft, rfftfreq, irfft
import numpy as np
from rich.table import Table
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import utils # from rad_ecg.scripts
import setup_globals # from rad_ecg.scripts 
from support import logger, console, log_time
import support

def load_graph_objects(datafile:str, filen:str, outputf:str):
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
        label_dict = {
            "P":(0, 7),
            "Q":(-11, -4),
            "R":(10, -4),
            "S":(-5, -15),
            "T":(0, 5),
            "P_on":(0, -10),
            "Q_on":(-11, -4),
            "T_on":(10, 0),
            "T_off":(10, 10),
        }
        for x, y in zip(x,y):
            label = f'{label[0]}' #:{y:.2f}
            plt.annotate(
                label,
                (x,y),
                textcoords="offset points",
                xytext=label_dict[label[0]],
                ha='center')
    #FUNCTION Valid Grouper
    def valid_grouper(arr):
        #TODO add docstrings
        #ecg_data['section_info']['valid'] passed in.
        sections = np.split(arr, np.where(np.diff(arr) != 0)[0] + 1)
        sect_lengths = [(x[0], len(x)) for x in sections]
        start_sects = np.where(np.diff(arr)!=0)[0] + 1
        start_sects = np.insert(start_sects, 0, 0)
        sect_tups = list(zip(start_sects, sect_lengths))
        sect_filt = list(filter(lambda x: x[1][0] == 1, sect_tups))
        sect_filt = [(x[0], x[1][1]) for x in sect_filt]
        return sect_filt

    # FUNCTION Update plot
    def update_plot(val):
        #Gameplan
        #When a plot updates, it will neeed to update the main plot
        #almost always.  The real test will be if there is anything
        #else I need it to update.  I would like it to update the other plots
        #if they exist.  
        #I also want it to be able to still do the normal labelling funcs...
        #Is it better to split this out...into a separate radio button situation

        #TODO add docstrings
        #clear the top chart

        #TODO Will need secondary update function for when both axis are present
        if configs["stump"] or configs["freq"]:
            if configs["freq"] and check_axis("freqs"):
                #Plot update routine for frequency 
                pass
            elif configs["stump"] and check_axis("stump"):
                #Plot update routine for frequency 
                pass

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

        ax_ecg.plot(range(start_w, end_w), wave[start_w:end_w], color='dodgerblue', label='ECG')
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
            [ax_ecg.scatter(inners[:, x], wave[inners[:, x]], label=peak_dict[x][0], color=peak_dict[x][1], alpha=0.8) for x in peak_dict.keys() if x !=2]  
            [add_cht_labels(inners[:, key], wave[inners[:, key].flatten()], ax_ecg, val[0]) for key, val in peak_dict.items() if key !=2]
        
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
        ax_ecg.set_title(f'ECG for idx {start_w:_d}:{end_w:_d} in sect {sect} ')

        fig.canvas.draw_idle()
        
    # FUNCTION Create Overlay plot
    def create_comparison_plot(slider_val):
        #TODO add docstrings
        #TODO put this into sidebar view
        sect = slider_val
        start_w = ecg_data['section_info'][sect]['start_point']
        end_w = ecg_data['section_info'][sect]['end_point']
        valid = ecg_data['section_info'][sect]['valid']
        sect_mean = np.mean(wave[start_w:end_w])
        inners = ecg_data['interior_peaks'][(ecg_data['interior_peaks'][:, 2] >= start_w) & (ecg_data['interior_peaks'][:, 2] <= end_w), :]
        R_peaks = inners[np.nonzero(inners[:, 2])[0], 2]
        P_onset = inners[np.nonzero(inners[:, 11])[0], 11]
        Q_onset = inners[np.nonzero(inners[:, 12])[0], 12]
        T_onset = inners[np.nonzero(inners[:, 13])[0], 13]
        T_offset = inners[np.nonzero(inners[:, 14])[0], 14]

        fig2, ax3 = plt.subplots(ncols=1, nrows=1, figsize=(12, 8))

        RR_diffs = int(np.mean(np.diff(R_peaks))//2)
        for idx, Rpeak in enumerate(R_peaks):
            ax3.plot(wave[Rpeak-RR_diffs:Rpeak+RR_diffs], label=f'ECG', color='dodgerblue', alpha=.5)
            ax3.scatter((P_onset[idx] - Rpeak) + RR_diffs , wave[P_onset[idx]], label='P Onset', s = 60, color='purple')
            ax3.scatter((Q_onset[idx] - Rpeak) + RR_diffs , wave[Q_onset[idx]], label='Q Onset', s = 60, color='darkgoldenrod')
            ax3.scatter((T_onset[idx] - Rpeak) + RR_diffs , wave[T_onset[idx]], label='T Onset', s = 60, color='teal')
            ax3.scatter((T_offset[idx] - Rpeak) + RR_diffs , wave[T_offset[idx]], label='T Offset', s = 60, color='orange')
        #Make a custom legend. 
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='P Onset', markerfacecolor='purple', markersize=15),
            Line2D([0], [0], marker='o', color='w', label='Q Onset', markerfacecolor='darkgoldenrod', markersize=15),
            Line2D([0], [0], marker='o', color='w', label='T Onset', markerfacecolor='teal', markersize=15),
            Line2D([0], [0], marker='o', color='w', label='T Offset', markerfacecolor='orange', markersize=15)
        ]
        ax3.set_ylabel('Voltage (mV)')
        ax3.set_xlabel('ECG index')
        
        ax3.legend(handles=legend_elements, loc='upper left')
        ax3.set_title(f'Overlayed QRS Complexes for section {sect} ', size=14)
        plt.show()


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

    #FUNCTION Frequency
    def frequencytown():
        global fs
        #removes all the labels from the frequency chart if they exist.  
        #Other wise it will remove the entire axis
        existlist = [(idx,axis._label) for idx, axis in enumerate(fig.get_axes()) if axis._label != ""]
        labels = list(map(lambda x: x[1]=="mainplot", existlist))
        if any(labels):
            ax_rem = existlist[labels.index(True)][0]
            fig.axes[ax_rem].remove()
        else:
            remove_axis(["freq_list", "var_small"])

        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, gs[0, :2])
        ax_wave = fig.add_subplot(inner_grid[0, :1], label = "var_small")
        ax_freq = fig.add_subplot(inner_grid[0, 1:2], label = "freq_list")
        
        ###  Plot the wave ###
        col = radio.value_selected
        sect = sect_slider.val
        start_w = ecg_data['section_info'][sect]['start_point']
        end_w = ecg_data['section_info'][sect]['end_point']
        wavesect = wave[start_w:end_w]
        ax_wave.plot(range(start_w, end_w), wave[start_w:end_w], label=col)
        ax_wave.set_ylim(wavesect.min() - wavesect.std()*2, wavesect.max() + wavesect.std()*2)
        ax_wave.set_xlim(start_w, end_w)
        ax_wave.set_ylabel('Voltage (mV)')
        ax_wave.set_xlabel('ECG index')
        ax_wave.set_title(f'ECG for idx {start_w:_d}:{end_w:_d} in sect {sect}', size = 12)
        ax_wave.legend(loc="upper right")

        #Plot the frequencies
        samp = wave[start_w:end_w].values
        fft_samp = np.abs(rfft(samp))
        freq_list = rfftfreq(len(samp), d=1/fs) #samp_freq is sampling rate
        freqs = fft_samp[0:int(len(samp)/2)]
        freq_l = freq_list[:int(len(samp)//2)]
        ax_freq.stem(freq_l, freqs, "b", markerfmt=" ", basefmt="-b")

        #Old way
        # X = np.fft.fft(wave[start_w:end_w])
        # N = len(X)
        # n = np.arange(N)
        # T = N/samp_freq
        # freq = n/T
        # ax_freq.stem(freq, np.abs(X), "b", markerfmt=" ", basefmt="-b")
        #different frequency zooms

        freqs_idx, peak_power = ss.find_peaks(freqs, height=freqs.mean()//10, distance=10)
        combined = list(zip(freq_list[freqs_idx], peak_power["peak_heights"]))
        sorted_p = sorted(combined, key=lambda x:x[1], reverse=True)[:10]
        # maybe use an annotate?
        for power in sorted_p:
            ax_freq.annotate(
                text=f"{power[0]:.1f}Hz",
                xy = (power[0]+0.3,(power[1]+power[1]*.02)),
                color="black", 
                weight="bold", fontsize=7, 
                ha="center", va="center"
        )
        if sorted_p:
            ax_freq.set_xlim(0, sorted_p[-1][0]*1.2)
            ax_freq.set_ylim(0, sorted_p[0][1]*1.2)
        else:
            ax_freq.set_xlim(0, 50)
        ax_freq.set_xlabel("Freq (Hz)")
        ax_freq.set_ylabel("Frequency Power")
        ax_freq.set_title(f"Top 10 frequencies found in sect {sect}", size=12)

    #FUNCTION Radio Button Actions
    def radiob_action(val):
        #Here we're going to switch states between where we have a single vs a 
        #double panel display.  
            #If i split for frequency, i want a new freq chart to the right, 
            #as well as the signal to my left.  (With the ability to still highlight
            #points / medians / Rpeaks. 
        #Also means i'll need a split for the comparison plot
        #
        sect = sect_slider.val
        start_w = ecg_data['section_info'][sect]['start_point']
        end_w = ecg_data['section_info'][sect]['end_point']
        if val != 'Frequency':
            if configs["freq"] and check_axis("freq_list"):
                remove_axis(["freq_list", "var_small"])
                update_plot("val")
                configs["freq"] = False
            
        if val != "Stumpy":
            if configs["stump"] and check_axis("overlays"):
                remove_axis(["overlays", "var_small", "dist_locs"])
                update_plot("val")
                configs["stump"] = False

        if val == 'Roll Median':	
            ax_ecg.plot(range(start_w, end_w), utils.roll_med(wave[start_w:end_w]), color='orange', label='Rolling Median')
            ax_ecg.legend(loc='upper left')

        if val == 'Add Inter':
            sect = sect_slider.val
            start_w = ecg_data['section_info'][sect]['start_point']
            end_w = ecg_data['section_info'][sect]['end_point']
            inners = ecg_data['interior_peaks'][(ecg_data['interior_peaks'][:, 2] >= start_w) & (ecg_data['interior_peaks'][:, 2] <= end_w), :]
            for key, val in peak_dict_ext.items():
                if inners[np.nonzero(inners[:, key])[0], key].size > 0:
                    ax_ecg.scatter(inners[:, key], wave[inners[:, key]], label=val[0], color=val[1], alpha=0.8)
            ax_ecg.set_title(f'All interior peaks for section {sect} ', size=14)

        if val == 'Hide Leg':
            ax_ecg.get_legend().remove()

        if val == 'Show R Valid':
            Rpeaks = ecg_data['peaks'][(ecg_data['peaks'][:, 0] >= start_w) & (ecg_data['peaks'][:, 0] <= end_w), :]
            for peak in range(Rpeaks.shape[0]):
                if Rpeaks[peak, 1]==0:
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

        if val == 'Overlay P':
            create_comparison_plot(sect_slider.val)

        if val == 'Frequency':
            configs["freq"] = True
            frequencytown()

        # if val == 'Stumpy':
        #     configs["stump"] = True
        #     stumpysearch()

        fig.canvas.draw_idle()    

    #FUNCTION Slide forward
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

    #FUNCTION Slide back
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
        "interior_peaks": np.genfromtxt(fpath+"_interior_peaks.csv", delimiter=",", dtype=np.int32, usecols=(range(15)))
    }

    #brokebarH plot for the background of the slider. 
    valid_sect = ecg_data['section_info']['valid']
    fig = plt.figure(figsize = (14, 8))
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[5, 1])
    plt.subplots_adjust(hspace=0.40) 
    ax_ecg = fig.add_subplot(gs[0, :2])
    first_sect = np.where(ecg_data['section_info']['valid']!=0)[0][0]
    start_w = ecg_data['section_info'][first_sect]['start_point']
    end_w = ecg_data['section_info'][first_sect]['end_point']
    ax_ecg.plot(range(start_w, end_w), wave[start_w:end_w], color='dodgerblue', label='mainplot')
    ax_ecg.set_xlim(start_w, end_w)
    ax_ecg.set_ylabel('Voltage (mV)')
    ax_ecg.set_xlabel('ECG index')
    peak_dict = {
            0:('P', 'green'),
            1:('Q', 'cyan'),
            2:('R', 'red'),
            3:('S', 'magenta'),
            4:('T', 'black')
    }
    peak_dict_ext = {
            11:('P_on', 'purple'),
            12:('Q_on', 'darkgoldenrod'), 
            13:('T_on', 'teal'), 
            14:('T_off', 'orange')
    }

    ax_ecg.legend(loc='upper left')

    ax_section = fig.add_subplot(gs[1, :2])
    ax_section.broken_barh(valid_grouper(valid_sect), (0,1), facecolors=('tab:blue'))
    ax_section.set_ylim(0, 1)
    ax_section.set_xlim(0, valid_sect.shape[0])

    sect_slider = Slider(ax_section, 
                        label='Sections',
                        valmin=first_sect, 
                        valmax=len(valid_sect), 
                        valinit=first_sect, 
                        valstep=1)

    #Invalid step axes placeholders
    axnext = plt.axes([0.595, 0.01, 0.15, 0.050])
    axprev = plt.axes([0.44, 0.01, 0.15, 0.050])

    #Singlestep axes placholders
    ax_sing_next = plt.axes([0.28, 0.01, 0.15, 0.050])
    ax_sing_prev = plt.axes([0.125, 0.01, 0.15, 0.050])

    #Jump section axes placeholders
    ax_jump_textb = plt.axes([0.88, 0.01, 0.06, 0.050])

    #Add axis container for radio buttons
    ax_radio = plt.axes([0.905, .45, 0.09, 0.20])

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
                    hovercolor='green')

    #Radio buttons
    radio = RadioButtons(ax_radio, ('Roll Median', 'Add Inter', 'Hide Leg', 'Overlay P', 'Show R Valid', 'Frequency', 'Stumpy Search'))

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
        Line2D([0], [0], marker='o', color='w', label=val[0], markerfacecolor=val[1], markersize=10) for val in peak_dict.values()
        ]

    ax_ecg.legend(handles=legend_elements, loc='upper left')

    failures = Counter(ecg_data['section_info']['fail_reason'])
    table = make_rich_table(failures)
    console.log(table)

    # quick log loading for checking
    # log = utils.load_log_results(lfpath)

    plt.show()
    plt.close()
    
def summarize_run():
    RMSSD = ecg_data['section_info'][['wave_section', 'RMSSD']]
    logger.info(f"{sorted(RMSSD, key= lambda x:x[1], reverse=True)[:20]}")

def main():
    global configs
    configs = setup_globals.load_config()
    configs["freq"], configs["stump"], configs["slider"] = False, False, True
    datafile = setup_globals.launch_tui(configs)
    global wave, fs
    wave, fs, filen, outputf = setup_globals.load_chartdata(configs, datafile, logger)
    load_graph_objects(datafile, filen, outputf)
    # summarize_run()

if __name__ == "__main__":
    main()

    #IDEA - Larger section clustering of smaller groups.  Or motif shifts
        #Could run it in the slider.py file.
    #IDEA - Or have a draggable band that switches your viewpoint to a histogram of the width of the band.
        #Like stumpy search function
    #TODO 
        #Add FFT / stumpy to main option button layout
        #have it reframe the layout the same way.  
        #In the distirbution plot below.  Maybe have the ability to refocus the main
        #chart based on the other area's.  (Think second slider below once vertical splits)
