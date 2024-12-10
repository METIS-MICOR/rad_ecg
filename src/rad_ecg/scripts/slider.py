import scipy.signal as ss
from scipy.fft import rfft, rfftfreq, irfft
import numpy as np
import wfdb
from collections import deque, Counter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import utils # from rad_ecg.scripts
import setup_globals# from rad_ecg.scripts 


def load_graph_objects(run:str, cam:str):
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


    def update_plot(val):
        #TODO add docstrings
        #clear the top chart
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

    def create_comparison_plot(slider_val):
        #TODO add docstrings
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


    def radiob_action(val):
        #TODO add docstrings
        sect = sect_slider.val
        start_w = ecg_data['section_info'][sect]['start_point']
        end_w = ecg_data['section_info'][sect]['end_point']
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

        fig.canvas.draw_idle()

        if val == 'Overlay P':
            create_comparison_plot(sect_slider.val)

    def move_slider_forward(vl):
        #TODO add docstrings
        curr_sect = sect_slider.val
        next_sect = np.where(ecg_data['section_info']['valid'][curr_sect+1:]==0)[0][0] + curr_sect+1
        sect_slider.set_val(next_sect)

    def move_slider_sing_forward(vl):
        #TODO add docstrings
        curr_sect = sect_slider.val
        sect_slider.set_val(curr_sect+1)

    def move_slider_back(vl):
        #TODO add docstrings
        curr_sect = sect_slider.val
        prev_sect = np.where(ecg_data['section_info']['valid'][:curr_sect]==0)[0][-1]
        sect_slider.set_val(prev_sect)

    def move_slider_sing_back(vl):
        #TODO add docstrings
        curr_sect = sect_slider.val
        sect_slider.set_val(curr_sect-1)

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
    cam_n = cam.split(".")[0].split("\\")[-1]
    fpath = f"./src/rad_ecg/data/output/{cam_n}/{run}"  
    lfpath = f"./src/rad_ecg/data/logs/{run}"

    global ecg_data
    ecg_data = {
        "peaks": np.genfromtxt(fpath+"_peaks.csv", delimiter=",", dtype=np.int32, usecols=(0, 1)),
        "section_info": np.genfromtxt(fpath+"_section_info.csv", delimiter=",", dtype=wave_sect_dtype),
        "interior_peaks": np.genfromtxt(fpath+"_interior_peaks.csv", delimiter=",", dtype=np.int32, usecols=(range(15)))
    }

    #brokebarH plot for the background of the slider. 
    valid_sect = ecg_data['section_info']['valid']
    fig = plt.figure(figsize = (14, 8))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[5, 1])
    plt.subplots_adjust(hspace=0.40) 
    ax_ecg = fig.add_subplot(gs[0, 0])
    first_sect = np.where(ecg_data['section_info']['valid']!=0)[0][0]
    start_w = ecg_data['section_info'][first_sect]['start_point']
    end_w = ecg_data['section_info'][first_sect]['end_point']
    ax_ecg.plot(range(start_w, end_w), wave[start_w:end_w], color='dodgerblue', label='ECG')
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

    ax_section = fig.add_subplot(gs[1, 0])
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
    radio = RadioButtons(ax_radio, ('Roll Median', 'Add Inter', 'Hide Leg', 'Overlay P', 'Show R Valid'))

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

    # quick log loading for checking
    # log = utils.load_log_results(lfpath)

    #IDEA - Have that sections logs loaded up to when you
    # navigate to it.  That would save some serious time.

    plt.show()
    plt.close()
    
def summarize_run():
    RMSSD = ecg_data['section_info'][['wave_section', 'RMSSD']]
    logger.info(f"{sorted(RMSSD, key= lambda x:x[1], reverse=True)[:20]}")


def main():
    run = "09-27-2024_20-56-36"  	#E3SJA-BNC61
    
    global logger
    logger = utils.load_logger(__name__)

    global wave, fs
    wave, fs, cam = setup_globals.load_chartdata(logger)
    load_graph_objects(run, cam)
    # summarize_run()

if __name__ == "__main__":
    main()

    #TODO - Brainstorm summary formats
    #TODO - Add stumpy signal abnormality search to CAMS

    #IDEA - FFT switch. 
        #Would be cool if you could switch from the slider to the FFT version of a section. 
    #IDEA Larger section clustering of smaller groups.  Or motif shifts
        #Could run it in the slider.py file.
    #IDEA Or have a draggable band that switches your viewpoint to a histogram of the width of the band.
        #Like stumpy search function

        