import scipy.signal as ss
from scipy.fft import rfft, rfftfreq, irfft
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
import rich
from rich.logging import RichHandler
from rich.console import Console

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


def load_logger(name:str):
	FORMAT = "%(asctime)s|%(levelname)s|%(funcName)s|%(lineno)d|%(message)s" #[%(name)s]
	FORMAT_RICH = "%(funcName)s|%(lineno)d|%(message)s"
	console = Console(color_system="truecolor")
	rh = RichHandler(level = logging.INFO, console=console)
	rh.setFormatter(logging.Formatter(FORMAT_RICH))

	#Set up basic config for logger
	logging.basicConfig(
		level=logging.INFO, 
		format=FORMAT, 
		datefmt="[%X]",
		handlers=[rh]
	)

	logger = logging.getLogger(name) 
	return logger


#FUNCTION Load Header Files
def get_records(folder:str)->list:
	"""Pulls the file info out of the data directory for file paths

	Args:
		p (str): [path to root data directory]

	Returns:
		dat_files (list): [List of dat names]
		mib_files (list): [List of mib names]
		head_files (list): [List of header names]
	"""

	# There are 3 files for each record
	#.dat = ECG data
	#.hea = header file (file info)
	#.mib = annotation file (beat annotations)
	
	#Get base directory
	# p = os.path.normpath(os.getcwd() + os.sep + os.pardir)
	p = os.getcwd()
	
	if p.endswith("scripts"):
		p = p[:-8]

	base_dir = os.path.join(p, "src" , "rad_ecg", "data", folder)
	dat_files, mib_files, head_files = [], [], []
	for subject in os.scandir(base_dir):
		if subject.is_dir():
			for file_idx in os.scandir(subject.path):
				if file_idx.name.endswith('.dat') and file_idx.is_file():
					dat_files.append(file_idx.path)

				elif file_idx.name.endswith('.mib') and file_idx.is_file():
					mib_files.append(file_idx.path)

				elif file_idx.name.endswith('.hea') and file_idx.is_file():
					head_files.append(file_idx.path)
	#leaving these here in case needed later #dat_files, mib_files, 
	return head_files

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
	#Above function not in use.  But another way To test signal instability might be
	#to count the slope changes. if the slope changes are above some expected value.
	#Then smooth the wave with a savgol

#NOTE Graph the peaks
#TODO - Redo this peaks graph as a generic graphing function. 
	#Make it operate off just the major r peaks, rolling med, and signal. 
	#We can have a boolean switch for more advanced features if 
	#they exist at the time its called

# def graph_peaks(R_peaks:tuple, start_p:int, end_p:int, rol_med:np.array, measures:dict):
# 	"""[Graphing function for peak analysis]

# 	Args:
# 		R_peaks (tuple): [List of the current windows R_peaks]
# 		start_p (int): [starting point]
# 		end_p (int): [end point]
# 		rol_med (np.array): [rolled median for the current window]
# 		measures (dict[dict]): [dictionary of PQRST measures]
# 	"""	
# 	#note for moving: you'll need the wave imported too. 

# 	fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(12,6))
# 	ax.set_ylim(min(wave[start_p:end_p])-0.2, max(wave[start_p:end_p])+0.2)
# 	plt.scatter(R_peaks[0]+start_p, R_peaks[1]['peak_heights'], marker="D", color='red')
# 	plt.plot(list(range(start_p, end_p)), wave[start_p:end_p], 'b')
# 	plt.plot(list(range(start_p, end_p)), rol_med, color='orange')
# 	plt.legend(['ECG', 'ECG rolling median'])
# 	plt.title(f'ECG for idx {start_p}:{end_p}')
# 	plt.xlabel("TimeSteps")
# 	plt.ylabel("Amplitude (mV)")

# 	#rotate tick labels 45 deg and format them to be more readable
# 	plt.xticks(ax.get_xticks(), labels = label_formatter(ax.get_xticks()) , rotation=-30)

# 	#[x] Fixed bug of looking at all measures by pulling out the last 20 instead,
# 		# then the filttering should perform similarly but much faster


# 	#Get the last 20 items in the measures dictionary
# 		#I go back 20 in case of peaks that are erroneous
# 		#They get filtered out anyway by only including those in the start_p -> end_p range
# 	last_20_dict = dict(list(measures.items())[-20:])

# 	S_peaks = [last_20_dict[p]["S_peak"] for p in last_20_dict if "S_peak" in last_20_dict[p] and p in range(start_p, end_p)]
# 	Q_peaks = [last_20_dict[p]["Q_peak"] for p in last_20_dict if "Q_peak" in last_20_dict[p] and p in range(start_p, end_p)]
# 	T_peaks = [last_20_dict[p]["T_peak"] for p in last_20_dict if "T_peak" in last_20_dict[p] and p in range(start_p, end_p)]
# 	P_peaks = [last_20_dict[p]["P_peak"] for p in last_20_dict if "P_peak" in last_20_dict[p] and p in range(start_p, end_p)]

# 	plt.scatter(x = Q_peaks, y = wave[Q_peaks], marker="o", s = 40, color='cyan')
# 	plt.scatter(x = S_peaks , y = wave[S_peaks], marker="*", s = 40, color='magenta')
# 	plt.scatter(x = T_peaks , y = wave[T_peaks], marker="p", s = 40, color='black')
# 	plt.scatter(x = P_peaks , y = wave[P_peaks], marker="o", s = 40, color='green')
# 	add_cht_labels(R_peaks[0]+start_p, R_peaks[1]['peak_heights'], plt, "R_peaks")
# 	add_cht_labels(Q_peaks, wave[Q_peaks].flatten(), plt, "Q_peaks")
# 	add_cht_labels(S_peaks, wave[S_peaks].flatten(), plt, "S_peaks")
# 	add_cht_labels(P_peaks, wave[P_peaks].flatten(), plt, "P_peaks")
# 	add_cht_labels(T_peaks, wave[T_peaks].flatten(), plt, "T_peaks")
	
# 	#NOTE Outlier plotting
# 	for k, v in last_20_dict.items():
# 		for k1, v1 in v.items():
# 			if k in range(start_p, end_p) and "_outlier" in k1:
# 				peak_outlier = f'{str(k1)[:6]}'
# 				plt.scatter(x = last_20_dict[k][peak_outlier], y = wave[last_20_dict[k][peak_outlier]], marker="X", s = 80, color='red')
# 				# plt.annotate(f'{v["outlier"]}', (peak_outlier, wave[peak_outlier]), textcoords="offset points", xytext=(0, -10), ha='center')
# 	def onSpacebar(event):
# 		"""When scanning ECG's, hit the spacebar if you want to mark a section to 
# 		come back too.  

# 		Args:
# 			event (_type_): type of key event that was triggereed
# 		"""		
# 		if event.key == " ": 
# 			#Get the title of the current figure (has the ECG index range)
# 			title = plt.gcf().axes[0].title._text
# 			logging.warning(f"Andy didn't like something here {title}")
 
# 	if os.getcwd().endswith("scripts"):   
# 		pass

# 	else:
# 		timer_running = True
# 		timer_debug = fig.canvas.new_timer (interval=2000)
# 		callback_id = timer_debug.add_callback(plt.close)
# 		# onspacebarw_tdb = partial(onSpacebar, timer_debug=timer_debug, timer_running=timer_running)
# 		fig.canvas.mpl_connect('key_press_event', onSpacebar)
# 		timer_debug.start()
# 	plt.show()	
# 	plt.close()
