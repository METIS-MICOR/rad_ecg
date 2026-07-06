#NOTE Import libraries
from rad_ecg.scripts import utils
from rad_ecg.scripts import support
import scipy.signal as ss
from scipy.fft import rfft, rfftfreq, irfft
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
from matplotlib.colors import rgb2hex
import time
import os
import logging
# from pathlib import Path
from collections import deque, Counter
from rich.logging import RichHandler
from rich.console import Console
from time import strftime
# from rich.progress import Progress
# from rich.live import Live
# from rich.panel import Panel
# from rich.table import Table
# from rich.layout import Layout
# from rich.theme import Theme
# from rich.progress import (
# 	Progress,
# 	BarColumn,
# 	SpinnerColumn,
# 	TextColumn,
# 	TimeRemainingColumn,
# 	MofNCompleteColumn,
# 	TimeElapsedColumn,
# )


current_date = strftime("%m-%d-%Y_%H-%M-%S")
FORMAT = "%(asctime)s|%(levelname)s|%(funcName)s|%(lineno)d|%(message)s" #[%(name)s]
FORMAT_RICH = "%(funcName)s|%(lineno)d|%(message)s"
console = Console(color_system="truecolor")
rh = RichHandler(level = logging.WARNING, console=console)
rh.setFormatter(logging.Formatter(FORMAT_RICH))

#Set up basic config for logger
logging.basicConfig(level=logging.INFO, 
					format=FORMAT,
					datefmt="[%X]",
					handlers=[
						rh,
						# logging.FileHandler(f"./src/rad_ecg/data/logs/{current_date}.log", mode="w")
						]
)

logger = logging.getLogger(__name__) 

#TERMINALSTUFF
#FUNCTION make_layout
# def make_layout() -> Layout:
# 	layout = Layout(name="root")
# 	layout.split(
# 		Layout(name="header", size=3), 
# 		Layout(name="main")
# 	)
# 	layout["main"].split_row(
# 		Layout(name="leftside"), 
# 		Layout(name="termoutput", ratio=2)
# 	)
# 	layout["leftside"].split_column(
# 		Layout(name="stats", ratio=2), 
# 		Layout(name="progbar")
# 	)
# 	return layout

#FUNCTION get_stats
# def get_stats(*sect_c) -> Table:
	
# 	rand_arr = np.random.randint(low=2, high=10, size=(10, 3)) + np.random.random((10, 3))
	
# 	stats_table = Table(
# 		expand=True,
# 		show_header=True,
# 		header_style="bold",
# 		title="[magenta][b]Last 5 sections HR data![/b]",
# 		highlight=True,
# 	)
# 	stats_table.add_column("Section", justify="right")
# 	stats_table.add_column("Avg HR", justify="center")
# 	stats_table.add_column("Std HR", justify="center")
# 	stats_table.add_column("RMSSD", justify="center")
# 	stats_table.add_column("NN50", justify="center")
# 	stats_table.add_column("PNN50", justify="center")
# 	if sect_c:
# 		#grab the last 5 valid 
# 		sect_c = sect_c[0]
# 		temp_arr = ecg_data["section_info"][sect_c-5:sect_c].copy()
# 		ids = np.nonzero(temp_arr["Avg_HR"])[0]
# 		for row in ids:
# 			stats_table.add_row(
# 				f"{temp_arr['wave_section'][row]}",
# 				f"{temp_arr['Avg_HR'][row]:.2f}",
# 				f"{temp_arr['SDNN'][row]:.2f}",
# 				f"{temp_arr['RMSSD'][row]:.2f}",
# 				f"{temp_arr['NN50'][row]:.2f}",
# 				f"{temp_arr['PNN50'][row]:.2f}",
# 			)
# 	else:
# 		for col in range(rand_arr.shape[1]):
# 			stats_table.add_row(
# 				f"Col {col}",
# 				f"{rand_arr[:, col].mean():.2f}",
# 				f"{rand_arr[:, col].std():.2f}",
# 				f"{rand_arr[:, col].max():.2f}",
# 				f"{rand_arr[:, col].min():.2f}",
# 			)

# 	return stats_table

#CLASS make_header
# class make_header:
# 	"""Display header with clock."""

# 	def __rich__(self) -> Panel:
# 		grid = Table.grid(expand=True)
# 		grid.add_column(justify="center", ratio=1)
# 		grid.add_column(justify="right")
# 		grid.add_row(
# 			"[b]ECG[/b] Detection application",
# 			datetime.now().ctime().replace(":", "[blink]:[/]"),
# 		)
# 		return Panel(grid, style="red on black")

#CLASS customer logging MainTableHandler
# class MainTableHandler(logging.Handler):
# 	def __init__(self, main_table: Table, layout: Layout, log_level: str):
# 		super().__init__()
# 		self.main_table = main_table
# 		self.log_list = []
# 		self.layout = layout
# 		self.log_format = "%(asctime)s-%(levelname)s-[%(message)s]"
# 		self.setLevel(logging.CRITICAL)
# 		#Could set colors for levels here. 
# 	#FUNCTION handler emit
# 	def emit(self, record):
# 		record.asctime = record.asctime.split(",")[0]
# 		#msg = self.format(record) #if you want just the message info switch comment lines
# 		msg = self.log_format % record.__dict__
# 		tsize = os.get_terminal_size().lines / 2
# 		if len(self.log_list) > int(tsize):
# 			self.log_list.append(msg)
# 			self.log_list.pop(0)
# 			mtable = redraw_main_table(self.log_list)
# 			self.layout["termoutput"].update(Panel(mtable, border_style="red"))
# 		else:
# 			self.main_table.add_row(msg)
# 			self.log_list.append(msg)

# #FUNCTION redraw_main_table
# def redraw_main_table(temp_list: list) -> Table:
# 	main_table = Table(
# 		expand=True,
# 		show_header=False,
# 		header_style="bold",
# 		title="[blue][b]Log Entries[/b]",
# 		highlight=True,
# 	)
# 	main_table.add_column("Log Entries")
# 	for row in temp_list:
# 		main_table.add_row(row)

# 	return main_table

# #FUNCTION get_file_handler
# def get_file_handler(log_dir: str) -> logging:
# 	log_format = "%(asctime)s|%(levelname)s|%(funcName)s|%(lineno)d|%(message)s" 
# 	log_file = log_dir
# 	file_handler = logging.FileHandler(log_file)
# 	file_handler.setFormatter(logging.Formatter(log_format))
# 	return file_handler

# #FUNCTION get_logger
# def get_logger(log_dir: str) -> logging:
# 	logger = logging.getLogger(__name__)
# 	logger.setLevel(logging.INFO)
# 	logger.addHandler(get_file_handler(log_dir))
# 	return logger

#FUNCTION Log time
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
#FUNCTION Consecutive valid peaks
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
		thres = np.where(freq_list < 18)[0][-1]
		outs = np.where(fft_samp[thres:int(len(samp)/2)] > fft_samp[0:thres].mean())[0]

		if outs.size >= 2:
			bad_sect_counter += 1
			#Already zero so you don't need to set this.  Ahhhh but they used to come in as one's.  
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
def section_stats(new_peaks_arr:np.array, section_counter:int, fs:float)->tuple:
	"""This function calculates the time domain stats for a given section. 

	Args:
		new_peaks_arr (np.array): Peaks for evaluation
		section_counter (int):Tracking what section we're in
		fs (float): Sampling frequency
	Returns:
		(tuple): Tuple of the HR stats for that section.
	"""		
	#First look and see if there's any peaks that are invalid (ie = 0). 
	#(Ignore the last peak as it's most likely invalid)
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
		#BUG - Lookback ranges
			#How far back does one need to look for each of these metrics.
			#may need to use the consecutive valid peaks funtion again here.  
			#Ask friday labgroup
			#HR
			#SDNN
			#RMSSD - 5 minute epoch convention (soroosh did 30 sec)
			#Keep a container of the last 15 epoch's
			#Look into heart rate turbulence.  
			#TODO:Add in QTVI to section metrics. 

		try:
			NN50 =  np.where(RR_diffs_time > 50)[0].shape[0]
			PNN50 = np.round((NN50 / RR_diffs.shape[0]) * 100, 2)
		except Exception as e:
			logger.warning(f'Unable to find NN50 {e}')
			NN50 = np.nan
			PNN50 = np.nan

		return (Avg_HR, SDNN, min_HR, max_HR, RMSSD, NN50, PNN50)
	
	#IDEA - .. PQRST std dev.  
		#Much like HRV looking at the variation of R peak to R peak.  
		#Is there any benefit in looking at the std dev of PQST peaks. 
		#Could run a clustering algorithm to look at outliers in a section. 
		#Or looking at a cluster variability? as an indication of a single
		# portion of the cycle malfunctioning.  Which would be very hard to 
		# see in the classical way of looking ECG's over time


	#Conditions: 
		# The section could still be valid and have erroneous points (rolling median)
		# But remember, this only gets called when the section is valid.  
		# So there wont' be peak sep/height errors, But possibly rolling median errors.

#FUNCTION Peak Validation Check
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

	#!Temp removal for testing
	# #Get the 25/75 quartiles and IQR
	# Boosted IQR to 80/20 - 7/12/22.  Better performance and less rolling
	#median errors on feasible sections
	# Q1 = np.quantile(med_arr, .10)
	# Q3 = np.quantile(med_arr, .90)
	Q1 = np.quantile(med_arr, .20)
	Q3 = np.quantile(med_arr, .80)
	
	IQR = Q3 - Q1

	# Test to see if IQR is lower than IQR_low_thresh This is to prevent hitting
	# a vanishing gradient for IQR.  
	# Previously the aglorithm would continually give roll violations  when the
	# after the signal got extremely quiet.  
	if IQR == IQR_low_thresh:
		low_counts += 1
		if low_counts > 6:
			IQR = 2*IQR
			logger.warning(f'Bumped up IQR 100% to {IQR:.4f} for section {cur_sect} low_count at {low_counts}')
		elif low_counts > 3: 
			IQR = IQR + .50*IQR
			logger.info(f'Bumped up IQR 50% to {IQR:.4f} for section {cur_sect} low_count at {low_counts}')

	elif IQR <= IQR_low_thresh:	
		IQR_low_thresh = IQR
	
	else:
		logger.info(f'IQR reset for section {cur_sect}')
		IQR_low_thresh = 1
		low_counts = 0

	#[x] Try Tom idea
	#Tom idea.  instead of widening IQR
	#monitor the difference between rolling median and average R_peaks. 

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

	#NOTE Soroosh Slope Check

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
		#TODO - Would it be worth making a boundary function here.  
		#You're using boundaries in 3 of your gates here.  So it could be useful.
	else:
		logging.critical(f"Uneven lengths of leftbases in sect {cur_sect}")
		fail_reas = "slope"
		peak_slope_check = False
		sect_valid = False

		#BUG - Edge case
			#Sometimes the last point change will be right next to the R peak.  (High freq vibration at the crest of the signal)
			#Thinking we put in another requirement here that the last inflection point also be below the rolling median. 
			#Removing those types of identifiers.
			#TODO - Fix above bug

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
				#Need to subtract start_idx from p0 and p1 to get correct indexing of rollmed
				#This way we're indexing the rolling median from the first peak in the analysis
				#To the end end of the chunk
			#If section from p0 to p1 is outside IQR range.  add to bad panda count
			##!Temp removal for testing
			# out_above = np.where(samp_section > (Q3 + 1.5*IQR))[0]
			# out_below = np.where(samp_section < (Q1 - 1.5*IQR))[0]

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
			#BUG - Global Avgs
			#I need a reset function for when the QRS lookback time gets over a certain amount.  
			#Otherwise it could get stuck on previous averages that it doesn't need. 
			#INstead a global average should be used and only based on peak sep and height.  As rolling median
			#will not be applicable if the base morphology of the wave changes. 
			#Encode fail reason
			if len(fail_reas) > 0:
				fail_reas = fail_reas + '|roll'
			else:
				fail_reas = 'roll'

			if plot_errors:
				fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(12,6))
				##!Temp removal for testing
				# plt.plot(range(start_idx, end_idx), wave_shift, label = 'ECG')
				plt.plot(range(start_idx, end_idx), wave[start_idx:end_idx], label = 'ECG')
				plt.plot(range(start_idx, end_idx), samp_roll_med, label='Rolling Median shifted')
				plt.scatter(new_peaks_arr[:, 0], peak_info['peak_heights'], marker='D', color='red', label='R peaks')
				plt.axhline(y=(np.quantile(samp_roll_med, .80)+1.5*IQR), color='magenta', linestyle='--', label='upper past roll med')
				plt.axhline(y=(np.quantile(samp_roll_med, .20)-1.5*IQR), color='red', linestyle='--', label='lower past roll med')
				plt.axhline(y=samp_roll_med.mean(), color = 'green', linestyle='--', label='rolling median mean (shifted)')
				ax.set_xticks(ax.get_xticks(), labels = utils.label_formatter(ax.get_xticks()) , rotation=-30)
				# plt.plot(samp_roll_med + (Q3+1.5*IQR), '--', color='green')
				# Use above if you want a bounded Quantile.  Might be worth
				# Using if the actual wave escapes those bounds 

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

	#NOTE Peak height check
	#BUG - Wandering R peaks
		#After reviewing many ECG's, I'm finding a common problem in checking R
		#peak heights.  Some people's ECG's have R peak heights that are quite
		#consistent.  others have a cyclical variance that have a large standard
		#deviation.  I'm thinking of ways around this without increasing the
		#search boundaries of scipy's find_peaks.  
		#IDEA
			#1.Use std outlier calc by the IQR of their peak heights. (Q1 + 1.5*IQR)
			#2.Use the distance from the R peak to the rolling median. If the slope is
			#sharp enough, it should hold decent separation.  (close to soroosh method)
	
	#orig
	# lower_bound = last_avg_peak_heights * 0.51 #Stock 0.6
	# upper_bound = last_avg_peak_heights * 4
	
	#idea1
	# Q1 = np.quantile(wave[new_peaks_arr[:, 0]], .25)
	# Q3 = np.quantile(wave[new_peaks_arr[:, 0]], .75)
	# IQR = Q3 - Q1

	# lower_bound = Q1 - (1.5 * IQR)
	# upper_bound = last_avg_peak_heights * 4
	# upper_bound = Q3 + (1.5 * IQR)
	# peak_height_check = np.any((peak_info['peak_heights'] < lower_bound ) | (peak_info['peak_heights'] > upper_bound))
	
	#idea2
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

	#NOTE Separation Check
	#Separation updated to 50% in each direction
	#Increased High end to 4x last_avg_peak_sep 5-31-23
	#[x] - Increase high end range
		#sudden changes from slow to fast heart rate. 
		#Long periods of repolarization for NN50
	#[x] - Backing down to 2x.  4x is too generous. 
	 
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
				#BUG - Possible error here if the last line is the sep error
			
			# plt.annotate(text='', xy=bad_sep_start_cords, xytext=bad_sep_fin_cords, arrowprops=dict(arrowstyle='<->'))
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

#FUNCTION Extract PQRST
def extract_PQRST(st_fn:tuple, 
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
		#was 0.1 . Needed a wider range here at higher heart rates when the other peaks beat with greater voltage.
		
		# reject_limit = threshold * (max(samp) - min(samp))
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
			#Old code but keeping for validation purposes
			#find the shortest distance
			# a = coef[0]	
			# b = coef[1]
			# c = a*point[0] + b*point[1]
			# d = np.abs((a * point[0] + b * point[1] + c)) / (np.sqrt(a * a + b * b)) 			
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
		#! Saved for testing
		# SQ_range = wave[temp_arr[temp_counter, 3]:temp_arr[temp_counter + 1, 1]]
		SQ_range = wave[samp_min:temp_arr[temp_counter + 1, 1]]

		#Do the same for the rolling median. 
		#! Saved for testing
		# filt_rol_med = rolled_med[temp_arr[temp_counter, 3]-st_fn[1]:temp_arr[temp_counter + 1] - st_fn[1]]
		filt_rol_med = rolled_med[samp_min-st_fn[1]:temp_arr[temp_counter + 1, 1] - st_fn[1]]

		#Subtract rolling median from wave to flatten it.
		#TODO.  Try scipy.detrend here too
		SQ_med_reduced = SQ_range - filt_rol_med


		#MEAS T Peak 
		try:
			RR_first_half = SQ_med_reduced[:(SQ_med_reduced.shape[0]//2)]
			#BUG - Noticing some T-peaks get very delayed.  to over the halfway
			#point through the r0 to r1 range.  Currently we're just looking in
			#the first half, but might need to increase that range to something
			#bigger.

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
	
	
	# # Graph check for PQRST peaks
	# # Note: Graph of the interior peaks after it has finished analyzing that section. 
	# #Quick plot check to make sure points are assigned correctly. 
	# fig, ax = plt.subplots(figsize=(12,8))
	# plt.plot(range(st_fn[1], st_fn[2]), wave[st_fn[1]:st_fn[2]], label='wave')
	# plt.plot(range(st_fn[1], st_fn[2]), rolled_med, label='median')
	# # Get nonzero indices of each peak in temp_arr
	# P_peaks = temp_arr[np.where(temp_arr[:, 0] != 0)[0], 0]
	# Q_peaks = temp_arr[np.where(temp_arr[:, 1] != 0)[0], 1]
	# R_peaks = temp_arr[np.where(temp_arr[:, 2] != 0)[0], 2]
	# S_peaks = temp_arr[np.where(temp_arr[:, 3] != 0)[0], 3]
	# T_peaks = temp_arr[np.where(temp_arr[:, 4] != 0)[0], 4]
	# # Graph em
	# plt.scatter(P_peaks, wave[P_peaks], label='P', marker="o", s = 40, color='green')
	# plt.scatter(Q_peaks, wave[Q_peaks], label='Q', marker="o", s = 40, color='cyan')
	# plt.scatter(R_peaks, wave[R_peaks], label='R', marker='d', s = 40, color='red')
	# plt.scatter(S_peaks, wave[S_peaks], label='S', marker="*", s = 40, color='magenta')
	# plt.scatter(T_peaks, wave[T_peaks], label='T', marker="p", s = 40, color='black')
	# #add the labels
	# utils.add_cht_labels(R_peaks, wave[R_peaks], plt, "R_peaks")
	# utils.add_cht_labels(Q_peaks, wave[Q_peaks], plt, "Q_peaks")
	# utils.add_cht_labels(S_peaks, wave[S_peaks], plt, "S_peaks")
	# utils.add_cht_labels(P_peaks, wave[P_peaks], plt, "P_peaks")
	# utils.add_cht_labels(T_peaks, wave[T_peaks], plt, "T_peaks")
	# #rotate tick labels 45 deg and format them to be more readable
	# plt.xticks(ax.get_xticks(), labels = utils.label_formatter(ax.get_xticks()) , rotation=-30)
	# plt.title(f'ECG waveform for indices {st_fn[1]:_d}:{st_fn[1]:_d}')
	# plt.xlabel("TimeSteps")
	# plt.ylabel("Amplitude (mV)")
	# plt.legend()
	# plt.show()
	# plt.close()

	
	#NOTE Segment Data  Extraction
	#The earlier iteration was looping between each R_peak to get its
	#consitutient peak values. This iteration moves on each individual peak.  
	peak_que = deque(new_peaks_arr[:, 0])
	#Get half the avg peak width for evaluation
	# avg_peak_diff = int(np.mean(np.diff(peak_que))//2)
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
			#!Might need to update the above when we change the Speak location
			#to a shorter duration.  
		#MEAS Q_onset
		# Q_shoulder = Q_onset
		#changed from 80 to 70 below - 7-23-22.  Occasionally catches steep
		#sections next to P peak
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

		#Temp plot to check Q onset.
		#Limit from P to Q
		#add rolling median
		#Highlight section looking backwards. 
		#Graph the 2nd deriv
		#isolate first point greater than mean of avg 2nd derivs
		# fig, ax = plt.subplots(figsize=(12, 7))
		# plt.plot(range(P_peak-1, R_peak), wave[P_peak-1:R_peak], label='P to R sect', color='dodgerblue')
		# plt.plot(range(slope_start, slope_end), wave[slope_start:slope_end], label='eval section', color='purple')
		# plt.plot(range(slope_start, slope_end), lil_grads, label='gradients', color='orange')
		# plt.scatter(temp_arr[temp_counter, 12], wave[temp_arr[temp_counter, 12]], label='Q onset')
		# plt.scatter(P_peak, wave[P_peak], label='P peak')
		# plt.scatter(Q_peak, wave[Q_peak], label='Q peak')

		
		# plt.xticks(ax.get_xticks(), labels = label_formatter(ax.get_xticks()) , rotation=-20)
		# plt.title(f'Q onset for indices {st_fn[1]:_d}:{st_fn[1]:_d}')
		# plt.xlabel("ECG Index")
		# plt.ylabel("Amplitude (mV)")
		# plt.legend(loc="upper left")
		# plt.show()
		# plt.close()

		#MEAS T onset
		#[x] Need to update slope_start here to samp_min.  Whiiiiich
		# probably means i need to store it as well.  ugh.  
		 
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

		#Quickplot for T onset
		# fig, ax = plt.subplots(figsize=(12, 7))
		# plt.plot(range(R_peak-1, T_peak), wave[R_peak-1:T_peak], label='R to T sect', color='dodgerblue')
		# plt.plot(range(slope_start, slope_end), wave[slope_start:slope_end], label='eval sect', color='purple')
		# plt.plot(range(R_peak, T_peak), rolled_med[R_peak-st_fn[1]:T_peak-st_fn[1]], label='rolling median', color='red')
		# plt.scatter(temp_arr[temp_counter, 13], wave[temp_arr[temp_counter, 13]], label='T onset')
		# plt.scatter(R_peak, wave[R_peak], label='R peak')
		# plt.scatter(T_peak, wave[T_peak], label='T peak')
		# plt.scatter(samp_min_dict[R_peak], wave[samp_min_dict[R_peak]], label='Sect sample min')
		# plt.xticks(ax.get_xticks(), labels = label_formatter(ax.get_xticks()) , rotation=-20)
		# plt.title(f'T onset for indices {st_fn[1]:_d}:{st_fn[1]:_d}')
		# plt.xlabel("ECG Index")
		# plt.ylabel("Amplitude (mV)")
		# plt.legend(loc="upper right")
		# plt.show()
		# plt.close()


		# S_shoulder = T_onset
		#OLD way below for S_ Shoulder extraction
		# lil_grads = np.gradient(lil_wave)
		# gre_mean_slopes = np.where(np.abs(lil_grads) >= np.mean(np.abs(lil_grads)))[0]
		# #splits the groups by where the changes in index are not equal to 1
		# groups = grouper(gre_mean_slopes)
		# # print(groups)
		# #Get the lengths of each group.  
		# lengths = Counter([x.size for x in groups])
		# #Get the largest length group
		# big_group = max(groups, key=lambda x : x.size)
		# #The max function will return it by size the first entry by esize.   So we need to check for equal 
		# #lengths.  In which case I'd want to take the second group.  Logic below
		# if lengths[len(big_group)] != 1:
		# 	big_group = sorted(groups, key=lambda x: (x.size, x[0]), reverse=True)[0]

		# #Lastly.  Check for if there is another slope change from the shoulder pos
		# #to the T_peak.  IF there is mark is at as the S_shoulder.  If not.  Take the 
		# #left base of the T_peak as the shoudler previously found. 
		# try:	
		# 	signs = np.sign(lil_grads[big_group[0]:])
		# 	if any(signs==-1):
		# 		# print('ping')
		# 		S_shoulder = slope_start + np.where(signs == -1)[0][0] + big_group[0]
		# 	else:
		# 		S_shoulder = slope_start + big_group[0]
		# except Exception as e:
		# 	logger.warning(f'Error = \n{e} on {R_peak:_d}')

		#MEAS QRS Complex
		#Add the QRS time in ms if both the onsets exist.
		if Q_onset and T_onset:
			temp_arr[temp_counter, 8] = int(1000*((T_onset - Q_onset)/fs))
			# temp_arr[temp_counter, 8] = int(1000*((temp_arr[temp_counter, 3] - Q_onset)/fs))

		# ? Checking histogram of returned QRS vals. 
		# hist, bins = np.histogram(ecg_data['interior_peaks'][np.where(ecg_data['interior_peaks'][:, 8] !=0)[0], 8], bins=10)
		# plt.figure(figsize=(16, 8))
		# plt.hist(ecg_data['interior_peaks'][ids[0], 8], bins)
		# plt.title(f'Histogram of QRS vals', fontsize=18)
		# plt.annotate(f'Avg QRS {np.mean(ecg_data["interior_peaks"][ids[0], 8]):.2f} ms', xy=(0.02, 0.95), xycoords='axes fraction', fontsize=12)
		# plt.annotate(f'Max QRS {np.max(ecg_data["interior_peaks"][ids[0], 8]):.2f} ms', xy=(0.02, 0.91), xycoords='axes fraction', fontsize=12)
		# plt.annotate(f'Min QRS {np.min(ecg_data["interior_peaks"][ids[0], 8]):.2f} ms', xy=(0.02, 0.87), xycoords='axes fraction', fontsize=12)

		#PR Interval
		#TODO - Need to do a test for a shoulder here.  Looking at whether or
		# not there is a negative to positive or negative to zero slope change
		# from P to Q.  Might need a better method for this but test it.  
			#See if the rolling median shows any behavior here. 

		# Filter the P to Q range. PR Segment goes from the end of the P peak
		# curvature, to the jerk/shoulder before the Q peak.
		
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
		
		# #quickplot for checking P_shoulders
		# fig, ax = plt.subplots(figsize=(12, 8))
		# st_g = P_peak - (2*srch_width)
		# end_g = P_peak + (2*srch_width)
		# max_gradient_idx = np.argmax(lil_grads)
		# plt.plot(range(st_g, end_g), wave[st_g:end_g], label='ECG')
		# plt.plot(range(slope_start, slope_end), lil_grads, label='gradients')
		# plt.scatter(P_peak, wave[P_peak], label='P peak')
		# plt.axvline(x=P_peak + srch_width, color='goldenrod', linestyle='--')
		# plt.axvline(x=P_peak - srch_width, color='goldenrod', linestyle='--')
		# plt.scatter(P_onset, wave[P_onset], color='purple', label='P onset')
		# plt.scatter(slope_start + max_gradient_idx, lil_grads[max_gradient_idx], label='Gradient max')
		# plt.hlines(wave[P_peak]+0.03, P_peak - srch_width, P_peak + srch_width, color='goldenrod', linestyle='--')
		# plt.annotate('Search Width', xy=(0.5, 0.83), xycoords='axes fraction', color='goldenrod', fontsize=14)
		# plt.xticks(ax.get_xticks(), labels = label_formatter(ax.get_xticks()) , rotation=-20)
		# plt.title(f'P onset for indices {st_fn[1]:_d}:{st_fn[1]:_d}')
		# plt.xlabel("ECG Index")
		# plt.ylabel("Amplitude (mV)")
		# plt.legend(loc="upper left")
		# plt.show()
		# plt.close()

		#dev chart
		# plt.plot(range(slope_start, slope_end), wave[slope_start:slope_end], label='ECG')
		# plt.plot(range(slope_start, slope_end), lil_grads, label='gradients')
		# plt.scatter(P_shoulder, wave[P_shoulder], color='red', label='ArgMax Gradient')
		# plt.show()
		# plt.scatter(slope_start + np.argmax(np.gradient(lil_grads)+1), wave[slope_start + np.argmax(np.gradient(lil_grads)+1)], label='argmax')
		# plt.xticks(ax.get_xticks(), labels = label_formatter(ax.get_xticks()) , rotation=-20)
		# plt.title(f'T onset for indices {st_fn[1]:_d}:{st_fn[1]:_d}')
		# plt.xlabel("ECG Index")
		# plt.ylabel("Amplitude (mV)")
		# plt.legend(loc="upper right")
		# plt.show()
		# plt.close()


		#MEAS ST Segment
		#ST segments are suppressed in this case as the higher heart rate obliterates them. 
		#TODO.  Find a healthy recording to test this on. 
		#TODO - Also will need a test in the future to see if the J point exists
		

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
		#quickplot for checking T_offset
		# fig, ax = plt.subplots(figsize=(12, 8))
		# st_g = T_peak - (2*srch_width)
		# end_g = T_peak + (2*srch_width)
		# max_gradient_idx = np.argmax(lil_grads)
		# plt.plot(range(st_g, end_g), wave[st_g:end_g], label='ECG', color='dodgerblue')
		# plt.plot(range(slope_start, slope_end), lil_grads, label='gradients', color='orange')
		# plt.scatter(T_peak, wave[T_peak], label='T peak', color='black')
		# plt.axvline(x=T_peak + srch_width*1.25, color='goldenrod', linestyle='--')
		# plt.axvline(x=T_peak - srch_width*1.25, color='goldenrod', linestyle='--')
		# plt.scatter(T_offset, wave[T_offset], color='orange', label='T offset')
		# plt.scatter(slope_start + max_gradient_idx, lil_grads[max_gradient_idx], label='Gradient max', color='orange')
		# plt.hlines(wave[T_peak]+0.01, T_peak - srch_width*1.25, T_peak + srch_width*1.25, color='goldenrod', linestyle='--')
		# plt.annotate('Search Width', xy=(0.40, 0.90), xycoords='axes fraction', color='goldenrod', fontsize=14)
		# plt.xticks(ax.get_xticks(), labels = label_formatter(ax.get_xticks()) , rotation=-20)
		# plt.title(f'P onset for indices {st_fn[1]:_d}:{st_fn[1]:_d}')
		# plt.xlabel("ECG Index")
		# plt.ylabel("Amplitude (mV)")
		# plt.legend(loc="upper left")
		# plt.show()
		# plt.close()

		#dev chart
		# fig = plt.figure(figsize=(16, 8))
		# plt.plot(range(slope_start, slope_end), wave[slope_start:slope_end], label='ECG')
		# plt.plot(range(slope_start, slope_end), lil_grads, label='gradients')
		# plt.scatter(T_offset, wave[T_offset], color='red', label='ArgMax Gradient')
		# plt.title(f'R_peak: {R_peak} Graph : {temp_counter}')
		# plt.show()

		# Shift the counter
		temp_counter += 1


	# Final plot check for interior peaks
	# fig, ax = plt.subplots(figsize=(12,8))
	# plt.plot(range(st_fn[1], st_fn[2]), wave[st_fn[1]:st_fn[2]], label='wave')
	# plt.plot(range(st_fn[1], st_fn[2]), rolled_med, label='rolling median')
	# # Get nonzero indices of each peak in temp_arr
	# P_peaks = temp_arr[np.where(temp_arr[:, 0] != 0)[0], 0]
	# Q_peaks = temp_arr[np.where(temp_arr[:, 1] != 0)[0], 1]
	# R_peaks = temp_arr[np.where(temp_arr[:, 2] != 0)[0], 2]
	# S_peaks = temp_arr[np.where(temp_arr[:, 3] != 0)[0], 3]
	# T_peaks = temp_arr[np.where(temp_arr[:, 4] != 0)[0], 4]

	# # Graph em
	# plt.scatter(P_peaks, wave[P_peaks], label='P', marker="o", s = 40, color='green')
	# plt.scatter(Q_peaks, wave[Q_peaks], label='Q', marker="o", s = 40, color='cyan')
	# plt.scatter(R_peaks, wave[R_peaks], label='R', marker='d', s = 40, color='red')
	# plt.scatter(S_peaks, wave[S_peaks], label='S', marker="*", s = 40, color='magenta')
	# plt.scatter(T_peaks, wave[T_peaks], label='T', marker="p", s = 40, color='black')

	# #Get nonzero indices of interior peak attributes 
	# P_onsets = temp_arr[np.where(temp_arr[:, 11] != 0)[0], 11]
	# Q_onsets = temp_arr[np.where(temp_arr[:, 12] != 0)[0], 12]
	# T_onsets = temp_arr[np.where(temp_arr[:, 13] != 0)[0], 13]
	# T_offsets = temp_arr[np.where(temp_arr[:, 14] != 0)[0], 14]
	
	# #graph em
	# plt.scatter(P_onsets, wave[P_onsets], label='P_onset', marker="o", s = 40, color='purple')
	# plt.scatter(Q_onsets, wave[Q_onsets], label='Q_onset', marker="o", s = 40, color='darkgoldenrod')
	# plt.scatter(T_onsets, wave[T_onsets], label='T_onset', marker='d', s = 40, color='teal')
	# plt.scatter(T_offsets, wave[T_offsets], label='T_offset', marker="*", s = 40, color='orange')
	
	# #add the labels
	# utils.add_cht_labels(R_peaks, wave[R_peaks], plt, "R_peaks")
	# utils.add_cht_labels(Q_peaks, wave[Q_peaks], plt, "Q_peaks")
	# utils.add_cht_labels(S_peaks, wave[S_peaks], plt, "S_peaks")
	# utils.add_cht_labels(P_peaks, wave[P_peaks], plt, "P_peaks")
	# utils.add_cht_labels(T_peaks, wave[T_peaks], plt, "T_peaks")
	# #rotate tick labels 45 deg and format them to be more readable
	# plt.xticks(ax.get_xticks(), labels = utils.label_formatter(ax.get_xticks()) , rotation=-30)
	# plt.title(f'ECG waveform for indices {st_fn[1]:_d}:{st_fn[1]:_d}')
	# plt.xlabel("TimeSteps")
	# plt.ylabel("Amplitude (mV)")
	# plt.legend()
	# plt.show()
	# plt.close()

	return temp_arr


#FUNCTION Main Peak Search
@log_time
def main_peak_search(
	ecg_data:dict,
	wave:np.array,
	fs:float,
	plot_fft:bool=False,
	plot_errors:bool=False
	):
	"""Detects R Peaks in the ECG wave.  Inputs peak positions into ecg_data dictionary.

	Args:
		ecg_data (dict): Dictionary of peaks and their info
		wave (np.array): Waveform to be analyzed
		fs (float): Sampling rate of the signal
		plot_fft (bool): boolean of whether or not to graph FFT
		plot_errors (bool): boolean of whether or not to plot errors

	Returns:
		ecg_data (dict): Dictionary with R_peak data
	"""
	#section tracking + invalid section tracking
	section_counter, invalid_sect_counter = 0, 0
	#Whether the wave is found
	found_wave = False
	#Whether dynamic STFT is in a countdown
	stft_loop_on = False
	stft_count = 0
	#Sample ranges to test the array stacking to ensure we're not getting slowdowns there. 
	stack_range = [x for x in range(0, 80000, 500)]
	#Stacking test for peak addition	
	
	@log_time
	def peak_stack_test(new_peaks_arr:np.array):
		return np.vstack((ecg_data['peaks'], new_peaks_arr)).astype(np.int32)

	#Set IQR data for start
	global IQR_low_thresh
	IQR_low_thresh = 1
	low_counts = 0


	#Load up a deque of start and end sections ( made by segment ECG)
	peak_que = deque(ecg_data['section_info'][['start_point', 'end_point']])
	# #TERMINALSTUFF
	# with Live(layout, console=console, refresh_per_second=4, screen=True) as live:
	# 	# Add MainTableHandler to logger
	# 	logger.addHandler(MainTableHandler(main_table, layout, logger.level))

	while len(peak_que) > 0:
		# time.sleep(0.1)
		curr_section = peak_que.popleft()
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
								distance = round(fs*(0.200)))  #Can't have a heart rate faster than 200ms

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
											plot_fft, 
											wave, 
											fs)

				# If you've found the wave and have sufficient num of peaks
				if sect_valid and R_peaks.size > 10:
					found_wave = True
					start_sect = section_counter
					ecg_data['section_info'][section_counter]['valid'] = 1
					logger.critical(f'Wave found at {start_p}:{end_p} in section {start_sect}')

					#Add the current  R peaks to the "peaks" and "interior_peaks" data container. 
					ecg_data['peaks'] = np.vstack((ecg_data['peaks'], new_peaks_arr)).astype(np.int32)

					#BUG - Not sure why i'm adding these at the moment.  Seeing as I didn't run the extract PQRST for this section. 
					int_peaks = np.zeros(shape=(new_peaks.shape[0], 15), dtype=np.int32)
					#Add the R peaks to the interior_peaks container. 
					int_peaks[:, 2] = new_peaks_arr[:, 0]
					ecg_data['interior_peaks'] = np.vstack((ecg_data['interior_peaks'], int_peaks))

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
			
			# if section_counter == 2802:
			# 	logger.critical(f'KENNY LOGGINS!!!???\nMarking trouble section')

			#Making sure we have enough historical data to scan backwards in time. 
			#Make sure the section counter is at least 10 ahead of the start_sect
			if section_counter < start_sect + 10:
				sect_valid, new_peaks_arr = STFT(
											new_peaks_arr, 
											peak_info, 
											rolled_med, 
											(section_counter, start_p, end_p), 
											plot_fft, 
											wave, 
											fs)
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
											plot_fft,
											wave,
											fs)
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
											plot_fft,
											wave,
											fs)

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
				sect_stats = section_stats(new_peaks_arr, section_counter, fs)
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
			
			# #TERMINALSTUFF
			# if section_counter % 20 == 0:
			# 	stats_table = get_stats(section_counter)
			# 	layout["stats"].update(Panel(stats_table, border_style="magenta"))

			#Advance section tracker to next section
			section_counter += 1
			logger.info(f'Section counter at {section_counter}')
			
			# #TERMINALSTUFF
			# #Update the progress bar
			# my_progress_bar.update(my_task, completed=section_counter)
			# live.refresh()

	return ecg_data

#Save ECG data to file and send confirmation email that the run is done. 
#FUNCTION Send Notification Email
def send_email(log_path:str):
	if os.getcwd().endswith('scripts'):
		pass
	else:
		with open(log_path, 'r') as f:
			for line in f.readlines()[-1:]:
				peak_search_runtime = line.split("|")[4].strip("")

		support.send_run_email(peak_search_runtime)
		logger.info('Runtime email sent')

#FUNCTION Save results
def save_results(ecg_data:dict):
	#Because structured arrays will do(ecg_data['section_info']) have mixed dtypes. You
	#have to feed the types back to the save routine when you save it.
	#(╯°□°）╯︵ ┻━┻

	#Export the CSV files
	logger.info("Savings CSV's")
	#Eventually need a folder existence check here.  If it doesn't, create it. 
	for x in ["peaks", "interior_peaks", "section_info"]:
		file_path = f"./data/csv/runs/{current_date}_{x}.csv"
		if x == "section_info":
			save_format = '%i, '*4 + '%s, ' + '%.2f, '*7
		else:
			save_format = '%i, '*ecg_data[x].shape[1]

		np.savetxt(
			fname = file_path,
			X = ecg_data[x], 
			fmt = save_format,
			delimiter=',',
		)
		logger.info(f'Saved {x} to {file_path}')
	
	# logger.warning(f'Size of rolling median as {ecg_data["rolling_med"].dtype} {sys.getsizeof(ecg_data["rolling_med"])*.000_001:.2f} MB')
	logger.critical(f"Wave section counts{np.unique(ecg_data['section_info']['valid'], return_counts=True)}")
	fail_counts = Counter(ecg_data['section_info']['fail_reason'])
	logger.critical(f"Fail reasons found:{list(fail_counts.items())}")

#NOTE START PROGRAM
def main():
	#NOTE Data Loader functions.  Load CAM files into inputdata folder. 
	head_files = utils.get_records('inputdata')

	#Need an input here to tell them which one to use.  
	logger.info("Which CAM would you like to import?")
	for idx, head in enumerate(head_files):
		name = head.split(".")[0].split("\\")[-1]
		logger.warning(f'file {idx}:\t{name}')
	header_chosen = input("Please choose a file number ")

	if not header_chosen.isnumeric():
		logger.critical(f'Incorrect file chosen, program terminating')
		exit()
	else:
		header_chosen = int(header_chosen)


	record = utils.load_signal_data(head_files[header_chosen])

	#ECG data
	global wave
	wave = record.p_signal
	
	#Frequency
	global fs
	fs = record.fs

	#Divide by signal length by freq(fs) for time domain
	# time_data = np.arange(0, len(wave), 1)/fs

	#Divide the wave into segments 
	wave_sections = utils.segment_ECG(wave, fs)[0:6000]

	#Setting mixed datatype (structured array) for ecg_data['section_info']
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

	#Base data container keys
	global ecg_data
	ecg_data = {
		'peaks': np.zeros(shape=(0, 2), dtype=np.int32),
		'rolling_med': np.zeros(shape=(wave.shape[0]), dtype=np.float32),
		'section_info': np.zeros(shape=(wave_sections.shape[0]), dtype=wave_sect_dtype),
		'interior_peaks': np.zeros(shape=(0, 15), dtype=np.int32)
	}

	ecg_data['section_info']['wave_section'] = np.arange(0, wave_sections.shape[0], 1)
	ecg_data['section_info']['start_point'] = wave_sections[:,0]
	ecg_data['section_info']['end_point'] = wave_sections[:,1]
	ecg_data['section_info']['valid'] = wave_sections[:,2]

	del wave_sections
	
	#NOTE Start R peak search
	ecg_data = main_peak_search( 
		ecg_data,
		wave, 
		fs,
		plot_fft=False, 
		plot_errors=True, 
	)

	log_path = f'./data/logs/{current_date}.log'
	send_email(log_path)
	save_results(ecg_data)
	logger.info('All done!')

		
if __name__ == "__main__":
	main()

# #TERMINALSTUFF
# #NOTE Making TUI 
# console = Console(color_system="truecolor")
# log_path = f"./data/logs/{current_date}_.log" #Path.cwd()
# logger = get_logger(log_path)

# main_table = Table(
# 	expand=True,
# 	show_header=False,
# 	header_style="bold",
# 	title="[blue][b]Log Entries[/b]",
# 	highlight=True,
# )
# main_table.add_column("Log Output")

# my_progress_bar = Progress(
# 	# SpinnerColumn(),
# 	TextColumn("{task.description}"),
# 	BarColumn(),
# 	"time elapsed:",
# 	TimeElapsedColumn(),
# 	TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
# )
# sect_size = ecg_data['section_info'].shape[0]
# my_task = my_progress_bar.add_task("find_peaks", total=int(sect_size))

# progress_table = Table.grid(expand=True)
# progress_table.add_row(
# 	Panel(
# 		my_progress_bar,
# 		title="ECG extraction progress bar",
# 		border_style="green",
# 		padding=(1, 1),
# 	)
# )

# stats_table = get_stats()
# layout = make_layout()
# layout["header"].update(make_header())
# layout["progbar"].update(Panel(progress_table, border_style="green"))
# layout["termoutput"].update(Panel(main_table, border_style="blue"))
# layout["stats"].update(Panel(stats_table, border_style="magenta"))


#https://en.wikipedia.org/wiki/QRS_complex#/media/File:SinusRhythmLabels.svg
#https://imotions.com/blog/learning/best-practice/heart-rate-variability/


