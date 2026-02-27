import time
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import multiprocessing
from numba import cuda
from typing import List
from kneed import KneeLocator
from collections import Counter
from itertools import cycle, chain
from pathlib import Path, PurePath
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, TextBox
import matplotlib.animation as animation
from dataclasses import dataclass, field
from rich import print as pprint
from rich.tree import Tree
from rich.text import Text
from rich.table import Table
from rich.theme import Theme
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
)
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, stft, welch, convolve, butter, filtfilt, savgol_filter
from scipy.stats import entropy, wasserstein_distance, pearsonr, probplot, boxcox, yeojohnson, norm, linregress

########################### Custom imports ###############################
from utils import segment_ECG
from setup_globals import walk_directory
from support import logger, console, log_time, NumpyArrayEncoder

########################### Sklearn metric / scaling imports ###############################
from shap import TreeExplainer
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import accuracy_score as ACC_SC
from sklearn.metrics import log_loss as LOG_LOSS
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as RSQUARED
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, make_scorer
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.utils.class_weight import compute_sample_weight

########################### Sklearn model imports #########################
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, GroupShuffleSplit, LeaveOneGroupOut, ShuffleSplit, StratifiedShuffleSplit
# from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import xgboost as xgb

########################### For GPU acceleration on XGBoost #########################
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

#CLASS BP_Feat
@dataclass
class BP_Feat():
    id          :str   = None #record index
    onset       :int   = None #Left trough of Systolic peak
    sbp_id      :int   = None #Sytolic Index
    dbp_id      :int   = None #Diastolic Index
    notch_id    :int   = None #Dicrotic notch
    SBP         :float = None #Systolic peak val
    DBP         :float = None #Diastolic trough val
    notch       :float = None #Notch val
    true_MAP    :float = None #MAP via integral
    ap_MAP      :float = None #MAP via formula
    shock_gap   :float = None #Diff of trueMAP and apMAP
    dni         :float = None #dicrotic Notch Index
    sys_sl      :float = None #systolic slope
    dia_sl      :float = None #diastolic slope
    ri          :float = None #resistive index
    pul_wid     :float = None #pulse width 
    p1          :float = None #Percussion Wave (P1)
    p2          :float = None #Tidal Wave (P2)
    p3          :float = None #Dicrotic Wave (P3)
    p1_p2       :float = None #Ratio of P1 to P2
    p1_p3       :float = None #Ratio of P1 to P3,
    aix         :float = None #Augmentation Index (AIx)
    ph_mor      :float = None #Mean Phase Angle (Morlet)
    ph_cgau     :float = None #Mean Phase Angle (Cgau1)
    lad_mean    :float = None #Mean Flow
    lad_sys_pk  :float = None #Mean systolic peak
    lad_dia_pk  :float = None #Mean diastolic peak
    lad_ds_rat  :float = None #Diastolic to Systolic Peak Ratio
    lad_dia_auc :float = None #Diastolic Flow Volume (AUC)
    cvr         :float = None #Coronary Vascular Resistance (MAP / LAD_mean)
    dcr         :float = None #Diastolic Coronary Resistance (DBP / LAD_dia_mean)
    lad_pi      :float = None #LAD Pulsatility Index
    lad_acc_sl  :float = None #Diastolic Acceleration Slope
    flow_div    :float = None #Carotid to LAD Flow Ratio 

#CLASS PigRad
class PigRAD:
    def __init__(self, npz_path):
        # load data / params
        self.npz_path      :Path  = npz_path
        self.view_eda      :bool  = False
        self.view_pig      :bool  = False
        self.view_models   :bool  = True
        self.fs            :float = 1000    #Hz
        self.windowsize    :int   = 8       #size of section window 
        self.batch_run     :bool  = isinstance(npz_path, list)
        # Multiple file pathing
        if self.batch_run:
            # Grab the parent directory of the first file in the batch list
            root_dir = Path(npz_path[0]).parent
            self.fp_base = root_dir / "batch_processed_output"
            self.fp_save = self.fp_base / "combined_batch_features.npz"

        else:
            # Single file pathing
            root_dir = Path(npz_path).parent
            stem_name = Path(npz_path).stem
            self.fp_base = root_dir / stem_name
            self.fp_save = self.fp_base / f"{stem_name}_feat.npz"
        
        # Create the target directory immediately so it's ready for saving/logging
        self.fp_base.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output folder set to: {self.fp_base}")
        
        # Load data (handles single and batch automatically)
        self.loader = SignalDataLoader(npz_path)
        
        # Master lists to hold data across all pigs
        self.all_avg_data :List = [] 
        self.avg_data     :List = []
        self.bp_data      :List[BP_Feat] = []
        self.gpu_devices  :list = [device.id for device in cuda.list_devices()]

        #TODO - Will need logic to automatically select leads
        self.ecg_lead     :str = 2 #self.pick_lead("ECG")      #2 pick the ECG lead
        self.lad_lead     :str = 1 #self.pick_lead("LAD")      #1 pick the Lad lead
        self.car_lead     :str = 6 #self.pick_lead("Carotid")  #6 pick the Carotid lead
        self.ss1_lead     :str = 4 #self.pick_lead("SS1")      #4 pick the SS1 lead
        
        self.avg_dtypes = [
            ('pig_id'     , 'U50'), #ID for the pig
            ('start'      , 'i4'),  #start index
            ('end'        , 'i4'),  #end index
            ('invalid'    , 'i4'),  #Is it an invalid section?
            ('shock_class', 'U4'),  #Shock Class
            ('HR'         , 'f4'),  #Heart Rate
            ############# Morphomics ################################
            ('SBP'        , 'f4'),  #Systolic Pressure
            ('DBP'        , 'f4'),  #Diastolic Pressure
            ('EBV'        , 'f4'),  #Estimated Blood Volume
            ('true_MAP'   , 'f4'),  #Mean Arterial Pressure (AUC)
            ('ap_MAP'     , 'f4'),  #Approximate Mean Arterial pressure (Formula)
            ('shock_gap'  , 'f4'),  #Difference between true and approximate MAP
            ('dni'        , 'f4'),  #Dichrotic Notch Index
            ('sys_sl'     , 'f4'),  #Systolic slope
            ('dia_sl'     , 'f4'),  #Diastolic slope
            ('ri'         , 'f4'),  #Resistive index
            ('pul_wid'    , 'f4'),  #Pulse width 
            ('p1'         , 'f4'),  #Percussion Wave (P1)
            ('p2'         , 'f4'),  #Tidal Wave (P2)
            ('p3'         , 'f4'),  #Dicrotic Wave (P3)
            ('p1_p2'      , 'f4'),  #Ratio of P1 to P2
            ('p1_p3'      , 'f4'),  #Ratio of P1 to P3,
            ('aix'        , 'f4'),  #Augmentation Index (AIx)
            # ####### LAD Flow Features ######################
            ('lad_mean'   , 'f4'),  # Mean LAD Flow
            ('lad_dia_pk' , 'f4'),  # Diastolic Peak Flow
            ('lad_sys_pk' , 'f4'),  # Systolic Peak Flow
            ('lad_ds_rat' , 'f4'),  # Diastolic to Systolic Peak Ratio
            ('lad_dia_auc', 'f4'),  # Diastolic Flow Volume (AUC)
            ('cvr'        , 'f4'),  # Coronary Vascular Resistance (MAP / LAD_mean)
            ('dcr'        , 'f4'),  # Diastolic Coronary Resistance (DBP / LAD_dia_mean)
            ('lad_pi'     , 'f4'),  # LAD Pulsatility Index
            ('lad_acc_sl' , 'f4'),  # Diastolic Acceleration Slope
            ('flow_div'   , 'f4'),  # Carotid to LAD Flow Ratio            
            ######### Frequency componenets ##################
            ('f0'         , 'f4'),  #Top Frequency (Fundamental)
            ('f1'         , 'f4'),  #Harmonic 1 (2nd biggest peak)
            ('f2'         , 'f4'),  #Harmonic 2 (3rd biggest peak)
            ('f3'         , 'f4'),  #Harmonic 3 (4th biggest peak)
            ('psd0'       , 'f4'),  #Amplitude of Top Freq
            ('psd1'       , 'f4'),  #Amplitude of Harmonic 1
            ('psd2'       , 'f4'),  #Amplitude of Harmonic 2
            ('psd3'       , 'f4'),  #Amplitude of Harmonic 3
            ('var_mor'    , 'f4'),  #Phase Variance (Morlet)
            ('var_cgau'   , 'f4'),  #Phase Variance (Cgau1)
            # ####### Signal Quality & Shift #################
            ('sqi_power'  , 'f4'),  # Spectral Purity (In-Band Power Ratio)
            ('sqi_entropy', 'f4'),  # Spectral Entropy
            # ('w_dist'     , 'f4'),  # Wasserstein Distance from Baseline
        ]
        #IDEA - feature - Pulse transit time?  

    def _derivative(self, signal:np.array, deriv:int=0)->tuple:
        """Calculates smoothed 0, 1st, and 2nd derivative using scipy's Savitzky-Golay filter.

        Args:
            signal (np.array): waveform

        Returns:
            (np.array): returns smoothed, 1st or 2nd derivative
        """        
        # Window length must be odd; approx 20-30ms is usually good for smoothing derivatives
        window = int(0.03 * self.fs) 
        if window % 2 == 0: 
            window += 1
        match deriv:
            case 0:
                return savgol_filter(signal, window_length=window, polyorder=3)
            case 1:
                return savgol_filter(signal, window_length=window, polyorder=3, deriv=1)
            case 2:
                return savgol_filter(signal, window_length=window, polyorder=3, deriv=2)
    
    def _integrate(self, signal:np.array)->float:
        """Apply integration of signal. Calculates area under curve. 

        Args:
            signal (np.array): waveform 

        Returns:
            float: AUC of signal
        """        
        return np.trapezoid(signal, dx=1.0/self.fs).item()
    
    def _bandpass_filt(self, data:np.array, lowcut:float=0.1, highcut:float=40.0, fs=1000.0, order:int=4)->np.array:
        """Apply Band Pass Filter

        Args:
            data (np.array): Signal to filter
            lowcut (float, optional): lowcut frequency. Defaults to 0.1.
            highcut (float, optional): highcut frequency. Defaults to 40.0.
            fs (float, optional): sampling rate. Defaults to 1000.0.
            order (int, optional): Order of the filter. Defaults to 4.

        Returns:
            np.array: Filtered signal
        """            
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def _low_pass_filt(self, data:np.array, lowcut:float=5, fs=1000.0, order:int=4)->np.array:
        """Apply Low pass filter

        Args:
            data (np.array): Signal to filter
            lowcut (float, optional): lowcut frequency. Defaults to 5.
            fs (float, optional): sampling rate. Defaults to 1000.0.
            order (int, optional): Order of the filter. Defaults to 4.

        Returns:
            np.array: Filtered signal
        """        
        nyq = 0.5 * fs
        cutoff = lowcut / nyq
        b, a = butter(order, cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    def _high_pass_filt(self, data:np.array, highcut:float=20, fs=1000.0, order:int=4)->np.array:
        """Apply high pass filter

        Args:
            data (np.array): Signal to filter
            highcut (float, optional): highcut frequency. Defaults to 20.
            fs (float, optional): sampling rate. Defaults to 1000.0.
            order (int, optional): Order of the filter. Defaults to 4.

        Returns:
            np.array: Filtered signal
        """        
        nyq = 0.5 * fs
        cutoff = highcut / nyq
        b, a = butter(order, cutoff, btype='high', analog=False)
        return filtfilt(b, a, data)
    
    def _yabig_meanie(self, values: list, precision: int = 4) -> float:
        """Safely calculates the mean, ignoring Nones/NaNs. Returns np.nan if empty."""
        clean_vals = [v for v in values if v is not None and not np.isnan(v)]
        if not clean_vals:
            return np.nan
        return np.round(np.nanmean(clean_vals), precision).item()
    
    def _find_sbp_info(self, ss1wave:np.array, sys_peak_idx:int) -> tuple:
        """
        Finds the true physiological onset (foot) of a pressure wave by analyzing 
        the first derivative (dp/dt) before the systolic peak.  Look for the greatest rate
        of change.
        """
        # Define a strict lookback window (e.g., 300ms max before the peak)
        # This prevents the search from bleeding into the previous beat's dicrotic notch
        lookback_samples = int(self.fs * 0.3)
        start_idx = max(0, sys_peak_idx - lookback_samples)
        
        # Extract the segment just before the peak
        segment = ss1wave[start_idx:sys_peak_idx]
        
        # Calculate the first derivative (dp/dt) of this segment
        if segment.size > 30: #was 2 for gradient
            # dpdt = np.gradient(segment)
            #update to smooth the signal
            dpdt = self._derivative(segment, 1)

        else:
            return None
        # Find the steepest part of the upstroke (maximum positive slope)
        max_slope_local_idx = np.argmax(dpdt)
        max_slope_val = dpdt[max_slope_local_idx]

        # Walk backwards from the steepest point until the slope flattens out.
        # Define "flat" as dropping below 5% of the maximum upstroke slope.
        flat_threshold = max_slope_val * 0.05
        
        onset_local_idx = max_slope_local_idx
        while onset_local_idx > 0 and dpdt[onset_local_idx] > flat_threshold:
            onset_local_idx -= 1
            
        # 5. Convert back to global coordinates
        true_onset_idx = start_idx + onset_local_idx
        
        return true_onset_idx

    def _process_single_beat(self, id:int, idx:int, peak:int, ss1wave:np.ndarray, carwave:np.ndarray, ladwave:np.ndarray, s_heights:dict, s_peaks:np.ndarray) -> BP_Feat:
        """Extracts features for a single beat.

        Args:
            id (int): _description_
            idx (int): _description_
            peak (int): _description_
            ss1wave (np.ndarray): _description_
            carwave (np.ndarray): _description_
            ladwave (np.ndarray): _description_
            s_heights (dict): _description_
            s_peaks (np.ndarray): _description_

        Returns:
            BP_Feat(dataclass): Beat object with generated features
        """
        bpf = BP_Feat()
        # ==========================================
        # --- Pressure Features ---
        # ==========================================
        # Safely assign indices. Encode section_peak as dual index
        bpf.id = str(idx) + "_" + str(id)                  
        bpf.sbp_id = peak.item()
        #BUG - Find peaks bases
            # Find peaks was failing in the later stages of hem shock when the
            # dicrotic notch gets below the diastolic valley Being that the
            # actual peak is finding correctly, 
            # We can just use the onset for the diastolic of the the next peak :tada:
            # Old Code for using find_peaks heights
            # bpf.dbp_id = s_heights["right_bases"][id].item()
            # bpf.onset = s_heights["left_bases"][id].item() if id == 0 else s_heights["right_bases"][id - 1].item()

        next_peak = s_peaks[id + 1].item()
        bpf.dbp_id = self._find_sbp_info(ss1wave, next_peak)
        bpf.onset = self._find_sbp_info(ss1wave, bpf.sbp_id)

        # Validate indices to prevent reversed slicing (sbp > dbp)
        if bpf.onset != None:
            if bpf.onset >= bpf.sbp_id or bpf.sbp_id >= bpf.dbp_id:
                return None
        else:
            return None
        
        # Pressures & Pulse Width
        bpf.SBP = ss1wave[bpf.sbp_id].item()
        bpf.DBP = ss1wave[bpf.dbp_id].item()
        bpf.pul_wid = (bpf.dbp_id - bpf.onset) / self.fs
        #TODO - Update pulse width to dicrotic notch height

        # Systolic Slope
        sys_run = (bpf.sbp_id - bpf.onset) / self.fs
        bpf.sys_sl = ((bpf.SBP - ss1wave[bpf.onset]) / sys_run).item() if sys_run > 0 else None
        #TODO - Ask Mark if we should upgrade this to DP/dt 
            #Could get that return from _find_sbp_info

        # Slices
        sub_notch = ss1wave[bpf.sbp_id:bpf.dbp_id]
        sub_full = ss1wave[bpf.onset:bpf.dbp_id]
        
        # MAP
        if sub_full.size > 0:
            bpf.true_MAP = self._integrate(sub_full) / (sub_full.size / self.fs)
            bpf.ap_MAP = bpf.DBP + (1/3) * (bpf.SBP - bpf.DBP)
            bpf.shock_gap = bpf.true_MAP - bpf.ap_MAP
        
        # Dicrotic Notch & DNI
        # Enforce an 50ms refractory period after the peak
        refractory_samples = int(self.fs * 0.05) 
        
        # Mask out the diastolic foot (highly concave-up valley at the end)
        # mask the last 25% of the descending limb, or a minimum of 50ms
        tail_samples = max(int(self.fs * 0.05), int(sub_notch.size * 0.25))
        
        # Ensure the slice is long enough to handle both masks safely
        if sub_notch.size > (refractory_samples + tail_samples + 3): 
            try:
                # Calculate 2nd deriv over the whole slice to avoid filter edge effects
                d2_notch = self._derivative(sub_notch, 2)
                
                # Apply the masks by setting the out-of-bounds 2nd derivative to negative infinity
                d2_notch[:refractory_samples] = -np.inf
                d2_notch[-tail_samples:] = -np.inf
                
                # Now the argmax is trapped in the middle of the slope where the notch lives
                bpf.notch_id = np.argmax(d2_notch).item()
                bpf.notch = sub_notch[bpf.notch_id].item()
                
                if bpf.notch and (bpf.SBP - bpf.DBP) > 0.1:
                    bpf.dni = (bpf.SBP - bpf.notch) / (bpf.SBP - bpf.DBP)

            except Exception as e:
                logger.warning(f"DNI/notch calculation failed: {e}")

        # Resistive Index (Carotid)
        if id + 1 < len(s_peaks): # Ensure we don't go out of bounds
            sub_car = carwave[bpf.onset:s_peaks[id+1]]
            if sub_car.size > 0:
                psv, edv = np.max(sub_car), sub_car[-1]
                bpf.ri = self.calc_RI(psv, edv)

        # Diastolic Slope
        if getattr(bpf, 'notch', None):
            notch_abs = bpf.sbp_id + bpf.notch_id
            y_dia = ss1wave[notch_abs:bpf.dbp_id]
            # Linregress requires at least 2 points
            if y_dia.size > 2: 
                try:
                    x_dia = np.arange(y_dia.size) / self.fs
                    bpf.dia_sl = linregress(y_dia, x_dia).slope.item()
                except Exception as e:
                    logger.warning(f"Diastolic Slope calculation failed: {e}")
                    bpf.dia_sl = None 
                    return bpf
        else:
            #If it can't find the notch, it can't do the rest of the calculations
            return bpf        
        # ==========================================
        # --- SS1 Features: P1, P3, P3, AIX ---
        # ==========================================
        p1_val, p2_val, p3_val = None, None, None

        # Absolute index of the notch
        # Define the entire systolic complex (onset to notch)
        sub_syst = ss1wave[bpf.onset:notch_abs]

        # Find true peaks in the systolic complex (Type A vs Type C)
        sys_peaks, _ = find_peaks(sub_syst)

        if len(sys_peaks) >= 2:
            # Multiple peaks: 
            # First is P1, Highest subsequent is P2
            p1_idx = sys_peaks[0]
            p2_idx = sys_peaks[1:][np.argmax(sub_syst[sys_peaks[1:]])]
            p1_val = sub_syst[p1_idx].item()
            p2_val = sub_syst[p2_idx].item()
        else:
            try:
                # 1st derivative of complex to find shoulders
                d1_sys_comp = self._derivative(sub_syst, 1)
                    
                # Only one true peak found. We must find the shoulder.
                main_peak_idx = np.argmax(sub_syst).item()
                # Check for early shoulder (P1 is shoulder, P2 is main peak)
                early_shoulders, _ = find_peaks(-d1_sys_comp[:main_peak_idx])
                # Check for late shoulder (P1 is main peak, P2 is shoulder)
                late_shoulders, _ = find_peaks(d1_sys_comp[main_peak_idx:])

                if len(early_shoulders) > 0:
                    p1_idx = early_shoulders[-1].item() 
                    p1_val = sub_syst[p1_idx].item()
                    p2_val = sub_syst[main_peak_idx].item()
                
                elif len(late_shoulders) > 0:
                    p1_val = sub_syst[main_peak_idx].item()
                    p2_idx = main_peak_idx + late_shoulders[0].item()
                    p2_val = sub_syst[p2_idx].item()
                else:
                    p1_val = sub_syst[main_peak_idx].item()
                    p2_val = p1_val

            except Exception as e:
                logger.info(f"shape of sub_syst = {sub_syst.shape}")
                logger.warning(f"{e}")

        # Find P3 (Dicrotic Wave) using Kneed
        diastolic_run = ss1wave[notch_abs:bpf.dbp_id]
        if len(diastolic_run) > 5:  
            # check if P3 is an obvious, true peak (common in deep shock)
            dias_peaks, _ = find_peaks(diastolic_run)
            
            if len(dias_peaks) > 0:
                # Grab the highest prominent peak in the diastolic run
                p3_idx = dias_peaks[np.argmax(diastolic_run[dias_peaks])]
                p3_val = diastolic_run[p3_idx].item()
                
            else:
                # Fallback: No true peak found. Look for a subtle bulge (knee).
                x_dias = np.arange(len(diastolic_run))
                # Dynamically determine the slope direction
                if diastolic_run[0] > diastolic_run[-1]:
                    run_direction = "decreasing"
                else:
                    run_direction = "increasing"
                
                try:
                    # Sensitivity parameter (default is 1.0)
                    kneedle = KneeLocator(
                        x=x_dias, 
                        y=diastolic_run, 
                        curve="concave", 
                        direction=run_direction,
                        S=1.0                  
                    )
                    if kneedle.knee is not None:
                        p3_val = diastolic_run[kneedle.knee].item()
                    else:
                        # Absolute fallback if kneed finds nothing
                        p3_val = np.max(diastolic_run).item()

                except Exception as e:
                    logger.warning(f"Kneed failed on P3 calc: {e}")
                    p3_val = np.max(diastolic_run).item()

        elif len(diastolic_run) > 0:
            p3_val = np.max(diastolic_run).item()
        # if its flat flat.  p3 is just the notch val
        else:
            p3_val = bpf.notch

        # Assign to dataclass & calculate ratios
        bpf.p1 = p1_val
        bpf.p2 = p2_val
        bpf.p3 = p3_val

        if p1_val is not None and p2_val is not None:
            # Avoid division by zero for P1/P2 ratio
            if p2_val != 0:
                bpf.p1_p2 = p1_val / p2_val
            
            # Avoid division by zero for Augmentation Index
            pulse_pressure_p1 = p1_val - bpf.DBP
            # Require at least a tiny pulse pressure
            if pulse_pressure_p1 > 0.1:  
                bpf.aix = (p2_val - bpf.DBP) / pulse_pressure_p1
            else:
                bpf.aix = None
        
        if p1_val and p3_val:
            bpf.p1_p3 = p1_val / p3_val

        # ==========================================
        # --- LAD Flow & Resistance Features ---
        # ==========================================
        sub_lad = ladwave[bpf.onset:bpf.dbp_id]
        sub_car_full = carwave[bpf.onset:bpf.dbp_id]
        
        # Define a physiological epsilon (e.g., 0.5 mL/min) to prevent exploding denominators
        eps = 0.5
        
        if sub_lad.size > 0:
            bpf.lad_mean = np.mean(sub_lad).item()
            
            # Flow Division Ratio & CVR & PI (Require lad_mean to clear the epsilon threshold)
            if abs(bpf.lad_mean) > eps:
                if sub_car_full.size > 0:
                    bpf.flow_div = np.mean(sub_car_full).item() / bpf.lad_mean
                
                if getattr(bpf, 'true_MAP', None):
                    bpf.cvr = bpf.true_MAP / bpf.lad_mean
                else:
                    bpf.cvr = None

                lad_max, lad_min = np.max(sub_lad).item(), np.min(sub_lad).item()
                bpf.lad_pi = (lad_max - lad_min) / bpf.lad_mean

            # Diastolic Phase Metrics
            if getattr(bpf, 'notch', None):
                notch_abs = bpf.sbp_id + bpf.notch_id
                lad_systole = ladwave[bpf.onset:notch_abs]
                lad_diastole = ladwave[notch_abs:bpf.dbp_id]
                
                if lad_systole.size > 0 and lad_diastole.size > 0:
                    bpf.lad_sys_pk = np.max(lad_systole).item()
                    bpf.lad_dia_pk = np.max(lad_diastole).item()
                    
                    # Diastolic/Systolic Ratio
                    if abs(bpf.lad_sys_pk) > eps:
                        bpf.lad_ds_rat = bpf.lad_dia_pk / bpf.lad_sys_pk
                     
                    # Diastolic Volume (AUC)
                    bpf.lad_dia_auc = self._integrate(lad_diastole)
                    
                    # Diastolic Coronary Resistance
                    lad_dia_mean = np.mean(lad_diastole).item()
                    if getattr(bpf, 'DBP', None) and abs(lad_dia_mean) > eps:
                        bpf.dcr = bpf.DBP / lad_dia_mean

                    # Diastolic Acceleration Slope 
                    # Enforce a minimum distance (e.g., 20ms) to prevent micro-run explosions
                    dia_pk_idx_rel = np.argmax(lad_diastole)
                    min_run_samples = int(self.fs * 0.02)
                    
                    if dia_pk_idx_rel > min_run_samples:
                        dia_run = dia_pk_idx_rel / self.fs
                        bpf.lad_acc_sl = ((bpf.lad_dia_pk - lad_diastole[0]) / dia_run).item()

        return bpf
 
    def calc_RI(self, psv:float, edv:float) -> float:
        """
        Calculates the Resistive Index (RI) from Peak Systolic Velocity (PSV) 
        and End-Diastolic Velocity (EDV).
        
        Args:
            psv (float): Peak systolic velocity.
            edv (float): End-diastolic velocity.
            
        Returns:
            float: The Resistive Index (RI).
        """
        if psv == 0:
            return None  # Avoid division by zero
        ri = (psv - edv) / psv
        return ri.item()
    
    def section_extract(self):
        """This is the main section for signal processing and feature creation. Updates the self.avg_data object
        """        
        # Progress bar for section iteration
        with Progress(
            TimeElapsedColumn(),
            SpinnerColumn(), 
            BarColumn(), 
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            freq_tool = CardiacFreqTools(fs=self.fs)
            jobtask = progress.add_task("Processing Pigs...", total=len(self.loader.records))
            menwithouthats = cycle([
                "Calculating features",
                "We can dance if we want to",
                "We can leave your friends behind",
                "'Cause your friends don't dance and if they don't dance",
                "Well they're, no friends of mine",
                "Say, we can go where we want to",
                "A place where they will never find",
                "And we can act like we come from out of this world",
                "Leave the real one far behind",
                "And we can dance, dance",
                "We can go when we want to",
                "Night is young and so am I",
                "And we can dress real neat from our hats to our feet",
                "And surprise them with a victory cry",
                "Say, we can act if we want to",
                "If we don't nobody will",
                "And you can act real rude and totally removed",
                "And I can act like an imbecile",
                "And say, We can dance, we can dance",
                "Everything's out of control",
                "We can dance, we can dance",
                "We're doing it from pole to pole",
                "We can dance, we can dance",
                "Everybody look at your hands",
                "We can dance, we can dance",
                "Everybody's taking the chance",
                "It's a safety dance",
                "Oh well it's safe to dance",
                "Yes it safe to dance",
                "We can dance if we want to",
                "We've got all your life and mine",
                "As long as we abuse it, never going to lose it",
                "Everything will work out right",
                "I say, we can dance if we want to",
                "We can leave your friends behind",
                "'Cause your friends don't dance, and if they don't dance",
                "Well they're no friends of mine",
                "I say, we can dance, we can dance",
                "Everything's out of control",
                "We can dance, we can dance",
                "We're doing it from pole to pole",
                "We can dance, we can dance",
                "Everybody look at your hands",
                "We can dance, we can dance",
                "Everybody's taking the chance",
                "Well it's safe to dance",
                "Yes it's safe to dance",
                "Well it's safe to dance",
                "Well it's safe to dance",
                "Yes it's safe to dance",
                "Well it's safe to dance",
                "Well it's safe to dance",
                "It's a safety dance",
                "Well it's a safety dance",
                "Oh it's a safety dance",
                "Oh it's a safety dance",
                "Well it's a safety dance"],
            )
            # Iterate every pig 
            for pig_id, record in self.loader.records.items():
                update_t = f"Processing Pig ID: {pig_id}"
                progress.update(task_id=jobtask, description=update_t)
                logger.info(update_t)
                full_data = record["data"]
                channels = record["channels"]
                logger.info(f"channels {channels}")
                #Section signals
                sections     :np.array = segment_ECG(full_data[channels[self.ecg_lead]], self.fs, self.windowsize)
                pig_avg_data :np.array = np.zeros(sections.shape[0], dtype=self.avg_dtypes)
                logger.info(f"sections shape: {sections.shape}")
                #Input section data into avg_data container
                pig_avg_data["pig_id"] = pig_id
                pig_avg_data["start"]  = sections[:, 0]
                pig_avg_data["end"]    = sections[:, 1]
                pig_avg_data["invalid"]  = sections[:, 2] #0=no, 1=yes
                del sections

                #Calc estimated blood volume for section. 
                if "EBV" in channels:
                    ebv_arr = full_data["EBV"].to_numpy()
                    #Normalize it
                    ebv_arr = ebv_arr / np.max(ebv_arr)
                    target_levels = {
                        "BL":(1.00),
                        "C1":(0.85, 1.00),   
                        "C2":(0.70, 0.85),  
                        "C3":(0.60, 0.70),   
                        "C4":(0.00, 0.60),   
                    }
                levels = list(target_levels.keys())
                conditions = [
                    ebv_arr >= 1.00,                       #Baseline - No blood loss
                    (ebv_arr >= 0.85) & (ebv_arr < 1.00),  #C1 - 0.85 <= ebv <= 1.0
                    (ebv_arr >= 0.70) & (ebv_arr < 0.85),  #C2 - 0.7 <= ebv <= 0.85
                    (ebv_arr >= 0.60) & (ebv_arr < 0.70),  #C3 - 0.6 <= ebv <= 0.7
                    ebv_arr < 0.60                         #C4 - 0.0 = ebv <= 0.6
                ]

                target = np.select(conditions, levels, default="UNKNOWN")
                logger.info(f"target counts\n {np.unique(target, return_counts=True)}")

                #bandpass the ecg, lad, and carotid signals
                for lead in [self.lad_lead, self.car_lead, self.ecg_lead]:
                    full_data[channels[lead]] = self._bandpass_filt(data=full_data[channels[lead]])
                    logger.info(f"{channels[lead]}")

                jobtask_proc = progress.add_task(f"{next(menwithouthats)}", total=pig_avg_data.shape[0])
                tot_beats:list[BP_Feat] = []
                
                for idx, section in enumerate(pig_avg_data):
                    start = section["start"].item()
                    end = section["end"].item()

                    #Find R peaks from ECG lead
                    # ecgwave = self.full_data[self.channels[self.ecg_lead]][start:end]
                    # e_peaks, e_heights = find_peaks(
                    #     x = ecgwave,
                    #     prominence = np.percentile(ecgwave, 95),  #99 -> stock
                    #     height = np.percentile(ecgwave, 90),      #95 -> stock
                    #     distance = round(self.fs*(0.200))           
                    # )

                    #Debug plot
                    # plt.plot(range(ecgwave.shape[0]), ecgwave.to_numpy())
                    # plt.scatter(e_peaks, e_heights["peak_heights"], color="red")

                    #Select section streams
                    ss1wave = full_data[channels[self.ss1_lead]][start:end].to_numpy()
                    ladwave = full_data[channels[self.lad_lead]][start:end]
                    carwave = full_data[channels[self.car_lead]][start:end]

                    #New feature
                    #TODO - Develop a feature routine that can evaluate if we have a biological signal. 
                        #IDEA - Welch's STFT for signal majority in the 0.5Hz to 15Hz band
                        #IDEA - Shannon Entropy
                        #IDEA - Wasserstein Distribution shift. 
                            #Not sure this will work. 
                    
                    # ---Signal Quality & Baseline Check ---
                    power_ratio, spec_entropy, _, _ = freq_tool.evaluate_signal(ss1wave)
                    # power_ratio, spec_entropy, w_dist, current_psd_norm = freq_tool.evaluate_signal(
                    #     ss1wave, baseline_psd_norm=baseline_psd_norm
                    # )
                    # Log the metrics to the dataset
                    pig_avg_data["sqi_power"][idx] = power_ratio
                    pig_avg_data["sqi_entropy"][idx] = spec_entropy
                    # pig_avg_data["w_dist"][idx] = w_dist if w_dist is not None else 0.0

                    # Threshold the signal to a particular power range and or spectral entropy
                    if power_ratio < 0.95 or spec_entropy > 0.50: #was 45 - right on the edge
                        logger.warning(f"sect {idx} rejected as noise. Power Ratio: {power_ratio:.2f}, Entropy: {spec_entropy:.2f}")
                        pig_avg_data["invalid"][idx] = 1
                        continue

                    # If this is our first valid physiological section, lock it in as the baseline
                    # if baseline_psd_norm is None:
                    #     baseline_psd_norm = current_psd_norm
                    #     pig_avg_data["w_dist"][idx] = 0.0 # Distance to itself is zero
                    #     logger.info(f"Baseline PSD locked at section {idx}")

                    prom_night = (np.percentile(ss1wave, 95) - np.percentile(ss1wave, 5)).item()
                    # logger.debug(f"prom: {prom_night*30:.2f}")
                    # first find systolic peaks
                    s_peaks, s_heights = find_peaks(
                        x = ss1wave,
                        prominence = prom_night * 0.30,      
                        height = np.percentile(ss1wave, 35), #Dropped from 50 to 35%
                        distance = int(self.fs*(0.20)),      #Upped from 10 to 20 (200bpm)
                        wlen = int(self.fs*3)
                    )
                    #BUG - left base
                        # the left bases aren't getting calculated correctly.  I tried to adjust wlen as a 
                        # width parameter, buuuut it didn't quite work
                    #BUG - Class 4 amplitude
                        #Also noticing the S peaks don't get identifed in later stages.  The dicrotic notch gets mislabeled
                        #but oddly its still able to find the onset correctly.  I guess the walkback is long enough to 
                        #still capture the preceeding S peak.  
                        #solution: lowering params as stated above

                    #Debug plot
                    # plt.plot(range(ss1wave.shape[0]), ss1wave.to_numpy())
                    # plt.scatter(s_peaks, s_heights["peak_heights"], color="red")
                    # plt.show()

                    #NOTE - Changing HR extraction to SS1. ECG is too unreliable.
                    if s_peaks.size < 3 or s_peaks.size > 60:
                        logger.info(f"sect {idx} peaks invalid for extract")
                        continue
                    else:
                        # ---  Phase Metric Extraction ---
                        # Choose a target freq (e.g., 2Hz matches typical cardiac rhythm)
                        target_f = 2.0 
                        
                        beat_phases_mor, var_mor = freq_tool.compute_section_phase_metric(
                            ss1wave, 
                            s_peaks[:-1], 
                            target_freq=target_f, 
                            wavelet='morlet'
                        )
                        beat_phases_cgau, var_cgau = freq_tool.compute_section_phase_metric(
                            ss1wave, 
                            s_peaks[:-1], 
                            target_freq=target_f, 
                            wavelet='cgau1'
                        )
                        
                        # Store section-level phase variance
                        pig_avg_data["var_mor"][idx] = var_mor
                        pig_avg_data["var_cgau"][idx] = var_cgau

                        # ---  HR Calculation ---
                        RR_diffs = np.diff(s_peaks)
                        HR = np.round((60 / (RR_diffs / self.fs)), 2)
                        if HR.size > 0:
                            pig_avg_data["HR"][idx] = int(np.nanmean(HR))
                        else:
                            logger.warning(f"Empty slice encountered in sect {idx}")
                    
                    ind_beats:list[BP_Feat] = []

                    #Initialize the baseline PSD for each pig
                    # baseline_psd_norm = None

                    for id, (peak, ph_m, ph_c) in enumerate(zip(s_peaks[:-1], beat_phases_mor, beat_phases_cgau)):
                        beat_feat = self._process_single_beat(id, idx, peak, ss1wave, carwave, ladwave, s_heights, s_peaks)
                        if beat_feat is not None:
                            # Assign the individual beat phases here
                            beat_feat.ph_mor = ph_m
                            beat_feat.ph_cgau = ph_c
                            
                            #Shift the indexes from the start.
                            beat_feat.onset += start
                            beat_feat.sbp_id += start
                            beat_feat.dbp_id += start
                            ind_beats.append(beat_feat)

                    if not ind_beats:
                        logger.warning(f"Sect {idx} produced no valid beats after processing.")
                        # Skip averaging to prevent the np.nanmean empty slice error
                        continue
                    
                    #Add beats to the total beats container
                    tot_beats.extend(ind_beats)    

                    # --- Morphomics / Pressure calculations ---
                    pig_avg_data["EBV"][idx]         = self._yabig_meanie(full_data["EBV"][start:end])
                    pig_avg_data["shock_class"][idx] = Counter(target[start:end]).most_common()[0][0].item()
                    pig_avg_data["dni"][idx]         = self._yabig_meanie([r.dni for r in ind_beats])
                    pig_avg_data["SBP"][idx]         = self._yabig_meanie([r.SBP for r in ind_beats])
                    pig_avg_data["DBP"][idx]         = self._yabig_meanie([r.DBP for r in ind_beats])
                    pig_avg_data["true_MAP"][idx]    = self._yabig_meanie([r.true_MAP for r in ind_beats])
                    pig_avg_data["ap_MAP"][idx]      = self._yabig_meanie([r.ap_MAP for r in ind_beats])
                    pig_avg_data["shock_gap"][idx]   = self._yabig_meanie([r.shock_gap for r in ind_beats])
                    pig_avg_data["sys_sl"][idx]      = self._yabig_meanie([r.sys_sl for r in ind_beats])
                    pig_avg_data["dia_sl"][idx]      = self._yabig_meanie([r.dia_sl for r in ind_beats])
                    pig_avg_data["ri"][idx]          = self._yabig_meanie([r.ri for r in ind_beats])
                    pig_avg_data["pul_wid"][idx]     = self._yabig_meanie([r.pul_wid for r in ind_beats])
                    pig_avg_data["p1"][idx]          = self._yabig_meanie([r.p1 for r in ind_beats])
                    pig_avg_data["p2"][idx]          = self._yabig_meanie([r.p2 for r in ind_beats])
                    pig_avg_data["p3"][idx]          = self._yabig_meanie([r.p3 for r in ind_beats])
                    pig_avg_data["p1_p2"][idx]       = self._yabig_meanie([r.p1_p2 for r in ind_beats])
                    pig_avg_data["p1_p3"][idx]       = self._yabig_meanie([r.p1_p3 for r in ind_beats])
                    pig_avg_data["aix"][idx]         = self._yabig_meanie([r.aix for r in ind_beats])
                    # --- LAD Flow calculations ---
                    pig_avg_data["lad_mean"][idx]    = self._yabig_meanie([r.lad_mean for r in ind_beats])
                    pig_avg_data["lad_dia_pk"][idx]  = self._yabig_meanie([r.lad_dia_pk for r in ind_beats])
                    pig_avg_data["lad_sys_pk"][idx]  = self._yabig_meanie([r.lad_sys_pk for r in ind_beats])
                    pig_avg_data["lad_ds_rat"][idx]  = self._yabig_meanie([r.lad_ds_rat for r in ind_beats])
                    pig_avg_data["lad_dia_auc"][idx] = self._yabig_meanie([r.lad_dia_auc for r in ind_beats])
                    pig_avg_data["cvr"][idx]         = self._yabig_meanie([r.cvr for r in ind_beats])
                    pig_avg_data["dcr"][idx]         = self._yabig_meanie([r.dcr for r in ind_beats])
                    pig_avg_data["lad_pi"][idx]      = self._yabig_meanie([r.lad_pi for r in ind_beats])
                    pig_avg_data["lad_acc_sl"][idx]  = self._yabig_meanie([r.lad_acc_sl for r in ind_beats])
                    pig_avg_data["flow_div"][idx]    = self._yabig_meanie([r.flow_div for r in ind_beats])
                    # --- STFT calculations ---
                        #ref link.   Is that our Johnny Morrison? It is! 
                        #https://shimingyoung.github.io/papers/morphomics_2019.pdf?hl=en-US
                    top_f, top_psd                   = freq_tool.STFT_extract(ss1wave, s_peaks)
                    pig_avg_data["f0"][idx]          = top_f[0]
                    pig_avg_data["f1"][idx]          = top_f[1]
                    pig_avg_data["f2"][idx]          = top_f[2]
                    pig_avg_data["f3"][idx]          = top_f[3]
                    pig_avg_data["psd0"][idx]        = top_psd[0]
                    pig_avg_data["psd1"][idx]        = top_psd[1]
                    pig_avg_data["psd2"][idx]        = top_psd[2]
                    pig_avg_data["psd3"][idx]        = top_psd[3]                        

                    #Move the progbar
                    progress.advance(jobtask_proc)

                #View the data
                if self.view_pig:
                    viewer = SignalGUI(
                        ss1_data = full_data[channels[self.ss1_lead]].to_numpy(), 
                        lad_data = full_data[channels[self.lad_lead]], 
                        beats = tot_beats, 
                        sampling_rate = self.fs,
                        window_size = int(self.fs * 5),  # Show 5 seconds of data at a time
                        pig_id = pig_id
                    )
                # Store the completed pig array in our master list
                self.all_avg_data.append(pig_avg_data)
                #Update the progbars
                progress.remove_task(jobtask_proc)
                progress.advance(jobtask)

        # After all pigs are processed, concatenate into a single master array
        self.avg_data = np.concatenate(self.all_avg_data)
        end_text = f"[bold green]File processing complete. Total sections: {self.avg_data.shape[0]}[/]"
        logger.info(end_text)
        console.print(end_text)    

    def create_features(self):
        self.section_extract()
        console.print("[bold green]Features created...[/]")
        # #Save Results
        # self.save_results()
        # console.print("[bold green]Features saved[/]")

    def run_pipeline(self):
        """Checks for existing save files. If found, loads them to save computation time.
        If not found, runs the feature creation and modeling pipeline
        """
        # 1. Check if files exist
        if self.fp_save.exists():
            console.print(f"[green]Found saved files for {self.lead}. Loading...[/]")
            
            # Load NPZ
            container = np.load(self.fp_save)
            cac = container['arr_0']
            console.print("[bold green]Data loaded. Launching Viewer...[/]")
            
            if self.view_pig:
                pass
                # Launch Viewer
                #TODO - Regime Viewer update
                    #Will need to update this. 

                # RegimeViewer(
                #     signal_data=self.full_data[self.lead].astype(np.float64),
                #     cac_data=cac,
                #     regime_locs=regime_locs,
                #     m=m,
                #     sampling_rate=self.fs,
                #     lead=self.lead
                # )
        else:
            console.print("[yellow]No saved data found. Running pipeline...[/]")
            console.print("[green]creating features...[/]")
            self.create_features()
            #Load up the EDA class
            eda = EDA(
                self.avg_data,
                self.avg_data.dtype.names,                
                self.fs, 
                self.gpu_devices, 
                self.fp_base,
                self.view_eda,
                self.view_models,
            )
            eda.clean_data()
            console.print("[green]prepping EDA...[/]")
            sel_cols = eda.feature_names[4:]
            #Sum basic stats
            eda.sum_stats(sel_cols, "Numeric Features")
            #graph features
            if eda.view_eda:
                #Plot eda charts
                for feature in sel_cols:
                    # eda.eda_plot("scatter", "EBV", feature)
                    eda.eda_plot("histogram", feature, None, eda.target)
                    eda.eda_plot("jointplot", "EBV", feature, eda.target)
                #Plot the heatmap
                eda.corr_heatmap(sel_cols=sel_cols)
                exit()

            console.print("[green]engineering features...[/]")
            engin = FeatureEngineering(eda)
            #select modeling columns of interest
            colsofinterest = [engin.data.columns[x] for x in range(4, engin.data.shape[1])]
            
            #Engineer your features here. available transforms below
            #log:  Log Transform
            #recip:Reciprocal
            #sqrt: square root
            #exp:  exponential - Good for right skew #!Broken
            #BoxC: Box Cox - Good for only pos val
            #YeoJ: Yeo-Johnson - Good for pos and neg val
            # Ex:
            # for feature in  ["psd0", "psd1", "psd2", "psd3"]:
            #     for transform in ["log", "recip", "sqrt", "BoxC", "YeoJ"]:
            #         engin.engineer(feature, False, True, transform)
            # engin.engineer("aix", True, False, "BoxC")
            # engin.engineer("psd0", True, False, "BoxC")
            # engin.engineer("psd1", True, False, "BoxC")
            # engin.engineer("psd2", True, False, "BoxC")
            # engin.engineer("psd3", True, False, "BoxC")
            #TODO - Normalization
                # Try normalizing some of the higher value metrics.  Being that we're using 
                # LOSO cross validation we might have wildly different heart rates
                # from pig to pig.  Which would cause the model to pay less attention
                # if it has var more variation. 
            norm_features = [
                "HR", "SBP", "DBP", "true_MAP", "lad_mean",
                "cvr", "sys_sl", "lad_acc_sl", "p1", "p2", "p3", 
                "f0", "f1", "f2", "f3", "lad_dia_pk", "lad_sys_pk"
            ]
            engin.normalize_subjects(norm_features)

            #reassign interest cols after transform
            colsofinterest = [engin.data.columns[x] for x in range(4, engin.data.shape[1])]
            
            #Remove unwanted features
            removecols = [
                "aix", "lad_mean", "cvr", 
                "flow_div", "lad_pi", #"var_mor", "var_cgau", 
                #"f0", "f1", "f2", "f3",
                "ap_MAP", "shock_gap"
            ]
            for col in removecols:
                if col in colsofinterest:
                    colsofinterest.pop(colsofinterest.index(col))

            #Scale your variables to the same scale.  Necessary for most machine learning applications. 
            #available sklearn scalers
            #s_scale : StandardScaler
            #m_scale : MinMaxScaler
            #r_scale : RobustScaler
            #q_scale : QuantileTransformer
            #p_scale : PowerTransformer
            scaler = "p_scale"

            #Next choose your cross validation scheme. Input `None` for no cross validation
            #kfold           : KFold Validation
            #stratkfold      : StratifiedKFold
            #groupkfold      : GroupKfold
            #groupshuffle    : GroupShuffleSplit
            #leaveonegroupout: LeaveOneGroupOut
            #leavepout       : Leave p out 
            #leaveoneout     : Leave one out
            #shuffle         : ShuffleSplit
            #stratshuffle    : StratifiedShuffleSplit
            cross_val = "leaveonegroupout"

            #Classifiers
            #'svm':LinearSVC
            #'rfc':RandomForestClassifier
            #'xgboost':XGBoostClassfier
            #'kneigh': KNeighborsClassifier
            console.print("[green]prepping data for training...[/]")
            dp = DataPrep(colsofinterest, scaler, cross_val, engin)
            modellist = ['svm', 'rfc', 'xgboost', 'kneigh']
            
            #split the training data #splits: test 25%, train 75% 
            [dp.data_prep(model, 0.25) for model in modellist]
            
            #Load the ModelTraining Class
            console.print("[green]training models...[/]")
            modeltraining = ModelTraining(dp)
            for model in modellist:
                console.print(f"training [green]{model}...[/]")
                modeltraining.get_data(model)
                modeltraining.fit(model)
                modeltraining.predict(model)
                modeltraining.validate(model)
                time.sleep(1)
            
            modeltraining.show_results(modellist, sort_des=False) 
            forest = ['rfc', 'xgboost']
            #Looking at feature importances
            for tree in forest: #Lol
                feats = modeltraining._models[tree].feature_importances_
                modeltraining.plot_feats(tree, colsofinterest, feats)
                modeltraining.SHAP(tree, colsofinterest)
            #Gridsearch
            modeltraining._grid_search("rfc", 10)
            #Ensemble?
            # ensemble = VotingClassifier(
            #     estimators=[
            #         ('knn', modeltraining._models['knn']), 
            #         ('xgb', modeltraining._models['xgb']), 
            #         ('rf', modeltraining._models['rfc'])
            #     ],
            #     voting='soft' # Uses predicted probabilities rather than hard labels
            # )
                

    def pick_lead(self, col:str) -> str:
        """Picks the lead you'd like to analyze

        Args:
            col (str): Lead you want to pick

        Raises:
            ValueError: Gotta pick an integer

        Returns:
            lead (str): the lead you picked!
        """
        tree = Tree(f":select channel:", guide_style="bold bright_blue")
        for idx, channel in enumerate(self.channels):
            tree.add(Text(f'{idx}:', 'blue') + Text(f'{channel} ', 'red'))
        pprint(tree)
        question = f"Please select the {col} channel\n"
        file_choice = console.input(f"{question}")
        if file_choice.isnumeric():
            pprint(f"lead {col} loaded")
            return int(file_choice)
        else:
            raise ValueError("Invalid selection")
        
    def save_results(self):
        """Saves the extracted feature data, including individual files if it's a batch."""
        
        # Save the concatenated master array (works for single or batch)
        np.savez_compressed(self.fp_save, self.avg_data)
        
        # Log the master file size
        mb_size = self.fp_save.stat().st_size / (1024 * 1024)
        logger.warning(f"Saved total feature array to {self.fp_save.name} ({mb_size:.2f} MB)")

        # if a batch, loop through and save the individual pig arrays
        if self.batch_run and hasattr(self, 'all_avg_data'):
            for pig_data in self.all_avg_data:
                # Ensure the array isn't empty before trying to save
                if len(pig_data) > 0:
                    # Grab the pig_id from the first row of this specific array
                    pig_id = str(pig_data["pig_id"][0]) 
                    
                    # Define the path inside the batch folder
                    individual_path = self.fp_base / f"{pig_id}_feat.npz"
                    
                    # Save it
                    np.savez_compressed(individual_path, pig_data)
                    
                    ind_mb_size = individual_path.stat().st_size / (1024 * 1024)
                    logger.info(f"Saved individual {pig_id} features to {individual_path.name} ({ind_mb_size:.2f} MB)")

#CLASS Data Loader
class SignalDataLoader:
    """Handles loading and structuring of the data. Will do single file and batch loading."""
    def __init__(self, file_path):
        self.is_batch = isinstance(file_path, list)
        # Dictionary mapping {pig_id: {'data': full_data, 'channels': channels, 'outcomes': outcomes}}
        self.records = {} 
        
        # Standardize to a list
        paths = file_path if self.is_batch else [file_path]
        
        for path in paths:
            pig_id = Path(path).stem
            data, channels, outcomes = self._load_single_file(str(path))
            self.records[pig_id] = {
                "data": data,
                "channels": channels,
                "outcomes": outcomes
            }

    def _load_single_file(self, path):
        """Core logic to load, extract, and structure data for a single file."""
        full_data = {}
        channels = []
        outcomes = None

        if path.endswith("npz"):
            container = np.load(path)
            files = container.files
            channels = self._identify_and_sort_channels(files)
            full_data = self._stitch_blocks(container, files, channels)
            
        elif path.endswith("pkl"):
            container = np.load(path, allow_pickle=True)
            full_data = container.to_dict(orient="series")
            channels = container.columns.to_list()
            if "ShockClass" in channels:
                outcomes = full_data.pop("ShockClass")
                channels.remove("ShockClass") # Cleaner than pop(index)
                
        return full_data, channels, outcomes

    def _identify_and_sort_channels(self):
        """
        Identifies unique channel names from NPZ keys and returns them 
        in a deterministic (alphabetical) order.
        """
        raw_names = set()
        for k in self.files:
            # Extract channel name from keys like 'ECG_block_1', 'HR_block_0'
            if '_block_' in k:
                name = k.split('_block_')[0]
                raw_names.add(name)
            else:
                # Catch-all for keys that don't follow the block naming convention
                raw_names.add(k)
        
        # Sort alphabetically to ensure the plot labels consistently map to the data indices
        return sorted(list(raw_names))
    
    def _stitch_blocks(self):
        full_data = {}
        for ch in self.channels:
            # Filter keys for this channel and sort by block index
            ch_blocks = sorted(
                [k for k in self.files if k.startswith(f"{ch}_block_")], 
                key=lambda x: int(x.split('_block_')[-1])
            )
            
            if ch_blocks:
                full_data[ch] = np.concatenate([self.container[b] for b in ch_blocks])
            else:
                # Fallback: if no blocks found, maybe it's a single file entry
                if ch in self.files:
                    full_data[ch] = self.container[ch]
        return full_data

#CLASS Advanced Viewer
class SignalGUI:
    """
    Interactive viewer for validating Systolic/Diastolic partitioning of LAD flow.
    Includes time-domain tracing, side-by-side Welch PSD frequency analysis, 
    live SQI metric tracking, and GIF export capabilities.
    """
    def __init__(
        self, 
        ss1_data     : np.array, 
        lad_data     : np.array, 
        beats        : list, 
        sampling_rate: float = 1000.0, 
        window_size  : int = 5000,
        pig_id       : str = None,
    ):
        """
        Args:
            ss1_data (np.array): SS1 pressure signal.
            lad_data (np.array): LAD flow signal.
            beats (list[BP_Feat]): List of extracted BP_Feat dataclasses.
            sampling_rate (float): Sampling frequency (fs) in Hz.
            window_size (int): Number of samples to show in the view (e.g., 5000 = 5 seconds at 1kHz).
            pig_id (str): Pig ID for titles and file exports.
        """
        # Data Setup 
        self.ss1 = ss1_data
        self.lad = lad_data
        self.beats = beats
        self.fs = sampling_rate
        self.pig_id = pig_id if pig_id else "Unknown_Subject"

        # State Settings 
        self.window_size = window_size
        self.current_pos = 0                                       # Start index of the current view window
        self.step_size = int(self.fs * 1.0)                        # 1-second jump for Next/Prev buttons
        self.playback_speed = 1.0                                  # Multiplier for live playback
        self.anim_step = int(self.fs * 0.05 * self.playback_speed) # Samples to advance per animation frame
        self.is_playing = False                                    # Toggle for live playback
        self.anim = None                                           # Holder for the FuncAnimation object
        
        # Frequency State Tracker: 0 = Off, 1 = Welch PSD
        self.freq_mode = 0 
        
        # Track transient Matplotlib elements (shading spans and scatter points) 
        # so they can be securely wiped and redrawn on every frame without memory leaks.
        self.spans = []
        self.scatters = []
        
        # Figure Initialization 
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click_jump) # Bind clicks on the nav bar
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)     # Bind keyboard shortcuts
        
        # Build the scaffolding and trigger the first draw
        self.setup_layout()
        self._init_axes_pool()
        self.update_view()
        plt.show()

    def setup_layout(self):
        """Defines the overarching GridSpec layout for the main plots vs the side control panel."""
        self.gs_main = gridspec.GridSpec(1, 2, width_ratios=[10, 2], figure=self.fig)
        self.gs_main.figure.suptitle(f"{self.pig_id}")
        
        # Left Panel: Main Plot Area (SS1, LAD, Navigator)
        self.gs_plots = gridspec.GridSpecFromSubplotSpec(
            3, 1, 
            subplot_spec=self.gs_main[0], 
            height_ratios=[2, 2, 0.5],
            hspace=0.35 
        )
        
        # Right Panel: Side Controls & Live Metrics Readout
        self.gs_side = gridspec.GridSpecFromSubplotSpec(
            8, 1, subplot_spec=self.gs_main[1], 
            height_ratios=[0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 4], 
            hspace=0.4
        )
        self.setup_controls()

    def setup_controls(self):
        """Initializes GUI buttons and text boxes into their respective grid slots."""
        #Buttons
        self.btn_play = Button(self.fig.add_subplot(self.gs_side[0]), 'Play / Pause')
        self.btn_prev = Button(self.fig.add_subplot(self.gs_side[1]), '< Prev (Left)')
        self.btn_next = Button(self.fig.add_subplot(self.gs_side[2]), 'Next > (Right)')
        ax_speed = self.fig.add_subplot(self.gs_side[3])
        self.txt_speed = TextBox(ax_speed, 'Speed: ', initial=str(self.playback_speed))
        ax_window = self.fig.add_subplot(self.gs_side[4])
        self.txt_window = TextBox(ax_window, 'Window: ', initial=str(self.window_size))
        self.btn_freq = Button(self.fig.add_subplot(self.gs_side[5]), 'Freq: OFF')
        self.btn_gif = Button(self.fig.add_subplot(self.gs_side[6]), 'Export GIF')
        
        # The metrics text needs an empty axis 
        self.ax_metrics = self.fig.add_subplot(self.gs_side[7])
        self.ax_metrics.axis('off')
        self.metric_text = self.ax_metrics.text(0.05, 0.95, "Metrics...", va='top', ha='left', fontsize=10, family='monospace')

        # Bind UI elements to their respective event functions
        self.btn_play.on_clicked(self.toggle_play)
        self.btn_prev.on_clicked(self.step_prev)
        self.btn_next.on_clicked(self.step_next)
        self.txt_speed.on_submit(self.update_speed)
        self.txt_window.on_submit(self.update_window_size)
        self.btn_freq.on_clicked(self.toggle_frequency)
        self.btn_gif.on_clicked(self.export_gif)

    def rebuild_layout(self):
        """
        Safely destroys and rebuilds the plotting axes when toggling View Modes.
        This prevents Matplotlib from stacking ghost axes on top of each other.
        """
        was_playing = self.is_playing
        if was_playing:
            self.toggle_play()
            
        # Remove existing axes securely from the figure
        for ax_name in ['ax_ss1', 'ax_ss1_freq', 'ax_lad', 'ax_lad_freq', 'ax_nav']:
            if hasattr(self, ax_name) and getattr(self, ax_name) is not None:
                getattr(self, ax_name).remove()
                setattr(self, ax_name, None)

        # Re-initialize the layout based on the new freq_mode state
        self._init_axes_pool()
        self.fig.canvas.draw_idle()
        self.update_view()
        
        # Resume animation if it was running before the layout change
        if was_playing:
            self.toggle_play()

    def toggle_frequency(self, event=None):
        """Cycles Frequency mode (Off -> PSD) and updates the UI button."""
        self.freq_mode = (self.freq_mode + 1) % 2
        labels = {0: "Freq: OFF", 1: "Freq: PSD"}
        self.btn_freq.label.set_text(labels[self.freq_mode])
        self.rebuild_layout()

    def _init_axes_pool(self):
        """
        Initializes axes dynamically based on the current frequency mode.
        If freq_mode == 1, it splits the main plot rows lengthwise to show Time and PSD side-by-side.
        """
        self.spans = []
        self.scatters = []

        # --- Base Axis Construction ---
        if self.freq_mode == 0:
            # Standard full-width time domain view
            self.ax_ss1 = self.fig.add_subplot(self.gs_plots[0])
            self.ax_lad = self.fig.add_subplot(self.gs_plots[1], sharex=self.ax_ss1)
            self.ax_ss1_freq, self.ax_lad_freq = None, None
        else:
            # Split rows lengthwise (1x2 grids) for side-by-side Time/PSD view
            gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=self.gs_plots[0], wspace=0.15)
            self.ax_ss1 = self.fig.add_subplot(gs_row1[0])
            self.ax_ss1_freq = self.fig.add_subplot(gs_row1[1])
            
            gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=self.gs_plots[1], wspace=0.15)
            self.ax_lad = self.fig.add_subplot(gs_row2[0], sharex=self.ax_ss1)
            self.ax_lad_freq = self.fig.add_subplot(gs_row2[1])

        # --- Configure SS1 Plot (Time Domain) ---
        self.line_ss1, = self.ax_ss1.plot([], [], color='black', lw=1.5, label="SS1 Pressure")
        self.ax_ss1.set_ylabel("Pressure (mmHg)")
        
        # Build Custom Legend for SS1
        ss1_legend_handles = [
            self.line_ss1,
            Patch(facecolor='lightcoral', alpha=0.3, label='Systole'),
            Patch(facecolor='dodgerblue', alpha=0.3, label='Diastole'),
            Line2D([0], [0], color='green', marker='>', linestyle='None', markersize=8, label='Onset'),
            Line2D([0], [0], color='red', marker='^', linestyle='None', markersize=8, label='Sys Peak'),
            Line2D([0], [0], color='purple', marker='v', linestyle='None', markersize=8, label='Dia Peak'),
            Line2D([0], [0], color='blue', marker='v', linestyle='None', markersize=8, label='Notch'),
        ]
        self.ax_ss1.legend(handles=ss1_legend_handles, loc="upper right", framealpha=0.9)
        
        # Center Beat Indicators (Floats just above the Y-axis using axes fraction transforms)
        trans = self.ax_ss1.get_xaxis_transform()
        self.center_mark_start, = self.ax_ss1.plot([], [], color='darkorange', lw=2, clip_on=False, transform=trans)
        self.center_mark_end, = self.ax_ss1.plot([], [], color='darkorange', lw=2, clip_on=False, transform=trans)
        self.center_mark_horiz, = self.ax_ss1.plot([], [], color='darkorange', lw=1, linestyle='--', clip_on=False, transform=trans)
        self.center_text = self.ax_ss1.text(0, 1.05, "Center Beat", transform=trans, ha='center', va='center', color='darkorange', fontweight='bold', fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8), clip_on=False)
        self.center_text.set_visible(False)

        # Configure LAD Plot (Time Domain)
        self.line_lad, = self.ax_lad.plot([], [], color='crimson', lw=1.5, label="LAD Flow")
        self.ax_lad.axhline(0, color='gray', linestyle='--', alpha=0.5) 
        self.ax_lad.set_ylabel("Flow (mL/min)")
        
        # Build Custom Legend for LAD
        lad_legend_handles = [
            self.line_lad,
            Line2D([0], [0], color='red', marker='^', linestyle='None', markersize=8, label='LAD Sys Peak'),
            Line2D([0], [0], color='blue', marker='^', linestyle='None', markersize=8, label='LAD Dia Peak')
        ]
        self.ax_lad.legend(handles=lad_legend_handles, loc="upper right", framealpha=0.9)

        # Configure Navigator Bar 
        self.ax_nav = self.fig.add_subplot(self.gs_plots[2])
        ds = max(1, len(self.ss1) // 5000) # Heavy downsampling so the nav bar renders instantly
        self.ax_nav.plot(np.arange(0, len(self.ss1), ds), self.ss1[::ds], color='gray', alpha=0.5)
        self.nav_cursor = self.ax_nav.axvline(self.current_pos, color='red', lw=2)
        self.ax_nav.set_yticks([])
        self.ax_nav.set_xlabel("Timeline (Click to Jump) | Spacebar to Pause")
        
        # Hide x labels for shared axes if we aren't in freq mode (where space is tight)
        if self.freq_mode == 0:
            plt.setp(self.ax_ss1.get_xticklabels(), visible=False)

    def update_view(self):
        """
        Main execution loop. Called on every frame/click. 
        Slices data, draws time traces, runs continuous SQI, and plots Welch PSD side-panels.
        """
        # Define window bounds
        s = self.current_pos
        e = min(s + self.window_size, len(self.ss1))
        x_data = np.arange(s, e)
        
        ss1_view = self.ss1[s:e]
        lad_view = self.lad[s:e]

        # Clean up old transient artist elements (prevents major memory leaks during playback)
        for span in self.spans: span.remove()
        for scat in self.scatters: scat.remove()
        self.spans.clear()
        self.scatters.clear()
        
        # Continuous Background SQI Calculation
        # This calculates the Power Ratio & Entropy for the *visible window* on every frame
        f_sqi = {"power_ratio": None, "entropy": None}
        if len(ss1_view) > 0:
            # Calculate Welch PSD (Caps segment length to max 2 seconds for resolution)
            nperseg_sqi = min(len(ss1_view), int(self.fs * 2))
            f_sqi_freq, psd_sqi = welch(ss1_view, fs=self.fs, nperseg=nperseg_sqi)
            total_pwr = np.sum(psd_sqi)
            
            if total_pwr > 0:
                sqi_band = (f_sqi_freq >= 0.5) & (f_sqi_freq <= 15.0)
                f_sqi["power_ratio"] = np.sum(psd_sqi[sqi_band]) / total_pwr
                
                # Normalize PSD to a probability distribution to calculate Shannon entropy
                psd_norm = psd_sqi / total_pwr
                f_sqi["entropy"] = entropy(psd_norm) / np.log(len(psd_norm))

        # Update Time Domain Line Traces
        self.line_ss1.set_data(x_data, ss1_view)
        self.line_lad.set_data(x_data, lad_view)

        # Rescale the Y-axis dynamically based on the current visible data limits
        if e > s:
            self.ax_ss1.set_xlim(s, e)
            self.ax_lad.set_xlim(s, e)
            self._apply_scale(self.ax_ss1, ss1_view)
            self._apply_scale(self.ax_lad, lad_view)

        # Draw Beat Indicators and Calculate Centered Phase 
        center_beat = None
        center_idx = s + (self.window_size // 2)

        for bpf in self.beats:
            # Verify the beat has the required anchors
            if getattr(bpf, 'onset', None) and getattr(bpf, 'dbp_id', None) and getattr(bpf, 'sbp_id', None):
                # Check if beat overlaps with current view
                if bpf.dbp_id > s and bpf.onset < e:
                    
                    # Capture the beat closest to the center for the metrics readout
                    if bpf.onset <= center_idx <= bpf.dbp_id:
                        center_beat = bpf
                    
                    # Calculate absolute index for the dicrotic notch
                    notch_abs = bpf.sbp_id + getattr(bpf, 'notch_id', 0) if getattr(bpf, 'notch_id', None) else None
                    if notch_abs:
                        # --- Shade Systole (Onset to Notch) ---
                        s_span_1 = self.ax_ss1.axvspan(bpf.onset, notch_abs, color='lightcoral', alpha=0.2)
                        s_span_2 = self.ax_lad.axvspan(bpf.onset, notch_abs, color='lightcoral', alpha=0.2)
                        # --- Shade Diastole (Notch to End/DBP) ---
                        d_span_1 = self.ax_ss1.axvspan(notch_abs, bpf.dbp_id, color='dodgerblue', alpha=0.2)
                        d_span_2 = self.ax_lad.axvspan(notch_abs, bpf.dbp_id, color='dodgerblue', alpha=0.2)
                        self.spans.extend([s_span_1, s_span_2, d_span_1, d_span_2])

                        # --- Plot Scatter Points for SS1 ---
                        sc1 = self.ax_ss1.scatter(bpf.sbp_id, self.ss1[bpf.sbp_id], color='red', zorder=5, marker='^')
                        sc2 = self.ax_ss1.scatter(notch_abs, self.ss1[notch_abs], color='blue', zorder=5, marker='v')
                        sc3 = self.ax_ss1.scatter(bpf.onset, self.ss1[bpf.onset], color='green', zorder=5, marker='>')
                        sc4 = self.ax_ss1.scatter(bpf.dbp_id, self.ss1[bpf.dbp_id], color='purple', zorder=5, marker='v')
                        self.scatters.extend([sc1, sc2, sc3, sc4])

                        # --- Plot Scatter Points for LAD ---
                        lad_systole = self.lad[bpf.onset:notch_abs]
                        lad_diastole = self.lad[notch_abs:bpf.dbp_id]
                        sys_idx_abs = bpf.onset + np.argmax(lad_systole) if lad_systole.size > 0 else bpf.onset
                        dia_idx_abs = notch_abs + np.argmax(lad_diastole) if lad_diastole.size > 0 else notch_abs
                        
                        sc5 = self.ax_lad.scatter(sys_idx_abs, self.lad[sys_idx_abs], color='red', zorder=5, marker='^')
                        sc6 = self.ax_lad.scatter(dia_idx_abs, self.lad[dia_idx_abs], color='blue', zorder=5, marker='^')
                        self.scatters.extend([sc5, sc6])

        # Toggle Center Beat Visual Indicators (The orange bracket floating over the plot)
        if center_beat:
            self.center_mark_start.set_data([center_beat.onset, center_beat.onset], [1.02, 1.08])
            self.center_mark_end.set_data([center_beat.dbp_id, center_beat.dbp_id], [1.02, 1.08])
            self.center_mark_horiz.set_data([center_beat.onset, center_beat.dbp_id], [1.05, 1.05])
            self.center_text.set_position(((center_beat.onset + center_beat.dbp_id) / 2, 1.05))
            self.center_mark_start.set_visible(True)
            self.center_mark_end.set_visible(True)
            self.center_mark_horiz.set_visible(True)
            self.center_text.set_visible(True)
        else:
            self.center_mark_start.set_visible(False)
            self.center_mark_end.set_visible(False)
            self.center_mark_horiz.set_visible(False)
            self.center_text.set_visible(False)

        # Side-By-Side Frequency Traces (If Toggled)
        if self.freq_mode == 1 and len(ss1_view) > 100:
            self.ax_ss1_freq.cla()
            self.ax_lad_freq.cla()
            
            # --- PSD (Welch) Mode ---
            f_s, p_s = welch(ss1_view, fs=self.fs, nperseg=nperseg_sqi)
            f_l, p_l = welch(lad_view, fs=self.fs, nperseg=nperseg_sqi)
            
            self.ax_ss1_freq.plot(f_s, p_s, color='darkviolet', lw=1.5)
            self.ax_lad_freq.plot(f_l, p_l, color='darkviolet', lw=1.5)
            self.ax_ss1_freq.fill_between(f_s, p_s, color='darkviolet', alpha=0.3)
            self.ax_lad_freq.fill_between(f_l, p_l, color='darkviolet', alpha=0.3)
            
            self.ax_ss1_freq.set_ylabel("Power / Hz")
            self.ax_lad_freq.set_ylabel("Power / Hz")
            
            # Crop X-axis to physiological frequency range (0 to 30 Hz)
            self.ax_ss1_freq.set_xlim(0, 30) 
            self.ax_lad_freq.set_xlim(0, 30)
            self.ax_ss1_freq.set_xlabel("Frequency (Hz)")
            self.ax_lad_freq.set_xlabel("Frequency (Hz)")

            self.ax_ss1_freq.set_title("SS1 Spectral Analysis", fontsize=10)
            self.ax_lad_freq.set_title("LAD Spectral Analysis", fontsize=10)

        # Final UI Updates
        # Move navigator dot
        self.nav_cursor.set_xdata([s + (self.window_size//2)])

        # Push metrics to side panel
        self.update_metrics_text(center_beat, f_sqi)
        self.fig.canvas.draw_idle()

    def update_metrics_text(self, bpf, f_sqi):
        """Updates the side panel with features from the centered beat and live SQI."""
        def fmt(val, prec=2):
            """Helper to safely format floats or return N/A if missing."""
            return f"{val:.{prec}f}" if val is not None else "N/A"
            
        text = f"--- Signal Quality ---\n"
        text += f"In-Band Pwr:  {fmt(f_sqi['power_ratio'], 2)}\n"
        text += f"Spec Entropy: {fmt(f_sqi['entropy'], 2)}\n\n"
        
        if bpf is None:
            text += "No complete beat\nin center view."
        else:
            text += "[CENTER BEAT]\n"
            text += "--- Hemodynamics ---\n"
            text += f"SBP:      {fmt(getattr(bpf, 'SBP', None), 1)}\n"
            text += f"DBP:      {fmt(getattr(bpf, 'DBP', None), 1)}\n"
            text += f"true_MAP: {fmt(getattr(bpf, 'true_MAP', None), 1)}\n"
            text += f"pul_Wid:  {fmt(getattr(bpf, 'pul_wid', None), 1)}\n\n"
            text += "--- Coronary Flow ---\n"
            text += f"Mean LAD: {fmt(getattr(bpf, 'lad_mean', None), 2)}\n"
            text += f"Sys Peak: {fmt(getattr(bpf, 'lad_sys_pk', None), 2)}\n"
            text += f"Dia Peak: {fmt(getattr(bpf, 'lad_dia_pk', None), 2)}\n"
            text += f"DS Ratio: {fmt(getattr(bpf, 'lad_ds_rat', None), 2)}\n"
            text += f"Dia AUC:  {fmt(getattr(bpf, 'lad_dia_auc', None), 2)}\n\n"
            text += "--- Resistance ---\n"
            text += f"CVR:      {fmt(getattr(bpf, 'cvr', None), 2)}\n"
            text += f"DCR:      {fmt(getattr(bpf, 'dcr', None), 2)}\n"
            text += f"Flow Div: {fmt(getattr(bpf, 'flow_div', None), 2)}\n"

        self.metric_text.set_text(text)

    def _apply_scale(self, ax, view_data):
        """Helper to dynamically autoscale the Y-axis of a plot based on visible data bounds with 15% padding."""
        if view_data.size > 1:
            v_min, v_max = np.min(view_data), np.max(view_data)
            pad = (v_max - v_min) * 0.15 if v_max != v_min else 0.1
            ax.set_ylim(v_min - pad, v_max + pad)

    # --- Interaction Events ---
    def step_next(self, event=None):
        """Advances view window by 1 time step."""
        self.current_pos = min(self.current_pos + self.step_size, len(self.ss1) - self.window_size)
        self.update_view()

    def step_prev(self, event=None):
        """Rewinds view window by 1 time step."""
        self.current_pos = max(0, self.current_pos - self.step_size)
        self.update_view()

    def on_click_jump(self, event):
        """Jumps directly to a point clicked on the navigator bar."""
        if event.inaxes == self.ax_nav:
            self.current_pos = int(event.xdata) - (self.window_size // 2)
            self.current_pos = max(0, min(self.current_pos, len(self.ss1) - self.window_size))
            self.update_view()

    def update_speed(self, text):
        """Handles user input in the Speed text box to control animation playback rate."""
        try: 
            self.playback_speed = float(text)
            self.anim_step = max(1, int(self.fs * 0.05 * self.playback_speed))
            console.print(f"Playback speed set to {self.playback_speed}x")
        except ValueError as v: 
            console.print(f"Invalid speed value: {v}. Please enter a number.")
            self.txt_speed.set_val(str(self.playback_speed))

    def update_window_size(self, text):
        """Handles user input in the Window text box to zoom in/out of the timeline."""
        try: 
            self.window_size = int(text)
            self.update_view()
        except ValueError as v: 
            print(f"Invalid window size: {v}")

    def on_key_press(self, event):
        """Keyboard shortcuts: Spacebar (Play/Pause), Left/Right Arrows (Step)."""
        if event.key == 'right': 
            self.step_next()
        elif event.key == 'left':
            self.step_prev()
        elif event.key == ' ':
            self.toggle_play()

    # Animation & Export 
    def toggle_play(self, event=None):
        """Starts or pauses the live timeline playback."""
        if self.is_playing:
            self.is_playing = False
            self.btn_play.label.set_text('Play')
            if self.anim: self.anim.event_source.stop()
        else:
            self.is_playing = True
            self.btn_play.label.set_text('Pause')
            if not self.anim:
                self.anim = animation.FuncAnimation(self.fig, self._animate_step, interval=50, blit=False, cache_frame_data=False)
            self.anim.event_source.start()

    def _animate_step(self, frame):
        """Advances the timeline automatically during playback."""
        if self.current_pos >= len(self.ss1) - self.window_size:
            self.toggle_play() 
            return
        self.current_pos = min(self.current_pos + self.anim_step, len(self.ss1) - self.window_size)
        self.update_view()

    def export_gif(self, event=None):
        """Exports the next 5 seconds of the timeline to a GIF file using Pillow."""
        console.print("Preparing GIF export... Please wait.")
        was_playing = self.is_playing
        if was_playing: self.toggle_play()

        original_pos = self.current_pos
        frames = 5 * 10
        step = int(self.fs / 10) 

        def gif_frame(i):
            self.current_pos = min(original_pos + (i * step), len(self.ss1) - self.window_size)
            self.update_view()

        export_anim = animation.FuncAnimation(self.fig, gif_frame, frames=frames, blit=False)
        filename = f"{self.pig_id}_{original_pos}.gif"
        
        try:
            export_anim.save(filename, writer=animation.PillowWriter(fps=10))
            console.print(f"Export complete! Saved as {filename}")
        except Exception as e:
            console.print(f"Failed to save GIF. Ensure Pillow is installed. Error: {e}")

        # Return UI to original state after export
        self.current_pos = original_pos
        self.update_view()
        if was_playing: 
            self.toggle_play()

#CLASS Wavelet / STFT / Phase Calculation Logic
class CardiacFreqTools:
    """Helper class for calculationing wavelets / STFT / Phase Conditions."""
    def __init__(self, fs=1000, bandwidth_parameter=8.0):
        self.fs = fs
        self.c = bandwidth_parameter
    
    #TODO - Update CardiacFreqTools with more stumpy 
        #Try extracting regime change in LAD.  Looking for inverted signals in
        #the wave form and possibly we can extract those. 

    def get_wavelet(self, wavelet_type:str, center_freq:float):
        """Generates the requested wavelet kernel

        Args:
            wavelet_type (str): Type of desired wavelet
            center_freq (float): Center frequency of wavelet

        Raises:
            ValueError: If you don't submit the right wavelet it will error

        Returns:
            _type_: wavelet
        """        
        w_desired = 2 * np.pi * center_freq
        s = self.c * self.fs / w_desired
        M = int(2 * 4 * s) + 1
        t = np.arange(-M//2 + 1, M//2 + 1)
        norm = 1 / np.sqrt(s)
        
        # Base complex exponential and gaussian window
        x = t / s
        gauss = np.exp(-0.5 * x**2)
        complex_exp = np.exp(1j * self.c * x)
        if wavelet_type == 'morlet':
            wavelet = norm * complex_exp * gauss
        elif wavelet_type == 'cgau1':
            # 1st derivative of the complex Gaussian
            wavelet = norm * (-x + 1j * self.c) * complex_exp * gauss
        else:
            raise ValueError(f"Wavelet {wavelet_type} not supported.")
            
        return wavelet

    def compute_section_phase_metric(self, signal:np.array, peaks:list, target_freq:float = 2.0, wavelet:str = 'morlet'):
        """
        Calculates the average phase angle for each beat and phase variance for the section.
        """
        if len(peaks) < 2:
            return [], np.nan
            
        #Apply CWT to the entire section at once (faster)
        kernel = self.get_wavelet(wavelet, target_freq)
        cwt_complex = convolve(signal, kernel, mode='same')
        full_phases = np.angle(cwt_complex)
        
        beat_phases = []
        pre_peak = int(0.2 * self.fs)
        beat_win = int(0.6 * self.fs)
        
        # Extract phase for each individual beat
        for p in peaks:
            start = p - pre_peak
            end = start + beat_win
            
            if start >= 0 and end < len(signal):
                beat_phase_array = full_phases[start:end]
                
                # Circular mean of phase for this specific beat
                mean_beat_phase = np.angle(np.mean(np.exp(1j * beat_phase_array)))
                beat_phases.append(mean_beat_phase)
                
        if not beat_phases:
            return [], np.nan
            
        # Calculate Circular Phase Variance for the entire section
        # R is the resultant vector length (0 to 1). Variance is 1 - R.
        # R = 
        R = np.abs(np.mean(np.exp(1j * np.array(beat_phases))))
        section_variance = 1 - R 
        
        return beat_phases, section_variance

    def STFT_extract(self, signal: np.array, peaks:np.array):
        """
        Calculates FFT for each R-R interval, averages the PSD across the section, 
        and extracts the top frequency and its first 3 harmonics.
        
        Args:
            signal (np.array): Waveform section
            peaks (np.array): Array of peak indices to chunk by
            
        Returns:
            tuple: (frequencies_array, psd_amplitudes_array) padded to length 4.
        """
        # Need at least two peaks to make an interval
        if len(peaks) < 2:
            return np.full(4, np.nan), np.full(4, np.nan)
            
        # To average FFTs of varying lengths, we need to zero-pad them to a consistent size.
        # 2 seconds of padding (2 * fs) gives 0.5Hz resolution.  Should fit the pig data well
        nfft = int(self.fs * 2.0) 
        freq_list = rfftfreq(nfft, d=1/self.fs)
        all_psd = []
        
        # Loop through each R-R interval
        for i in range(len(peaks) - 1):
            p0 = peaks[i]
            p1 = peaks[i+1]
            samp = signal[p0:p1]
            
            if len(samp) == 0:
                continue
                
            # Perform FFT with zero-padding to nfft so the frequency bins align
            fft_samp = np.abs(rfft(samp, n=nfft))
            psd = fft_samp ** 2  # Square the amplitude to get Power Spectral Density
            all_psd.append(psd)
            
        if not all_psd:
            return np.full(4, np.nan), np.full(4, np.nan)
            
        # average the PSDs across all intervals in this section
        mean_psd = np.mean(all_psd, axis=0)
        
        # scipy find peaks 
        freq_res = freq_list[1] - freq_list[0]
        min_dist = max(1, int(0.5 / freq_res))
        peaks_idx, _ = find_peaks(
            mean_psd, 
            distance=min_dist, 
            prominence=np.percentile(mean_psd, 50)
        )
        
        # Handle edge cases where signal is flat/dead
        if len(peaks_idx) == 0:
            return np.full(4, np.nan), np.full(4, np.nan)
            
        #Sort peaks by PSD amplitude in descending order
        sorted_peak_indices = peaks_idx[np.argsort(mean_psd[peaks_idx])][::-1]
        
        #Extract top 4 peaks (Fundamental + 3 "harmonics")
        top_indices = sorted_peak_indices[:4]
        top_f = freq_list[top_indices]
        top_psd = mean_psd[top_indices]
        
        #Pad with NaNs if fewer than 4 distinct peaks were found
        if len(top_f) < 4:
            pad_size = 4 - len(top_f)
            top_f = np.pad(top_f, (0, pad_size), constant_values=np.nan)
            top_psd = np.pad(top_psd, (0, pad_size), constant_values=np.nan)
            
        return top_f, top_psd
    
    def evaluate_signal(self, signal: np.array, baseline_psd_norm=None) -> tuple:
        """
        Evaluates if the signal is physiological or just noise. Uses Welch's method to estimate the power spectral density independently of peak finding.
        
        Args:
            signal (np.array): The raw waveform section.
            baseline_psd_norm (np.array, optional): A normalized baseline PSD for Wasserstein distance.
            
        Returns:
            tuple: (in_band_ratio, spectral_entropy, wasserstein_dist, current_psd_norm)
        """
        # Calculate PSD using Welch's method - robust to random noise spikes
        # Using 2-second segments with 50% overlap for smooth spectrum
        nperseg = int(self.fs * 2.0)
        freqs, psd = welch(signal, fs=self.fs, nperseg=nperseg)
        
        total_power = np.sum(psd)
        if total_power == 0:
            return 0.0, 1.0, np.nan, None

        # --- In-Band Power Ratio ---
        # Look for power specifically in the 0.5 Hz to 15.0 Hz band (HR + Harmonics)
        band_mask = (freqs >= 0.1) & (freqs <= 15.0)
        in_band_power = np.sum(psd[band_mask])
        power_ratio = in_band_power / total_power
        
        # --- Spectral Entropy ---
        # Normalize the PSD so it sums to 1 (like a probability distribution)
        psd_norm = psd / total_power
        
        # Calculate Shannon entropy and normalize it (0-1)
        # 0 = Single pure sine wave, 1 = Pure flat white noise
        spec_entropy = entropy(psd_norm)
        norm_spec_entropy = spec_entropy / np.log(len(psd_norm))
        
        # --- Wasserstein Distance (Distribution Shift) ---
        #NOTE - Save for later
            #Try to see if we can do it with shannon entropy / PSD.  
            #Use wasserstein as a backup
            #IDEA - What if you use the target label to calculate teh wassterstein distance for each class.  
                #Giving us a distribution distance from the previous class over time????  Not sure that helps
            
        w_dist = np.nan
        if baseline_psd_norm is not None:
            # Treat the frequencies as the "locations" and normalized PSD as the "weights"
            w_dist = wasserstein_distance(freqs, freqs, u_weights=psd_norm, v_weights=baseline_psd_norm)

        return power_ratio, norm_spec_entropy, w_dist, psd_norm

#CLASS EDA
class EDA(object):
    def __init__(
            self,
            avg_data:np.array,
            col_names:list,
            fs:float,
            gpu_devices:list,
            fp:str,
            view_eda:bool,
            view_models:bool
        ):
        self.data:pd.DataFrame = pd.DataFrame(avg_data)
        self.feature_names:list = list(col_names)
        self.fs:float = fs
        self.fp_base:Path = fp
        self.view_eda = view_eda
        self.view_models = view_models
        self.gpu_devices:list = gpu_devices
        self.task:str = "classification"
        self.target:pd.Series = None
        self.target_name:str = "shock_class"
        self.target_names:list = ["BL", "C1", "C2", "C3", "C4"]
        self.rev_target_dict:dict = {
            0:"BL",
            1:"C1",
            2:"C2",
            3:"C3",
            4:"C4"
        }

    #FUNCTION clean_data
    def clean_data(self):
        #imputate sections needing it
        # for col in self.feature_names[4:]:
        #     self.imputate("mean", col)
        
        #Replace zeros with nan's
        self.data.iloc[:, 1:].replace(0, np.nan, inplace=True)

        #Display nulls
        self.print_nulls(False)
        
        #Drop nulls
        self.drop_nulls(self.feature_names[5:])

        #Drop zero vals
        self.drop_zeros(self.feature_names[5:])
        
        #Drop outliers (features, IQR range)
        self.drop_outliers(self.feature_names[5:], 10)

        #Drop col used to make target, and any cols we don't want.  PSD definitely not
        if not self.view_eda:
            for col in ["EBV", "psd0", "psd1", "psd2", "psd3"]:
                self.data.pop(col)
                self.feature_names.pop(self.feature_names.index(col))
                logger.info(f"removed col {col}")
        
        #Get rid of these cols for modeling. 
        else:
            for col in ["psd0", "psd1", "psd2", "psd3"]:
                self.data.pop(col)
                self.feature_names.pop(self.feature_names.index(col))
                logger.info(f"removed col {col}")

        #Drop the target column.
        self.target = self.data.pop("shock_class")
        self.feature_names.pop(self.feature_names.index("shock_class"))
        logger.info(f"assigned target {self.target.name}")
        
        #Check nulls
        self.print_nulls(False)

    #FUNCTION Imputation
    def imputate(self, imptype:str, col:str):
        """Function for imputing missing data.  
        Will be adding others in the future. 

        Args:
            imptype (str): Type of imputation you want
            col (str): column you want to perform it on
        """		
        if imptype == "mean":
            self.data[col].fillna(self.data[col].mean(), inplace=True)

        elif imptype == "median":
            self.data[col].fillna(self.data[col].median(), inplace=True)

        elif imptype == "mode":
            self.data[col].fillna(self.data[col].mode(), inplace=True)

    #FUNCTION drop_nulls
    def drop_nulls(self, col:str|list=None):
        """Null Dropping routine. Use at your own risk.  If a target column is provided, it will drop the column.  
        If a list is provided, it will drop each item in the list.  If no target provided, this will drop all rows with a null. 
        (switch axis=1 if you want by all columns)

        Args:
            col (str | list, optional): Column or list of columns to drop. Defaults to None.
        """
        
        logger.info(f'Shape before drop {self.data.shape}')
        if isinstance(col, str):
            self.data.dropna(subset=[col], how="any", inplace=True)
        elif isinstance(col, list):
            self.data.dropna(subset=col, how="any", inplace=True)
        else:
            self.data.dropna(axis=0, subset=self.data, how='any', inplace=True)
        logger.info(f'Shape after drop {self.data.shape}')
    
    #FUNCTION drop_zeros
    def drop_zeros(self, col: str | list = None):
        """Zero Dropping routine.
        If a target column is provided, it drops rows where that column is 0.  
        If a list is provided, it drops rows where ANY of those columns are 0.  
        If no target provided, it drops rows where ANY column is 0. 

        Args:
            col (str | list, optional): Column or list of columns to check for zeros. Defaults to None.
        """
        
        logger.info(f'Shape before drop {self.data.shape}')
        
        if isinstance(col, str):
            # Keep rows where the specific column does NOT equal 0
            self.data = self.data[self.data[col] != 0]
            
        elif isinstance(col, list):
            # Keep rows where ALL of the specified columns do NOT equal 0. (equivalent to dropping if ANY of them are 0)
            self.data = self.data[(self.data[col] != 0).all(axis=1)]
            
        else:
            # If col is None, check the entire DataFrame
            self.data = self.data[(self.data != 0).all(axis=1)]
            
        logger.info(f'Shape after drop {self.data.shape}')

    #FUNCTION drop_outliers
    def drop_outliers(self, col: str | list = None, degree: int = 6):
        """
        Outlier Dropping routine based on 6x IQR.
        Identifies the Interquartile Range (IQR) and removes rows where values 
        fall outside of (Q1 - 6*IQR) or (Q3 + 6*IQR).
        
        Args:
            col (str | list, optional): Column or list of columns to check for outliers.
            degree (int, optional): Degree of IQR you want if not 6 IQR
        """
        if col is None:
            return

        logger.info(f'Shape before dropping outliers: {self.data.shape}')
        
        # Standardize input to a list
        cols_to_check = [col] if isinstance(col, str) else col
        
        for c in cols_to_check:
            if c in self.data.columns:
                # Calculate Q1 (25th percentile) and Q3 (75th percentile)
                Q1 = self.data[c].quantile(0.25)
                Q3 = self.data[c].quantile(0.75)
                
                # Calculate IQR and bounds
                IQR = Q3 - Q1
                lower_bound = Q1 - (degree * IQR)
                upper_bound = Q3 + (degree * IQR)
                
                # Create a mask for rows to keep: within bounds OR missing (NaN)
                # We keep NaNs here so they don't accidentally get dropped 
                mask = ((self.data[c] >= lower_bound) & (self.data[c] <= upper_bound)) | self.data[c].isna()
                
                # Apply mask
                self.data = self.data[mask]
                
        logger.info(f'Shape after dropping outliers: {self.data.shape}')

    #FUNCTION print_nulls
    def print_nulls(self, plotg=False):
        """Checks for nan's.  Color codes output to verify if over 30% of the
        data is missing.  Prints results to a Rich Table

        Args:
            plotg (bool, optional): Whether or not to plot the missing values
            Defaults to False.
        """		
        #column | nulls | % of Total | Over 30 % Nulls
        #str    | int   | float      | Boolean

        #Adding in a Rich Table for printing.
        table = Table(title="Null Report")
        table.add_column("Column", justify="right", no_wrap=True)
        table.add_column("Null Count", justify="center")
        table.add_column("Null %", justify="center")
        table.add_column("Over 30%", justify="center")

        for x in range(0, len(self.data.columns), 20):
            subslice = self.data.iloc[:, x:x+20]
            cols = list(subslice.columns)
            nulls = subslice.isnull().sum()
            perc = round(nulls / subslice.shape[0], 2)
            over30 = [True if x > .30 else False for x in perc]
            end_sect = False
            for ss in range(len(cols)):
                if ss == len(cols)-1:
                    end_sect = True
                if not over30[ss]:
                    table.add_row(
                        cols[ss], 
                        f"{nulls.iloc[ss]:.0f}", 
                        f"{perc.iloc[ss]:.2%}", 
                        f"{str(over30[ss])}", 
                        style="white on blue", 
                        end_section=end_sect
                    )
                else:
                    table.add_row(
                        cols[ss], 
                        f"{nulls.iloc[ss]:.0f}", 
                        f"{perc.iloc[ss]:.2%}", 
                        f"{str(over30[ss])}", 
                        style="red on white", 
                        end_section=end_sect
                    )
        console.log("Printing Null Table")
        console.print(table)

        if plotg:
            console.log("plotting null visualization")
            #Print of a chart of the NA values.  Solid black square = no nulls.
            plt.figure(figsize=(10, 8))
            plt.imshow(self.data.isna(), aspect="auto", interpolation="nearest", cmap="gray")
            plt.xlabel("Column Number")
            plt.ylabel("Sample Number")
            plt.title("Null Visualization")
            plt.show()
            plt.close()

    #FUNCTION sum_stats
    def sum_stats(self, stat_list:list, title=str):
        """Accepts a list of features you want to be summarized. 
        Manipulate the .agg function below to return your desired format.

        Args:
            stat_list (list): List of feature names
            title (str): What you want to call the plot
        """		
        #Add a rich table for results. 
        table = Table(title=title)
        table.add_column("Measure Name", style="green", justify="right")
        table.add_column("mean", style="sky_blue3", justify="center")
        table.add_column("std", style="turquoise2", justify="center")
        table.add_column("max", style="yellow", justify="center")
        table.add_column("min", style="gold3", justify="center")
        table.add_column("count", style="cyan", justify="center")

        for col in stat_list:
            _mean, _stddev, _max, _min, _count = self.data.loc[self.data[col] != 0, col].agg(["mean", "std", "max", "min", "count"]).T
            table.add_row(
                col,
                f"{_mean:.2f}",
                f"{_stddev:.2f}",
                f"{_max:.2f}",
                f"{_min:.2f}",
                f"{_count:.0f}",
            )
        console.log(f"printing table for features :\n{stat_list}")
        console.print(table)

    #FUNCTION num_features
    def num_features(self, plotg:bool=False, print_stats:bool=True):
        """ isolates numeric features and does a quick plot of them. 		

        Args:
            plotg (bool, optional): Whether to plot. Defaults to False.
            print_stats (bool, optional): Whether to show table stats. Defaults to False.
        """		
        self.num_df = self.data.select_dtypes(include='number')

        #Add a rich table for results. 
        table = Table(title="Num Feature Report")
        table.add_column("Measure Name", style="green", justify="right")
        table.add_column("mean", style="sky_blue3", justify="center")
        table.add_column("std", style="turquoise2", justify="center")
        table.add_column("max", style="yellow", justify="center")
        table.add_column("min", style="gold3", justify="center")
        table.add_column("count", style="cyan", justify="center")
        table.add_column("nulls\nfound?", justify="center", no_wrap=False)
        # self.num_df.iloc[:, idx:idx+20].agg(["count", np.mean, np.std, max, min]).T

        if print_stats:
            for idx in range(0, self.num_df.shape[1], 40):
                subslice = self.num_df.iloc[:, idx:idx+40]
                cols = list(subslice.columns)
                for ss in range(len(cols)):
                    if ss == len(cols)-1:
                        end_sect = True
                    else:
                        end_sect = False
                    _count, _mean, _stddev, _max, _min = subslice.iloc[:, ss].agg(["count", np.nanmean, np.nanstd, max, min]).T
                    if _count == subslice.shape[0]:
                        nulls = "[bold green]No"
                    else:
                        nulls = "[bold red]Yes"
                    table.add_row(
                        cols[ss],
                        f"{_mean:.2f}",
                        f"{_stddev:.2f}",
                        f"{_max:.2f}",
                        f"{_min:.2f}",
                        f"{_count:.0f}",
                        nulls,
                        end_section=end_sect
                    )
                    if end_sect:
                        colnames = [table.columns[x].header for x in range(len(table.columns))]
                        table.add_row(
                            colnames[0],
                            colnames[1],
                            colnames[2],
                            colnames[3],
                            colnames[4],
                            colnames[5],
                            colnames[6],
                            style="white",
                            end_section=end_sect
                        )
                    
            # logger.info(self.num_df.iloc[:, idx:idx+20].agg(["count", np.mean, np.std, max, min]).T)
            console.log("Printing Num Feature Table")
            console.print(table)
        if plotg:
            for idx in range(0, len(self.num_df.columns), 40):
                self.num_df.iloc[:, idx:idx+40].plot(
                    lw=0, 
                    marker=".", 
                    subplots=True, 
                    layout=(-1, 3),
                    figsize=(12, 12), 
                    markersize=8
                )
                plt.tight_layout()
                plt.show()

    #FUNCTION cat_features
    # def cat_features(self, plotg:bool=False, print_stats:bool=True):
    # 	""" isolates categorical features and does a quick plot of them

    # 	Args:
    # 		plotg (bool, optional): Whether to plot. Defaults to False.
    # 		print_stats (bool, optional): Whether to show table stats. Defaults to False.
    # 	"""
    # 	self.cat_df = self.data.select_dtypes(exclude=['number', 'datetime'])

    # 	# if plotg:
    # 	# 	for x in range(0, len(self.data.columns), 20):
    # 	# 		self.cat_df.iloc[:, idx:idx+40].plot(
    # 	# 								lw=0, marker=".", 
    # 	# 								subplots=True, layout=(-1, 3),
    # 	# 								figsize=(12, 12), markersize=3
    # 	# 		)
    # 	# 		plt.show()
    # 	# 		print('\n\n')
    # 	#IDEA - Cat plotting

    # 	if print_stats:
    # 		#Add a rich table for results. 
    # 		table = Table(title="Cat Feature Report", expand=True)
    # 		table.add_column("Col Name", style="green", justify="right")
    # 		table.add_column("count", style="cyan", justify="center")
    # 		table.add_column("unique", style="sky_blue3", justify="center")
    # 		table.add_column("mf val", style="yellow", justify="center", no_wrap=False)
    # 		table.add_column("mf count", style="gold3", justify="center")
            
    # 		for idx in range(0, self.cat_df.shape[1], 40):
    # 			subslice = self.cat_df.iloc[:, idx:idx+40]
    # 			cols = list(subslice.columns)
    # 			for ss in range(len(cols)):
    # 				if ss == len(cols) - 1:
    # 					end_sect = True
    # 				else:
    # 					end_sect = False
    # 				_count, _unique, _tval, _tfreq = subslice.iloc[:, ss].describe().T
    # 				table.add_row(
    # 					f'{cols[ss]}',
    # 					f'{_count}',
    # 					f'{_unique}', 
    # 					f'{_tval}',
    # 					f'{_tfreq}',
    # 					end_section = end_sect
    # 				)
    # 				if end_sect:
    # 					colnames = [table.columns[x].header for x in range(len(table.columns))]
    # 					table.add_row(
    # 						colnames[0],
    # 						colnames[1],
    # 						colnames[2],
    # 						colnames[3],
    # 						colnames[4],
    # 						style="white",
    # 						end_section=end_sect)

    # 		console = Console()
    # 		console.log("Printing Cat Feature Table")
    # 		console.print(table)

    #FUNCTION heatmap
    def corr_heatmap(self, sel_cols:list):
        """Generates correlation heatmap of numeric variables.

        Args:
            sel_cols (list): columns you want to submit for a heatmap
        """		
        #! Up to the user to submit the correct columns for heatmaps (ie - numeric)
        #if you didn't select any columns, it will select all the numeric
        #columns for you. 

        if not sel_cols:
            sel_cols = self.data.select_dtypes(include='number')

        self.num_corr = self.data[sel_cols].corr(method="spearman") 

        #!Caution
        #If you want to automate null dropping  you can do so below		
        #Find the corr cols that are null
        # more_drop_cols = list(self.num_corr.isnull().columns)

        #Drop em!
        # self.num_df.drop(more_drop_cols, axis=1, inplace=True)

        #Make correlation chart
        fig = plt.figure(figsize=(12, 8))
        mask = np.triu(np.ones_like(self.num_corr, dtype=bool))
        heatmap = sns.heatmap(self.num_corr,
            mask=mask,
            vmin=-1, 
            vmax=1, 
            annot=True, 
            annot_kws={
                'fontsize':6,
            },
            fmt='.1f',
            cmap='RdYlGn')
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':16}, pad=12)
        if self.fp_base:
            fig.savefig(PurePath(self.fp_base, Path("heatmap.png")), dpi=300, bbox_inches='tight')
        timer_error = fig.canvas.new_timer(interval = 3000)
        timer_error.single_shot = True
        timer_cid = timer_error.add_callback(plt.close, fig)
        timer_error.start()
        plt.show()
        plt.close()

    #FUNCTION eda_plot
    def eda_plot(self, 
        plot_type:str="histogram",
        feat_1:str=False, 
        feat_2:str=False, 
        group:pd.Series=None
        ):
        """Basic plotting method for EDA class

        Plot types:
        - scatterplot
        - histogram
        - jointplot
        - pairplot

        To use the plotting method, try something like this.

        `explore.eda_plot("scatterplot", feat_1, feat_2, group)`

        The last variable, group, is what controls groupings.  Submit a
        categorical column, it will group the chart accordingly. Submit
        False, and it will just give you the standard chart.

        Args:
            plot_type(str): type of plot you want graphed. 
            feat_1 (str or bool): feature of interest (col name)
            feat_2 (str or bool, optional): feature of interest (col name)
            group (pd.Series or bool, optional): Column you want to group on. Usually
                a categorical. Defaults to False.
        """		
        def onSpacebar(event):
            """When scanning ECG's, hit the spacebar if keep the chart from closing. 

            Args:
                event (_type_): accepts the key event.  In this case its looking for the spacebar.
            """	
            if event.key == " ": 
                timer_error.stop()
                timer_error.remove_callback(timer_cid)
                logger.warning(f'Timer stopped')
            #quick correlation
            if not isinstance(feat_1, bool) and not isinstance(feat_2, bool):
                self.corr = self.data[feat_1].corr(self.data[feat_2])
                logger.warning(f'correlation of {feat_1} and {feat_2}: {self.corr:.2f}')

        #Generates repeatable colordict for all values in the group
        if not isinstance(group, bool):
            #Get the size of the target responses. (how many are there)
            num_groups = np.unique(group).size
                #NOTE, best to use MPL's sequential colormaps as the ligthness is
                #monotonically increasing in intensity of the color
                #Avail cmaps here.  
                #https://matplotlib.org/stable/tutorials/colors/colormaps.html
            #Get the colormap
            color_cmap = mpl.colormaps["Paired"]
            #Generate a hex code for the color.  
            color_str = [mpl.colors.rgb2hex(color_cmap(i)) for i in range(num_groups)]
            #Now make a dictionary of the activities and their hex color.
            colcyc = color_str[:num_groups]
            cycol = cycle(colcyc)
            group_color_dict = {x:next(cycol) for x in np.unique(group)}

            ###### 
            #if the target isn't in the data, this add's it to a temp dataframe
            #for hue/group mapping
            #BUG - This code smells here.  come back and rewrite
                #Solutions:
                #1. change the dataset source (for all plots mind you)
                    #to the self.dataset.
                        #Wouldn't give you updated nulls if you dropped any)
                        #I'd like to keep that original dataset in tact if possible. 
                #2. Use the weird logic i came up with below.  It works ok. 
                    #ideaflow: iF the grouped target var isn't in the datset (ie
                    #the target), it adds it back in so i can assign color hue's
                    #(usually for classification)
                #3. Maybe brainstorm with Tom tomorrow about solutions here. 
                    #nothing coming to mind right away because I want to keep the original 
                    #dataset untouched.  

            hue_col = group.name

            if hue_col not in self.data.columns.tolist():
                _comb_df = pd.concat([self.data, group], axis=1)
            else:
                _comb_df = self.data
        else:
            _comb_df = self.data
        
        cur_col_idx = _comb_df.columns.tolist()
        if plot_type == "scatter":
            logger.info(f'plotting scatterplot\nfor {feat_1} and {feat_2}')
            fig = plt.figure(figsize=(8, 8))
            if isinstance(group, bool):
                assert _comb_df[feat_1].shape[0] == _comb_df[feat_2].shape[0], "Shape of feat_1 and feat_2 dont match"
                plt.scatter(
                    _comb_df[feat_1], 
                    _comb_df[feat_2],
                    )
                plt.title(f'Scatter of {feat_1} by {feat_2}')
            else:
                for grp in group_color_dict.keys():
                    #indexes for the group of eval. use that for indexing the feat_2 array
                    idxmask = np.where(_comb_df[hue_col]==grp)[0]
                    if hue_col == self.target.name:
                        plt.scatter(
                            _comb_df.iloc[idxmask, cur_col_idx.index(feat_1)], 
                            _comb_df.iloc[idxmask, cur_col_idx.index(feat_2)], 
                            c=group_color_dict[grp],
                            label=self.rev_target_dict[grp]
                        )
                    else:
                        plt.scatter(
                            _comb_df.iloc[idxmask, cur_col_idx.index(feat_1)], 
                            _comb_df.iloc[idxmask, cur_col_idx.index(feat_2)], 
                            c=group_color_dict[grp],
                            label=grp
                        )
                title = f"{feat_1}_{feat_2}_by_{hue_col}"
                plt.title(f"{title}")
                plt.legend(loc='upper right')

            plt.xlabel(f'{feat_1}')
            plt.ylabel(f'{feat_2}')
            if self.fp_base:
                fig.savefig(Path(f"{self.fp_base + title}.png"), dpi=300)
            timer_error = fig.canvas.new_timer(interval = 3000)
            timer_error.single_shot = True
            timer_cid = timer_error.add_callback(plt.close, fig)
            spacejam = fig.canvas.mpl_connect('key_press_event', onSpacebar)
            timer_error.start()
            plt.show()
            plt.close()

        #If its a histogram
        if plot_type == "histogram":
            fig, (ax_hist, ax_box) = plt.subplots(
                2, 
                sharex=True, 
                figsize=(10, 8), 
                gridspec_kw={"height_ratios": (.85, .15)}
            )
            if not isinstance(group, bool):
                if hue_col == self.target.name:
                    # group_color_dict = {self.rev_target_dict[k]:v for k, v in group_color_dict.items()}
                    sns.histplot(
                        data = _comb_df, 
                        x=feat_1, 
                        ax=ax_hist, 
                        hue=group,
                        palette=group_color_dict,
                        multiple='stack'
                        )
                else:
                    sns.histplot(
                        data = _comb_df, 
                        x=feat_1, 
                        ax=ax_hist, 
                        hue=group,
                        palette=group_color_dict,
                        multiple='stack'
                        )
                logger.info(f'plotting histogram for\n{feat_1} grouped by {hue_col}')
            else:
                sns.histplot(
                    data = _comb_df, 
                    x=feat_1, 
                    ax=ax_hist, 
                    )
                logger.info(f'plotting histogram for\n{feat_1}')

            sns.boxplot(data = _comb_df, x = _comb_df[feat_1], ax=ax_box)
            title = f"Histogram Boxplot of {feat_1}"
            ax_hist.set_title(f"{title}")
            ax_hist.set_xlabel(f'Distribution of {feat_1}')
            ax_hist.set_ylabel('Count')
            ax_box.set_xlabel('')
            if self.fp_base:
                fig.savefig(PurePath(self.fp_base, Path(f"{title}.png")), dpi=300)
            timer_error = fig.canvas.new_timer(interval = 3000)
            timer_error.single_shot = True
            timer_cid = timer_error.add_callback(plt.close, fig)
            spacejam = fig.canvas.mpl_connect('key_press_event', onSpacebar)
            timer_error.start()
            plt.show()
            plt.close()

        #If its a pairplot
        if plot_type == "pairplot":
            #select all columns in groups of 6 (visually any more and the plot becomes too crowded)
                #Problem is that it screws with the logic of the pairplot
            for colnum in range(0, _comb_df.shape[1]-1, 6): 
                cols = _comb_df.iloc[:, colnum:colnum+6].columns.tolist()
                #BUG - Code smells here .  must fix
                if (hue_col in cols) and (hue_col == self.target.name):
                    cols.pop(cols.index(hue_col))
                if not isinstance(group, bool):
                    label_list = sorted(Counter(group).keys())
                    pg = sns.PairGrid(
                        data = _comb_df, 
                        vars = cols,
                        hue = hue_col,
                        hue_order = label_list,
                        palette = group_color_dict,
                        diag_sharey = False, 
                    
                        #old code from pairplot
                        # hue_order=np.unique(group),
                        # kind='reg',
                        # diag_kind='kde',
                        # plot_kws={
                        # 	'color':group_color_dict.values(),
                        # 	'line_kws':{'color':'red'}
                        # 	},
                        # diag_kws={
                        # 	'color':group_color_dict
                        # }
                        # height=10,
                        # aspect=5
                    )
                    pg.map_diag(sns.histplot, multiple="stack", element="step")
                    pg.map_offdiag(sns.scatterplot)
                    #Had to make a custom legend because sns legend being annoying
                    legend_elements = [
                        Line2D([0], [0], 
                         marker = 'o', 
                        color = 'w', 
                        label = val[0],
                        markerfacecolor = val[1], 
                        markersize = 10) for val in group_color_dict.items()
                    ]
                    pg.figure.get_axes()[-1].legend(
                        handles=legend_elements,
                        loc='upper right', 
                        # bbox_to_anchor = (0.98, 0.15),
                        fancybox=True,
                        shadow=True)

                    logger.info(f'plotting pairplot for\n{cols}\ngrouped by {hue_col}')
                else:
                    sns.pairplot(
                        data = _comb_df.iloc[:, colnum:colnum+6], 
                        kind='reg',
                        diag_kind='kde',
                        # diag_kws={'color':'dodgerblue'}
                        plot_kws={'color':'blue','line_kws':{'color':'red'}},
                        # height=10,
                        # aspect=5
                        )
                    logger.info(f'plotting pairplot for\n{cols}')
                if self.fp_base:
                    pg.savefig(PurePath(self.fp_base, Path(f"{title}.png")), dpi=300, bbox_inches='tight')
                timer_error = fig.canvas.new_timer(interval = 3000)
                timer_error.single_shot = True
                timer_cid = timer_error.add_callback(plt.close, fig)
                spacejam = fig.canvas.mpl_connect('key_press_event', onSpacebar)
                timer_error.start()
                plt.show()
                plt.close()

            # ax_hist.set_xlabel(f'Distribution of {feat_1}')
            # ax_hist.set_ylabel('Count')

        #If its a jointplot
        if plot_type == "jointplot":
            if not isinstance(group, bool):
                # mapped_df = _comb_df.copy()
                if hue_col == self.target.name:
                    # group_color_dict = {self.rev_target_dict[v]:k for k, v in group_color_dict.items()}
                    # inv_t_dict = {v: k for k, v in self.target_dict.items()}
                    # mapped_df[hue_col] = mapped_df[hue_col].map(inv_t_dict)
                    # group_color_dict = {inv_t_dict[int(k)]:v for k, v in group_color_dict.items()}
                    label_list = self.target_names
                    hue_target = _comb_df[hue_col] #.map(self.rev_target_dict)
                else:
                    label_list = sorted(Counter(group).keys())
                    hue_target = _comb_df[hue_col]

                logger.info(f'plotting jointplot for\n{feat_1} and {feat_2}\ngrouped by {hue_col}')
                jplot = sns.jointplot(
                    data = _comb_df,
                    x = feat_1, 
                    y = feat_2,
                    hue = hue_target,
                    kind = 'scatter',
                    hue_order = label_list,
                    palette=group_color_dict,
                    s = 50
                    )
                for label, color in group_color_dict.items():
                    sns.regplot(
                        data = _comb_df.iloc[np.where(hue_target==label)[0], :],
                        ax = jplot.ax_joint,
                        x = feat_1, 
                        y = feat_2,
                        color=color,
                        label=label
                    )

            else:
                logger.info(f'plotting jointplot for\n{feat_1} and {feat_2}')
                jplot = sns.jointplot(
                    data = _comb_df,
                    x = feat_1, 
                    y = feat_2,
                    kind = 'reg',
                    space = 0,
                )
            if feat_1 == "EBV":
                jplot.ax_joint.invert_xaxis()

            title = f"Jointplot for {feat_1} "
            if feat_2:
                title += f"and {feat_2} "
            if isinstance(group, pd.Series):
                title += f"by {group.name} "
            #Set the title and adjust it up a bit
            jplot.figure.suptitle(f"{title}", y=0.98)
            jplot.figure.subplots_adjust(top=0.95)
            if self.fp_base:
                jplot.savefig(PurePath(self.fp_base, Path(f"{title}.png")), dpi=300)

            timer_error = jplot.figure.canvas.new_timer(interval = 3000)
            timer_error.single_shot = True
            timer_cid = timer_error.add_callback(plt.close, jplot.figure)
            spacejam = jplot.figure.canvas.mpl_connect('key_press_event', onSpacebar)
            timer_error.start()
            plt.show()
            plt.close()

#CLASS Feature Engineering
class FeatureEngineering(EDA):
    def __init__(self, eda:object):
        if eda:
            self.data = eda.data
            self.feature_names = eda.feature_names
            self.target = eda.target
            self.target_names = eda.target_names
            self.gpu_devices = eda.gpu_devices
            self.task = eda.task
            self.rev_target_dict = eda.rev_target_dict
            self.fp_base = eda.fp_base
            self.view_models = eda.view_models

        else:
            super().__init__(self)
            EDA.clean_data(self)
        
        """		
        Inputs:
            data = pd dataframe of the feature columns
            target = pd series of the target column
            target_names = 	1D np.array of targets
            feature_names = 1D np.array of column names. 
            data_description = str description of the dataset.(Print with sep="\n")
            file_name = filename for when you want to export results. 
        Args:
            task (str): machine learning task you want to implement
            dataset (dict): sklearns dictionary import of varibles/names/filenames
        """	

    #FUNCTION cos_sim
    def get_cos_sim(
        self, 
        vec1:list, 
        vec2:list, 
        impute:str="None",
        dtype:str="float"
                ):
        """Calculates cosine similarity of two vectors. 
            -Will dilineate between a text cos sim and 
            numerical cos sim by the dtype argument

        Args:
            vec1 (list): array of first vector. 
            vec2 (list): array of second vector. 
            impute (str): Whether or not to impute the values with mean
            dtype (str): What type of cos sim you want to run

        Returns:
            float: Return the cosine similarity of both vectors
        """
        if dtype == 'str':
            from sklearn.feature_extraction.text import TfidfVectorizer

        else:
            if impute:
                v1mean = np.nanmean(vec1)
                v2mean = np.nanmean(vec2)
                vec1 = np.where(np.isnan(vec1), v1mean, vec1)
                vec2 = np.where(np.isnan(vec2), v2mean, vec2)

            dp = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dp / (norm1 * norm2)

    # #FUNCTION categorical_encoding
    #def categorical_encoding(self, enc:str, feat:str, order:list=None):
    #     """Note, can only drop single columns at a time. I'll 
    #     Eventually make it do multiple

    #     Args:
    #         enc (str): _description_
    #         feat (str): _description_
    #         order (list, optional): _description_. Defaults to None.

    #     Raises:
    #         ValueError: _description_
    #     """		
    #     if isinstance(enc, str):
    #         enc_dict = {
    #             "onehot" :OneHotEncoder(
    #                 categories="auto",
    #                 drop=None,
    #                 sparse_output=False,
    #                 dtype=int,
    #                 handle_unknown="error",
    #                 min_frequency=None,
    #                 max_categories=None)
    #             ,
    #             "ordinal":OrdinalEncoder(
    #                 categories=[order],
    #                 dtype=np.float64,
    #                 # handle_unknown="error",
    #                 min_frequency=None,
    #                 max_categories=None)
    #         }
    #         encoder = enc_dict.get(enc)
    #         if not encoder:
    #             raise ValueError(f"Encoder not loaded, check before continuing")

    #         #Fit and transform the column (Needs to reshaped to 2d for transform)
    #         arr = encoder.fit_transform(self.data[feat].to_numpy().reshape(-1, 1))
    #         if enc == "onehot":
    #             #grab columns 
    #             ndf_cols = encoder.categories_[0].tolist()
    #             ndf_cols = ["oh_" + x for x in ndf_cols]
    #             #Add to dataset
    #             new_df = pd.DataFrame(arr, index = self.data.index, columns=ndf_cols)
    #             #Add colnames to feature_names
    #             self.feature_names.extend(ndf_cols)
            
    #         elif enc == "ordinal":
    #             #Make trans col name
    #             nn = "ord_" + feat
    #             new_df = pd.DataFrame(arr, index = self.data.index, columns=[nn])
    #             self.feature_names.append(nn)

    #         #Add the new col/cols
    #         self.data = pd.concat([self.data, new_df], axis=1)

    #         #Drop said feature of transform from dataset
    #         self.data.drop([feat], axis=1, inplace=True)
    #         self.feature_names.pop(self.feature_names.index(feat))
    #         logger.info(f"Feature: {feat} has been encoded with {encoder.__class__()} ")

    def normalize_subjects(self, colsofinterest:str|list):
        """This function is for normalizing each individual pigs data across columns that have a larger than normal variance.

        Args:
            colsofinterest (str | list): Cols you want to normalize
        """        
        logger.info(f'Columns to normalize {colsofinterest}')
        #if its a string (single column) turn it into a list
        if isinstance(colsofinterest, str):
            columns = [columns]

        #Get the id's
        pigpen = self.data["pig_id"].unique()
        
        #Normalize per pig data. 
        for col in colsofinterest:
            norm_col = f"{col}_delta"
            self.data[norm_col] = np.nan
            for pig in pigpen:
                # Isolate this specific pig's data
                sub_mask = self.data.pig_id == pig
                dual_mask = sub_mask & (self.target == "BL")
                
                #Calc baseline mean
                if dual_mask.any():
                    baseline_m = self.data.loc[dual_mask, col].mean()
                else:
                    #If you can't find a BL mask.  Fall back to the first 10% of
                    #the data to build your baseline.
                    sub_indices = self.data[pig].index
                    fallback_cutoff = max(1, len(sub_indices) // 10)
                    baseline_ind = sub_indices[:fallback_cutoff]
                    baseline_m = self.data.loc[baseline_ind, col].mean()
                
                #Normalize (Current - Baseline) / absolute(Baseline)
                vals = self.data.loc[sub_mask, col]
                self.data.loc[sub_mask, norm_col] = (vals - baseline_m) / (abs(baseline_m) + 1e-6) #tiny shift so no divide by zero errors

            self.data.drop([col], axis=1, inplace=True)
            self.feature_names.pop(self.feature_names.index(col))
            self.feature_names.append(norm_col)
        super().sum_stats(self.feature_names[4:], "Normalized Features")
        logger.info(f'Columns normalized {[x for x in self.feature_names[4:] if "_delta" in x]}')

    #FUNCTION engineer
    def engineer(self, features:list, transform:bool, display:bool, trans:str):
        """Feature Engineering function.  This function allows you to explore 
        individual column transformations.  

        Args:
            features (list): list of str (or one str) features you want to transform
            transform (bool): Whether you want to transform the column and drop the original column
            display (bool): boolean of if logger should show the transform
            trans (str):  What type of transformation you want

        """
        def transform_col(self, feature:str, trans:str):
            if trans == "log":
                tran_col = np.log(self.data[feature])

            elif trans == "recip":
                tran_col = np.reciprocal(self.data[feature])

            elif trans == "sqrt":
                tran_col = np.sqrt(self.data[feature])

            #FIXME
            #removing for short term.  Getting wierd errors. 
            # elif trans == "exp":
            # 	tran_col = np.exp(self.data[feature])

            elif trans == "BoxC":
                tran_col = boxcox(self.data[feature])
                tran_col = pd.Series(tran_col[0])

            elif trans == "YeoJ":
                tran_col = yeojohnson(self.data[feature])
                tran_col = pd.Series(tran_col[0])
        
            return tran_col
        
        def probability_plot(self, col_name:str, trans_name:str):
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize = (10, 8))
            probplot(
                self.data[col_name], 
                dist=norm,
                plot=ax1
            )
            probplot(
                self.data[trans_name], 
                dist=norm,
                plot=ax2
            )
            ax1.set_title(f'Probability plot\n{col_name}')
            ax2.set_title(f'Probability plot\n{trans_name}')
            plt.show()
            plt.close()

        target = False
        #If its a single feature, it will come through as a string
        if isinstance(features, str):
            #If single column.  Repeat without the list comp
            #FIXME - Find a cleaner way to do this.  Really don't like 
            #repeating code more than once.  
            feature = features
            trans_col = transform_col(self, feature, trans)
            trans_name = trans + "_" + feature 
            self.data[trans_name] = trans_col.values

            if not transform and display:
                #we don't want to store the data, but we'd like to look at
                # it. Calls EDA histogram plot from EDA class and plots
                #then drops the column out
                logger.info(f'Distribution before {trans} transform\nfor {feature}')
                super().eda_plot("histogram", feature, False, target)
                logger.info(f'Distribution after {trans} transform\nfor {feature}')
                super().eda_plot("histogram", trans_name, False, target)
                logger.info(f'Probability plot for\n{feature}\nbefore-after\n{trans} transformation')
                probability_plot(self, feature, trans_name)
                
                self.data.drop(columns=trans_name, inplace=True)
                
            elif transform:
                #If we want charts, show them.  If we're running a ton of
                #models and don't want to see it everytime, set to false
                if display:
                    logger.info(f'Distribution before transform for {feature}')
                    super().eda_plot("histogram", feature, False, target)
                    logger.info(f'Distribution after transform for {feature}')
                    super().eda_plot("histogram", trans_name, False, target)
                    logger.info(f'Probability plot for\n{feature}\nbefore-after\n{trans} transformation')
                    probability_plot(self, feature, trans_name)

                self.data.drop(columns=feature, inplace=True)
                self.feature_names.pop(self.feature_names.index(feature))
                self.feature_names.append(trans_name)

        # if its a list, then it will transform each feature
        elif isinstance(features, list):
            for feature in features:
                trans_col = transform_col(self, feature, trans)
                trans_name = trans + "_" + feature 
                self.data[trans_name] = trans_col.values

                if not transform and display:
                    #we don't want to store the data, but we'd like to look at
                    # it. Calls EDA histogram plot from EDA class and plots
                    #then drops the column out
                    logger.info(f'Distribution before transform for {feature}')
                    super().eda_plot("histogram", feature, False, target)
                    logger.info(f'Distribution after transform for {feature}')
                    super().eda_plot("histogram", trans_name, False, target)
                    logger.info(f'Probability plot for\n{feature}\nbefore-after\n{trans} transformation')
                    probability_plot(self, feature, trans_name)

                    self.data.drop(columns=trans_name, inplace=True)
                    
                elif transform:
                    #If we want charts, show them.  If we're running a ton of
                    #models and don't want to see it everytime, set to false
                    if display:
                        logger.info(f'Distribution before transform for {feature}')
                        super().eda_plot("histogram", feature, False, target)
                        logger.info(f'Distribution after transform for {feature}')
                        super().eda_plot("histogram", trans_name, False, target)
                        logger.info(f'Probability plot for\n{feature}\nbefore-after\n{trans} transformation')
                        probability_plot(self, feature, trans_name)

                    self.data.drop(columns=feature, inplace=True)
                    self.feature_names.pop(self.feature_names.index(feature))
                    self.feature_names.append(trans_name)

        else:
            logger.warning(f'{trans} not found in transformers. Check and try again')

#CLASS DataPrep
class DataPrep(object):
    def __init__(self, features:list, scaler:str=False, cross_val:str=False, engin:object=False):  
        """This is the initalization in between feature engineering and modeltraining.
        
        Logic:
            1. Loads empty dict's for _performance, _predictions, and _models
            2. if the feature engineering object (engin) is passed into the intialization
               constructer, It will pull the features you've generated from there. 
            3. if it is submitted without an object, the data will inherit all data / dataset
               description data form the EDA class.  Allowing flexibility in modeling with
               and without engineered features. 

        Args:
            features (list):List of features you want to model
            scaler (str, optional):Type of scaler you want used. Defaults to False
            cross_val (str, optional):Cross validation scheme you want.  Defaults to False
            engin (object, optional):Feature Engineering object. Defaults to False.
        """		
        self._performance = {}
        self._predictions = {}
        self._models = {}
        self._traind = {}
        self.scaled = False
        self.scaler = scaler
        self.cross_val = cross_val
        
        if engin:
            self.data = engin.data[features]
            self.groups = engin.data["pig_id"].to_numpy() # np.unique(engin.data["pig_id"])
            self.feature_names = features
            self.target = engin.target
            self.target_names = engin.target_names
            self.task = engin.task
            self.gpu_devices = engin.gpu_devices
            self.rev_target_dict = engin.rev_target_dict
            self.fp_base = engin.fp_base
            self.view_models = engin.view_models

        else:
            EDA.__init__(self) 
            EDA.clean_data(self)
            self.data = self.data[features]
            self.feature_names = features

        logger.info(f"Modeling task: {self.task}")
        logger.info(f'Dataset features:{self.feature_names}')
        logger.info(f'Dataset target:{self.target.name}')
        logger.info(f'Dataset Shape:{self.data.shape}')
        logger.info(f'Target shape:{self.target.shape}')
        if self.cross_val:
            logger.info(f'Cross Validation:{self.cross_val}')

    #FUNCTION dataprep
    def data_prep(
            self, 
            model_name:str, 
            split:float,
            model_category:str=None, 
            category_value:str=None
        ):
        """Prepares the DataPrep object to accept model parameters and categories
        Logic:
            1. Sets the split
            2. Sets the empty dictionaries for the _models, _predictions, _performance
            3. Sets the X and y for features and target respectively
            4. Scales the data if the algorithm calls for it.
            5. Performs the split of test and train datasets

        Args:
            model_name (str): abbreviated name of the model
            split (float): What test train split % that you want. (input = decimal %)
            scale (str): What scaler to use. 
            cross_val (str): Cross validation scheme
            model_category (str, optional): Grouping (categorical) of model target you'd like. Defaults to None.
            category_value (any, optional): What value you're targeting. Defaults to None.
        """		
          
        self.split = split
        self.model_category = {}
        self._performance[model_name] = {}
        self._predictions[model_name] = {}
        self._models[model_name] = {}
        self._traind[model_name] = {}

        if model_category != None and category_value != None:	
            self.model_category[model_name] = model_category
            self.category_value[model_name] = category_value

            #If the category's doesn't exist in the model results.  Add them. 
            if category_value not in self._predictions: 
                self._predictions[model_name][category_value]= {}
            if category_value not in self._performance:
                self._performance[model_name][category_value] = {}
            if category_value not in self._models:
                self._models[model_name][category_value] = {}

            self.data_cat = self.data[self.data[model_category] == category_value]
            self._traind[model_name]["X"] = self.data_cat[self.feature_names]
            self._traind[model_name]["y"] = self.data_cat[self.target]
            
        else:
            self.category_value = None
            self._traind[model_name]["X"] = self.data[self.feature_names].to_numpy()
            if self.task == "classification":
                switch_dict = {y:x for x, y in self.rev_target_dict.items()}
                self._traind[model_name]["y"] = self.target.map(switch_dict).to_numpy()
            else:
                self._traind[model_name]["y"] = self.target

        #FUNCTION scalers
        #Models that don't need scaling
            #Tree-based algo's
            #Lda, NB 

        #Tips on choosing Scalers below
        #StandardScaler(with_mean=True)
            #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
            #Assumes normal distribution.  If not, needs a transformation to
            #a normal dist then, standardscaler.
            #sensitive to outliers. 
        
        #MinMaxScaler(feature_range=(0, 1))
            # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
            # sensitive to outliers. 
            
        #RobustScaler(quantile_range=(0.25, 0.75), with_scaling=True)
            # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler
            # Use this one if you've got outliers that you can't remove. (or alot of them)
            # Scales by quantile ranges

        #QuantileTransformer()
            # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#
            # Scales by transforming to a normal or uniform distribution
            # Nonlinear transformation, so it could distort correlations. 
            # Also known as a rankscaler. 

        #PowerTransformer()
            #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html#sklearn.preprocessing.PowerTransformer
            #parametric, monotonic transformation to fit a Gaussian Distribution
            #finds optimal scaling, for stabalizing variance and skewness. 
            #Supports box-cox (strictly positive) and yeo-johnson transforms (pos and neg values)

        # MEAS Test Train Split
        if self.cross_val in ["groupkfold", "leaveonegroupout", "groupshuffle"]:
            # Test train split Group-aware outer split
            gss = GroupShuffleSplit(n_splits=1, test_size=split, random_state=42)
            
            # Get the indices for train and test based on groups
            train_idx, test_idx = next(gss.split(
                self._traind[model_name]["X"], 
                self._traind[model_name]["y"], 
                groups=self.groups)
            )

            # Assign the training/testing datasets
            self._traind[model_name]["X_train"] = self._traind[model_name]["X"][train_idx]
            self._traind[model_name]["X_test"] = self._traind[model_name]["X"][test_idx]
            self._traind[model_name]["y_train"] = self._traind[model_name]["y"][train_idx]
            self._traind[model_name]["y_test"] = self._traind[model_name]["y"][test_idx]
            
            # Save the training groups so CV knows which pig is which during the inner loop
            self._traind[model_name]["groups_train"] = self.groups[train_idx]
        else:
            # If its not group validation normal test split
            X_train, X_test, y_train, y_test = train_test_split(
                self._traind[model_name]["X"], 
                self._traind[model_name]["y"], 
                random_state=42, test_size=split
            )
            self._traind[model_name]["X_train"] = X_train 
            self._traind[model_name]["y_train"] = y_train 
            self._traind[model_name]["X_test"] = X_test
            self._traind[model_name]["y_test"] = y_test

        # Scale the Data 
        if isinstance(self.scaler, str):
            scaler_dict = {
                "r_scale":RobustScaler(quantile_range=(0.25, 0.75), with_scaling=True),
                "s_scale":StandardScaler(), 
                "m_scale":MinMaxScaler(feature_range=(0, 1)),
                "p_scale":PowerTransformer(method='yeo-johnson', standardize=True),
                "q_scale":QuantileTransformer(output_distribution='normal', random_state=42)
            }
            scaler = scaler_dict.get(self.scaler)
            if not scaler:
                raise ValueError(f"Scaler not loaded, check before continuing")
            
            #Fit/transform the scaler only on the training data
            self._traind[model_name]["X_train"] = scaler.fit_transform(self._traind[model_name]["X_train"])
            
            # Apply the scaling parameters to the test set (DO NOT FIT AGAIN)
            self._traind[model_name]["X_test"] = scaler.transform(self._traind[model_name]["X_test"])
            
            # (Optional) If you need the full 'X' array scaled for other reasons, scale it now 
            # using the rules learned from the training set, though usually X_train/X_test is enough.
            # self._traind[model_name]["X"] = scaler.transform(self._traind[model_name]["X"])
            
            self.scaled = True
            logger.info(f"{model_name}'s data has been scaled with {scaler.__class__.__name__}")

#CLASS Model Training
class ModelTraining(object):
    def __init__(self, dataprep):
        """This is the initalization for modeltraining.  It will inherit objects from the 
        
        Logic:
            1. Loads empty dict's for _performance, _predictions, and _models
            2. if the feature engineering object (engin) is passed into the intialization
               constructer, It will pull the features you've generated from there. 
            3. if it is submitted without an object, the data will inherit all
               data from the EDA class.  Allowing flexibility in modeling with
               and without engineered features. 

            Note:  This doesn't inherit automatically from the DataPrep class because you might
            create the DataPrep class with / or without engineered data.  If the DataPrep class is 
            initialized when empty, it will resort to the EDA class for its data
            sourcing, and thereby forefeit any engineered features from being
            input into the model. 
            
        Args:
            dataprep (object): dataprep class of how you want the data set up for modeling. 
        """
        self._models = dataprep._models
        self._predictions = dataprep._predictions
        self._performance = dataprep._performance
        self._traind = dataprep._traind
        self.category_value = dataprep.category_value
        self.feature_names = dataprep.feature_names
        self.split = dataprep.split
        self.groups_train = dataprep.groups
        self.target_names = dataprep.target_names
        self.gpu_devices = dataprep.gpu_devices
        self.task = dataprep.task
        self.cross_val = dataprep.cross_val
        self.fp_base = dataprep.fp_base
        self.CV_func = None
        self.view_models = dataprep.view_models
        self.class_weights = {}

        #MEAS Model params
        self._model_params = {
            "rfc":{
                "model_name":"RandomForestClassifier  ",
                "model_type":"classification",
                "scoring_metric":"accuracy",
                #link to params
                #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
                "base_params":{
                    "n_estimators":100,                 #int | 100		
                    "criterion":"gini",                 #str | gini
                    "max_depth":None,                   #int
                    "min_samples_split":2,              #int | 2
                    "min_samples_leaf":1,               #int | 1
                    "min_weight_fraction_leaf":0.0,     #float | 0.0
                    "max_features":10,                  #str | "sqft"
                    "max_leaf_nodes":None,              #int | None
                    "min_impurity_decrease":0.0,        #float | 0.0
                    "bootstrap":True,                   #bool | True
                    "n_jobs":None,                      #int | None
                    "random_state":42,                  #int | Answer to everything in the universe
                    "warm_start":False,                 #bool | False
                    "class_weight":"balanced"            #Treat target as ordinal
                },
                "init_params":{
                    "n_estimators":100,                 #int | 100		
                    "criterion":"gini",                 #str | gini
                    "max_depth":None,                   #int
                    "min_samples_split":2,              #int | 2
                    "min_samples_leaf":1,               #int | 1
                    "min_weight_fraction_leaf":0.0,     #float | 0.0
                    "max_features":10,                  #str | "sqft"
                    "max_leaf_nodes":None,              #int | None
                    "min_impurity_decrease":0.0,        #float | 0.0
                    "bootstrap":True,                   #bool | True
                    "n_jobs":None,                      #int | None
                    "random_state":42,                  #int | Answer to everything in the universe
                    "warm_start":False,                 #bool | False
                    "class_weight":"balanced"            #Treat target as ordinal
                },
                "grid_srch_params":{
                    "n_estimators":range(5, 200, 10),
                    "criterion":["gini", "entropy", "log_loss"],
                    "min_samples_split":range(5, 50),            
                    "min_samples_leaf":range(5, 50),             
                    # "max_features":["sqrt", "log2", None]
                }
            },
            "kneigh":{
                "model_name":"KNeighborsClassifier  ", 
                "model_type":"classification",
                "scoring_metric":"accuracy",
                #link to params
                #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
                "base_params":{
                    "n_neighbors":5,                   #int | 100		
                    "weights":"uniform",                #str | uniform
                    "algorithm":"auto",                 #str | auto
                    "leaf_size":30,                     #int | 30
                    "p":2,                              #int | 2
                    "metric":"minkowski",               #str | minkowski
                    "metric_params":None,               #dict | None
                    "n_jobs":None,                      #int | None
                    "weights":"distance"                #Treat target as ordinal
                },
                "init_params":{
                    "n_neighbors":5,                   #int | 100		
                    "weights":"uniform",                #str | uniform
                    "algorithm":"auto",                 #str | auto
                    "leaf_size":30,                     #int | 30
                    "p":2,                              #int | 2
                    "metric":"minkowski",               #str | minkowski
                    "metric_params":None,               #dict | None
                    "n_jobs":None,                      #int | None
                    "weights":"distance"                #Treat target as ordinal
                },
                "grid_srch_params":{
                    "n_estimators":range(5, 200, 10),
                    "weights":["uniform", "distance"],
                    "algorithm":["auto", "ball_tree", "kd_tree", "brute"],
                    "leaf_size":range(5, 50),             
                }
            },
            "svm":{
                #Notes. 
                    #
                "model_name":"OneVsRestClassifier(SVM)  ",
                "model_type":"classification",
                "scoring_metric":"accuracy",
                #link to params
                #https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html#sklearn.multiclass.OneVsRestClassifier
                #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
                "base_params":{
                    "C":1.0,						
                    "kernel":"rbf",		       #str
                    "degree":3,                #str
                    "gamma":"scale",
                    "max_iter":1000,
                    "decision_function_shape":"ovr",
                    "random_state":42,
                    "class_weight":"balanced"           #Treat target as ordinal
                },
                "init_params":{
                    "C":0.8,						
                    "kernel":"poly",		       #str
                    "degree":3,                #str
                    "gamma":"scale",
                    "max_iter":10000,
                    "decision_function_shape":"ovr",
                    "random_state":42,
                    "class_weight":"balanced"           #Treat target as ordinal
                },
                "grid_srch_params":{
                    "C":np.arange(0, 1.1, 0.1),
                    "kernel":["linear", "poly", "rbf", "sigmoid", "precomputed"],
                    "n":np.arange(1000, 10000, 500)
                }
            },
            "xgboost":{
                "model_name":"XGBClassifier  ",
                "model_type":"classification",
                "scoring_metric":"accuracy",
                #link to params
                #https://xgboost.readthedocs.io/en/stable/parameter.html
                "base_params":{
                    "booster":"gbtree",
                    "device":"cpu",
                    "gamma":0,
                    "objective":"multi:softmax",
                    "max_depth":6,
                    "learning_rate": 0.3,
                    "num_class":5,
                },
                "init_params":{
                    "booster":"gbtree",
                    "device":"cpu",
                    "gamma":0,
                    "objective":"multi:softmax",
                    "max_depth":6,
                    "learning_rate": 0.3,
                    "num_class":5,
                },
                "grid_srch_params":{
                    "learning_rate":np.arange(0, 1.1, 0.1),
                    # "min_child_weight": np.arange(0, 10),
                    "gamma": np.arange(0, 5, 0.5),
                    # "subsample": np.arange(0.6, 1.0, 0.2),
                    # "colsample_bytree": np.arange(0.6, 1.0, 0.2),
                    "max_depth": np.arange(0, 10, 1),
                    "n_estimators": np.arange(0, 500, 50),      # number of trees
                }
            }
        }

    #FUNCTION get_data
    def get_data(self, model_name:str):
        """Unpacks training and test data

        Args:
            model_name (str): Name of model
        """		
        self.X_train = self._traind[model_name]["X_train"]
        self.X_test = self._traind[model_name]["X_test"]
        self.y_train = self._traind[model_name]["y_train"]
        self.y_test = self._traind[model_name]["y_test"]
        self.X = self._traind[model_name]["X"]
        self.y = self._traind[model_name]["y"]
        if self.cross_val in ["groupkfold", "leaveonegroupout", "groupshuffle"]:
            self.groups_train = self._traind[model_name].get("groups_train", None)

    #FUNCTION Load Model
    def load_model(self, model_name:str):
        """Loads model with initial parameters

        Args:
            model_name (str): _description_

        Returns:
            model: Model ready for training
        """			
        params = self._model_params[model_name]['init_params']
        ####################  classification Models ##################### 
        match model_name:
            case 'svm':
                kernel = SVC(**params)
                model = OneVsRestClassifier(
                    estimator=kernel,
                    n_jobs=-1
                )
                return model
            case 'rfc':
                return RandomForestClassifier(**params)
            case 'xgboost':
                #Due to xgboost not having a class parameter.  We have to save it and feed it into the fit function.... THANKS
                self.class_weights = compute_sample_weight("balanced", y=self._traind[model_name]["y_train"])
                return XGBClassifier(**params)
            case 'kneigh':
                return KNeighborsClassifier(**params)

    #FUNCTION models fit
    @log_time
    def fit(self, model_name:str):
        """This module handles the fit functions for each of the sklearn models. 
        Logic:\n
            1. Extracts model parameters from _model_params dictionary.\n
            2. Unpacks said dictionary, into the model being run.\n

        Args:
            model_name (str): abbreviated name of the model to run
            
        """
        #MEAS Model training \ Param loading
        ####################  Model Load  ##############################		
        self.model = ModelTraining.load_model(self, model_name)

        ####################  Fitting  ##############################
        logger.info(f'{model_name}: fitting model')
        
        #For super fun spinner action in your terminal.
        progress = Progress(
            SpinnerColumn(
                spinner_name="pong",
                speed = 1.2, 
                finished_text="fit complete in",
            ),
            "time elapsed:",
            TimeElapsedColumn(),
        )

        with progress:
            task = progress.add_task("Fitting Model", total=1)
            if model_name == "xgboost":
                self.model.fit(self.X_train, self.y_train, sample_weight=self.class_weights)
            else:    
                self.model.fit(self.X_train, self.y_train)
            progress.update(task, advance=1)

        # self.model.fit(self.X_train, self.y_train)
        
        if self.category_value != None:
            self._models[model_name][self.category_value] = self.model
        else:
            self._models[model_name] = self.model
        logger.info(f"fit complete for {model_name}")

    #FUNCTION predict
    @log_time
    def predict(self, model_name):
        """Fits the model in question
        Note:
            Will add an additional key for category if that is desired. 

        Args:
            model_name (str): abbreviated name of the model
        """		
        logger.info(f'{model_name}: making predictions')

        if self.category_value != None:
            self._predictions[model_name][self.category_value] = self._models[model_name][self.category_value].predict(self.X_test)
        else:
            self._predictions[model_name] = self._models[model_name].predict(self.X_test)
        
    #FUNCTION validate
    def validate(self, model_name):
        """This module handles the model metrics and which to run.   It
        summarizes model metrics and outputs them into a rich table as those
        look better than plain ol print statements.

        Logic:
            1. Pull out the parameters of the model it just ran. 
            2. Create a rich table for results storage. 
            3. Identify which task route to take for which metric to run.
                if Regressor
                    - Calculate desired metric
                    - Provide Model summary

                if Classification
                    - Generate a classification report, and confusion matrix.
                    - Format and print model results as a rich table.

        Args:
            model_name (str): Name of model
        """
        def load_cross_val(cv_name:str):
                cv_validators = {
                    "kfold"           :KFold(n_splits=10, shuffle=True, random_state=42),
                    "stratkfold"      :StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    # "leaveoneout"   :LeaveOneOut(),
                    "groupkfold"      :GroupKFold(n_splits=min(5, len(np.unique(self.groups_train)))),
                    "leaveonegroupout":LeaveOneGroupOut(),
                    "groupshuffle"    :GroupShuffleSplit(n_splits=5, test_size=0.25, random_state=42),
                    "shuffle"         :ShuffleSplit(n_splits=10, test_size=0.25, train_size=0.5, random_state=42),
                    "stratshuffle"    :StratifiedShuffleSplit(n_splits=10, test_size=0.25, train_size=0.5, random_state=42)
                }
                return cv_validators[cv_name]

        #FUNCTION custom_confusion_matrix
        def custom_confusion_matrix(y_true, y_pred, display_labels=None, model_name="Model", positive_class_first=False):
            """
            Plots a custom confusion matrix with percentages and counts. 
            Automatically handles both binary and multi-class classification.
            """
            def onSpacebar(event):
                """When plotting, hit the spacebar if keep the chart from closing. 

                Args:
                    event (_type_): accepts the key event.  In this case its looking for the spacebar.
                """	
                if event.key == " ": 
                    timer_error.stop()
                    timer_error.remove_callback(timer_cid)
                    logger.warning(f'Timer stopped')

            # Generate base confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            n_classes = cm.shape[0]
            
            # Determine if binary or multiclass
            is_binary = n_classes == 2
            
            # Format counts and percentages for all cells
            counts = [f"{x:0.0f}" for x in cm.flatten()]
            perc = [f"{x / np.sum(cm):0.2%}" for x in cm.flatten()]
            
            # Handle label formatting based on classification type
            if is_binary:
                if positive_class_first:
                    # Flipped matrix: [[TP, FN], [FP, TN]]
                    cm = np.flip(cm)
                    cats = ["True +", "False -", "False +", "True -"]
                    if display_labels is not None:
                        display_labels = display_labels[::-1]
                else:
                    # Standard sklearn matrix: [[TN, FP], [FN, TP]]
                    cats = ["True -", "False +", "False -", "True +"]
                    
                # Combine Categories, Counts, and Percentages
                labs = [f"{cat}\n{count}\n{p}" for cat, count, p in zip(cats, counts, perc)]
            else:
                # For multi-class, we just use Counts and Percentages (No TP/TN/FP/FN categories)
                labs = [f"{count}\n{p}" for count, p in zip(counts, perc)]

            # reshape labels to match the matrix dimensions
            labs = np.asarray(labs).reshape(n_classes, n_classes)
            
            # resize the figure based on the number of classes
            fig_size = max(6, n_classes * 1.5)
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            
            # Create the Heatmap
            sns.heatmap(
                cm, 
                ax=ax, 
                annot=labs, 
                fmt="", 
                annot_kws={'fontsize': 14},
                cmap='Blues',
                xticklabels=display_labels if display_labels is not None else "auto",
                yticklabels=display_labels if display_labels is not None else "auto"
            )
            
            # Set titles and labels
            plt.title(f"{model_name.upper()} Confusion Matrix", fontsize=20, pad=20)
            plt.xlabel("Predicted Label", fontsize=14)
            plt.ylabel("True Label", fontsize=14)
            
            # Rotate tick marks so multi-class labels don't overlap
            plt.xticks(rotation=45, ha="right", fontsize=12)
            plt.yticks(rotation=0, fontsize=12)
            
            plt.tight_layout()
            if self.view_models:
                if self.fp_base:
                    fig.savefig(PurePath(self.fp_base, Path(f"{model_name}_CM.png")), dpi=300)
                timer_error = fig.figure.canvas.new_timer(interval = 3000)
                timer_error.single_shot = True
                timer_cid = timer_error.add_callback(plt.close, fig.figure)
                spacejam = fig.figure.canvas.mpl_connect('key_press_event', onSpacebar)
                timer_error.start()
                plt.show()
                plt.close()

        #FUNCTION classification_report
        def make_cls_report(y_true, y_pred, display_labels=None):
            report = classification_report(
                y_true, 
                y_pred, 
                labels = np.unique(y_pred), 
                target_names=display_labels,
                zero_division=False
            )
            #BUG - Unrepresented classes throw a div by zero error.  Look into this later. 
            body = report.split("\n\n")
            header = body[0]
            rows = [body[x].split("\n") for x in range(1, len(body))]
            rows_flat = list(chain(*rows))
            table = Table(title = f'Classification report', header_style="Blue on white")
            table.add_column(header, justify='center', style='white on blue')
            for row in rows_flat:
                table.add_row(row)
            console.print(table)

        #FUNCTION ROC_AUC
        def roc_auc_curves(self, model:str):
            def onSpacebar(event):
                """When plotting, hit the spacebar if keep the chart from closing. 

                Args:
                    event (_type_): accepts the key event.  In this case its looking for the spacebar.
                """	
                if event.key == " ": 
                    timer_error.stop()
                    timer_error.remove_callback(timer_cid)
                    logger.warning(f'Timer stopped')

            #Get the size of the target responses. (how many are there)
            num_groups = np.unique(self._traind[model]["y"]).size
            #Get the colormap
            color_cmap = plt.cm.get_cmap('Paired', num_groups) 
            #Generate a hex code for the color.  
            color_str = [mpl.colors.rgb2hex(color_cmap(i)) for i in range(color_cmap.N)]
            #Now make a dictionary of the activities and their hex color.
            colcyc = color_str[:num_groups]
            cycol = cycle(colcyc)
            group_color_dict = {x:next(cycol) for x in self.target_names}

            #Using one vs rest scheme for Aucroc
            test_prob = self._models[model].predict_proba(self._traind[model]["X_test"])
            fpr, tpr, auc_s = {}, {}, {}
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
            for cls in range(len(self.target_names)):  #BUG - Possible hardcoding here to a numbered target class. 
                fpr[cls], tpr[cls], _ = roc_curve(self._traind[model]["y_test"], test_prob[:, cls], pos_label=cls)
                auc_s[cls] = auc(fpr[cls], tpr[cls])
                # roc_auc_s[cls] = roc_auc_score(
                # 	y_true = self._traind[model]["y_test"],
                # 	y_score = test_prob[:, cls],
                # 	average = "macro",
                # 	multi_class = "ovr",
                # 	labels = cls
                # )
                plt.plot(
                    fpr[cls], 
                    tpr[cls], 
                    linestyle="--",
                    color = group_color_dict[self.target_names[cls]],
                    label = f"ROC curve for {self.target_names[cls]} vs rest AUC:{auc_s[cls]:.2f}" 
                )

            plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
            plt.title(f'{model.upper()} ROC Curve')
            plt.legend(loc="lower right")
            timer_error = fig.figure.canvas.new_timer(interval = 3000)
            timer_error.single_shot = True
            timer_cid = timer_error.add_callback(plt.close, fig.figure)
            spacejam = fig.figure.canvas.mpl_connect('key_press_event', onSpacebar)
            timer_error.start()
            plt.show()
            plt.close()

        #FUNCTION cv_roc_auc_curves
        def cv_roc_auc_curves(model_name:str):
            def onSpacebar(event):
                """When plotting, hit the spacebar if keep the chart from closing. 

                Args:
                    event (_type_): accepts the key event.  In this case its looking for the spacebar.
                """	
                if event.key == " ": 
                    timer_error.stop()
                    timer_error.remove_callback(timer_cid)
                    logger.warning(f'Timer stopped')

            logger.info(f"Generating CV ROC AUC curves for {model_name}...")
            
            # Re-initialize model and CV splitter
            freshmodel = ModelTraining.load_model(self, model_name)
            CV_func = load_cross_val(self.cross_val) 
            
            # Set up the common X-axis (False Positive Rate) for interpolation
            mean_fpr = np.linspace(0, 1, 100)
            
            # Setup dictionaries to hold True Positive Rates and AUCs across folds
            n_classes = len(self.target_names)
            tprs = {i: [] for i in range(n_classes)}
            aucs = {i: [] for i in range(n_classes)}
            
            # Unpack data
            X_data = self.X_train
            y_data = self.y_train
            groups_data = self.groups_train
            
            # Iterate through the CV folds
            #BUG - had to hardcode the group here 
            for fold, (train_ix, test_ix) in enumerate(CV_func.split(X_data, y_data, groups=groups_data)):
                # Fit the model on the training fold
                freshmodel.fit(X_data[train_ix], y_data[train_ix])
                # Predict probabilities on the testing fold
                probas_ = freshmodel.predict_proba(X_data[test_ix])
                
                # Compute ROC for each class (One-vs-Rest)
                for cls in range(n_classes):
                    # Binarize the target for the current class
                    y_test_bin = (y_data[test_ix] == cls).astype(int) 
                    
                    # Check if there are actually any positive samples for this class in this fold
                    if np.sum(y_test_bin) == 0:
                        logger.warning(f"Fold {fold}: Class '{self.target_names[cls]}' missing from test set. Skipping ROC for this fold.")
                        continue # Skip to the next class
                    
                    # Calculate ROC metrics
                    fpr, tpr, _ = roc_curve(y_test_bin, probas_[:, cls])
                    
                    # Interpolate the TPR to our common mean_fpr scale
                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0 # Force curve to start at 0
                    
                    tprs[cls].append(interp_tpr)
                    aucs[cls].append(auc(fpr, tpr))
                    
            # Set up the plot
            fig, ax = plt.subplots(figsize=(10, 8))
            color_cmap = plt.get_cmap('Paired', n_classes) 
            
            # Plot Mean ROC and standard deviation for each class
            for cls in range(n_classes):
                # Ensure we actually have data to plot for this class
                if len(tprs[cls]) == 0:
                    logger.warning(f"Class '{self.target_names[cls]}' had no test samples across any folds. Cannot plot.")
                    continue
                
                # Calculate mean TPR and AUC
                mean_tpr = np.mean(tprs[cls], axis=0)
                mean_tpr[-1] = 1.0 # Force curve to end at 1
                mean_auc = auc(mean_fpr, mean_tpr)
                std_auc = np.std(aucs[cls])
                
                # Plot the solid mean line
                ax.plot(
                    mean_fpr, mean_tpr, 
                    color=color_cmap(cls), 
                    lw=2, 
                    label=f"Mean ROC {self.target_names[cls]} (AUC = {mean_auc:.2f} \u00B1 {std_auc:.2f})"
                )
                
                # Plot the shaded standard deviation band
                std_tpr = np.std(tprs[cls], axis=0)
                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                ax.fill_between(
                    mean_fpr, tprs_lower, tprs_upper, 
                    color=color_cmap(cls), alpha=0.1
                )

            # Add chance line and formatting
            ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black", label="Chance (AUC = 0.5)")
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.set_title(f"Cross-Validated ROC-AUC (OvR) - {model_name.upper()}", fontsize=14)
            ax.legend(loc="lower right")
            
            plt.tight_layout()
            if self.view_models:
                if self.fp_base:
                    fig.savefig(PurePath(self.fp_base, Path(f"{model_name}_AUCROC.png")), dpi=300)
                timer_error = fig.figure.canvas.new_timer(interval = 3000)
                timer_error.single_shot = True
                timer_cid = timer_error.add_callback(plt.close, fig.figure)
                spacejam = fig.figure.canvas.mpl_connect('key_press_event', onSpacebar)
                timer_error.start()
                plt.show()
                plt.close()

        #FUNCTION classification summary
        def classification_summary(model_name:str, y_pred:np.array, cv_class:str=False):
        #######################Confusion Matrix and classification report##########################
            labels = self.target_names
            no_proba = ["svm", ""]

            #Call confusion matrix
            logger.info(f'{model_name} confusion matrix')
            custom_confusion_matrix(self.y_test, y_pred, display_labels=labels, model_name=model_name)
            #Call classification report
            logger.info(f'{model_name} classification report')
            make_cls_report(self.y_test, y_pred, display_labels=labels)
            
            #Generate ROC curves for non CV runs. 
            if not cv_class:
                if model_name not in no_proba:
                    roc_auc_curves(self, model_name)
            else:
                if model_name not in no_proba:
                    cv_roc_auc_curves(model_name)

        #FUNCTION No Crossval
        def no_cv_scoring(y_pred:np.array, cat_bool:bool, table)->float:
            #I'm not sure why i'm keeping no cross validation as an option, but here we are. 
            scoring_dict = {
                #regression
                # "rsme"    : MSE(self.y_test, y_pred, squared=False),
                # "mse"     : MSE(self.y_test, y_pred),
                # "mae"     : MAE(self.y_test, y_pred),
                # "rsquared": RSQUARED(self.y_test, y_pred),
                #classification

                "accuracy": ACC_SC(self.y_test, y_pred),
                "logloss" : LOG_LOSS(self.y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(self.y_test, y_pred),
                "f1_weighted": f1_score(self.y_test, y_pred, average='weighted'),
                "f1_macro": f1_score(self.y_test, y_pred, average='macro'),
                "mcc": matthews_corrcoef(self.y_test, y_pred)
                #clustering
            }
            if self.task == "regression":
                logger.info(f'{model_name}: Calculating {metric} for {self.task}')
                if cat_bool:
                    self._performance[self.category_value][model_name][metric.upper()] = scoring_dict[metric]
                    scores = self._performance[self.category_value][model_name][metric.upper()]
                    table.add_column(f'{scores:^.2f}', justify="center", style="white on blue")
                    
                else:
                    self._performance[model_name][metric.upper()] = scoring_dict[metric]
                    scores = self._performance[model_name][metric.upper()]
                    table.add_column(f'{scores:^.2f}', justify="center", style="white on blue")

            if self.task == "classification":
                logger.info(f'{model_name}: Calculating {metric} for {self.task}')
                if cat_bool:
                    self._performance[self.category_value][model_name][metric.upper()] = scoring_dict[metric]
                    scores = self._performance[self.category_value][model_name][metric.upper()]
                    table.add_column(f'{scores:^.2%}', justify="center", style="white on blue")
                    
                else:	
                    self._performance[model_name][metric.upper()] = scoring_dict[metric]
                    scores = self._performance[model_name][metric.upper()]
                    table.add_column(f'{scores:^.2%}', justify="center", style="white on blue")
                    
                classification_summary(model_name, y_pred)
                return scores
            
            else:

                return None

        #FUNCTION With Cross Validation
        def cv_scoring(y_pred:np.array, cat_bool:bool, model:str, table)->float:
            """_summary_

            Args:
                y_pred (np.array): _description_
                cat_bool (bool): _description_
                model (str): _description_
                table (_type_): _description_

            Returns:
                float: _description_
            """

            def generate_cv_predictions(freshmodel, CV_func, X_data:np.array, y_data:np.array, groups_data:np.array):#-> Tuple[list, list]
                actual_t = np.array([])
                predicted_t = np.array([])
                #BUG - Had to hardcode groups into the split.  REFACTOR eventually
                for train_ix, test_ix in CV_func.split(X = X_data, y = y_data, groups=groups_data):
                    train_X, train_y, test_X, test_y = X_data[train_ix], y_data[train_ix], X_data[test_ix], y_data[test_ix]
                    freshmodel.fit(train_X, train_y)
                    predicted_labels = freshmodel.predict(test_X)
                    predicted_t = np.append(predicted_t, predicted_labels)
                    actual_t = np.append(actual_t, test_y)

                return predicted_t, actual_t

            #Load a fresh untrained model and score it.
            freshmodel = ModelTraining.load_model(self, model_name)

            #Load Cross Validation
            CV_func = load_cross_val(self.cross_val)

            #Validate
            if self.cross_val in ["groupkfold", "leaveonegroupout", "groupshuffle"]:
                scores = cross_validate(
                    freshmodel, self.X_train, self.y_train, 
                    groups=self.groups_train, cv=CV_func
                )["test_score"]
            else:
                scores = cross_validate(
                    freshmodel, self.X_train, self.y_train, cv=CV_func
                )["test_score"]
            
            #reload model untrained model for cross_validation predictions
            freshmodel = ModelTraining.load_model(self, model_name)

            #Load Cross Validation
            CV_func = load_cross_val(self.cross_val)

            #Generate new predictions based on cross validated data.
            #BUG - Had to hardcode groups into the cv preds.  REFACTOR eventually
            y_pred, y_target = generate_cv_predictions(
                freshmodel, CV_func, self.X_train, self.y_train, self.groups_train
            )

            #Store them in the modeltraining object
            self._predictions[model_name] = y_pred
            self.y_test = y_target
        
            #IDEA
                #Do we want a permutation test at the end of cross validation to
                #see if the distributions changed? aka did the model find any
                #real relation to the inputs

                #? Two fold inner and outer CV? Make a custom scorer??

            if cat_bool:
                self._performance[self.category_value][model_name][metric.upper()] = scores
            else:
                self._performance[model_name][metric.upper()] = (scores.mean(), scores.std())

            #Add them to the table. 
            table.add_column(f'{scores.mean():^.2f}', justify="center", style="white on blue")
            
            if self.task == "classification":
                logger.info(f'{model_name}: Calculating model summary')
                classification_summary(model_name, y_pred, True)
                
            return scores.mean()

        ###################### METRIC CENTRAL ##################################################
        metric = self._model_params[model_name]["scoring_metric"]
        #Grab the model parameters used.
        params = {k:v for k, v in self.model.get_params().items()}
        #Make a results table
        table = Table(title = str(self.model.__class__).split(" ")[1].split(".")[-1].rstrip(">'"), header_style="white on blue")
        table.add_column(metric.upper(), justify="right", style="white on blue")

        #Grab predictions
        cat_bool = self.category_value != None
        if cat_bool:
            y_pred = self._predictions[self.category_value][model_name]
        else:
            y_pred = self._predictions[model_name]
        
        if self.cross_val:
            #Call cross validation function
            scores = cv_scoring(y_pred, cat_bool, model_name, table)
            table.add_row("CV:", f"{self.cross_val}", end_section=True)

        else:
            #Call regular holdout scoring function
            scores = no_cv_scoring(y_pred, cat_bool, table)
            table.add_row("Test holdout", f"{self.split:.0%}", end_section=True)
        

        #Add the model parameters to the table
        table.add_row("Params:", "", end_section=True)
        [table.add_row(k, str(v)) for k, v in params.items()]

        #Print them to the console
        logger.info(f'{model_name}: model results')
        console.print(table)

        #TODO.  Add in prec/recall chart as found here. 
        #https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_display_object_visualization.html#sphx-glr-auto-examples-miscellaneous-plot-display-object-visualization-py

    #FUNCTION show_results
    def show_results(self, modellist:list, sort_des:bool=False):
        #Make a results table
        #Structure will be:
        #modelname | metric | score

        table = Table(title = "Model Results Table", header_style="white on blue")
        table.add_column("model name", justify="right", style="white on blue")
        table.add_column("metric", justify="center", style="white on blue")
        table.add_column("score", justify="center", style="white on blue")

        cat_bool = self.category_value != None
        #TODO - cat_bool + Metric units
            #Code in for category selection here too. 
            #Also need to reformat below to not be hardecoded to task. 
                #temp fix for now, but ultimately i'd it to format the metric
                #as it is supposed to be reported. 
        _templist = []
        for model in modellist:
            model_name = str(self._model_params[model]['model_name'])[:-2]
            metric = self._model_params[model]['scoring_metric']
            score = self._performance[model][metric.upper()]
            if self.task == "regression":
                score = f'{score :.2f}'
            elif self.task == "classification":
                if self.cross_val:
                    score, std = score[0], score[1]
                    score = f'Mean: {score:.2%} +/-:{std:.2%}'
                else:
                    score = f'{score:.2%}'
                    
            _templist.append((model_name, metric.upper(), score))
            
        if sort_des:
            _templist = sorted(_templist, key=lambda x: x[2], reverse=True)
        else:
            _templist = sorted(_templist, key=lambda x: x[2], reverse=False)

        [table.add_row(model_name, metric, score) for (model_name, metric, score) in _templist]
        logger.info('model results')
        console.print(table)

    #FUNCTION importance plot
    def plot_feats(self, model:str, features:list, imps:list):
        def onSpacebar(event):
            """When plotting, hit the spacebar if keep the chart from closing. 

            Args:
                event (_type_): accepts the key event.  In this case its looking for the spacebar.
            """	
            if event.key == " ": 
                timer_error.stop()
                timer_error.remove_callback(timer_cid)
                logger.warning(f'Timer stopped')
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (10, 8))
        feat_imp = sorted(zip(features, imps), key=lambda x: -x[1])[:20]
        dfeats = pd.DataFrame(data = feat_imp, columns=["Name", "Imp"])
        barchart = plt.barh(
            y=dfeats["Name"], 
            height=0.8,
            width=dfeats["Imp"],
        )
        ax.invert_yaxis()
        plt.title(f"{model} Top 20 feature importance", fontsize=14)
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature Name")
        if self.view_models:
            if self.fp_base:
                fig.savefig(PurePath(self.fp_base, Path(f"{model}_feat.png")), dpi=300)
            timer_error = fig.canvas.new_timer(interval = 3000)
            timer_error.single_shot = True
            timer_cid = timer_error.add_callback(plt.close, fig)
            spacejam = fig.canvas.mpl_connect('key_press_event', onSpacebar)
            timer_error.start()
            plt.show()

    #FUNCTION importance plot
    def SHAP(self, model:str, features:list):
        """Generates and plots SHAP values to help with model variable influence

        Args:
            model (str): Model to be evaluated
            features (list): Features to analyze
        """        
        def onSpacebar(event):
            """When plotting, hit the spacebar if keep the chart from closing. 

            Args:
                event (_type_): accepts the key event.  In this case its looking for the spacebar.
            """	
            if event.key == " ": 
                timer_error.stop()
                timer_error.remove_callback(timer_cid)
                logger.warning(f'Timer stopped')

        #Variable definition
        plots_to_show = ["bar",] #"waterfall", "violin"
        X_train = self._traind[model]["X_train"]
        fitted_model = self._models[model]

        #Load the trained model into the tree explainer
        tree_explainer = shap.TreeExplainer(fitted_model)
        
        # Generate the Explanation object (modern API)
        # This replaces the need for raw shap_values in most modern plots
        explanation = tree_explainer(X_train, self._predictions[model])
        
        # Fallback raw values (legacy) for plots that haven't migrated yet
        shap_values_raw = tree_explainer.shap_values(X_train, self._predictions[model])

        if "bar" in plots_to_show:
            fig = plt.figure()
            shap.summary_plot(shap_values_raw, X_train, feature_names=features, plot_type="bar", show=False)
            fig.figure.suptitle(f"{model} SHAP Feature Importance (Bar)", y=0.98, size=12)
            fig.figure.subplots_adjust(top=0.95)
            if self.view_models:
                if self.fp_base:
                    fig.savefig(PurePath(self.fp_base, Path(f"{model}_shap_bar.png")), dpi=300)
                timer_error = fig.canvas.new_timer(interval = 3000)
                timer_error.single_shot = True
                timer_cid = timer_error.add_callback(plt.close, fig)
                spacejam = fig.canvas.mpl_connect('key_press_event', onSpacebar)
                timer_error.start()
                plt.show()

        if "violin" in plots_to_show:
            fig = plt.figure()
            # Summary plot still uses raw values for violin type
            shap.summary_plot(shap_values_raw, X_train, feature_names=features, plot_type="violin", color="coolwarm", show=False)
            fig.figure.suptitle(f"{model} Violin of Feature Importance", size=12)
            if self.view_models:
                if self.fp_base:
                    fig.savefig(PurePath(self.fp_base, Path(f"{model}_shap_violin.png")), dpi=300)
                timer_error = fig.canvas.new_timer(interval = 3000)
                timer_error.single_shot = True
                timer_cid = timer_error.add_callback(plt.close, fig)
                spacejam = fig.canvas.mpl_connect('key_press_event', onSpacebar)
                timer_error.start()
                plt.show()

        if "beeswarm" in plots_to_show:
            fig = plt.figure()
            shap.plots.beeswarm(explanation, show=False)
            plt.title(f"{model} SHAP Beeswarm Summary", size=12)
            if self.view_models:
                if self.fp_base:
                    fig.savefig(PurePath(self.fp_base, Path(f"{model}_shap_beeswarm.png")), dpi=300)
                timer_error = fig.canvas.new_timer(interval = 3000)
                timer_error.single_shot = True
                timer_cid = timer_error.add_callback(plt.close, fig)
                spacejam = fig.canvas.mpl_connect('key_press_event', onSpacebar)
                timer_error.start()
                plt.show()

        if "heatmap" in plots_to_show:
            fig = plt.figure()
            shap.plots.heatmap(explanation, show=False)
            plt.title(f"{model} SHAP Heatmap", size=12)
            if self.view_models:
                if self.fp_base:
                    fig.savefig(PurePath(self.fp_base, Path(f"{model}_shap_heatmap.png")), dpi=300)
                timer_error = fig.canvas.new_timer(interval = 3000)
                timer_error.single_shot = True
                timer_cid = timer_error.add_callback(plt.close, fig)
                spacejam = fig.canvas.mpl_connect('key_press_event', onSpacebar)
                timer_error.start()
                plt.show()

        if "scatter" in plots_to_show:
            fig = plt.figure()
            # Defaults to showing the first feature's dependence, colored by the strongest interacting feature
            top_feature = features[0] if features else 0
            shap.plots.scatter(explanation[:, top_feature], color=explanation, show=False)
            plt.title(f"{model} SHAP Scatter/Dependence", size=12)
            if self.view_models:
                if self.fp_base:
                    fig.savefig(PurePath(self.fp_base, Path(f"{model}_shap_scatter.png")), dpi=300)
                timer_error = fig.canvas.new_timer(interval = 3000)
                timer_error.single_shot = True
                timer_cid = timer_error.add_callback(plt.close, fig)
                spacejam = fig.canvas.mpl_connect('key_press_event', onSpacebar)
                timer_error.start()
                plt.show()


        if "waterfall" in plots_to_show:
            try:
                fig = plt.figure()
                # Check if the explanation object is 3D (Multi-class/Multi-output)
                if len(explanation.shape) == 3:
                    # explanation[instance_index, feature_slice, class_index]
                    # We'll plot the explanation for the 1st observation (0), all features (:), and the 1st class (0)
                    target_class = 0 
                    shap.plots.waterfall(explanation[0, :, target_class], show=False)
                    fig.figure.suptitle(f"{model} Waterfall Plot (Instance 0, Class {target_class})", size=12)
                else:
                    # Standard 2D explanation (Binary classification or Regression)
                    shap.plots.waterfall(explanation[0], show=False)
                    fig.figure.suptitle(f"{model} Waterfall Plot (Instance 0)", size=12)
                if self.view_models:                    
                    if self.fp_base:
                        fig.savefig(PurePath(self.fp_base, Path(f"{model}_shap_waterfall.png")), dpi=300)
                    timer_error = fig.canvas.new_timer(interval = 3000)
                    timer_error.single_shot = True
                    timer_cid = timer_error.add_callback(plt.close, fig)
                    spacejam = fig.canvas.mpl_connect('key_press_event', onSpacebar)
                    timer_error.start()
                    plt.show()

                
            except Exception as e:
                logger.warning(f"Waterfall plot failed: {e}")

    #FUNCTION _grid_search
    @log_time
    def _grid_search(self, model_name: str, folds: int):
        from sklearn.model_selection import GridSearchCV
        console.print(f'{model_name} grid search initiated')
        base_clf = self._models[model_name]
        params = self._model_params[model_name]["grid_srch_params"]
        metric = self._model_params[model_name]["scoring_metric"]
        
        # Determine GPU availability and worker count
        num_gpus = len(self.gpu_devices)
        if num_gpus > 0:
            # 1 worker per GPU. You can multiply this (e.g., num_gpus * 2) 
            # if your models are small and fit in VRAM concurrently.
            n_workers = num_gpus 
            # Wrap the classifier so each worker gets its own GPU
            clf = MultiGPU(base_clf, self.gpu_devices)
            # Prefix the param grid keys because of the wrapper
            params = {f"estimator__{k}": v for k, v in params.items()}
        else:
            n_workers = -1 # Fall back to all CPUs if no GPUs
            clf = base_clf
        # Define all the metrics you want to track during the search
        scoring_dict = {
            "accuracy": make_scorer(ACC_SC),
            "balanced_accuracy": make_scorer(balanced_accuracy_score),
            "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0.0),
            "f1_macro": make_scorer(f1_score, average="macro", zero_division=0.0),
            "mcc": make_scorer(matthews_corrcoef)
        }
        
        # --- Determine the correct CV Strategy ---
        group_splitters = ["groupkfold", "leaveonegroupout", "groupshuffle"]
        if self.cross_val == "leaveonegroupout":
            cv_strategy = LeaveOneGroupOut()
        elif self.cross_val == "groupkfold":
            cv_strategy = GroupKFold(n_splits=folds)
        elif self.cross_val == "groupshuffle":
            cv_strategy = GroupShuffleSplit(n_splits=folds, test_size=0.25, random_state=42)
        else:
            cv_strategy = folds # Fall back to standard integer for KFold/StratifiedKFold

        grid = GridSearchCV(
            clf, 
            n_jobs=n_workers, 
            param_grid=params, 
            cv=cv_strategy, 
            scoring=scoring_dict,
            refit=metric
        )
        
        # For super fun spinner action in your terminal.
        progress = Progress(
            SpinnerColumn(
                spinner_name="shark",
                speed=1.2, 
                finished_text="searching parameters",
            ),
            "time elapsed:",
            TimeElapsedColumn(),
        )
        
        with progress:
            task = progress.add_task("Fitting Model", total=1)
            if self.cross_val in group_splitters:
                grid.fit(self.X_train, self.y_train, groups=self.groups_train)
            else:
                grid.fit(self.X_train, self.y_train)
                
            progress.update(task, advance=1)

        logger.info(f"{model_name} best params\n{grid.best_params_}")
        logger.info(f"{model_name} best {metric}: {grid.best_score_:.2%}")
        
        # file saving block
        fp = self.fp_base / "gridresults.txt"
        current_time = time.strftime("%m-%d-%Y %H:%M:%S", time.localtime())
        
        with open(fp, "a") as savef:
            savef.write(f"\n{'='*40}\n")
            savef.write(f"Gridsearch ran on {current_time}\n")
            savef.write(f"Model {model_name} using {self.cross_val} ({grid.cv} folds)\n")
            savef.write(f"Optimized for: {metric}\n")
            
            # Write the best primary score
            savef.write(f"\nBest {metric} Score: {grid.best_score_:.2%}\n")
            
            # --- Log the other metrics for the best model ---
            # grid.best_index_ tells us which hyperparameter combination won
            best_idx = grid.best_index_
            savef.write("Corresponding metrics for this model:\n")
            for metric_name in scoring_dict.keys():
                if metric_name != metric:
                    # Look up the mean test score for this specific metric and parameter index
                    other_score = grid.cv_results_[f'mean_test_{metric_name}'][best_idx]
                    savef.write(f"  - {metric_name}: {other_score:.2%}\n")
            
            # Strip 'estimator__' from the saved best params for readability
            clean_params = {k.replace('estimator__', ''): v for k, v in grid.best_params_.items()}
            savef.write(f"\nParameters:\n{clean_params}\n")

        return grid

#CLASS Multi GPU routing
class MultiGPU(BaseEstimator):
    """
    Wraps an estimator to assign it to a specific GPU based on the joblib worker ID.
    Routes GPU parameters to supported models (XGBoost) and falls back 
    to CPU for standard sklearn models. Features CuPy interception for 
    zero-copy XGBoost data transfers.
    """
    def __init__(self, estimator, gpu_devices):
        self.estimator = estimator
        self.gpu_devices = gpu_devices
        self.gpu_id = None # Track which GPU this worker is using

    def fit(self, X, y, **kwargs):
        # Determine which joblib worker process we are in
        ident = multiprocessing.current_process()._identity
        worker_id = ident[0] - 1 if ident else 0
        
        if self.gpu_devices:
            gpu_idx = worker_id % len(self.gpu_devices)
            self.gpu_id = self.gpu_devices[gpu_idx]
            
            # Handle XGBoost directly
            if isinstance(self.estimator, xgb.XGBClassifier):
                self.estimator.set_params(device=f"cuda:{self.gpu_id}")
                
                # Move data to this specific GPU
                if CUPY_AVAILABLE:
                    with cp.cuda.Device(self.gpu_id):
                        X_gpu = cp.asarray(X)
                        y_gpu = cp.asarray(y)
                        self.estimator.fit(X_gpu, y_gpu, **kwargs)
                        return self
                
            # Handle OneVsRestClassifier (For SVMs)
            elif isinstance(self.estimator, OneVsRestClassifier):
                base_est = self.estimator.estimator
                if isinstance(base_est, xgb.XGBClassifier):
                    base_est.set_params(device=f"cuda:{self.gpu_id}")
                    self.estimator.estimator = base_est
                    # Note: don't pass CuPy arrays to OneVsRestClassifier because will crash on GPU arrays.

            # Handle Scikit-Learn CPU Models (RF, KNN)
            elif isinstance(self.estimator, (RandomForestClassifier, KNeighborsClassifier)):
                if hasattr(self.estimator, 'n_jobs'):
                    self.estimator.set_params(n_jobs=1)

        # Fallback for CPU models or if CuPy isn't installed
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        # Move prediction data to GPU, then pull results back to CPU
        if CUPY_AVAILABLE and self.gpu_id is not None and isinstance(self.estimator, xgb.XGBClassifier):
            with cp.cuda.Device(self.gpu_id):
                X_gpu = cp.asarray(X)
                preds = self.estimator.predict(X_gpu)
                return cp.asnumpy(preds) # Send back to CPU for sklearn's scorer
                
        return self.estimator.predict(X)
        
    def predict_proba(self, X):
        # Move prediction data to GPU, then pull results back to CPU
        if CUPY_AVAILABLE and self.gpu_id is not None and isinstance(self.estimator, xgb.XGBClassifier):
            with cp.cuda.Device(self.gpu_id):
                X_gpu = cp.asarray(X)
                preds = self.estimator.predict_proba(X_gpu)
                return cp.asnumpy(preds) # Send back to CPU for sklearn's scorer
                
        return self.estimator.predict_proba(X)
        
    def score(self, X, y):
        return self.estimator.score(X, y)
        
    def __getattr__(self, name):
        return getattr(self.estimator, name)
    
def load_choices(fp:str, batch_process:bool=False):
    """Loads whatever file you pick

    Args:
        fp (str): file path
        batch_process (bool): whether we're loading all the files

    Raises:
        ValueError: If you don't give it a numeric selection in single select, it errors

    Returns:
        files (list|str): File(s) we want to load
    """    
    try:
        tree = Tree(f":open_file_folder: [link file://{fp}]{fp}", guide_style="bold bright_blue")
        walk_directory(Path(fp), tree)
        pprint(tree)

    except Exception as e:
        logger.warning(f"{e}")
    
    if not batch_process:
        question = "What file would you like to load?\n"
        file_choice = console.input(f"{question}")
        if file_choice.isnumeric():
            files = sorted(f for f in Path(str(fp)).iterdir() if f.is_file())
            return files[int(file_choice)]
        else:
            raise ValueError("Invalid choice")
    else:
        return sorted(f for f in Path(str(fp)).iterdir() if f.is_file())

# --- Entry Point ---
@log_time
def main():
    fp:Path = Path.cwd() / "src/rad_ecg/data/datasets/JT"
    batch_process:bool = True
    selected = load_choices(fp, batch_process)
    rad = PigRAD(selected)
    rad.run_pipeline()

if __name__ == "__main__":
    main()

#Problem statement.  
# We're looking to classify the 4 stages of hemorhagic shock. 
# We'll look for 5 regime changes and hope for the best!  

#Workflow
#File choice and signal loading
#Pigrad initialization kicks off stumpy matrix profile calculation
#Runs FLUSS algorithm to look for semantic shifts in the morphology of the signal
#Loads stumpy CAC () curve results to RegimeViewer matplotlib GUI
#During that load, we calculate the phase variance over time with each beat
#To do that we need to isolate the beats and then set a standard window before and after
#Align all of them and then look for the variance in the aligned beat to run the wavelet over.
#Gives the phase variance a kind of a step curve.

#Good results
#sep-3-24

#NOTES 2-10-26
#FLUSS not performing as well as I'd like. 
#Possible problems. 
    #an m that changes throughout the signal because the signal I want to look at is too long.  
        # Could section the ecg according to the phase labels in BSOS data.  
    
    #It also could be that the  periodicity of the carotid flow is so strong it completely obliterates
    #any smaller signal change.  Which we did see in the freqwuency spectrum as the power for that signal
    #was really large.  

    #Additionally, euclidean distance's might break down in this instance because the change isn't immediate.  
    #It's gradual over time which FLUSS won't be able to see.

    #Mortlet Wavelet might not be suitable (meant for ecg's not flow traces)
    #debauchies 4/ symlet 5 and gaussian may be more appropriate

#IDEA - New Modeling path
#What about shooting for a change point detection algorithm.  BOCPD (Bayesian optimized change point detection) might work here.  
#Proposed outline
#1. Downsample if necessary
#2. apply a zero-phase butterworth bandpass filter (0.5 - 30 Hz) on the carotid and LAD traces in order to remove wander and artifacts. 
#3. dicrotic notch index (DNI)
    # find the r peak. Use scipy find_peaks
    # find the systolic peak(SBP) and diastolic trough (DBP)
    # use the second deriv to get the local maxima (aka the dichrotic notch)
    # dni =  (Pnotch - DBP) / (SBP - DBP)
    # supposedly falls off quickly from hem stages 2 and up.
#4. Pulse wave reflection ratios
    #p1 - percussion wave - initial upstroke by lv ejection
    #p2 - tidal wave - reflection from the upper body and renal
    #p3 - dicrotic wave - reflection from the lower body. 
#5. Systolic + Diastolic Slopes 
    #max slope - max value of the first derivative during the upstroke. 
        #gets greater in class 1.  decreases in following
    #Decay time constant
        #for an exponential decay func p(t) = P0e^-t/T to the diastolic portion - notch to end diastole
#6. Use AUC for calculating MAP
#7  Calculate shannon energy maybe?
#8. Calculate diastolic retrograde fraction
    # Don't really understand this one, need to come back. 
#9.  Maybe use a clustering approach for labeling sections
#10. Throw it all at an XGBOOST and look at feature importance. 
#11. Couldn't hurt to verify feature importance with some SHAP values

#Early models to try. 
#https://computationalphysiology.github.io/circulation/examples/regazzoni_scipy.html