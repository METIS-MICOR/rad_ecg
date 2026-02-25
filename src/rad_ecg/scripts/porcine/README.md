Data from the ACT1 protocol, curated for investigating changes in LAD/Carotid flow waveforms during hemorrhage.
Selected data spans from 5 minutes before start of hemorrhage to end of hemorrhage.

Columns
---
name | description | Units
|:--- |:-------:|---|
| Time | Time from start of hemorrhage | [seconds] |
| LAD | Flow in the left anterior descending (LAD) artery. Positive flow is towards the heart | [mL/min]|
| ECG1 | ECG lead 1. Placement may be animal dependent | [mV] |
| ECG2 | ECG lead 2. Placement may be animal dependent | [mV] |
| SS1 (SP200) | Pressure in the carotid artery | [mmHg] |
| SS2 (SP200) | Pressure in the axillary artery (probably...) | [mmHg] |
| Carotid (TS420)| Flow in the carotid artery. Positive flow is towards the brain | [mL/min] |
| EBV: Estimated Blood Volume (EBV)| Based off of metadata collected during the surgery | [mL] |
| ShockClass | Class of hypovolemic shock. Based on EBV. Categories are as follows, with EBV0 denoting EBV at baseline | [categorical] |
| BL | EBV=EBV0 | [ebv] |
| Class1 | 0.85*EBV0<=EBV<EBV0 | [ebv] |
| Class2 | 0.7*EBV0<=EBV<0.85*EBV0 | [ebv] |
| Class3 | 0.6*EBV0<=EBV<0.7*EBV0 | [ebv] |
| Class4 | EBV<0.6*EBV0 | [ebv] |



Feature Descriptions

name | datatype | description
|:--- |:-------:|---|
|######### | Morphomics / Pressure features | ##################|
|start       | i4 |  start index |
|end         | i4 |  end index |
|valid       | i4 |  valid Section |
|shock_class | U4 |  Shock Class |
|HR          | i4 |  Heart Rate |
|SBP         | i4 |  Systolic Pressure - Pressure at systolic peak |
|DBP         | i4 |  Diastolic Pressure - Pressure at diastolic trough |
|EBV         | i4 |  Estimated Blood Volume - Created feature of estimated blood volume over time |
|true_MAP    | f4 |  Mean Arterial Pressure (AUC)  |
|ap_MAP      | f4 |  Approximate Mean Arterial pressure (Formula) |
|shock_gap   | f4 |  Difference between true and approximate MAP |
|dni         | f4 |  Dichrotic Notch Index - Dichrotic notch index (value)  |
|sys_sl      | f4 |  Systolic slope |
|dia_sl      | f4 |  Diastolic slope |
|ri          | f4 |  Resistive index | RI = (Flow_max - Flow_min) / Flow_max
|pul_wid     | f4 |  Pulse width - difference between the foot of the pressure wave and the dicrotic notch, or simply as Pulse Pressure ($SBP - DBP$). |
|p1          | f4 |  Percussion Wave (P1) - First prominent peak in systole forward-traveling wave generated the moment the left ventricle forcefully contracts and ejects blood into the aorta. |
|p2          | f4 |  Tidal Wave (P2) - 2nd peak in systole | As P1 wave travels down the arterial tree, it hits the bifurcations and high-resistance arterioles in the periphery. This causes the wave to reflect and bounce back toward the heart. P2 is that reflected wave arriving |
|p3          | f4 |  Dicrotic Wave (P3)  |	Once systole ends, the aortic valve snaps shut (causing the notch). The stretched, elastic walls of the aorta then naturally recoil. This recoil acts like a secondary pump, creating the P3 wave to push blood forward through the body and into the coronary arteries while the heart is resting |
|p1_p2       | f4 |  Ratio of P1 to P2 - ratios of the amplitudes above |
|p1_p3       | f4 |  Ratio of P1 to P3, - ratios of the amplitudes above |
|aix         | f4 |  Augmentation Index (AIx) - difference between the second (tidal) and first (percussion) systolic peaks, expressed as a percentage of the pulse  |pressure: $AIx = \frac{P2 - P1}{SBP - DBP}$
| ######### | STFT Frequency features | #################### | 
|f0         | f4 |  STFT Top Frequency (Fundamental) |
|f1         | f4 |  STFT Harmonic 1 (2nd biggest peak) |
|f2         | f4 |  STFT Harmonic 2 (3rd biggest peak) |
|f3         | f4 |  STFT Harmonic 3 (4th biggest peak) |
|psd0       | f4 |  STFT Amplitude of Top Freq |
|psd1       | f4 |  STFT Amplitude of Harmonic 1 |
|psd2       | f4 |  STFT Amplitude of Harmonic 2 |
|psd3       | f4 |  STFT Amplitude of Harmonic 3 |
|var_mor    | f4 |  Wavelet Phase Variance (Morlet). fit morlet wavelet to each beat and averaged its phase variance per section |
|var_cgau   | f4 |  Wavelet Phase Variance (Cgau1) fit gaussian wavelet to each beat and averaged its phase variance per section |
|######## | LAD Flow Features | ###################### |
|lad_mean    | f4 |  Mean LAD Flow
|lad_dia_pk  | f4 |  Diastolic Peak Flow
|lad_sys_pk  | f4 |  Systolic Peak Flow
|lad_ds_rat  | f4 |  Diastolic to Systolic Peak Ratio
|lad_dia_auc | f4 |  Diastolic Flow Volume (AUC)
|cvr         | f4 |  Coronary Vascular Resistance (MAP / LAD_mean)
|dcr         | f4 |  Diastolic Coronary Resistance (DBP / LAD_dia_mean)
|lad_pi      | f4 |  LAD Pulsatility Index (LAD_maxflow - LAD_minflow) / LAD_mean
|lad_acc_sl  | f4 |  Diastolic Acceleration Slope
|flow_div    | f4 |  Carotid to LAD Flow Ratio (LAD_mean_carotid / LAD_mean)