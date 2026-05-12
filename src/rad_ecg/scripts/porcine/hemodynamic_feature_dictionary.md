# Hemodynamic and Morphomic Feature Dictionary

This document serves as a comprehensive reference for all physiological, morphomic, and spectral features extracted by the `PigRAD` pipeline. Features are calculated on a per-beat basis and averaged over a rolling window (e.g., 8 seconds), or calculated globally across the entire window.

---

## 1. Pressure Morphomics (SS1 Lead)
These features characterize the morphology of the arterial blood pressure wave.

* **SBP (Systolic Blood Pressure):** The maximum pressure amplitude achieved during the systolic phase of the cardiac cycle.
    * *Formula:* $\max(P(t))$ *for $t$ in systole*
* **DBP (Diastolic Blood Pressure):** The minimum pressure amplitude in the cardiac cycle, occurring at the foot (onset) of the beat or the end of diastole.
* **true_MAP (True Mean Arterial Pressure):** The true geometric mean pressure over the entire cardiac cycle, calculated via the definite integral (Area Under the Curve) of the pressure wave.
    * *Formula:* $\frac{1}{T} \int_{0}^{T} P(t) dt$
* **ap_MAP (Approximate Mean Arterial Pressure):** The standard clinical approximation of MAP.
    * *Formula:* $DBP + \frac{1}{3}(SBP - DBP)$
* **shock_gap:** The numerical difference between the true geometric MAP and the clinically approximated MAP. Divergence often indicates morphomic collapse.
    * *Formula:* $true\_MAP - ap\_MAP$
* **dni (Dicrotic Notch Index):** A normalized measure of the dicrotic notch depth relative to the total pulse pressure.
    * *Formula:* $\frac{SBP - Notch\_Pressure}{SBP - DBP}$
* **sys_sl (Systolic Slope):** The rate of pressure generation (analogous to maximal $dP/dt$) during the upstroke of systole.
    * *Formula:* $\frac{SBP - P_{onset}}{t_{SBP} - t_{onset}}$
* **sys_sl_len (Systolic Slope Length):** The time duration from the onset of the pressure wave to the systolic peak.
    * *Formula:* $t_{SBP} - t_{onset}$
* **dia_sl (Diastolic Slope):** The linear rate of pressure decay during the diastolic run (from the dicrotic notch to the subsequent end-diastole).
    * *Formula:* Linear Regression Slope ($m$) of $P(t) = mt + b$ for $t \in [t_{notch}, t_{end}]$
* **pul_wid (Pulse Width at FWHM):** The Full-Width at Half-Maximum of the systolic peak. It measures the time duration the pressure wave stays above 50% of the pulse pressure.
    * *Formula:* $t_{right} - t_{left}$ where $P(t) = P_{onset} + 0.5 	imes (SBP - P_{onset})$
* **p1 (Percussion Wave):** The amplitude of the first sub-peak (shoulder) in the systolic complex.
* **p2 (Tidal Wave):** The amplitude of the second sub-peak in the systolic complex (often the absolute SBP in healthy subjects).
* **p3 (Dicrotic Wave):** The amplitude of the rebound wave occurring immediately after the dicrotic notch in diastole.
* **p1_p2 (P1/P2 Ratio):** *Formula:* $\frac{P1}{P2}$
* **p1_p3 (P1/P3 Ratio):** *Formula:* $\frac{P1}{P3}$
* **aix (Augmentation Index):** A measure of arterial stiffness and wave reflection.
    * *Formula:* $\frac{P2 - DBP}{P1 - DBP}$

---

## 2. Coronary & Systemic Flow (LAD & Carotid Leads)
These features assess myocardial perfusion, flow directionality, and systemic resistance.

* **lad_mean (Mean LAD Flow):** The average volumetric flow rate through the Left Anterior Descending artery across the entire beat.
* **lad_sys_pk (LAD Systolic Peak):** The maximal flow rate during the systolic ejection phase (onset to notch).
* **lad_dia_pk (LAD Diastolic Peak):** The maximal flow rate during the diastolic phase. In healthy subjects, coronary flow is predominantly diastolic.
* **lad_ds_rat (Diastolic/Systolic Ratio):** The ratio of diastolic peak flow to systolic peak flow.
    * *Formula:* $\frac{lad\_dia\_pk}{lad\_sys\_pk}$
* **lad_dia_net (Diastolic Net Volume):** The Area Under the Curve (total volume) of LAD flow specifically during the diastolic phase.
    * *Formula:* $\int_{t_{notch}}^{t_{end}} Flow_{LAD}(t) dt$
* **lad_dia_neg (Diastolic Negative Volume):** The total volume of *retrograde* (backwards) flow in the LAD during diastole.
    * *Formula:* $\int_{t_{notch}}^{t_{end}} | \min(0, Flow_{LAD}(t)) | dt$
* **cvr (Coronary Vascular Resistance):** The global resistance of the coronary vascular bed.
    * *Formula:* $\frac{true\_MAP}{lad\_mean}$
* **dcr (Diastolic Coronary Resistance):** Coronary resistance specifically during the diastolic perfusion window.
    * *Formula:* $\frac{DBP}{Mean\_Diastolic\_LAD\_Flow}$
* **lad_pi (LAD Pulsatility Index):** A quantification of the pulsatile energy in the coronary flow.
    * *Formula:* $\frac{\max(LAD) - \min(LAD)}{lad\_mean}$
* **lad_acc_sl (LAD Diastolic Acceleration Slope):** The rate at which coronary flow accelerates to its diastolic peak immediately following aortic valve closure.
    * *Formula:* $\frac{lad\_dia\_pk - Flow_{LAD}(t_{notch})}{t_{dia\_pk} - t_{notch}}$
* **flow_div (Flow Division Ratio):** A systemic macro-metric comparing blood prioritized to the brain (Carotid) versus the heart (LAD).
    * *Formula:* $\frac{Mean\_Carotid\_Flow}{lad\_mean}$
* **retro_flow (Carotid Retrograde Flow):** The total volume (AUC) of negative/backwards flow in the Carotid artery during diastole.
    * *Formula:* $\int_{t_{notch}}^{t_{end}} | \min(0, Flow_{Carotid}(t)) | dt$
* **ri (Carotid Resistive Index):** A measure of downstream vascular resistance in the systemic/cerebral circulation.
    * *Formula:* $\frac{Peak\_Systolic\_Velocity - End\_Diastolic\_Velocity}{Peak\_Systolic\_Velocity}$

---

## 3. Spectral & Phase Features
Extracted using Short-Time Fourier Transforms (STFT) and Continuous Wavelet Transforms (CWT).

* **f0, f1, f2, f3 (Harmonic Frequencies):** The fundamental frequency ($f0$, highly correlated to heart rate) and its first three harmonic peaks ($f1, f2, f3$), extracted via Welch's Power Spectral Density.
* **psd0, psd1, psd2, psd3 (Harmonic Power):** The absolute power (amplitude) associated with $f0, f1, f2,$ and $f3$.
* **var_mor (Phase Variance - Morlet):** Measures the phase stability (rhythmicity) of the beats across a window. Evaluated at a 2.0 Hz center frequency using a Morlet complex wavelet.
    * *Formula:* $1 - R$, where $R = \left| \frac{1}{N} \sum e^{i \theta_n} \right|$ (Circular mean vector length)
* **var_cgau (Phase Variance - Gaussian):** Phase variance evaluated using the 1st derivative of a complex Gaussian wavelet (`cgau1`).

---

## 4. Signal Quality Indices (SQI)
Used primarily for data rejection and pipeline integrity, ensuring models are not trained on noise or machine-calibration artifacts.

* **sqi_power (In-Band Power Ratio):** The proportion of total signal energy contained within physiological bounds (0.5 Hz to 15.0 Hz). 
    * *Formula:* $\frac{\sum_{f=0.5}^{15} PSD(f)}{\sum_{f=0}^{\infty} PSD(f)}$
* **sqi_entropy (Normalized Spectral Entropy):** The Shannon entropy of the power spectrum. A value near 0 indicates a pure sine wave; a value near 1.0 indicates pure, flat white noise. 
    * *Formula:* $\frac{-\sum (P(f) \ln P(f))}{\ln(N)}$
