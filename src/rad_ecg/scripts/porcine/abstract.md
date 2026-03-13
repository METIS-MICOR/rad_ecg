##  NYAS 2026 Abstract Draft

### Title
Classification of Hemorrhagic Shock Stages in a Porcine Model Using Machine Learning and Hemodynamic Modeling. Andrew Heroy, MS: Lead Data Scientist, Dominic Nathan: PhD : Biomedical Engineering, Jack Nelson,PhD,(TODO - input Jack info), John Green, MD: (TODO input JT info), Pat Walker, MD (TODO - Input Pat Info), Jonathan Morrison, MD (TODO-Input Johnny’s info) Nate Ehat, MS: Biostatistician, Mark C. Haigney, MD:MiCOR Director/Head of Cardiology at USUHS Military Cardiovascular Outcomes Research Program (MiCOR) (TODO - Add anyone else I forgot)

### Body 
Hemorrhagic shock remains a dominant cause of potentially preventable battlefield death, with many casualties deteriorating before definitive surgical control despite aggressive point-of-injury care. In far-forward environments, monitoring is constrained to a small number of low-fidelity physiologic channels that are intermittently available, artifact-prone, and frequently corrupted by motion, evacuation, and combat conditions. A central challenge—often underemphasized—is that raw physiologic waveforms acquired during progressive hemorrhage contain substantial noise: sensor dropout, motion artifact, electrical interference, and non-physiologic transients that, if allowed into downstream analysis, degrade both feature extraction and model performance. Prior work has shown that rich hemodynamic information is embedded in continuous arterial waveform morphology and frequency content during hemorrhage, and that features such as reduced pulsatility and shifting frequency–power spectra can distinguish shock severity. However, realizing these gains requires robust, automated signal quality assessment to separate true physiologic signal from noise before feature extraction. Recent machine-learning approaches leveraging noisy physiologic time series have demonstrated improved prediction of trauma mortality, hemodynamic decompensation, and hemorrhage risk, yet few have explicitly addressed how noise rejection itself shapes classification accuracy. Building on this literature, we developed and tested an automated noise-tagging pipeline—driven by spectral entropy and in-band power concentration—to gate signal quality before extracting stage-specific signatures of hemorrhagic shock from high-resolution hemodynamic data. In a porcine model of controlled hemorrhage, we aimed to classify four stages of hemorrhagic shock, defined by progressive estimated blood loss, and to quantify how automated noise rejection improves classification performance.

### Methods
Data were collected from 12 porcine subjects undergoing controlled hemorrhage. A known bleed rate allowed for continuous estimation of blood loss over time, defining four progressive stages of hemorrhagic shock as the target classification variable. Because raw hemodynamic waveforms are heavily contaminated by artifact, we developed an automated signal quality index (SQI) pipeline applied to every analysis window prior to feature extraction. For each segment, Welch’s method estimated the power spectral density (PSD), from which two complementary metrics were computed: (1) In-Band Power Ratio—the fraction of total spectral power within the 0.01–15 Hz physiologic band (spanning heart rate fundamentals and harmonics), and (2) Normalized Shannon Entropy of the PSD distribution, scaled to [0, 1], where values near 0 indicate a concentrated, periodic (physiologic) spectrum and values near 1 indicate flat, noise-dominated energy spread. Segments failing either threshold (in-band power < 95% or spectral entropy > 0.50) were automatically tagged as noise and excluded, preventing corrupted data from propagating into feature extraction or model training. The same SQI metrics were used upstream for automated lead selection via Levenshtein name-matching combined with spectral quality scoring. From validated segments, we extracted 43 features across three domains: pressure morphomics (systolic/diastolic pressures, mean arterial pressure, augmentation index, dicrotic notch index), coronary (LAD) and carotid flow (retrograde flow fraction, coronary and vascular resistance indices), and the frequency domain (phase variance, harmonic amplitudes). To classify shock stages, a suite of machine learning models—Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Random Forest, and XGBoost—were trained, with their probabilities fused via a Voting Classifier. Concurrently, a mechanistic approach using partial differential equations (2D Windkessel model) is being developed to model carotid flow within the aortic arch.

### Results
Automated noise rejection via the Shannon Entropy and In-Band Power SQI pipeline was critical to achieving stable classification performance. Without it, artifact-corrupted segments introduced spurious feature values that degraded model accuracy across subjects. With the SQI gate in place, each model was evaluated using Leave-One-Subject-Out (LOSO) cross-validation with Power Transformer scaling. The models achieved the following mean classification accuracies across the four shock stages: One-vs-Rest SVM (43.80% +/- 17.14%), KNN (45.15% +/- 12.84%), Voting Classifier (54.43% +/- 13.30%), Random Forest (54.95% +/- 15.06%), and XGBoost, the top performer (58.02% +/- 16.91%). Notably, the best results were achieved after tuning the in-band frequency range to 0.01–15 Hz, which captured low-frequency hemodynamic oscillations lost with a narrower passband. The parallel PDE approach remains in early development but shows potential for providing mechanistic physiological context alongside the data-driven models.

### Conclusions
Automated, frequency-domain noise rejection using Shannon Entropy and In-Band Power Ratio is essential for reliable machine learning classification of hemorrhagic shock stages from continuous hemodynamic waveforms. By gating signal quality before feature extraction, the pipeline prevents artifact-driven misclassification and enables robust performance under realistic, noisy acquisition conditions. XGBoost and ensemble methods achieve the strongest stage-level accuracy, and the combination of spectral noise tagging with data-driven modeling offers a practical path toward real-time shock staging in prolonged field care environments.

### Disclaimer 
This abstract presents a novel machine learning framework for classifying Hemorrhagic Shock. The author(s) declare that they have no conflict of interest relevant to this manuscript.  The opinions and assertions expressed herein are those of the author(s) and do not reflect the official policy or position of the Uniformed Services University of the Health Sciences or the Department of War.

### NYAS Requirements

Abstracts should be relevant to the scientific topic of the symposium and must contain primary scientific data. 

Only one abstract should be submitted per presenter. 

Limit of 1,500 characters (including spaces, not including title/affiliations). 
Provide a brief title using Title Case**
**Title Case: Capitalize the first word and all major words (nouns, pronouns, adjectives, verbs, adverbs). Do not use capital letters for prepositions, articles, or conjunctions unless one is the first word.)

Author(s):
Type first name, last name, and middle initial in Title Case.
Do not use periods in degree abbreviations (example: use PhD, not Ph.D.)
Affiliation(s): Type using Title Case and include the following information
Institution, City, State (if applicable), and Country (Do not use abbreviations for states. For example: use New York, not NY).
If multiple affiliations are to be indicated, list each separated by a semicolon.
Numbers should be in numerical order preceding each affiliation without a space between the number and the first word. Numbering should be in brackets to show who belongs with each affiliation. E.g., Robert Smith [1], Martin Rose [1], and Jane Doe [2]. 
If an author has multiple affiliations, use numbering after the last degree separated by commas with no spaces (example: John Doe, MD, PhD[1,2]).
Text:
Place special abbreviations in parentheses after the full word, the first time they appear.
Special characters should be spelled out where possible (e.g. alpha not α).


