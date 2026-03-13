##  NYAS 2026 Abstract Draft

### Title
Classification of Hemorrhagic Shock in a Porcine Model Using Machine Learning and Hemodynamic Modeling. Andrew Heroy, MSc: [1], Jack Nelson,PhD:[1], Dominic Nathan: PhD[1], John Green, MD, FS, FACP [1, 2], Pat Walker, MD[1,2], Jonathan Morrison, MD [1,2], Nate Ehat, MSc:[1], Mark C. Haigney, MD:[1]
USUHS Military Cardiovascular Outcomes Research Program (MiCOR), Bethesda, MD, USUHS Battlefield Shock and Organ Support (BSOS) Bethesda, MD


### Body 
Hemorrhagic shock (HS) remains a preventable cause of death, with casualties deteriorating before definitive surgical control despite point-of-injury care. In a porcine model (n=12) of controlled hemorrhage, we classified four stages of HS to quantify how automated noise rejection improves classification.

### Methods
A known bleed rate defined the four HS stages. We developed a Signal Quality Index (SQI) using Welch’s method to estimate Power Spectral Density (PSD). Two metrics gated the signal: (1) In-Band Power Ratio (0.5–15 Hz) and (2) Normalized Shannon Entropy. Segments failing thresholds (in-band power <95% or entropy >0.50) were excluded. 43 features (morphomics, flow, frequency) were extracted. SVM, KNN, Random Forest, and XGBoost models were evaluated via Leave-One-Subject-Out cross-validation. A mechanistic 2D Windkessel PDE model is also being developed to provide physiological context for carotid flow.

### Results
The models achieved the following mean classification accuracies: One-vs-Rest SVM (43.80% +/- 17.14%), KNN (45.15% +/- 12.84%), Voting Classifier (54.43% +/- 13.30%), Random Forest (54.95% +/- 15.06%), and XGBoost (58.02% +/- 16.91%). The parallel PDE approach remains in early development but shows great potential model improvement.

### Conclusions
Automated noise rejection is essential for robust classification of HS stages from hemodynamic waveforms. Our research offers an explainable, practical path toward real-time triage in far-forward environments.

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


