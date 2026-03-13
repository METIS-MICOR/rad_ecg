####  NYAS 2026 Abstract Draft

### Title
Classification of Hemorrhagic Shock Stages in a Porcine Model Using Machine Learning and Hemodynamic Modeling. Andrew Heroy, MS: Lead Data Scientist, Dominic Nathan: PhD : Biomedical Engineering, Jack Nelson,PhD,(TODO - input Jack info), John Green, MD: (TODO input JT info), Pat Walker, MD (TODO - Input Pat Info), Jonathan Morrison, MD (TODO-Input Johnny’s info) Nate Ehat, MS: Biostatistician, Mark C. Haigney, MD:MiCOR Director/Head of Cardiology at USUHS Military Cardiovascular Outcomes Research Program (MiCOR) (TODO - Add anyone else I forgot)

### Body 
Introduction: Hemorrhagic shock is a leading cause of preventable death in trauma casualties. Early, accurate identification of shock severity is critical for guiding treatment, especially in the battlefield arena where time is of the utmost importance. The purpose of this study is to classify the four stages of hemorrhagic shock based on progressive estimated blood loss, utilizing advanced continuous hemodynamic, frequency and flow feature extraction in a porcine model.

### Methods
Data were collected from 12 porcine subjects undergoing controlled hemorrhage. A known bleed rate allowed for the calculation of continuous estimated blood loss over time. This blood loss percentage defined the 4 progressive stages of hemorrhagic shock, which serve as our target classification variable. We extracted features from continuous signals across three domains: morphomics/pressure (e.g., systolic/diastolic pressures, mean arterial pressure, augmentation index, dicrotic notch index), coronary (LAD) and carotid flow features (retrograde flow, coronary, vascular resistance), and the frequency realm (phase variance, in-band power, shannon entropy). To classify shock stages, two parallel modeling approaches were developed. First, a suite of machine learning (ML) models were trained on the extracted 43 features.  These models were; a Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Random Forest, XGBoost. The probabilities from all four models were then given to a final, ensemble Voting Classifier. Concurrently, a mathematical approach utilizing partial differential equations (2D Windkessel) is being developed to model carotid flow within the aortic arch.

### Results
Validation of these techniques demonstrates promising predictive capabilities within the ML framework. Each model was run using LOSO (Leave One Subject Out) cross validation and scaled with sklearns PowerTransformer.  The models achieved the following mean classification accuracies across the four shock stages: One-vs-Rest SVM (43.80% +/- 17.14%), KNN (45.15% +/- 12.84%), Voting Classifier (54.43% +/- 13.30%), Random Forest (54.95% +/- 15.06%), and XGBoost, which emerged as the top-performing model (58.02% +/- 16.91%). While the ML models successfully captured the complex relationships in the hemodynamic data streams, the parallel differential equation approach remains in early development but shows tandem potential for providing mechanistic physiological insights. 
### Conclusions
Machine learning techniques, particularly XGBoost and ensemble methods, demonstrate early efficacy in classifying the four stages of hemorrhagic shock using high-fidelity pressure, frequency, and flow metrics. Integrating these robust data-driven models with mechanistic PDE modeling may further enhance real-time shock staging, providing critical early warnings for prolonged field care.

### Disclaimer 
This abstract presents a novel machine learning framework for classifying Hemorrhagic Shock. The author(s) declare that they have no conflict of interest relevant to this manuscript.  The opinions and assertions expressed herein are those of the author(s) and do not reflect the official policy or position of the Uniformed Services University of the Health Sciences or the Department of War..



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


