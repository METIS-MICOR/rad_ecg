Main Functions Outline
------------

|# | Function | scope | Description |
|:--- |:-------------|:-------------:|:-------------:|
|1| log_time      |small | Measures how long any function takes to run|
|2| roll_med     |small | This calculates a rolling median of a HR signal |
|3| section_finder |small | This is a debugging tool. Namely for when you'd like to search through a debugged session for where a section might be after inspecting the log file. |
|4| segment_ECG  |small | For splitting up the ECG into equal length sections (currently with 20% overlap) |
|5| consec_valid_peaks |small| This function is for scanning back in time for the most recent HR data to validate a future section of data.  (think of them as localized averages) The main key to remember is that they're all *consecutive*.  Meaning they all need to be valid peaks next to each other to avoid any erroneous/biased averages | 
|6| peak_val_check |large| This function is one of the gates that checks whether or not a section has drifted outside of the 3 basic wave parameters we use for stability checking.  1. Rolling median violation,  2. R peak height and 3. R peak spacing.
|7| STFT| large| This section calculates the FFT of any given R peak to R peak transition in a section. |
|8| section stats| small | Calculates HR, avg_HR, min, max std, nn50, qtvi, 
|9| Rpeak search| large | This is the first search function in the process of analyzing a section of wave.  Has many parts, but this is the main function that gets called to iterate through the cam. Maybe rename this too, handles way more than just R peaks.  |
|10| extract_PQRST| yuge | This function extracts the PQST peaks once the R peaks have been identified by scipy| 

