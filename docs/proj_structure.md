# Project Organization

------------
    |────opens_ecg
    |    │
    |    ├── docs                       <- Documentation for the detector
    |    |   ├── images                 <- Base images for README
    |    |   ├── func_outline.md        <- Outline of main functions
    |    |   └── proj_structure.md      <- Current file
    |    |── src
    |    |   └── rad_ecg       
    |    |       |
    |    |       ├── data
    |    |       |   ├── inputdata      <- Single Lead ECG Data goes here
    |    |       |   ├── logs           <- Log outputs for each ECG run
    |    |       |   ├── sample         <- scipy sample electrocardiogram
    |    |       |   └── output              
    |    |       |       ├── reports        <- Generated analysis as HTML, PDF, LaTeX, etc.
    |    │       |       ├── figures        <- Generated graphics and figures for reporting
    |    |       |       └── peak_data      <- csv's for interior peaks, R peaks, and section data
    |    |       |
    |    |       └── scripts
    |    |           ├── __init__.py        <- Package initialization py file
    |    |           ├── peak_detect_v3.py  <- The meat and potatoes 
    |    |           ├── setup_globals.py   <- Global variable loader
    |    |           ├── slider.py          <- Interactive plot to view results
    |    |           ├── support.py         <- Emailing function.(Emails when cam is analyzed)
    |    |           └── utils.py           <- Graphing functions, file loaders, non-essential funcs
    |    │
    |    ├── tests
    |    |   ├── __init__.py            <- Initialization py file
    |    |   ├── test_basic.py          <- Testing rolling median, segmentation and section_finder functions
    |    |   ├── test_detect.py         <- Testing main peak extraction process
    |    |   ├── test_imports.py        <- Testing imports
    |    |   ├── test_stft.py           <- Testing Short Time Fourier Transform
    |    |   └── test_version.py        <- Testing version 
    |    |
    |    ├── .gitignore                 <- Basic gitignore to prevent accidental file uplaod to github.
    |    ├── LICENSE                    <- MIT
    |    ├── poetry.lock                <- Locked file of libarary dependancy versions
    |    ├── pyproject.toml             <- Houses all project information / main libraries
    |    ├── README.md                  <- The top-level README for developers using this project.
    |    └── requirements.txt           <- Base requirements.txt file for non-poetry environment use

Link to example
https://raw.githubusercontent.com/shaygeller/Normalization_vs_Standardization/master/README.md
Will be using poetry to package, so distribution will be cleaner.  

## Base Data Containers

- ie what stores the data as it traverses the ECG
- its stored as a dictionary (ecg_data) with these keys
  1. peaks
  2. rolling_med
  3. section_info
  4. interior_peaks

<style> table {margin-left: 0 !important;} </style>
<font size=5> <h1 style="text-align: left;">peaks</h1></font>
|col idx | col name | data type |
|:--- |:---|:---:|
|0| R peak index | int32
|1| Peak validity | int32

<font size=5> <h1 style="text-align: left;">rolling_med</h1></font>
|col idx | col name | data type |
|:--- |:---|:---:|
|0| wave index | float32
|1| rolling median | float32

<font size=5> <h1 style="text-align: left;">section_info</h1></font>
|col idx | col name | data type |
|:--- |:---|:---:|
|0 | idx of wave section            | int32 |
|1 | start_point of wave section    | int32 |
|2 | end_point of wave section      | int32 |
|3 | valid section                  | int32 |
|4 | Avg HR                         | float32 |
|5 | Min HR                         | float32 |
|6 | Max HR                         | float32 |
|7 | STD HR                         | float32 |
|8 | NN50                           | float32 |
|9 | PNN50                          | float32 |
|10| Section failure encoding       | str |

<font size=5> <h1 style="text-align: left;">interior_peaks</h1></font>
|col idx | col name | val type | data type |
|:--- |:---|:---:|:---:|
| 0 | P peak        | idx  | int32 |
| 1 | Q peak        | idx  | int32 |
| 2 | R peak        | idx  | int32 |
| 3 | S peak        | idx  | int32 |
| 4 | T peak        | idx  | int32 |
| 5 | Valid_QRS     | bool | int32 |
| 6 | PR Interval   | ms   | int32 |
| 7 | PR Segment    | ms   | int32 |
| 8 | QRS Complex   | ms   | int32 |
| 9 | ST Segment    | ms   | int32 |
| 10| QT Interval   | ms   | int32 |
| 11| P_onset       | idx  | int32 |
| 12| Q_onset       | idx  | int32 |
| 13| T_onset       | idx  | int32 |
| 14| T_offset      | idx  | int32 |
| 15| J_point       | idx  | int32 |