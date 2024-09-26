

<h1 align="center">
  <b>Robust Agile Detector (RAD_ECG) </b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.11-8bf230.svg" /></a>
      <a href="https://numpy.org/doc/">
        <img src="https://img.shields.io/badge/Numpy-1.25-8bf230.svg" /></a>
      <a href="https://pandas.pydata.org/docs/index.html">
        <img src="https://img.shields.io/badge/Pandas-1.5.0-8bf230.svg" /></a>
      <a href="https://rich.readthedocs.io/en/stable/">
        <img src="https://img.shields.io/badge/Rich-13.8.0-8bf230.svg" /></a>
      <a href="https://wfdb.readthedocs.io/en/latest/">              
        <img src="https://img.shields.io/badge/wfdb-4.1.2-8bf230.svg" /></a>
      <a href="https://docs.scipy.org/doc/scipy/">            
        <img src="https://img.shields.io/badge/Scipy-1.14.0-8bf230.svg" /></a>
      <a href="https://matplotlib.org/stable/index.html">            
        <img src="https://img.shields.io/badge/Matplotlib-3.9.0-8bf230.svg" /></a>
      <a href="https://cloud.google.com/storage/docs">            
        <img src="https://img.shields.io/badge/GCS-2.18.0-8bf230.svg" /></a>
</p>


## Requirements

- python = "^3.11"
- numpy = "^1.25.0"
- pandas = "^1.5.0"
- rich = "^13.8.0"
- wfdb = "^4.1.2"
- scipy = "^1.14.0"
- matplotlib = "^3.9.0"
- google-cloud-storage = "^2.18.2"

# Project setup *without* Poetry

Launch VSCode if that is IDE of choice.
`CTRL + ~` will open a new terminal
Navigate to the directory where the repo has been cloned

```terminal
git clone https://github.com/METIS-MICOR/rad_ecg.git
cd rad_ecg
python -m venv .venv

#activate the environment in your terminal 
#On Windows
.venv\Scripts\activate.bat

#On Mac
source .venv/bin/activate
```

Before next step, ensure you see the environment name to the left of your command prompt.  If you see it and the path file to your current directory, then the environment is activated.  If you don't activate it, and start installing things.  You'll install all the `requirements.txt` libraries into your base python environment. Which will lead to dependency problems down the road.  I promise.

Once activated, install the required libraries.

```terminal
pip install -r requirements.txt
```

# Project setup with Poetry

## How to check Poetry installation

In your terminal, navigate to your root folder.

To make sure poetry is installed on your system. Type the following into your terminal

```terminal
poetry -V
poetry self update
```

If poetry is not installed, do so in order to continue


To spawn a new poetry .venv

```terminal
poetry shell
```

To install libraries

```terminal
poetry install
```

This will read from the poetry lock file that is included
in this repo and install all necessary packagage versions.  Should other
versions be needed, the project TOML file will be utilized and packages updated according to your system requirements.  

To view the current libraries installed

```terminal
poetry show
```

To view only top level library requirements

```terminal
poetry show -T
```


if you see a `version` returned, you have Poetry installed.  The second command is to update poetry if its installed. (Always a good idea). If not, follow this [link](https://python-poetry.org/docs/) and follow installation commands for your systems requirements. If on windows, we recommend the `powershell` option for easiest installation. Using pip to install poetry will lead to problems down the road and we do not recommend that option.  It needs to be installed separately from your standard python installation to manage your many python installations.  `Note: Python 2.7 is not supported`

### Environment storage

Some prefer Poetry's default storage method of storing environments in one location on your system.  The default storage are nested under the `{cache_dir}/virtualenvs`.  See the below image for general system location of the cache.

![Cache Directory](docs/images/p_cach_dir.png)

If you want to store you virtual environment locally.  Set this global configuration flag below once poetry is installed.  This will now search for whatever environments you have in the root folder before trying any global versions of the environment in the cache.

```terminal
poetry config virtualenvs.in-project true
```

For general instruction as to poetry's functionality and commands, please see read through poetry's [cli documentation](https://python-poetry.org/docs/cli/)

Before running the extraction script, adjust/amend the config.json in the root of this folder to your runtime requirements. 

1. Do you want to plot the fourier transforms when a signal is lost?
  - "plot_fft":false
2. Do you want to plot errors as they occur in extraction
  - "plot_errors":false
3. What sampling frequency was used with the data (Hz)
  - "samp_freq:170
4. Do you want to display the terminal dashboard during extraction  
  - "live_term":false
5. Is this porcine data?  
  - "porcine":false
6. Provide data path
  - "data_path"  :"/src/pathtoyourdata/onyourmachine"

To run the extraction program, run the command

```terminal
poetry run python peak_detect_v3.py
```
## Andy Todo List

- [ ] Add overall progbar with standard log output
- [ ] Add instructions on README for GCP usage
- [ ] Add instructions for complete package installation
- [x] Adjust logger logic to run off global calls
- [x] Update README
- [x] Add JSON runtime config 