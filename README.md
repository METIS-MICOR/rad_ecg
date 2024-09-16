

<h1 align="center">
  <b>Robust Agile Detector (RAD_ECG) </b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.8-ff69b4.svg" /></a>
</p>


## Requirements

- python >= 3.8
- poetry >=1.70
- scipy >= 1.9.3
- numpy >= 1.22.0
- pandas >= 1.4.0
- matplotlib >= 3.73
- rich >= 13.7.0

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

- Do you want to plot the fourier transforms when a signal is lost?
  - "plot_fft":false,   
- Do you want to plot errors as they occur in extraction
  - "plot_errors":false
- In Hz.  What sampling frequency was used with the data
  - "samp_freq:170
- Do you want to display the terminal dashboard during extraction  
  - "live_term":false,
- Is this porcine data?  
  - "porcine":false,
- Provide data path
  - "data_path"  :"/src/pathtoyourdata/onyourmachine"

To run the extraction program, run the command

```terminal
poetry run python peak_detect_v3.py
```
## Andy Todo List

[ ] - Add overall progbar with standard log output
[ ] - Add JSON runtime config 
[ ] - Update README
[ ] - Package and upload to pypy