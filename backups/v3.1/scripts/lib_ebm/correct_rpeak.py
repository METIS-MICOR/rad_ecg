import datetime
import pandas as pd
import re
import numpy as np
from numba import njit
from tools import load_header

def read_rri(rri_file: str):
    """read data from .rri file

    Parameters
    ----------
    rri_file : str
        File path

    Returns
    -------
    rposition : DataFrame
        rposition (integer) and marker (only for librash)

    freq : integer

    starttime : datetime

    """
    i = 0
    with open(rri_file, 'r') as f:
        f.readline()
        f.readline()

        rr_startdate = f.readline().split('=')[1]
        rr_starttime = f.readline().split('=')[1]

        file_starttime = rr_startdate + " " + rr_starttime
        start_time = re.findall(r"\d+", file_starttime)
        start_time = list(map(int, start_time))
        try:
            start_time = datetime.datetime(*start_time)
        except ValueError:
            start_time = datetime.datetime(*(start_time[2::-1] + start_time[3:]))

        freq = eval(f.readline().split('=')[1])

        while f.readline()[0] != '-':
            i += 1

    df = pd.read_csv(rri_file, skiprows=6+i, sep="\t", header=None)
    return df, freq, start_time


def correct_rposition(rposition, freq, rposition_starttime, ebm_header):
    """correct rposition array

    Parameters
    ----------
    rposition : np.ndarray[int]
        rposition
    freq : integer

    rposition_starttime : datetime

    ebm_header :
        ebm header

    Returns
    -------
    corrected_rposition : np.ndarray[int]

    Raises
    ------
    IndexError
        no ECG or EKG signal in ebm_header
    """
    @njit
    def _correction(rposition, start_i, end_i):
        correct_rposition = np.zeros(rposition.shape, dtype=np.int64)
        j = 0
        for i in range(rposition.size):
            while not start_i[j] <= rposition[i] < end_i[j]:
                j += 1
            rposition_tmp = rposition[i] - (rposition[i] - start_i[j]) * rate_correction
            correct_rposition[i] = round(rposition_tmp)
        return correct_rposition

    find_ecg = False
    for i, channel_name in enumerate(ebm_header["channels"]):
        if channel_name.find("EKG") >= 0 or channel_name.find("ECG") >= 0:
            find_ecg = True
            break

    if not find_ecg:
        raise IndexError("Can't find ECG channel.")

    starttimes = ebm_header["starttime"][i]
    rate_correction = ebm_header["sec_error"][i]
    recording_length = np.array(ebm_header["length"][i])
    start_s = np.array([(ebm_start - rposition_starttime).total_seconds() for ebm_start in starttimes])
    start_i = start_s * freq
    end_i = start_i + recording_length

    corrected_rposition = _correction(rposition, start_i, end_i)
    return corrected_rposition


def correct_rrifile(rri_file: str, header: dict, new_rripath: str):
    """correct rri file

    Parameters
    ----------
    rri_file : str
        path of old rri file
    header : dict
        ebm header
    new_rripath : str
        where to save the new rri file
    """
    rposition_df, freq, starttime = read_rri(rri_file)
    corrected_rposition = correct_rposition(rposition_df.iloc[:, 0].values, freq, starttime, header)
    rposition_df.iloc[:, 0] = corrected_rposition
    with open(new_rripath, "w") as new_f, open(rri_file, "r") as old_f:
        line = old_f.readline()[:-1]
        line += " (corrected by Yaopeng)\n"
        while line[0] != "-":
            new_f.write(line)
            line = old_f.readline()
        new_f.write(line)

    rposition_df.to_csv(new_rripath, mode="a", header=False, sep="\t", index=False)


if __name__ == '__main__':
    import os
    header_folder = "" # 
    header_files = os.listdir(header_folder)
    rri_folder = "" #
    rri_files = os.listdir(rri_folder)
    new_rri_folder = "" #
    for filename in rri_files:
        if filename.split("_")[0] + "_header.pickle" in header_files:
            old_rripath = rri_folder + filename
            ebm_header = header_folder + filename.split("_")[0] + "_header.pickle"
            new_rripath = new_rri_folder + filename
            header = load_header(ebm_header)
            correct_rrifile(old_rripath, header, new_rripath)