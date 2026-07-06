import pyedflib
import os
import pickle
import numpy as np
from scipy.interpolate import interp1d
from math import ceil
from pyebmreader import ebmreader


def get_ebm_headers(ebmfolder, **options):
    """
    Get channel names, starttimes, recording length (# of datapoints), rate corrections (sec_error), and frequencies from
    a folder.
    Parameters
    ----------
    ebmfolder : [str]
        path of embla folder
    
    save : [bool], optional, default: False
        whether to save the header or not

    savefolder : [str], optional, default: ./
        where to save the header

    id : [str], optional
        patient id.
    
    Returns
    -------
    [dict]
        {
            "channels": list of [str],
            "starttime": list of list of datetime
            "length": list of list of integer
            "sec_error": list of double
            "frequency": list of double
        }
    """
    files = os.listdir(ebmfolder)
    signals = list(filter(lambda x: x[-3:] == "ebm", files))
    channel_names = list(map(lambda x: x[:-4], signals))
    headers = {
        "channels": channel_names,
        "starttime": [],
        "length": [],
        "sec_error": [],
        "frequency": []
    }
    for filename in channel_names:
        filename = filename + ".ebm"
        filepath = os.path.join(ebmfolder, filename)
        _, signal_header = ebmreader(filepath, True)  # load the header from ebm files
        headers["starttime"].append(signal_header["starttime"])
        headers["length"].append(signal_header["length"])
        headers["sec_error"].append(signal_header["sec_error"])
        headers["frequency"].append(signal_header["frequency"])
    try:
        issave = options["save"]
        if issave is True:
            try:
                save_path = options["savefolder"]
            except KeyError:
                save_path = "./"
            try:
                patient_id = options["id"]
            except KeyError:
                index = ebmfolder.find("SL")
                patient_id = ebmfolder[index:index + 5]
            save_path = os.path.join(save_path, patient_id + "_header.pickle")
            f = open(save_path, "wb")
            pickle.dump(headers, f)
            f.close()
    except KeyError:
        pass
    return headers


def load_header(filepath):
    """Load the header file

    Parameters
    ----------
    filepath : [str]
        Path of the header file

    Returns
    -------
    [dict]
        header
    """
    f = open(filepath, "rb")
    headers = pickle.load(f)
    german_c = ["ä", "ö", "ü", "ß",  "Ä", "Ü", "Ö"]
    english_c = ["a", "o", "v", "b", "A", "U", "O"]
    for i, name in enumerate(headers["channels"]):
        for g, e in zip(german_c, english_c):
            name = name.replace(g, e)
            if len(name) >= 17:
                name = name[:16]
        headers["channels"][i] = name
    f.close()
    return headers


def correct_signal(signal,
                   freq,
                   channelname,
                   edfstarttime,
                   ebmheader,
                   tol=0.01):
    """correct one signal from edf by using header from ebm

    Parameters
    ----------
    signal : np.ndarray
        signal from edf
    freq : int
        signal's frequency
    channelname : string
        name of the channel
    edfstarttime : datetime.datetime

    ebmheader : Dict
        return from get_ebm_headers
    tol : float, optional
        if the maximum shift is smaller than the tolerance, do nothing to the signal, by default 0.01

    Returns
    -------
    [type]
        [description]
    """
    try:
        i = ebmheader["channels"].index(channelname)
    except ValueError:
        print(f"No such channel:{channelname} in ebm header")
        return
    ebm_starttimes = ebmheader["starttime"][i]
    recording_length = ebmheader["length"][i]
    sec_error = ebmheader["sec_error"][i]
    if sec_error == 0:
        return signal
    ebm_freq = ebmheader["frequency"][i]
    max_shift = np.max(recording_length) / ebm_freq * sec_error
    if max_shift <= tol:
        # if the maximu shift is less than tolerance, do nothing.
        return signal
    assert int(ebm_freq) == int(freq), f"{channelname} channel,\
    Frequencies from EDF file and EBM file are not the same."

    t = []
    real_signal = []
    for j, ebmstarttime in enumerate(ebm_starttimes):
        length = recording_length[j]
        time_diff = ebmstarttime - edfstarttime
        start_index = round(time_diff.total_seconds() * freq)
        end_index = length + start_index

        if end_index <= 0 or start_index >= len(signal):
            continue

        start_index = max(0, start_index)
        end_index = min(len(signal), end_index)

        if len(ebm_starttimes) == 1:
            x = np.arange(start_index, end_index, 1, dtype=np.int)
            y = signal[x]
        elif j == 0:
            x = np.arange(start_index, end_index + 1, 1, dtype=np.int)
            y = signal[x]
            y[-1] = 0
        elif j == len(ebm_starttimes) - 1:
            x = np.arange(start_index - 1, end_index, dtype=np.int)
            y = signal[x]
            y[0] = 0
        else:
            x = np.arange(start_index - 1, end_index + 1, dtype=np.int)
            y = signal[x]
            y[0] = 0
            y[-1] = 0
        x = (-(x - x[0]) * sec_error + x)
        t.append(x)
        real_signal.append(y)
    if len(t) == 0:
        return signal
    t = np.concatenate(t)
    real_signal = np.concatenate(real_signal)
    interpolate_f = interp1d(t,
                             real_signal,
                             bounds_error=False,
                             fill_value=0,
                             assume_sorted=True)
    last_t = int(t[-1] / freq) * freq
    resampled_t = np.arange(int(last_t))
    return interpolate_f(resampled_t)


def correct_edf(edfpath, ebmheader, saveas="./out.edf"):
    """Correct edf based on header from ebm

    Parameters
    ----------
    edfpath : str
        path of the old edf
    ebmheader : [type]
        return value from get_ebm_headers
    saveas : str, optional
        path of the new edf, by default "./out.edf"
    """
    with pyedflib.EdfReader(edfpath) as old_edf:
        edf_header = old_edf.getHeader()
        #         events = f_test.readAnnotations()
        signal_headers = old_edf.getSignalHeaders()
        edf_starttime = old_edf.getStartdatetime()

        signals = []
        num_of_signal = len(signal_headers)
        duration = np.zeros(num_of_signal)

        for i in range(num_of_signal):
            signal_tmp = old_edf.readSignal(i)
            label, freq = signal_headers[i]["label"], signal_headers[i][
                "sample_rate"]
            corrected_signal = correct_signal(signal_tmp, freq, label,
                                              edf_starttime, ebmheader)
            signals.append(corrected_signal)
            duration[i] = corrected_signal.size // freq
        max_duration = max(duration)
        for i in range(num_of_signal):
            freq = signal_headers[i]["sample_rate"]
            num_of_zeros = int((max_duration - duration[i]) * freq)
            if num_of_zeros != 0:
                signals[i] = np.concatenate(
                    [signals[i], np.zeros(num_of_zeros)])
    with pyedflib.EdfWriter(saveas, num_of_signal) as new_edf:
        new_edf.setHeader(edf_header)
        new_edf.setSignalHeaders(signal_headers)
        new_edf.writeSamples(signals)