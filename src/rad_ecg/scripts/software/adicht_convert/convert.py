import os
import adi
import json
import numpy as np
import pandas as pd
import datetime
from pathlib import PurePath, Path
import time
import logging

#FUNCTION Log time
################################# Timing Func ####################################
def log_time(fn):
    """Decorator timing function.  Accepts any function and returns a logging
    statement with the amount of time it took to run. DJ, I use this code everywhere still.  Thank you bud!

    Args:
        fn (function): Input function you want to time
    """	
    def inner(*args, **kwargs):
        tnow = time.time()
        out = fn(*args, **kwargs)
        te = time.time()
        took = round(te - tnow, 2)
        if took <= 60:
            logging.warning(f"{fn.__name__} ran in {took:.2f}s")
        elif took <= 3600:
            logging.warning(f"{fn.__name__} ran in {(took)/60:.2f}m")		
        else:
            logging.warning(f"{fn.__name__} ran in {(took)/3600:.2f}h")
        return out
    return inner

#CLASS Numpy encoder
class CustomEncoder(json.JSONEncoder):
    """Custom numpy JSON Encoder.  Takes in any type from an array and formats it to something that can be JSON serialized. Source Code found here. https://pynative.com/python-serialize-numpy-ndarray-into-json/
    
    Args:
        json (object): Json serialized format
    """	
    def default(self, obj):
        match obj:
            case np.integer():
                return int(obj)
            case np.floating():
                return float(obj)
            case np.ndarray():
                return obj.tolist()
            case pd.DataFrame():
                return obj.to_json(orient='records')
            case dict():
                return obj.__dict__
            case datetime.datetime():
                return datetime.datetime.strftime(obj, "%m-%d-%Y_%H-%M-%S")
            case _:
                return super(CustomEncoder, self).default(obj)

@log_time
def get_adi_data(file_path:str):
    """
    Reads an .adicht file and returns a dictionary of NumPy arrays.
    Structure: { 'ChannelName': [block1_array, block2_array, ...] }
    """
    fp = str(file_path)
    f = adi.read_file(fp)
    data_dict = {}

    num_blocks = getattr(f, 'n_records', 0)
    if num_blocks == 0 and len(f.channels) > 0:
        num_blocks = f.channels[0].n_segments

    for ch in f.channels:
        # Create a list to hold all blocks for this specific channel
        channel_data = []
        
        for block_idx in range(1, num_blocks + 1):
            try:
                # Get data as a numpy array
                segment = ch.get_data(block_idx)
                channel_data.append(segment)
            except Exception:
                # Handle cases where a block might be empty
                channel_data.append(np.array([]))
        
        # Store using the channel's actual name from LabChart
        data_dict[ch.name] = channel_data
        
    return data_dict

@log_time
def save_array(data: dict, fp: str):
    output_path = Path(fp).with_suffix('.npz')
    flat_data = {}
    for channel_name, blocks in data.items():
        for i, block_array in enumerate(blocks):
            # Clean the channel name for filenames (remove spaces/special chars)
            clean_name = "".join([c if c.isalnum() else "_" for c in channel_name])
            key = f"{clean_name}_block_{i+1}"
            flat_data[key] = block_array

    # 3. Save the flattened dictionary
    np.savez_compressed(output_path, **flat_data)
    
    # Log the size for peace of mind
    mb_size = os.path.getsize(output_path) / (1024 * 1024)
    logging.warning(f"Saved {output_path.name} ({mb_size:.2f} MB)")

def run_batch_convert():
    fp = "src/rad_ecg/data/datasets/sharc_fem/"
    inputfiles = Path(fp + "/base/sharc").iterdir()
    for adict_f in inputfiles:
        file_path = PurePath(Path.cwd(), Path(adict_f))
        my_data = get_adi_data(file_path)
        save_array(my_data, PurePath(Path.cwd(), Path(fp), Path("converted"), Path(adict_f.stem)))
        logging.info(f"array {file_path} converted")

if __name__ == "__main__":
    data = run_batch_convert()
    logging.info("Directory converted")