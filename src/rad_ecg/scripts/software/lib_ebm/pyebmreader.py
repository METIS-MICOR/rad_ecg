"""
==============================
Embla file reader from python
==============================

This script is a python version of : https://github.com/gpiantoni/hgse_private/blob/master/ebmread.m

Version 0.21

Author: Yaopeng Ma    mayaope@biu.ac.il

date: 3.20.2021

"""
import numpy as np
from struct import unpack
import datetime
import matplotlib.pyplot as plt

# parameter define
EBM_RAWDATA_HEADER = 'Embla data file'
EBM_RESULTS_HEADER = 'Embla results file'

EBM_R_VERSION = b'\x80'
EBM_R_SUBJECT_INFO = b'\xd0'
EBM_R_RATECORR = b'\x8a'
EBM_R_HEADER = b'\x81'
EBM_R_TIME = b'\x84'
EBM_R_CHANNEL = b'\x85'
EBM_R_SAMPLING_RATE = b'\x86'
EBM_R_UNIT_GAIN = b'\x87'
EBM_R_CHANNEL_NAME = b'\x90'
EBM_R_DATA = b'\x20'
EBM_UNKNOWN_DATASIZE = b'\xff\xff\xff\xff'
EBM_END_OF_SIGNATURE = b'\x1A'
EBM_MAX_SIGNATURE_SIZE = 80
EBM_MAC_ENDIAN = b'\xff'
EBM_INTEL_ENDIAN = b'\x00'

ERROR_UNKNOWN_SIGNATURE = 'This is not a Embla data file'
ERROR_FILE_NOT_FOUND = 'Could not open data file'
ERROR_UNKNOWN = 'Failure in reading the file'
ERROR_CANCEL = 'Operation was canceled'

endian = 'big'
# Big endian by default

SIZE_uchar = 1
SIZE_char = 1
SIZE_ulong = 4
SIZE_long = 4
SIZE_int8 = 1
SIZE_int16 = 2


def unpack_one(num_type, buffer, endian):
    se = ">" if endian == "big" else "<"
    return unpack(se + num_type, buffer)[0]


def cal_stoptime(starttime, deltat):
    seconds = int(deltat)
    microsec = (deltat - seconds) * 1e6
    dt = datetime.timedelta(seconds=seconds, microseconds=microsec)
    return starttime + dt


def ebmreader(filepath, onlyheader=False):
    with open(filepath, "rb") as f:
        header = {}
        signature = []
        signature.append(f.read(SIZE_char))
        i = 0
        while signature[
                -1] != EBM_END_OF_SIGNATURE and i < EBM_MAX_SIGNATURE_SIZE - 1:
            i = i + 1
            signature.append(f.read(SIZE_char))
        signature = "".join(map(lambda x: x.decode("windows-1252"), signature))

        assert i != EBM_MAX_SIGNATURE_SIZE - 1, ERROR_UNKNOWN_SIGNATURE
        assert EBM_RAWDATA_HEADER in signature, ERROR_UNKNOWN_SIGNATURE

        ch = f.read(SIZE_char)
        if ch == EBM_MAC_ENDIAN:
            endian = "big"
        elif ch == EBM_INTEL_ENDIAN:
            endian = "little"
        wideId = 1
        # Store the position of the start of the block structure
        # If this is not a file with 8 bit block IDs then we will change
        # this again.
        firstBlockOffset = f.tell()
        ch = f.read(SIZE_uchar)
        if ch == b"\xff":
            ch = f.read(SIZE_ulong)
            if ch == b"\xff\xff\xff\xff":
                # we have 32 bit block IDs so we skip the rest of the
                # 32 byte header and store the position of the block
                # structure which should start right after.

                firstBlockOffset = firstBlockOffset + 31
                wideId = 1
        f.seek(firstBlockOffset, 0)

        # find  the data block
        rec = 0
        recnum = -1
        header["starttime"] = []
        header["stoptime"] = []
        header["length"] = []
        data = []
        while True:
            if wideId != 0:
                rec = unpack_one("L", f.read(SIZE_ulong), endian)
            else:
                rec = unpack_one("B", f.read(SIZE_uchar), endian)
            recSize = unpack_one("l", f.read(SIZE_long), endian)
            recPos = f.tell()
            if rec == int.from_bytes(EBM_R_VERSION, endian):
                minor = int.from_bytes(f.read(SIZE_int8), endian)
                major = int.from_bytes(f.read(SIZE_int8), endian)
                header["fileversion"] = major + 0.01 * minor
            if rec == int.from_bytes(EBM_R_SUBJECT_INFO, endian):
                tmp = f.read(SIZE_int8 * recSize)
                header["subjectinfo"] = tmp.decode("windows-1252").rstrip(
                    "\x00")
            if rec == int.from_bytes(EBM_R_HEADER, endian):
                tmp = f.read(SIZE_int8 * recSize)
                header["extra"] = tmp.decode("windows-1252").rstrip("\x00")
            if rec == int.from_bytes(EBM_R_TIME, endian):
                year = unpack_one("h", f.read(SIZE_int16), endian)
                month = int.from_bytes(f.read(SIZE_int8), endian)
                day = int.from_bytes(f.read(SIZE_int8), endian)
                hour = int.from_bytes(f.read(SIZE_int8), endian)
                minute = int.from_bytes(f.read(SIZE_int8), endian)
                second = int.from_bytes(f.read(SIZE_int8), endian)
                hsec = int.from_bytes(f.read(SIZE_int8), endian) * 10000

                times_data = (year, month, day, hour, minute, second, hsec)
                recnum = recnum + 1
                header["starttime"].append(datetime.datetime(*times_data))
            if rec == int.from_bytes(EBM_R_CHANNEL, endian):
                header["channel"] = unpack_one("h", f.read(SIZE_int16), endian)

            if rec == int.from_bytes(EBM_R_SAMPLING_RATE, endian):
                header["frequency"] = unpack_one("L", f.read(SIZE_long),
                                                 endian) / 1000
            if rec == int.from_bytes(EBM_R_RATECORR, endian):
                header["sec_error"] = unpack_one("d", f.read(8), endian)
            if rec == int.from_bytes(EBM_R_UNIT_GAIN, endian):
                header["unitgain"] = unpack_one("l", f.read(SIZE_long),
                                                endian) * 1e-9
            if rec == int.from_bytes(EBM_R_CHANNEL_NAME, endian):
                tmp = f.read(recSize * SIZE_int8)
                header["channelname"] = tmp.decode("windows-1252").strip(
                    "\x00")

            # read data
            if rec == int.from_bytes(EBM_R_DATA, endian):
                if onlyheader is False:
                    newdata = f.read(recSize)
                    newdata = np.frombuffer(newdata, np.int16)
                    newdata = newdata * header["unitgain"]
                    data.append(newdata)
                else:
                    f.seek(recSize, 1)
                current = header["starttime"][recnum]
                header["stoptime"].append(
                    cal_stoptime(current, recSize / 2 / header["frequency"]))
                header["length"].append(recSize // 2)
            b = f.read(1)
            if not b:
                break
            else:
                f.seek(recPos + recSize, 0)
        if len(header["stoptime"]) > 1:
            header["interrupt length"] = [
                (stop - start).seconds
                for start, stop in zip(header["starttime"], header["stoptime"])
            ]
    return data, header


if __name__ == "__main__":
    

    data, header = ebmreader(
        r"C:\githubrepo\MiCOR\opens_ecg\src\rad_ecg\data\datasets\methadone\EKG_OPAQ 445 - 6_29_2008.ebm",
        onlyheader=True)
    print(header["starttime"])
    print(header["length"])
    print(data)

    # plt.figure()
    # plt.plot(data[-1][:])
    # plt.show()