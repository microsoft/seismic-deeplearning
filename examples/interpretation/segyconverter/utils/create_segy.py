import segyio
import numpy as np
from glob import glob
from os import listdir
import os
import pandas as pd
import re
import matplotlib.pyplot as pyplot


def parse_trace_headers(segyfile, n_traces):
    '''
    Parse the segy file trace headers into a pandas dataframe.
    Column names are defined from segyio internal tracefield
    One row per trace
    '''
    # Get all header keys
    headers = segyio.tracefield.keys
    # Initialize dataframe with trace id as index and headers as columns
    df = pd.DataFrame(index=range(1, n_traces + 1),
                      columns=headers.keys())
    # Fill dataframe with all header values
    for k, v in headers.items():
        df[k] = segyfile.attributes(v)[:]
    return df



def parse_text_header(segyfile):
    '''
    Format segy text header into a readable, clean dict
    '''
    raw_header = segyio.tools.wrap(segyfile.text[0])
    # Cut on C*int pattern
    cut_header = re.split(r'C ', raw_header)[1::]
    # Remove end of line return
    text_header = [x.replace('\n', ' ') for x in cut_header]
    text_header[-1] = text_header[-1][:-2]
    # Format in dict
    clean_header = {}
    i = 1
    for item in text_header:
        key = "C" + str(i).rjust(2, '0')
        i += 1
        clean_header[key] = item
    return clean_header



def show_segy_details(segyfile):
    with segyio.open(segyfile, ignore_geometry=True) as segy:
        segydf = parse_trace_headers(segy, segy.tracecount)
        print(f"Loaded from file {segyfile}")
        print(f"\tTracecount: {segy.tracecount}")
        print(f"\tData Shape: {segydf.shape}")
        print(f"\tSample length: {len(segy.samples)}")
        pyplot.figure(figsize=(10,6))
        pyplot.scatter(segydf[['INLINE_3D']],segydf[['CROSSLINE_3D']], marker=",")
        pyplot.xlabel('inline')
        pyplot.ylabel('crossline')
        pyplot.show()

    

    
def load_segy_with_geometry(segyfile):
    try:
        segy = segyio.open(segyfile, ignore_geometry=False)
        segy.mmap()
        print(f"Loaded with geometry: {segyfile} :")
        print(f"\tNum samples per trace: {len(segy.samples)}")
        print(f"\tNum traces in file: {segy.tracecount}")
    except ValueError as ex:
        print(f"Load failed with geometry: {segyfile} :")
        print(ex)
        
        
        
def create_segy_file(masklambda, filename, sorting=segyio.TraceSortingFormat.INLINE_SORTING, ilinerange=[10,50], xlinerange=[100,300]):
    spec = segyio.spec()

    # to create a file from nothing, we need to tell segyio about the structure of
    # the file, i.e. its inline numbers, crossline numbers, etc. You can also add
    # more structural information, but offsets etc. have sensible defautls. This is
    # the absolute minimal specification for a N-by-M volume
    spec.sorting = 2
    spec.format = 1
    spec.samples = range(int(10))
    spec.ilines = range(*map(int, ilinerange))
    spec.xlines = range(*map(int, xlinerange))
    print(f"Written to {filename}")
    print(f"\tinlines: {len(spec.ilines)}")
    print(f"\tcrosslines: {len(spec.xlines)}")

    with segyio.create(filename, spec) as f:
        # one inline consists of 50 traces
        # which in turn consists of 2000 samples
        step = 0.00001
        start = step * len(spec.samples)
        # fill a trace with predictable values: left-of-comma is the inline
        # number. Immediately right of comma is the crossline number
        # the rightmost digits is the index of the sample in that trace meaning
        # looking up an inline's i's jth crosslines' k should be roughly equal
        # to i.j0k
        trace = np.linspace(-1,1,len(spec.samples),True,dtype=np.single)

        if sorting == segyio.TraceSortingFormat.INLINE_SORTING:
            # Write the file trace-by-trace and update headers with iline, xline
            # and offset
            tr = 0
            for il in spec.ilines:
                for xl in spec.xlines:
                    if masklambda(il, xl):
                        f.header[tr] = {
                            segyio.su.offset: 1,
                            segyio.su.iline: il,
                            segyio.su.xline: xl
                        }
                        f.trace[tr] = trace * ((xl / 100.0) + il)
                        tr += 1
                      
            f.bin.update(
                tsort=segyio.TraceSortingFormat.CROSSLINE_SORTING
            )
        else:
            # Write the file trace-by-trace and update headers with iline, xline
            # and offset
            tr = 0
            for il in spec.ilines:
                for xl in spec.xlines:
                    if masklambda(il, xl):
                        f.header[tr] = {
                            segyio.su.offset: 1,
                            segyio.su.iline: il,
                            segyio.su.xline: xl
                        }
                        f.trace[tr] = trace + (xl / 100.0) + il
                        tr += 1
                 
            f.bin.update(
                tsort=segyio.TraceSortingFormat.INLINE_SORTING
            )
        print(f"\ttraces: {tr}")