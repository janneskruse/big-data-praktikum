############# Import the necessary modules #############
#python basemodules and jupyter modules
import os
import sys
import shutil
import psutil
import multiprocessing as mp
import re
from datetime import datetime, timedelta
from operator import itemgetter
import fnmatch
from dotenv import load_dotenv

#benchmarking
import time
import cProfile
import pstats
import io

# data handling
import h5py
import xarray as xr
import dask.array as da
import numpy as np
import pandas as pd
import scipy
from scipy import signal, fft
import pyfftw
import pyfftw.interfaces.dask_fft as dafft
import pickle
import zarr

############# Parse the command line arguments #############
total_cpus = int(mp.cpu_count())


############# Base paths and folder names #############
# get the base path of the repository
repo_dir = os.popen('git rev-parse --show-toplevel').read().strip()
###load the .env file
load_dotenv(dotenv_path=f"{repo_dir}/.env")

###########get the environment vairables#########
base=os.getenv("BASE_FOLDER")
zarr_base=os.getenv("ZARR_BASE_FOLDER")
print(f"Base data folder:{base}")

startTime = time.time()


############# Define the functions #############
def get_sorted_folders (base):
    """
    Groups folders by date and sorts them chronologically.
    
    Args:
        base (str): The base folder to search for days.
        
    Returns:
        list: A list of dates in the format "YYYYMMDD".
    """
    
    # Change to the base directory
    os.chdir(base)
    folders = os.listdir()
    
    # Define the date pattern
    date_pattern = re.compile(r"(\d{8})_?\d*")  # Match the date in the folder name
    date_folders = {}

    # Group folders by date
    for folder in folders:
        match = date_pattern.match(folder)
        if match:
            date_str = match.group(1)
            if date_str in date_folders:
                date_folders[date_str].append(folder)
            else:
                date_folders[date_str] = [folder]

    print("Number of folders before moving files:", len(folders))
    
    # Sort folders within each date group
    for date in date_folders:
        date_folders[date].sort(key=lambda x: (x.split('_')[0], int(x.split('_')[1]) if '_' in x else 0))
    
    # Combine folders for each date where there are multiple folders
    for date, folders in date_folders.items():
        if len(folders) > 1:
            combine_folders_with_same_date(folders)
    
    os.chdir(base)
    print("Number of folders after moving files:", len(os.listdir()))        
            
    # sort the dates
    date_folders = dict(sorted(date_folders.items(), key=lambda x: x[0]))        
    
    return list(date_folders.keys())

def combine_folders_with_same_date(folders):
    """
    Combines folders with the same date into one folder.

    Args:
        base (str): The base folder to search for days.
    """
    primary_folder = folders[0]
    folder_path = os.path.join(base, folders[1])
    primary_folder_path = os.path.join(base, primary_folder)
    
    # Move contents to the primary folder
    files=os.listdir(folder_path)
        
    pool=mp.Pool(mp.cpu_count())
    pool.starmap(move_files, [(filename, folder_path, primary_folder_path) for filename in files])
    pool.close()
    pool.join()
    
    # Remove the now-empty folder
    os.system(f"rm -rf {folder_path}")
    print(f"Combined {folders[1]} into {primary_folder}.") 

def move_files(filename, folder_path, primary_folder_path):
    """
    Moves a file from a folder to the primary folder.
    
    Args:
        filename (str): The name of the file to move.
        folder_path (str): The path to the folder containing the file.
        primary_folder_path (str): The path to the primary folder.
    """
    if not os.path.exists(os.path.join(primary_folder_path, filename)): # Check if the file already exists in the primary folder
        join=os.path.join(folder_path, filename)
        shutil.move(join, primary_folder_path) # Move the file to the primary folder
    # else:
    #     print("already exists")

def extract_timestamp(filename):
    """
    Extracts the timestamp from a DAS-h5-file's filename.
    
    Args:
        filename (str): The filename to extract the timestamp from.
        
    Returns:
        str: The timestamp part of the filename.
    """
    # for the format is 'rhone1khz_UTC_yyyymmdd_hhmmss.ms.h5'
    timestamp_part = filename.split('_')[2] + filename.split('_')[3]
    return timestamp_part

def get_filenames(folder, base):
    """
    Collects the filenames in the data folder and sorts them by time.

    Args:
        folder (str): The folder to search for files.

    Returns:
        dict: A dictionary where keys are integers and values are filenames, sorted chronologically.
    """
    # Change to the folder directory
    folder_path = os.path.join(base, folder)
    os.chdir(folder_path)
    files=os.listdir()
    
    # filter files to only get those where the filename contains 'rhone2das' or 'rhone1khz'
    # Use fnmatch.filter to find matches for both patterns in a single pass
    pattern_matches = fnmatch.filter(files, '*rhone2das*') + fnmatch.filter(files, '*rhone1khz*')
    
    # Sort the files by timestamp
    sorted_files = sorted(pattern_matches, key=extract_timestamp)
    
    return  sorted_files 

def channel_fourier(data, args, taper, positions):
    """
    Applies the Fourier transform to segments of the DAS records using the pyFFTW library.
    
    Args:
        data (np.array): The DAS data.
        args (dict): The arguments for the Fourier transform.
            Args requires the following keys:
                "fft_dtype" (str): The data type for the Fourier transform.
                "num_frequency_points" (int): The number of frequency points.
                "start_channel_index" (int): The start channel index.
                "end_channel_index" (int): The end channel index.
                "seg_len" (int): The segment length.
                "hop" (int): The hop size.
                "n_samples" (int): The number of samples.
        taper (np.array): The taper function.
        positions (np.array): The positions of the segments in the data.
        
    Returns:
        np.array: The Fourier transformed segments.
    """
    
    # Unpack the arguments
    seg_len = args["seg_len"]
    end_channel_index, start_channel_index = args["end_channel_index"], args["start_channel_index"]
    fft_dtype = args["fft_dtype"]  # dtype: float32
    n_segments = positions.shape[0]

    # Pre-allocate the segments array
    segments = ([data[pos:pos+seg_len] for pos in positions])
    segments = [seg.T[start_channel_index:end_channel_index] for seg in segments]
    Fsegs = np.zeros((n_segments, end_channel_index - start_channel_index, num_frequency_points), dtype=fft_dtype)  # empty float32 array
    
    # Pre-allocate the input array for FFTW
    fft_input = pyfftw.empty_aligned(seg_len, dtype=fft_dtype)
    # Create the FFTW object
    fft_object = pyfftw.builders.rfft(fft_input)  # , planner_effort='FFTW_ESTIMATE') #, threads=mp.cpu_count()//2)
    
    # Prepare slices for efficient slicing
    channel_slice = slice(start_channel_index, end_channel_index)
    
    for i in range(n_segments):
        for channel_number, channel in enumerate(segments[i]):
            fft_input[:] = taper * channel  # Apply taper
            #np.multiply(taper, channel, out=fft_input)  # Apply taper
            fft_output = fft_object()  # Execute FFT
            fourier_transformed = (10 * np.log(np.abs(fft_output) ** 2 + 1e-10))[:num_frequency_points]  # Compute power spectrum
            fourier_transformed[0] = 0  # Remove DC component (average value of the signal)
            Fsegs[i][channel_number] = fourier_transformed
    
    return Fsegs  # return the Fourier transformed segments

 

def create_spectro_segment(file_index, args, filelist):
    """
    Creates a spectrogram segment from a file.
    
    Args:
        file_index (int): The index of the file.
        args (dict): The arguments for the Fourier transform.
            Args requires the following keys:
                "n_files" (int): The number of files.
                "seg_len" (int): The segment length.
                "hop" (int): The hop size.
                "n_samples" (int): The number of samples.
        filelist (list): The list of file names.
        
    Returns:
        np.array: The Fourier transformed segments.
        int: The number of segments.
    """
    
    # chunk args
    n_files=args["n_files"]
    seg_len=args["seg_len"]
    hop=args["hop"]
    n_samples=args["n_samples"]
    filename=filelist[file_index]
    float_type=args["fft_dtype"]
    
    #taper function
    taper = signal.windows.tukey(seg_len, 0.25)  # reduces the amplitude of the discontinuities at the boundaries, thereby reducing spectral leakage.
    
    # Load the data
    # xr_h5 = xr.open_dataset(filename, engine='h5netcdf', backend_kwargs={'phony_dims': 'access'})
    # data = xr_h5["Acoustic"].compute().values.astype(float_type)
    
    f = h5py.File(filename,'r')
    dset=f['Acoustic']
    data = np.array(dset)
    
    if file_index < n_files - 1:
        #xr_h5_2 = xr.open_dataset(filelist[file_index + 1], engine='h5netcdf', backend_kwargs={'phony_dims': 'access'})
        #data_2 = xr_h5_2["Acoustic"].compute().values.astype(float_type)
        f = h5py.File(filelist[file_index+1],'r')
        dset=f['Acoustic']
        data_2 = np.array(dset)
        data = np.concatenate((data, data_2[0:seg_len]), axis=0)
    
    next_file_index = file_index+1
    file_pos = file_index * n_samples

    # If the current file is not the last one
    if file_index != n_files-1:
        # Calculate the starting positions of each segment in the data
        # first segment: (next_file_index-1)*n_samples/hop, rounded up
        # last segment: (next_file_index*n_samples-1)/hop, rounded down
        positions = np.arange(np.ceil((file_index)*n_samples/hop), np.floor((next_file_index*n_samples-1)/hop)+1, dtype=int)*hop - file_pos # scaled by the hop size and offset by the file position
    else:
        # If last one, start: (next_file_index*n_samples-seg_len)/hop
        # to ensure that the last segment doesn't extend beyond the end of the data
        positions = np.arange(np.ceil((file_index)*n_samples/hop), np.floor((next_file_index*n_samples-seg_len)/hop)+1, dtype=int)*hop - file_pos

    
    # Filter positions to ensure valid segments
    positions = positions[positions + seg_len <= len(data)]
    
    start=time.time()
    Fsegs = channel_fourier(data, args, taper, positions)
    print(f"Time taken for fft of {filename}: {time.time()-start}")
    
    return Fsegs, positions.shape[0]




##########Base settings#########
#granularity of spectrogram
freq_res = int(os.getenv("FREQ_RES", 1)) # frequency resolution in Hz
time_res = float(os.getenv("TIME_RES", 0.1)) # time res in seconds
float_type = os.getenv("FLOAT_TYPE", 'float32')

# section
channel_distance = int(os.getenv("CHANNEL_DISTANCE", 4)) # distance between channels in meters
cable_start = int(os.getenv("CABLE_START", 0)) # cable section to be processed (in meters) - 0==start
cable_end = int(os.getenv("CABLE_END", 9200)) # cable section to be processed (in meters) - 0==start
start_channel_index, end_channel_index = cable_start // channel_distance, cable_end // channel_distance # channel distances to indices
expected_channels = end_channel_index - start_channel_index # expected number of channels

# Additional parameters:
file_length = int(os.getenv("FILE_LENGTH", 30)) # Length of a single h5 file in seconds.
sample_freq = int(os.getenv("SAMPLE_FREQ", 1000)) # Sampling frequency in Hz of the recorded data.
freq_max = int(os.getenv("FREQ_MAX", 100)) # maximum frequency cut off value for the analysis
seg_length=1/freq_res #calculate window length corresponding to freq_res
n_samples = file_length*sample_freq #number of samples in one file/total number of data points available in one file
num_frequency_points = int(seg_length*freq_max+1)
seg_sample_len=int(seg_length*sample_freq) # how many time points should be in one processing window
n_segments_file=int(2*(file_length/seg_length)) # amount of segments for the desired window length
location_coords = np.arange(cable_start, cable_end, 4) # channel locations
freq_coords=scipy.fft.rfftfreq(int(sample_freq/freq_res), 1/sample_freq)[:num_frequency_points] # frequency coordinates
hop = int(time_res*sample_freq) # hop size - how many samples to skip between segments

#fft input arguments
args = {
    "fft_dtype": float_type,
    "num_frequency_points" : num_frequency_points,
    "start_channel_index" : start_channel_index,
    "end_channel_index" : end_channel_index,
    "seg_len" : seg_sample_len,
    "hop" : hop,
    "n_samples" : n_samples,
    "seg_length" : seg_length,
    "expected_channels" : expected_channels
}


##########Main#########
if __name__=='__main__':
    
    # set the folder
    folders=get_sorted_folders(base)
    folder=folders[0]
    
    #path and name of resulting zarr-formatted data cube.
    zarr_name = f"cryo_cube_{folder}.zarr"
    zarr_path = f"{zarr_base}/{zarr_name}"
    
    os.chdir(base) # change to the base directory
    print(f"base folder: {os.getcwd()}")
    while os.path.exists(zarr_path) and folders:  # Check if folders is not empty
        folder = folders.pop(0)  # remove and return the first element
        zarr_name = f"cryo_cube_{folder}.zarr"
        zarr_path = f"{zarr_base}/{zarr_name}"
    
    if not folders:
        print("No more folders to process.")
        sys.exit(0)
    else:
        print(f"Processing folder {folder}")
    
    #get the day and month
    day=folder[6:8]
    month=folder[4:6]
    
    # print the settings
    print(20*"*")
    print("Max number of CPUs: ", total_cpus)
    print(f"Processed day: {day}.{month}.2020")
    print(f"Time resolution: {time_res} sec")
    print(f"Frequency resolution: {freq_res} Hz")
    print(f"Resulting overlap: {1-hop/seg_sample_len}")
    print(f"Expected number of channels: {expected_channels}")
    print(10*"*")
    
    # get the filenames and the total amount of segments
    filenames = get_filenames(folder, base) #[0:10]
    n_files=len(filenames)
    args["n_files"] = n_files
    print("Number of files:", n_files)
    print("filenames", filenames)
    n_segments_total = int(np.floor((n_files*n_samples-seg_sample_len)/hop))+1 # total amount of segments

    print(f"Creating zarr shape...")
    # creating zarr shape
    z_shape=(n_segments_total, expected_channels, num_frequency_points) 
    z_chunks=(n_segments_file,expected_channels,num_frequency_points)

    print("Creating metadata...")
    start=time.time()

    # Generate time coordinates based on the first file
    dummy_file_path=os.path.join(base, folder, filenames[0])
    dummy_xr = xr.open_dataset(filenames[0], engine='h5netcdf', backend_kwargs={'phony_dims': 'access'})
    attr = dummy_xr['Acoustic'].attrs
    start_time = np.datetime64(attr["ISO8601 Timestamp"], 'ns') # Get the start time of the first file
    time_res_ms = time_res * 1000  # Convert time_res from seconds to milliseconds
    time_coords = start_time + np.arange(n_segments_total) * np.timedelta64(int(time_res_ms), 'ms') 
    
    
    fft_dask = da.zeros(z_shape, chunks=z_chunks, dtype=float_type) # create an empty dask array

    xr_zarr = xr.Dataset(
        {
            "fft": (["time", "channel", "frequency"], fft_dask),
        },
        coords={
            "time": time_coords,
            "channel": location_coords,
            "frequency": freq_coords,
        },
    )
    print(xr_zarr)
        
    print(f"metadata created in {time.time()-start}s:")

    #xarray dataset to zarr
    print(f"Creating and writing empty {zarr_path} with metadata...")
    start=time.time()
    xr_zarr.to_zarr(zarr_path, mode='w', consolidated=True)
    print(f"zarr created in {time.time()-start}s")
     
    # In the following lines, multiple cpu-cores calculate
    # a fft for each file simultanously.
    # Before that we split the whole data to be processed in to not overload the memory!

    # Determine available system memory 
    available_memory = psutil.virtual_memory().available * 0.8  # Use 80% of available memory (let's reserve some memory for the system and other processes)

    # Calculate how many files can be processed simultaneously
    memory_per_file = os.path.getsize(dummy_file_path)*1.5 # 1.5 times the size of the file, assuming some overhead
    print("Memory per file (MB):", memory_per_file / (1024**2))
    files_at_once = int(available_memory / memory_per_file)
    files_at_once = max(1, files_at_once) # Ensure that at least one file is processed at once

    # Calculate the number of divisions 
    n_div = max(1, n_files // files_at_once)
    index_list = np.arange(n_files)

    # set split up
    if n_files > files_at_once:
        split_up = np.array_split(index_list, n_div)
    else:
        split_up = [index_list]

    # Define the number of cores to be 90% of available cores, rounded down
    n_cores = int(total_cpus * 0.9) // 1
    print("Number of cores used:", n_cores)

    startT = time.time() # start the timer

    running_index=0
    for liste in split_up:
        
        print(f"Starting FFT for split_up liste {liste}")
        # Start the local timer
        start = time.time() 
        
        # multiprocessing the fft calculation
        pool=mp.Pool(n_cores)
        fft_results=pool.starmap(create_spectro_segment, [(i, args, filenames) for i in liste])
        pool.close()
        pool.join()
        
        # Print the time taken to process the files
        end=time.time() # end the local timer
        fft_time=end-start
        time_per_file=fft_time/len(liste)
        
        print(20*"*")
        print(f"Time taken for fft from splitup: {fft_time}")
        print(f"Time per File: {time_per_file}")
        print(20*"*")
        
        fft_results = list(fft_results) # convert the map object to a list
        
        print("Writing liste to zarr...")
        start=time.time()
        
        # Convert results to Dask arrays and store them
        for i in liste:
            Fsegs, nseg = fft_results[i-int(liste[0])]
            nseg = int(nseg)
            dask_array = da.from_array(Fsegs, chunks=(nseg, expected_channels, num_frequency_points))
            xr_zarr["fft"][running_index:running_index+nseg] = dask_array
            running_index += nseg
            
        xr_zarr.to_zarr(zarr_path, mode='a', consolidated=True)
        print(f"Wrote FFT to zarr using Dask in {time.time()-start}s for split_up {liste}")
        
    print(20*"*")
    print("Calculation completed")
    print("Total computation time in seconds:", time.time()-startTime) 
    print("Computation time in seconds for fft and Zarr storage:", time.time()-startT)
    print("Number of processed files:", n_files)
    print("Number of used cores:", n_cores)
    print(20*"*")
    
    # submit the script again
    if len(folders)>30: #
        print(f"Submitting the script again to process the next folder {folders[0]}")
        os.chdir(f"{repo_dir}/code/slurm")
        #os.system(f"sbatch 02_fft_pipeline.sh") #uncomment this to autmatically submit the next script for the next folder when finished
    
    