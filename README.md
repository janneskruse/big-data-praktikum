# Big-Data-Praktikum

Use Jupyterlab Software to load FFTW 3.3.10
    - This needs to be somehow implemented into slurm later on
    
## Installation
The instructions on how to get the scripts running with the correct conda environment can be found [here](./code/condaenv).

## Understanding of the h5 Files

1. H5 Files:

- We are working with H5 files that store time-series data from a DAS (Distributed Acoustic Sensing)

2. Acoustic Attribute:

- Each H5 file contains an attribute called Acoustic.
- This Acoustic attribute contains the recorded data, which is likely a 2D array where one dimension represents time and the other represents spatial channels.

3. Timesteps and Duration:

- Each H5 file covers a duration of 30 seconds.
- Given a sampling frequency of 1000 Hz, there should be 30,000 timesteps (or samples) in each file (since 30 seconds * 1000 samples/second = 30,000 samples).
- Most of the files have only 12000-13000 rows

4. Data Structure:

- Each timestep corresponds to a row in the 2D array, and each column in that row corresponds to a different channel (spatial location).
- Therefore, we have a 2D array of shape (30000, number_of_channels).

5. Segmentation:

- The data is divided into segments for time-frequency analysis.
- Each segment is a subset of the 30,000 timesteps. For instance, if the segment length corresponds to 1 second, each segment would contain 1000 timesteps.
- Segments can overlap based on the hop size defined in your code. This overlap helps in capturing more continuous changes in the signal.

6. Channel-wise Analysis:

- Within each segment, you further analyze the data channel by channel.
- Each segment contains data from all channels for that specific time window.

7. Fourier Transformation:

- We apply the Fourier transform (FFT) to each channel within each segment.
- The FFT is applied to transform the time-domain data into the frequency domain, giving us the frequency content of the signal for that specific segment and channel.