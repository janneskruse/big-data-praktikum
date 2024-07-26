# Big-Data-Praktikum

Git Repository zum Kurs [Big Data Praktikum](https://git.informatik.uni-leipzig.de/dbs/big-data-praktikum/-/tree/master). Zum Thema Signal Processing on Cryoseismological data from Rhônegletscher (Switzerland) gehöriges Git Repo: [CryoCube](https://github.com/JosepinaU/CryoCube).

---
authors: Louis Trinkle, Jannes Kruse

---
    
## Repo structure
All the code developed during the course can be found inside [code](./code/).
Inside [code](./code/). there is thre folders:

- [condaenv](./code/condaenv/) to setup the conda environment for all the scripts present
- [notebooks](./code/notebooks) with all the notebooks explaining the pipelines and to visualize the cubes. This includes a [script](./code/notebooks/AWS_streaming.ipynb) to stream an example cube from an AWS S3 Bucket uploaded in the following pipeline: .
- [slurm](./code/slurm/) containing the python pipelines for uploading the files to the cluster from the corresponding cloud or storage location, creating the cube including a fourier and wavelet transformation and a pipeline to upload resulting cubes to an AWS S3 Bucket

The key hardcoded variables, including password access to AWS or the Nextcloud are stored in an .env file loaded by teh scripts. To be able to run the scripts, you would have to create a new .env file and copy the following into it:

```
BASE_FOLDER="/work/le837wmue-Rhone-download/le837wmue-Rhone_download-1720747219/DAS_2020"
ZARR_BASE_FOLDER="/work/ju554xqou-cryo_cube"
FLOAT_TYPE=float32
FREQ_RES=1
FREQ_MAX=100
TIME_RES=0.1
FILE_LENGTH=30
SAMPLE_FREQ=1000
CHANNEL_DISTANCE=4
CABLE_START=0
CABLE_END=9200
AWS_KEY=#yourawskey
NEXTCLOUD_BASE = "https://cloud.scadsai.uni-leipzig.de/index.php/s/gozxE5r9YdwGL8w/download"
NEXTCLOUD_USERNAME=#youremail@example.de
NEXTCLOUD_PW=""
LD_LIBRARY_PATH=/home/sc.uni-leipzig.de/ju554xqou/.conda/envs/rhoneCube/lib:$LD_LIBRARY_PATH
```

Be aware, that the password for AWS and the cloud you will have to provide yourself.
The LD_LIBRARY_PATH needs to be set for the libfftw3f.so.3 library which we installed with conda. This is needed for the wavelet transformation we implemented.
You can get the path by running the following in your activated conda environment:
```bash
find $CONDA_PREFIX -name "libfftw3f.so.3"
```


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

## AWS Set-up

1. Create an AWS-Account
2. Obtain an [Access Key](https://us-east-1.console.aws.amazon.com/iam/home#/security_credentials/access-key-wizard) 
3. add it to your config AWS file - follow Option 3 for this [tutorial](https://wellarchitectedlabs.com/common/documentation/aws_credentials/).
4. Create S3 Bucket and upload .zarr following the S3Upload Notebook here.
5. Set the Bucket Policy to public. To do so, add this:

```(json)
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadForAllObjects",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::your_s3_bucket/*"
        }
    ]
}
```