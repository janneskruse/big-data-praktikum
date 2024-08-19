# Big-Data-Praktikum

Git Repository for the class [Big Data Praktikum](https://git.informatik.uni-leipzig.de/dbs/big-data-praktikum/-/tree/master) of University of Leipzig. Original Codebase corresponding to the topic Signal Processing on Cryoseismological data from Rhônegletscher (Switzerland): [CryoCube](https://github.com/JosepinaU/CryoCube).

---
authors: Louis Trinkle, Jannes Kruse

---
    
## Repo structure
All the code developed during the course can be found inside [code](./code/).
Inside [code](./code/) there is the folders:

- [condaenv](./code/condaenv/) to setup the conda environment for all the scripts present
- [notebooks](./code/notebooks) with all the notebooks explaining the pipelines and to visualize the cubes. This includes a [script](./code/notebooks/04_streamingCube.ipynb) to stream an example cube from an AWS S3 Bucket uploaded in the following pipeline: [03_S3Upload.ipynb(./code/notebooks/03_S3Upload.ipynb)
- [slurm](./code/slurm/) containing the python pipelines for uploading the files to the cluster from the corresponding cloud or storage location, creating the cube including a fourier (and wavelet transformation - would hang after a certain amount of channels) and a pipeline to upload the resulting cubes to an AWS S3 Bucket

The key hardcoded variables, including password access to AWS or the Nextcloud are stored in an .env file loaded by the scripts. To be able to run the scripts, you would have to create a new .env file in the root of this git repository and copy the following into it:

```
BASE_FOLDER=<Your download folder with the DAS data e.g. "/work/le837wmue-Rhone-download/le837wmue-Rhone_download-1720747219/DAS_2020">
ZARR_BASE_FOLDER= <Your download folder with the DAS data e.g. "/work/ju554xqou-cryo_cube">
FLOAT_TYPE=float32
FREQ_RES=1
FREQ_MAX=100
TIME_RES=0.1
FILE_LENGTH=30
SAMPLE_FREQ=200
AWS_ACCOUNT_ID = <Your AWS account id>
AWS_ACCESS_KEY_ID = <Your access key>
AWS_SECRET_ACCESS_KEY = <Your secret key>
AWS_REGION = eu-north-1
AWS_MAX_MEMORY_MB = 5000
AWS_BUCKET_NAME = rhone-glacier-das
NEXTCLOUD_BASE = "https://cloud.scadsai.uni-leipzig.de/index.php/s/gozxE5r9YdwGL8w/download"
NEXTCLOUD_USERNAME=<youremail@example.de>
NEXTCLOUD_PW=""
LD_LIBRARY_PATH=~/.conda/envs/rhoneCube/lib:$LD_LIBRARY_PATH
```

Be aware, that the keys and ids for AWS and the cloud you will have to provide yourself.
The LD_LIBRARY_PATH (on linux like systems) needs to be set  for the libfftw3f.so.3 library to be loaded, which we installed with conda. This is needed for the wavelet transformation implemented.
You can get the path by running the following in your activated conda environment:
```bash
find $CONDA_PREFIX -name "libfftw3f.so.3"
```

Instructions on the AWS setup you can find [below](#aws-set-up).


## Installation
The instructions on how to get the scripts running with the correct conda environment can be found [here](./code/condaenv).

## AWS Set-up

1. Create an AWS-Account and add the account id to the .env file
2. Obtain an [Access Key](https://us-east-1.console.aws.amazon.com/iam/home#/security_credentials/access-key-wizard) 
3. Add the obtained AWS access key id and the AWS secret access key to your .env file (variables from above)
4. Create the S3 Bucket and upload the .zarr cubes following the S3Upload Notebook [here](./code/notebooks/S3Upload.ipynb).
There is a pipeline to multithread the upload with slurm as well:  [03_s3_upload.py](./code/slurm/03_s3_upload.py).

