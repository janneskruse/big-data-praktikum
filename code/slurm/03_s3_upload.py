############# Import the necessary modules #############
#python basemodules and jupyter modules
import time
import logging
import os
import sys
import subprocess
import json
import glob
import re
import multiprocessing as mp
from dotenv import load_dotenv
# get the base path of the repository
repo_dir = os.popen('git rev-parse --show-toplevel').read().strip()
###load the .env file
load_dotenv(dotenv_path=f"{repo_dir}/.env")


# AWS SDK modules
import boto3
from botocore.exceptions import ClientError


############# Create .aws folder with credentials and config file #############
os.chdir(os.path.expanduser("~"))
if not os.path.exists(".aws"):
    os.mkdir(".aws")
os.chdir(".aws")
# Create a credentials file with the access key and secret key from the .env file
with open("credentials", "w") as file:
    file.write(f"[default]\naws_access_key_id = {os.getenv('AWS_ACCESS_KEY_ID')}\naws_secret_access_key = {os.getenv('AWS_SECRET_ACCESS_KEY')}")
# Create a config file with the region from the .env file and json as output format
with open("config", "w") as file:
    file.write(f"[default]\nregion = {os.getenv('AWS_REGION')}\noutput = json")
#print(f"Credentials and config files created in {os.getcwd()}: {os.listdir()}")
#read credentials file
# with open("credentials", "r") as file:
#     print(file.read())

############## calculate the number of cores for distributed processing
total_cpus = int(sys.argv[1])
n_cores = int(total_cpus * 0.9) // 1
print("Number of cores used:", n_cores)

############# Get the AWS parameters from .env #############
def convert_to_valid_bucket_name(original_name: str) -> str:
    """
    Convert a string to a valid S3 bucket name.
    
    :param original_name: The original name to convert.
    :return: The converted name.
    """
    
    # Convert to lowercase
    bucket_name = original_name.lower()

    # Replace underscores and spaces with hyphens
    bucket_name = re.sub(r'[_\s]+', '-', bucket_name)

    # Remove any character that isn't lowercase letter, number, or hyphen
    bucket_name = re.sub(r'[^a-z0-9-]', '', bucket_name)

    # Ensure the name starts and ends with a letter or number
    bucket_name = re.sub(r'(^-|-$)', '', bucket_name)

    # Trim the name to 63 characters if too long
    bucket_name = bucket_name[:63]

    # Ensure the name is at least 3 characters long
    if len(bucket_name) < 3:
        bucket_name = bucket_name.ljust(3, 'a')  # Pad with 'a' if too short

    return bucket_name

# Set the bucket name and memory limit
os.chdir(repo_dir)
bucket_name = convert_to_valid_bucket_name(os.getenv("BUCKET_NAME", "rhone-glacier-das"))
memory_limit=int(os.getenv('AWS_MAX_MEMORY_MB', '0'))
region=os.getenv('AWS_REGION', 'eu-north-1')
print(f"Bucket name: {bucket_name}")

############# Setup an AWS client #############
s3_client = boto3.client('s3', region_name=region)
print(f"Client created: {s3_client}")



############# Define the s3 bucket functions #############
def bucket_exists(bucket_name: str, s3_client: boto3.client) -> bool:
    """Check if an S3 bucket with the specified name already exists.

    :param bucket_name: Name of the bucket to check
    :param s3_client: Boto3 S3 client
    :return: True if the bucket exists, else False
    """
    
    try:
        # List all buckets
        response = s3_client.list_buckets()
        # Check if the bucket exists in the list of buckets
        for bucket in response['Buckets']:
            if bucket['Name'] == bucket_name:
                return True
        return False
    except ClientError as e:
        logging.error(e)
        return False
    
def create_bucket(bucket_name: str, s3_client: boto3.client, region: str) -> bool:
    """Create an S3 bucket in a specified region

    If a region is not specified, the bucket is created in the S3 default
    region (us-east-1).

    :param bucket_name: Bucket to create
    :param s3_client: Boto3 S3 client
    :param region: String region to create bucket in, e.g., 'us-west-2'
    :return: True if bucket created, else False
    """


    # Check if the bucket already exists
    if bucket_exists(bucket_name, s3_client):
        print(f"Bucket {bucket_name} already exists. Probable URL: https://s3.{region}.amazonaws.com/{bucket_name}")
        return True

    try:
        # Specify the region for the bucket
        location = {'LocationConstraint': region}
        s3_client.create_bucket(Bucket=bucket_name,
                                CreateBucketConfiguration=location)
        print(f"Bucket {bucket_name} created successfully.")
        
    except ClientError as e:
        logging.error(e)
        return False
    return True

def update_public_access_block(bucket_name: str, s3_client: boto3.client):
    """Update the public access block settings for an S3 bucket."""
    try:
        # Define the public access block configuration
        public_access_block_config = {
            'BlockPublicAcls': False,  # Set to False to allow public ACLs
            'IgnorePublicAcls': False,  # Set to False to respect public ACLs
            'BlockPublicPolicy': False,  # Set to False to allow public policies
            'RestrictPublicBuckets': False  # Set to False to not restrict public access
        }

        # Apply the public access block configuration to the bucket
        s3_client.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration=public_access_block_config
        )
        print(f"Public access block settings updated for bucket {bucket_name}.")
        return True
    except ClientError as e:
        print(f"Error updating public access block settings for bucket {bucket_name}: {e}")
        return False

def set_bucket_policy(bucket_name: str, s3_client: boto3.client) -> bool:
    """
    Set the bucket policy to allow specified AWS accounts to upload objects and 
    make all objects in the bucket publicly readable.

    :param bucket_name: Bucket to set the policy on
    :param s3_client: Boto3 S3 client
    :return: True if the policy was set, else False
    """

    # Define the bucket policy
    bucket_policy = {
        # Specify the version of the policy language
        "Version": "2012-10-17",
        # Define the statement section, which is a list of policy statements
        "Statement": [
            {
                # Allows everyone to read objects
                "Sid": "AllowPublicRead",  #Statement ID for identifying this statement, useful for managing policies with multiple statements
                "Effect": "Allow",
                "Principal": "*", #{"AWS": [f"arn:aws:iam::{os.getenv('AWS_ACCOUNT_ID')}:root",]}, # AWS account id
                "Action": ["s3:GetObject","s3:GetObjectAcl"], # operation
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"], #bucket resource to change
                "Condition": {
                     "Bool": {
                         "aws:SecureTransport":"true"
                     }
                },
            }
        ]
    }

    try:
        # Apply the bucket policy
        s3_client.put_bucket_policy(Bucket=bucket_name, Policy=json.dumps(bucket_policy), ExpectedBucketOwner=os.getenv('AWS_ACCOUNT_ID'))
        print(f'Bucket policy set successfully for {bucket_name}.')
        return True
    except ClientError as e:
        print(f"Error setting bucket policy for {bucket_name}: {e}")
        return False


# create the bucket
print(20*"*")
print(f"Creating bucket {bucket_name} in {region} region.")
bucket_exists=create_bucket(bucket_name, s3_client, region)
print(20*"*")

# set the bucket policy
if bucket_exists:
    if update_public_access_block(bucket_name, s3_client):
        print(f"Setting bucket policy for {bucket_name}.")
        set_bucket_policy(bucket_name, s3_client)
    else:
        print(f"Could not set public access block for bucket {bucket_name}")
else:
    print(f"Bucket {bucket_name} does not exist. Policy not set.")
print(20*"*")




############# Check the current buckets and compare the memory usage  #############
# define a function to get the memory usage of a bucket
def get_bucket_memory_usage(bucket_name: str, s3_client: boto3.client) -> int:
    """
    Calculate the total memory usage of an S3 bucket.

    :param bucket_name: The name of the S3 bucket.
    :param s3_client: The Boto3 S3 client.
    
    :return: The total memory usage in bytes.
    """

    total_size = 0
    continuation_token = None

    while True:
        # Use list_objects_v2 to handle large numbers of objects
        if continuation_token:
            response = s3_client.list_objects_v2(Bucket=bucket_name, ContinuationToken=continuation_token)
        else:
            response = s3_client.list_objects_v2(Bucket=bucket_name)

        # Check if the response contains 'Contents'
        if 'Contents' in response:
            for obj in response['Contents']:
                total_size += obj['Size']

        # Check for continuation token to handle paginated responses
        if 'NextContinuationToken' in response:
            continuation_token = response['NextContinuationToken']
        else:
            break

    return total_size


# Output the bucket names
response = s3_client.list_buckets()
print('Existing buckets:')
bucket_memory=[]
if response['Buckets']:
    for bucket in response['Buckets']:
        bucket_name = bucket["Name"]
        total_size = get_bucket_memory_usage(bucket_name, s3_client)
        total_size_mb = total_size / (1024 * 1024)  # Convert bytes to megabytes
        bucket_memory.append((bucket_name,total_size_mb))
        print(f'{bucket_name}: {total_size_mb:.2f} MB')
else:
    print('No buckets exist')
total_aws_memory=sum([x[1] for x in bucket_memory])

print(20*"*")
print(f"AWS memory limit set to: {memory_limit} MB")
print(f"Memory space used on AWS in MB: {total_aws_memory}")
print(20*"*")






###########final pipeline fucntions to upload the cubes ###############
def get_cube_memory(cube_name):
    """
    Calculate the memory usage of a cube directory.

    This function uses the `du` command to get the size of the directory
    in bytes, and then converts it to megabytes (MB).

    :param cube_name: The name of the cube directory.
    :return: The memory usage of the cube in megabytes (MB).
    """
    # Get the size of the directory
    result = subprocess.run(['du', '-sb', cube_name], stdout=subprocess.PIPE, text=True)
    cube_memory = int(result.stdout.split()[0])
    # Turn cube_memory into MB
    cube_memory_mb = cube_memory / (1024 * 1024)
    print(f"Cube memory in MB: {cube_memory_mb}")
    return cube_memory_mb

# Define the function to upload a directory to an S3 bucket
def upload_directory(directory_name, bucket_name, region, s3_prefix=''):
    """Upload a directory to an S3 bucket

    :param directory_name: Directory to upload
    :param bucket: Bucket to upload to
    :param s3_prefix: S3 prefix for the uploaded files
    :return: True if directory was uploaded, else False
    """
    s3_client = boto3.client('s3')

    for root, dirs, files in os.walk(directory_name):
        for file in files:
            file_path = os.path.join(root, file)
            s3_path = os.path.relpath(file_path, os.path.dirname(directory_name)) #set directory root

            if s3_prefix:
                s3_path = os.path.join(s3_prefix, s3_path)

            try:
                s3_client.upload_file(file_path, bucket_name ,s3_path) #, ExtraArgs={'ACL':'public-read'})
                print(f'Successfully uploaded {file_path}')
            except ClientError as e:
                logging.error(e)
                return False
    print(f"Zarr cube available at: https://s3.{region}.amazonaws.com/{bucket_name}/{directory_name}")
    return True


    
def upload_pipeline(cube_name, bucket_name, total_aws_memory, memory_limit, region):
    """
    Upload a cube to an S3 bucket if within memory limits.

    This function checks if the memory usage of a cube is within the 
    specified memory limit. If so, it creates an S3 bucket, sets it public, 
    and uploads the cube. It reports success or failure at each step.

    :param cube_name: The name of the cube directory to upload.
    :param bucket_name: The name of the S3 bucket to upload to.
    :param total_aws_memory: The total memory usage of all S3 buckets.
    :param memory_limit: The maximum memory usage allowed for all S3 buckets.
    
    :return: None
    """
    cube_memory_mb = get_cube_memory(cube_name)
    
    print(f"Attempting to upload the cube {cube_name} to the new bucket: {bucket_name}")
    

    if total_aws_memory + cube_memory_mb < (int(memory_limit)-total_aws_memory):
        if upload_directory(cube_name, bucket_name, region=region):
            print(f'Successfully uploaded directory {cube_name} to bucket {bucket}.')
        else:
            print(f'Failed to upload directory {cube_name} to bucket {bucket}.')
    else:
        print(f"Bucket {bucket} would exceed the memory limit of {memory_limit} MB. Total memory usage is {total_aws_memory} MB.")

        
# Define the cube to upload and its memory
zarr_base= os.getenv("ZARR_BASE_FOLDER") #repo_dir
os.chdir(zarr_base)
cubes=glob.glob("*.zarr")

total_cube_memory=0
cube_index=0
while cube_index < len(cubes) and total_cube_memory < (memory_limit-total_aws_memory):
    cube_memory = get_cube_memory(cubes[cube_index])
    # Check if cube would exceed the memory limit
    if total_cube_memory + cube_memory < memory_limit:
        total_cube_memory += cube_memory
        cube_index += 1
    else:
        # Stop if memory limit would be exceeded
        break
cubes_to_upload=cubes[0:cube_index]
print(f"List of cubes to upload: {cubes_to_upload}")


############## multithreaded upload of the cubes ##############
print("Uploading all the cubes...")
print(20*"*")
start=time.time()
# multithreading the zarr upload
with mp.pool.ThreadPool(n_cores) as pool:
    pool.starmap(upload_pipeline, [(cube_name, bucket_name, total_aws_memory, memory_limit, region) for cube_name in cubes_to_upload])
print(f"Total uploading time in seconds:{time.time()-start}") 
print(f"Average uploading time per file in seconds:{(time.time()-start)/len(cubes_to_upload)}") 
print(20*"*")