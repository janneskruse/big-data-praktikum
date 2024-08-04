import requests
import os
import zipfile
import multiprocessing
from tqdm import tqdm
from datetime import datetime
import zipfile
import logging
from multiprocessing import Pool, cpu_count
from collections import defaultdict

######## set the URL and credentials ########
# get the base path of the git repository
repo_dir = os.popen('git rev-parse --show-toplevel').read().strip()
###load the .env file
load_dotenv(dotenv_path=f"{repo_dir}/.env")

#get the environment vairables
base=os.getenv("BASE_FOLDER")

# URL for the whole dataset:
base_url = os.getenv("NEXTCLOUD_BASE")
# set the credentials
username = os.getenv("NEXTCLOUD_USERNAME")
password = os.getenv("NEXTCLOUD_PW")

print(f"Cloudbase: {base_url}")

######## set the storage path ########
# Create the directory if it does not exist
storage_path = base #"/work/le837wmue-Rhone_download"
os.makedirs(storage_path, exist_ok=True)

def unzip_file(zip_path, extract_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"{datetime.now().strftime('%Y%m%d%H%M')}: File unzipped successfully at", extract_path)
        os.remove(zip_path)
        print(f"{datetime.now().strftime('%Y%m%d%H%M')}: Zip file {zip_path} deleted successfully")
    except Exception as e:
        print(f"Error unzipping file {zip_path}: {e}")

def request_folder():
    url= base_url
    zip_path = os.path.join(storage_path, "DAS_2020.zip")
    extract_path = storage_path
    
    try:
        response = requests.get(url, auth=(username, password), stream=True)
    
        if response.status_code == 200:
            print(f"{datetime.now().strftime('%Y%m%d%H%M')}: File downloading...")
            
            size = int(response.headers.get("Content-Length", 0))
            
            progress = tqdm(response.iter_content(1024*1024*100), f"Downloading {zip_path}", total=size, unit="B", unit_scale=True, unit_divisor=1024)
            
            with open(zip_path, 'wb') as file:
                for data in progress.iterable:
                    file.write(data)
                    progress.update(len(data))      
            
            print(f"{datetime.now().strftime('%Y%m%d%H%M')}: File saved successfully at", zip_path)  

            # Unzip the file
            unzip_file(zip_path, extract_path)

            zip_files = [os.path.join(root, file) for root, dirs, files in os.walk(extract_path) for file in files if file.endswith('.zip')]
            print(f"Found {len(zip_files)} zip files in the directory")

            if zip_files:
                num_processes = min(len(zip_files)//2, multiprocessing.cpu_count())
                with multiprocessing.Pool(processes=num_processes) as pool:
                    for zip_file in zip_files:
                        pool.apply_async(unzip_file, (zip_file, os.path.splitext(zip_file)[0]))

        else:
            print(f"{datetime.now().strftime('%Y%m%d%H%M')}: File {zip_path} not found. Status code:", response.status_code)
    except Exception as e:
        print(f"Error in request_folder: {e}")

request_folder()


# Configure logging
logging.basicConfig(filename='unzip_rhone.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')

def extract_folder(folder_info):
    zip_path, extract_path, folder_files = folder_info
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in folder_files:
            zip_ref.extract(file, extract_path)
            logging.info(f'Extracted {file.filename}')

def unzip_folders(zip_path, extract_path):
    logging.info(f'Starting to unzip {zip_path} to {extract_path}')
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get the list of files and group them by top-level directory
        file_list = zip_ref.infolist()
        folders = defaultdict(list)
        
        for file in file_list:
            top_level_folder = file.filename.split('/')[0]
            folders[top_level_folder].append(file)
        
        total_folders = len(folders)
        logging.info(f'Number of top-level folders to extract: {total_folders}')
        
        # Prepare data for parallel processing
        folder_info_list = [(zip_path, extract_path, files) for files in folders.values()]
        
        # Extract folders with progress bar and parallel processing
        with tqdm(total=total_folders, unit='folder') as pbar:
            with Pool(cpu_count()) as pool:
                for _ in pool.imap_unordered(extract_folder, folder_info_list):
                    pbar.update(1)
    
    logging.info('Unzipping completed.')

if __name__ == "__main__":
    zip_path = "/work/le837wmue-Rhone_download/DAS_2020.zip"
    extract_path = "/work/le837wmue-Rhone_download/DAS_2020"
    
    # Create extract_path if it does not exist
    os.makedirs(extract_path, exist_ok=True)
    
    unzip_folders(zip_path, extract_path)
    
