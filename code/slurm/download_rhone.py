import requests
import os
import zipfile
import multiprocessing
from tqdm import tqdm
from datetime import datetime

base_url = "https://cloud.scadsai.uni-leipzig.de/index.php/s/gozxE5r9YdwGL8w/download"
username = "j.kruse@studserv.uni-leipzig.de"
password = ""

storage_path = "/work/le837wmue-Rhone_download"
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