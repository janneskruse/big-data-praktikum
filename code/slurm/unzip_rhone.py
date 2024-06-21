import zipfile
import os
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import defaultdict

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
