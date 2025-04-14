# utils/download_dataset.py

import os
import urllib.request
import tarfile

def download_and_extract(url, dest):
    if not os.path.exists(dest):
        os.makedirs(dest)
    filename = url.split("/")[-1]
    file_path = os.path.join(dest, filename)

    if not os.path.exists(file_path):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")
    
    print("Extracting dataset...")
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=dest)
    print("Extraction complete.")

if __name__ == "__main__":
    DATA_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    DEST_DIR = "./data"
    download_and_extract(DATA_URL, DEST_DIR)
