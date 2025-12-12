import requests
import zipfile
import os
import sys
from tqdm import tqdm
from bs4 import BeautifulSoup

# Add core directory to path to import preprocess module
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
from core._01_preprocess import preprocess_image_stack

def download_large_file_from_google_drive(file_id, destination):
    session = requests.Session()

    # Step 1: get the virus scan warning page
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = session.get(URL)

    # Step 2: parse HTML for form action + hidden fields
    soup = BeautifulSoup(response.text, "html.parser")
    form = soup.find("form", {"id": "download-form"})
    if form is None:
        # Sometimes Google Drive returns the file directly if it's small enough or no virus scan warning
        # Check if we got the file content directly
        if 'Content-Disposition' in response.headers:
             print("Downloading source images (direct)...")
             with open(destination, "wb") as f:
                f.write(response.content)
             print("Download completed:", destination)
             return
        else:
             raise Exception("Could not find download form. Google may have changed the page structure.")

    download_url = form["action"]
    inputs = form.find_all("input")

    # Collect required POST parameters
    params = {}
    for inp in inputs:
        name = inp.get("name")
        value = inp.get("value")
        if name:
            params[name] = value

    # Step 3: send final download request
    print("Downloading source images...")
    final_response = session.get(download_url, params=params, stream=True)
    final_response.raise_for_status()

    # Try to get content length (may be missing for some responses)
    total_bytes = final_response.headers.get('Content-Length')
    try:
        total_bytes = int(total_bytes) if total_bytes is not None else None
    except ValueError:
        total_bytes = None

    # Provide a simple console progress bar.    
    with open(destination, 'wb') as f, tqdm(total=total_bytes, unit='B', unit_scale=True, desc='Downloading') as pbar:
        for chunk in final_response.iter_content(32768):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    print("Download completed:", destination)


def extract_zip(zip_path, extract_to):
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    print("Extraction completed.")

def precompute_cache(data_dir):
    print("Precomputing alignment cache for all datasets...")
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found.")
        return

    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    
    for folder in tqdm(folders, desc="Processing datasets"):
        folder_path = os.path.join(data_dir, folder)
        try:
            # This will load, align, and save to cache automatically
            preprocess_image_stack(folder_path, use_cache=True)
        except Exception as e:
            print(f"Failed to process {folder}: {e}")

if __name__ == "__main__":
    FILE_ID = "1Ld-aduENwICbDshjeG9-WEuLaJ7B1XYu"
    
    # Ensure we are in the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    download_path = "data_tmp.zip"
    
    # 1. Download
    if not os.path.exists("data") or not os.listdir("data"):
        os.makedirs("data", exist_ok=True)
        download_large_file_from_google_drive(FILE_ID, download_path)
        # 2. Extract
        extract_zip(download_path, "data") 
        if os.path.exists(download_path):
            os.remove(download_path)
    else:
        print("Data folder already exists and is not empty, skipping download.")

    # 3. Precompute Cache
    precompute_cache("data")
    
    print("Initialization complete! You can now run 'python gui.py'")

