import requests
from tqdm import tqdm
import os


def download_file(url, target_directory, file_size=None):
    """A simple function to download a file from a URL with a progress bar using only the requests library"""
    local_filename = url.split('/')[-1]

    with requests.get(url, stream=True) as r:
        if file_size is not None:
            progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True, desc=f"{local_filename}")
        else:
            total_size = int(r.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"{local_filename}")

        with open(os.path.join(target_directory, local_filename), 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                progress_bar.update(len(chunk))

    progress_bar.close()
