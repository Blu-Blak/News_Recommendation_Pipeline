import os
import subprocess
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        if zipfile.is_zipfile(dest_path):
            print(f"File {dest_path} already exists and is a valid zip, skipping download.")
            return
        else:
            print(f"File {dest_path} exists but is not a valid zip (likely a previous failed download). Deleting...")
            dest_path.unlink()
            
    print(f"Downloading {url} to {dest_path} using wget...")
    
    wget_cmd = ["wget", "-c", "-O", str(dest_path)]
    if "huggingface.co" in url and "HF_TOKEN" in os.environ:
        print("Using HF_TOKEN for authentication...")
        wget_cmd.append(f"--header=Authorization: Bearer {os.environ['HF_TOKEN']}")
    wget_cmd.append(url)
    
    subprocess.run(wget_cmd, check=True)
    
    if not zipfile.is_zipfile(dest_path):
        raise RuntimeError(f"Downloaded file {dest_path} is still not a valid zip file. The download failed or auth token is invalid.")

def extract_zip(zip_path: Path, extract_to: Path):
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"Directory {extract_to} already contains files, skipping extraction.")
        return
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def download_mind(raw_dir: Path):
    repo_id = "yjw1029/MIND"
    files = ["MINDsmall_train.zip", "MINDsmall_dev.zip"]
    
    for filename in files:
        zip_path = raw_dir / filename
        extract_dir = raw_dir / filename.replace(".zip", "")
        
        if zip_path.exists() and zipfile.is_zipfile(zip_path):
            print(f"File {zip_path} already exists and is a valid zip, skipping download.")
        else:
            if zip_path.exists():
                print(f"File {zip_path} exists but is invalid. Deleting...")
                zip_path.unlink()
            print(f"Downloading {filename} from HuggingFace...")
            token = os.environ.get("HF_TOKEN")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=str(raw_dir),
                token=token
            )
            
        extract_zip(zip_path, extract_dir)

def download_mind_large(raw_dir: Path):
    repo_id = "yjw1029/MIND"
    filename = "MINDlarge_test.zip"
    
    zip_path = raw_dir / filename
    extract_dir = raw_dir / filename.replace(".zip", "")
    
    if zip_path.exists() and zipfile.is_zipfile(zip_path):
        print(f"File {zip_path} already exists and is a valid zip, skipping download.")
    else:
        if zip_path.exists():
            print(f"File {zip_path} exists but is invalid. Deleting...")
            zip_path.unlink()
        print(f"Downloading {filename} from HuggingFace...")
        token = os.environ.get("HF_TOKEN")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(raw_dir),
            token=token
        )
        
    extract_zip(zip_path, extract_dir)

def download_ebnerd(raw_dir: Path, scale: str = "demo"):
    urls = {
        f"ebnerd_{scale}.zip": f"https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_{scale}.zip",
        "Ekstra_Bladet_word2vec.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_word2vec.zip",
        "google_bert_base_multilingual_cased.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/google_bert_base_multilingual_cased.zip"
    }
    for filename, url in urls.items():
        zip_path = raw_dir / filename
        extract_dir = raw_dir / filename.replace(".zip", "")
        download_file(url, zip_path)
        extract_zip(zip_path, extract_dir)

def download_ebnerd_large(raw_dir: Path):
    urls = {
        "ebnerd_large.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip",
        "ebnerd_testset.zip": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip"
    }
    for filename, url in urls.items():
        zip_path = raw_dir / filename
        extract_dir = raw_dir / filename.replace(".zip", "")
        download_file(url, zip_path)
        extract_zip(zip_path, extract_dir)

if __name__ == "__main__":
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    download_mind(raw_dir)
    download_ebnerd(raw_dir, scale="demo")
