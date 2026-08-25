import zipfile
from pathlib import Path

submissions_dir = Path("outputs/submissions")
if not submissions_dir.exists():
    print("outputs/submissions directory not found!")
    exit(1)

zip_files = [f for f in submissions_dir.glob("*.zip") if not f.name.endswith("_temp.zip")]
if not zip_files:
    print("No .zip files found in outputs/submissions!")
    exit(0)

print("Compressing submission zip files using multi-algorithm pass (BZIP2 + LZMA) for <50MB size...\n")
for zpath in zip_files:
    orig_size = zpath.stat().st_size / (1024 * 1024)
    print(f"Processing '{zpath.name}' (Current Size: {orig_size:.2f} MB)...")
    
    # 1. Read uncompressed file from existing zip
    with zipfile.ZipFile(zpath, 'r') as zip_in:
        internal_filename = zip_in.namelist()[0]
        content = zip_in.read(internal_filename)
        
    uncompressed_mb = len(content) / (1024 * 1024)
    print(f"  Uncompressed Text Size: {uncompressed_mb:.2f} MB")
    
    # 2. Test BZIP2 (Burrows-Wheeler Transform - optimal for numerical ASCII)
    bz2_temp = zpath.with_name(zpath.stem + "_bz2_temp.zip")
    with zipfile.ZipFile(bz2_temp, 'w', compression=zipfile.ZIP_BZIP2) as zip_out:
        zip_out.writestr(internal_filename, content)
    bz2_size = bz2_temp.stat().st_size / (1024 * 1024)
    print(f"  -> BZIP2 Compression: {bz2_size:.2f} MB")
    
    # 3. Test LZMA
    lzma_temp = zpath.with_name(zpath.stem + "_lzma_temp.zip")
    with zipfile.ZipFile(lzma_temp, 'w', compression=zipfile.ZIP_LZMA) as zip_out:
        zip_out.writestr(internal_filename, content)
    lzma_size = lzma_temp.stat().st_size / (1024 * 1024)
    print(f"  -> LZMA Compression:  {lzma_size:.2f} MB")
    
    # Select smallest zip and overwrite original
    if bz2_size <= lzma_size:
        bz2_temp.replace(zpath)
        if lzma_temp.exists(): lzma_temp.unlink()
        best_size = bz2_size
        alg = "BZIP2"
    else:
        lzma_temp.replace(zpath)
        if bz2_temp.exists(): bz2_temp.unlink()
        best_size = lzma_size
        alg = "LZMA"
        
    print(f"  => SUCCESS: Saved '{zpath.name}' using {alg} at {best_size:.2f} MB\n")

print("All submission zip files successfully compressed!")
