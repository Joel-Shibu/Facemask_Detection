import os
import zipfile
import hashlib

def verify_integrity(file_path, expected_md5):
    """Verify the MD5 checksum of the downloaded file"""
    if not os.path.exists(file_path):
        return False
    
    print("Verifying download integrity...")
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest().upper() == expected_md5.upper()

def main():
    DATASET_URL = "https://www.kaggle.com/datasets/joel102005/face-mask-detection-dataset/download"
    EXPECTED_MD5 = "59E4213E72781B5F387FF1BBE87FD3CF"
    ZIP_FILE = "face-mask-detection-dataset.zip"
    
    print("="*60)
    print("Face Mask Detection Dataset Download")
    print("="*60)
    print("\nPlease follow these steps to download the dataset:")
    print(f"1. Visit: {DATASET_URL}")
    print("2. Click the 'Download' button (requires Kaggle login)")
    print("3. Save the file to this directory")
    print(f"4. The file should be named: {ZIP_FILE}")
    
    input("\nPress Enter after you've downloaded the file...")
    
    if not os.path.exists(ZIP_FILE):
        print(f"\nError: {ZIP_FILE} not found in the current directory.")
        print("Please make sure you've downloaded the file and it's in the same directory as this script.")
        return
    
    if not verify_integrity(ZIP_FILE, EXPECTED_MD5):
        print("\nError: The downloaded file is corrupted or incomplete.")
        print("Please delete the file and try downloading again.")
        return
    
    print("\nFile verified successfully! Extracting...")
    
    # Create dataset directory if it doesn't exist
    os.makedirs('dataset', exist_ok=True)
    
    # Extract the zip file
    try:
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall('dataset')
        print("Extraction completed successfully!")
        print(f"Dataset is available in the 'dataset' directory.")
        
        # Clean up
        os.remove(ZIP_FILE)
        print(f"Removed {ZIP_FILE} to save space.")
        
    except Exception as e:
        print(f"Error extracting the file: {e}")
        print("Please make sure the file is not corrupted and try again.")

if __name__ == "__main__":
    main()