import gdown
import zipfile
import os
import shutil

def download_and_extract_zip(drive_url, output_path, extract_to):
    """
    Downloads a ZIP file from Google Drive, extracts it, and removes any extra folder.
    
    Args:
        drive_url (str): Direct download link to the file on Google Drive.
        output_path (str): Local path to save the downloaded ZIP file.
        extract_to (str): Directory to extract the contents.
    """
    try:
        # Step 1: Download the ZIP file
        gdown.download(drive_url, output_path, quiet=False, fuzzy=True)
        print(f"Downloaded file saved to: {output_path}")
        
        # Step 2: Check if the file is a ZIP file
        if not zipfile.is_zipfile(output_path):
            raise zipfile.BadZipFile("Downloaded file is not a valid ZIP file.")
        
        # Step 3: Extract the ZIP file
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted contents to: {extract_to}")
        
        # Step 4: Move files if an extra folder is created
        for root, dirs, files in os.walk(extract_to):
            # If there's an extra folder, move its contents up and remove the folder
            for folder in dirs:
                folder_path = os.path.join(root, folder)
                for file in os.listdir(folder_path):
                    shutil.move(os.path.join(folder_path, file), extract_to)
                os.rmdir(folder_path)  # Remove the now-empty folder
            break  # Only process the top-level directory

    except zipfile.BadZipFile as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Replace with your Google Drive shareable link
    google_drive_url = "https://drive.google.com/file/d/10ApjOcTo6tjO34q5et-ij-tbrSm0Waqc/view?usp=drive_link"
    output_file = "data/raw/data.zip"
    extract_dir = "data/interim"
    
    # Create output directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)
    
    # Download and extract the ZIP file
    download_and_extract_zip(google_drive_url, output_file, extract_to=extract_dir)
