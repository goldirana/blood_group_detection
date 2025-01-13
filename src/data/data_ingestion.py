import gdown
import zipfile
import os
import shutil
from src import logger
from src.config.configuration import ConfigurationManager


class DataIngestion:
    def __init__(self, config):
        self.config=config

    
    def download_and_extract_zip(self):
        """
        Downloads a ZIP file from Google Drive, extracts it, and removes any extra folder.
        
        Args:
            drive_url (str): Direct download link to the file on Google Drive.
            output_path (str): Local path to save the downloaded ZIP file.
            extract_to (str): Directory to extract the contents.
        """
        try:
            # Step 1: Download the ZIP file
            gdown.download(self.config.google_drive_url, 
                        self.config.output_path, quiet=False, fuzzy=True)
            print(f"Downloaded file saved to: {self.config.output_path}")
            
            # Step 2: Check if the file is a ZIP file
            if not zipfile.is_zipfile(self.config.output_path):
                raise zipfile.BadZipFile("Downloaded file is not a valid ZIP file.")
            
            # Step 3: Extract the ZIP file
            with zipfile.ZipFile(self.config.output_path, 'r') as zip_ref:
                zip_ref.extractall(self.config.extract_to)
            print(f"Extracted contents to: {self.config.extract_to}")
            
            # Step 4: Move files if an extra folder is created
            for root, dirs, files in os.walk(self.config.extract_to):
                # If there's an extra folder, move its contents up and remove the folder
                for folder in dirs:
                    folder_path = os.path.join(root, folder)
                    for file in os.listdir(folder_path):
                        shutil.move(os.path.join(folder_path, file), self.config.extract_to)
                    os.rmdir(folder_path)  # Remove the now-empty folder
                break  # Only process the top-level directory

        except zipfile.BadZipFile as e:
            print(e)
        except Exception as e:
            print(f"An error occurred: {e}")



if __name__ == "__main__":
    config_manager = ConfigurationManager()
    config_params = config_manager.get_data_ingestion_config()
    ob = DataIngestion(config_params)
    ob.download_and_extract_zip()