import matplotlib.pyplot as plt
from src import logger
from src.utils.common import read_yaml, log_error, create_directory
from dataclasses import dataclass
from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
import cv2
import numpy as np
import os
from typing import List, Union, Optional
import sys
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import List
from src.config.configuration import ConfigurationManager


class ImagePreprocessing:
    def __init__(self, config):
        self.config=config

    @log_error(faliure_message="Image Preprocessing")
    def process_image(self, image):
        # corrected_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # corrected_image = corrected_image.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=70.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(image)
        
        # Apply erosion to thin and darken lines
        kernel = np.ones((1, 1), np.uint8)
        thinned_image = cv2.erode(enhanced_image, kernel, iterations=200)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
        enhanced_contrast = clahe.apply(thinned_image)

        smoothed = cv2.GaussianBlur(enhanced_contrast, (3, 3), 23)
        # Step 3: Sharpen the image
        sharpened = cv2.addWeighted(enhanced_contrast,2, smoothed, -0.5, 0)

        denoised_image = cv2.fastNlMeansDenoising(sharpened, None, h=20, 
                                                templateWindowSize=5, 
                                                searchWindowSize=25)
        kernel = np.ones((1, 1), np.uint8)
        erode_image = cv2.erode(denoised_image, kernel, iterations=200)
        return erode_image
    
    @log_error(faliure_message="Reading Image")
    def read_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        return image
    
    @log_error(sucess_message=None, faliure_message="Get folder path")
    def get_folder_path(self, main_folder: str) -> List[str]:
        """
        To get sub folder names of given folder 
        Return:
        list: 
        """
        folder_names = list(os.listdir(main_folder)) # returns the folder name
        print(folder_names)
        folder_with_path = [] # to save foldername with base path
        
        for sub_folder in folder_names:
            if os.path.isdir(os.path.join(main_folder, sub_folder)) != True:
                logger.info(f"{sub_folder} not a directory\nRemoving {sub_folder}") # to remove reduant files 
            else:
                folder_with_path.append(os.path.join(main_folder, # base path
                                                     sub_folder)) # image_folder_name
        return folder_with_path

    @log_error(faliure_message="Failed in saving image")
    def save_image(self, image, save_path) -> None:
        create_directory(save_path, is_extension_present=True)
        cv2.imwrite(save_path, image)
        

    def apply_image_operations(self, image_path) -> None:
        try:
            # Read, process, and save the image
            image = self.read_image(image_path)
            image = self.process_image(image)
            self.save_image(image, image_path.replace("raw", "interim"))
            
        except Exception as e:
            return f"Error processing {image_path}: {e}"

    @log_error(faliure_message="Directory not found")
    def get_all_image_path(self, folder_path: List[str]):
        """
        To get the base image path with image_name
        Args:
            folder_path(List[Str]): List containing sub directory names
        Returns:
            all_image_path: List of str; image_path/image_name.extension
        """
        all_image_path = []
        for folder in folder_path:
            image_names = os.listdir(folder)
            for image_name in image_names:
                image_path = os.path.join(folder, image_name)
                all_image_path.append(image_path)
        return all_image_path

def main(func,all_image_path):
    """
    Apply func on all the images using multiprocessing
    Returns None
    """
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(func, all_image_path), total=len(all_image_path)))

if __name__ == "__main__":
    config_manager = ConfigurationManager()
    config_params = config_manager.get_image_preprocessing_config()

    ob = ImagePreprocessing(config_params)
    folder_path = ob.get_folder_path(config_params.raw_path) #-> raw folder with sub directory

    all_image_path = ob.get_all_image_path(folder_path) # image name with path
    main(ob.apply_image_operations, all_image_path)
