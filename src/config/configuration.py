from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src import logger
from src.utils.common import read_yaml
from src.entity.config_entity import (DataIngestionConfig,
                                      ImagePreprocessingConfig)


class ConfigurationManager:
    def __init__(self):
        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARAMS_FILE_PATH)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        params = DataIngestionConfig(
            google_drive_url=self.config.data.google_drive_url,
            download_path=self.config.data.download_path,
            extract_to=self.config.data.extract_to
            )
        return params 
    
    def get_image_preprocessing_config(self):
        params=ImagePreprocessingConfig(
            raw_path=self.config.data.extract_to,
            interim_path=self.config.data.interim_path
        )
        return params