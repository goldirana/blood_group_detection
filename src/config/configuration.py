from src.constants import (CONFIG_FILE_PATH, PARAMS_FILE_PATH,
                           VANILLA_FILE_PATH)
from src import logger
from src.utils.common import read_yaml
from src.entity.config_entity import (DataIngestionConfig,
                                      ImagePreprocessingConfig,
                                      SplitFolderConfig,
                                      VanillaModelConfig,
                                      ImageTransformationConfig)


class ConfigurationManager:
    def __init__(self):
        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARAMS_FILE_PATH)
        
        # Architecture params
        self.vanilla=read_yaml(VANILLA_FILE_PATH) # get vanilla params

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
    
    def get_split_folder_config(self):
        params = SplitFolderConfig(
            interim_path=self.config.data.interim_path,
            processed_path=self.config.data.processed_path,
            size_ratio=(self.config.data.train_size,
                        self.config.data.test_size,
                        self.config.data.val_size)          
        )
        return params
    
    def get_image_transformation_params(self)->ImageTransformationConfig:
        params=ImageTransformationConfig(
            image_shape=[self.params.image_transformation.height,
                        self.params.image_transformation.width],

            mean=self.params.image_transformation.mean,
            std=self.params.image_transformation.std,
            # Directory params

            train_path=self.config.data.train_path,
            batch_size=self.params.image_transformation.batch_size,
            shuffle=self.params.image_transformation.shuffle,
            num_workers=self.params.image_transformation.workers
        )
        return params

    def get_vanilla_architecture_params(self) -> VanillaModelConfig:
        params=VanillaModelConfig(
            # Getting params from params.yaml
            image_height=self.params.image_params.height,
            image_width=self.params.image_params.width,

            # Getting vanilla params from vanilla_params.yaml
            layer_1=self.vanilla.layers.first, # access vanilla architecture layer 1 params
            layer_2=self.vanilla.layers.second # access vanilla architecture layer 2 params

        )
        return params