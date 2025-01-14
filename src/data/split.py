from dataclasses import dataclass
from src.utils.common import split_image_folder
from src.config.configuration import ConfigurationManager


class SplitFolder:
    def __init__(self, config):
        self.config=config

    def create_train_test(self):
        split_image_folder(self.config.size_ratio, base_dir=self.config.interim_path,
                           target_dir=self.config.processed_path)
        

if __name__ == "__main__":
    config_manager = ConfigurationManager()
    config_params = config_manager.get_split_folder_config()

    ob = SplitFolder(config_params)
    ob.create_train_test()