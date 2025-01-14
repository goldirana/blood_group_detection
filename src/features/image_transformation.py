from src.config.configuration import ConfigurationManager
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from src import logger



class ImageTransformation:
    def __init__(self, config):
        self.config=config

    def get_image_transformer(self, data="train"):
        if data != "train":
            transform = transforms.Compose(
                [transforms.Resize(self.config.image_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.mean, std=self.config.std)]
            )
            return transform
        else: # transformation for test or validation
            logger.info(f"Initializing transformation on {data} data")
            transform = transforms.Compose([
                transforms.Resize(self.config.image_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.mean, std=self.config.std)
            ])
            return transform
        
    def get_data_loader(self, image_folder_path, transform):
        image_dataset = ImageFolder(image_folder_path, transform=transform)

        image_loader = DataLoader(image_dataset,
                                  batch_size=self.config.batch_size,
                                  shuffle=self.config.shuffle,
                                  num_workers=self.config.num_workers)
        
        return image_loader
        
if __name__ == "__main__":
    config_manager = ConfigurationManager()
    config_params = config_manager.get_image_transformation_params()

    ob = ImageTransformation(config_params)
    train_transformer = ob.get_image_transformer(data="train")
    train_loader = ob.get_data_loader("data/processed/train", train_transformer)

