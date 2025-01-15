from src.constants import (CONFIG_FILE_PATH, PARAMS_FILE_PATH,
                        VANILLA_FILE_PATH)
from src.features.image_transformation import ImageTransformation
from src.config.configuration import ConfigurationManager
from src.models.vanilla.model import VanillaModel
from src import logger
from tqdm import tqdm

from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import optim

from importlib import import_module
import torch
import mlflow
import os


class ModelTraining:
    def __init__(self, config):
        self.config=config
        self.device= str("gpu" if torch.cuda.is_available() else "cpu")
    
    def get_optimizer(self):
        module = import_module("torch.optim")
        optimizer = getattr(module, self.config.optimizer)
        return optimizer
    
    def get_criterion(self):
        module = import_module("torch.nn")
        criterion = getattr(module, self.config.criterion)
        return criterion
    
    def get_model(self, basemodule= None, 
                        submodule= None,
                        modelclass= None) -> type:
        """
        Args:
            basemodule(str): Directory where models are present
            submodule(str): name of model
            model_class(str): Class name of model
        Return: 
            object of t
        """
        basemodule= basemodule or self.config.basemodule 
        submodule= submodule or self.config.submodule
        modelclass= modelclass or self.config.modelclass

        module = import_module(f"{basemodule}.{submodule}")
        model = getattr(module, modelclass)
        return model
    
    def get_model_training_attributes(self, model=None, optimizer=None, criterion=None):
        """
        Returns the model attributes which is used to build the model
        Args:
            model(type): model which needs to be trained
            optimizer(type): model's optimizer
            criterion(type): model's criterion
        Return:
            Tuple: (model, optimizer, criterion ) 
        """
        try:
            optimizer = self.get_optimizer() or optimizer
            criterion = self.get_criterion() or criterion
            model = self.get_model() or criterion
            return model, optimizer, criterion
        except Exception as e:
            logger.error(e)
            raise e
    
    def train(self, train_loader, model_params, model=None,  criterion=None, optimizer=None):
               
        try:

            model = model(model_params)
            optimizer = optimizer(model.parameters(), **self.config.optimizer_args)
            criterion = criterion()
            logger.info("Model Attributes initalized")
        except Exception as e:
            logger.error("Error occured while initalizing model")
            raise e
        
        print("Starting training loop...")
        for epoch in tqdm(range(self.config.epochs), desc="Training Epochs"):
            model.train()
            
            if len(train_loader) > 0:
                for images, labels in train_loader:
                    print("Batch size:", images.size())
                    break
            else:
                print("Data loader is empty. Check your dataset and path.")
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            logger.info(f'Epoch [{epoch+1}/{2}], Loss: {loss.item():.4f}')
            print(f'Epoch [{epoch+1}/{2}], Loss: {loss.item():.4f}')


if __name__ == "__main__":
    config_manager = ConfigurationManager()
    config_params = config_manager
    


    # Data Loader
    image_transformation_params = config_manager.get_image_transformation_config()
    image_transformation = ImageTransformation(image_transformation_params)

    train_transformer = image_transformation.get_image_transformer(data="train") # perform image_transformation on training dataset
    train_loader = image_transformation.get_data_loader(config_manager.config.data.train_path, 
                                                        train_transformer)
    # # Model Training
    try:
        logger.info("Model Training Intialized")
        model_training_params = config_manager.get_model_training_config()
        trainer = ModelTraining(model_training_params)

        # Model Architecture
        model_params = config_manager.get_architecture_config()
        model, optimizer, criterion  = trainer.get_model_training_attributes()
        logger.info(f"Initializing Model: {model.__name__}")
        
        # Training Loop
        trainer.train(train_loader, model_params, model, criterion, optimizer)
    
    except Exception as e:
        logger.error("Error occured in model training")
        raise e