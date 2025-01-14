from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    google_drive_url: str
    download_path: str
    extract_to: str


@dataclass
class ImagePreprocessingConfig:
    raw_path: str
    interim_path: str


@dataclass
class SplitFolderConfig:
    interim_path: str
    processed_path: str
    size_ratio: tuple


@dataclass
class VanillaModelConfig:
    image_height: int
    image_width: int
    layer_1: dict
    layer_2: dict


@dataclass
class ImageTransformationConfig:
    # Image params
    image_shape: list
    mean: int
    std: int
    # Directory params
    train_path: str
    batch_size: int
    shuffle: bool
    num_workers:int
