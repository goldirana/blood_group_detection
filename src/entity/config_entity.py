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
