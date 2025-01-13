from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    google_drive_url: str
    download_path: str
    extract_to: str