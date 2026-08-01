# Dataset Subsystem (`app/ai/dataset/`)

## Purpose
Manages environmental sound dataset ingestion, split generation, and class label indexing.

## Key Components
- **`dataset_loader.py`**: Contains `BaseDatasetLoader` interface contract and `AudioDatasetLoader` for datasets such as **ESC-50** or **UrbanSound8K**.

## Standard Class Categories
- Fire Alarm / Smoke Detector
- Baby Crying
- Doorbell
- Door Knock
- Car Horn / Vehicles
- Siren
- Speech / Commands
- Background Ambient Noise
