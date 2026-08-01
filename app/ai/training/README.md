# Model Training Subsystem (`app/ai/training/`)

## Purpose
Orchestrates offline model training, validation split evaluation, hyperparameter tuning, and lightweight quantization / export to TFLite format for microcontrollers/mobile runtime.

## Interfaces
- `BaseTrainer`: Abstract Base Class defining `train()` and `export_model()`.
- `ModelTrainer`: Training execution engine wrapper.
