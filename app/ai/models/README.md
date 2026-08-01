# AI Model Architecture Package (`app/ai/models/`)

## Purpose
Establishes model abstraction layers and dynamic factory instantiation for sound classification models.

## Key Classes
- **`BaseSoundClassifier`**: Abstract Base Class defining standard `load_model()`, `predict()`, and `is_loaded` contracts.
- **`ModelFactory`**: Factory design pattern allowing dynamic registration and creation of model backends (`TFLite`, `PyTorch`, `ONNX`, `Mock`).
- **`StarterMockClassifier`**: Lightweight starter implementation for unit testing and initial system pipeline validation.

## Supported Model Formats (Future Phases)
- TFLite (Optimized for edge runtime)
- ONNX Runtime
- PyTorch / TensorFlow H5
