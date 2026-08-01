# Outputs Storage Directory (`app/outputs/`)

## Purpose
Serves as the designated directory for generated output artifacts:
- Quantized TFLite model binaries (`sound_classifier.tflite`)
- Exported ONNX or HDF5 weights
- Evaluated confusion matrix plots and ROC curves
- Exported spectrogram images

> [!NOTE]
> Artifact files in this directory (except `README.md`) are ignored by version control (`.gitignore`).
