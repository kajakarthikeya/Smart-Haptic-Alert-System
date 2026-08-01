# AI Subsystem (`app/ai/`)

## Purpose
The `app/ai/` directory contains the machine learning pipeline components organized according to Clean Architecture for environmental sound detection.

## Structure Overview

```
app/ai/
├── dataset/             # Dataset loading & ingestion interfaces
├── preprocessing/       # Signal normalization, filtering & framing
├── feature_extraction/ # Mel-spectrogram & log-frequency feature generators
├── models/              # Model abstraction contracts & ModelFactory
├── training/            # Pipeline trainer & evaluation routines
├── inference/           # Real-time inference engine
└── utils/               # Machine learning metrics & evaluation helpers
```

## Clean Architecture Contracts
All AI pipeline components derive from Abstract Base Classes (`abc.ABC`) to ensure models, feature extractors, and preprocessors can be updated or swapped (e.g., PyTorch to TFLite or ONNX) without refactoring downstream business logic.
