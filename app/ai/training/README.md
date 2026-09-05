# Model Training Subsystem (`app/ai/training/`)

## 1. Purpose
The **Model Training Subsystem** trains deep neural network classifiers for environmental sound recognition. It consumes pre-extracted, normalized, and stratified feature datasets (`dataset_splits.npz`), manages training lifecycles with validation loops and callbacks, supports balanced class weighting, saves checkpoints, and logs training history and visual metrics.

---

## 2. Architecture & Components

```
app/ai/training/
├── __init__.py           # Public module interface exports
├── data_loader.py        # TrainingDataLoader & TrainingDataset container
├── exceptions.py         # Domain-specific training exceptions
├── pipeline.py           # TrainingPipeline automated orchestrator
├── trainer.py            # BaseTrainer & ModelTrainer execution engine
└── visualizer.py         # TrainingVisualizer for accuracy & loss curves
```

### Key Classes
- `TrainingDataLoader`: Loads `dataset_splits.npz` and `class_names.json`, validates dimensions (`184 x 173`), expands channel axes to `(N, 184, 173, 1)`, detects NaNs and Infs, verifies labels, and calculates balanced class weights.
- `ModelTrainer`: Subclasses `BaseTrainer`. Manages mini-batch training with `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau`. Persists `sound_classifier_best.keras` and `sound_classifier_final.keras`.
- `TrainingVisualizer`: Renders high-resolution training/validation accuracy curves, loss curves, and composite comparison figures to `app/outputs/model_training/`.
- `TrainingPipeline`: Orchestrates end-to-end training, validation, artifact saving, and post-training inference verification.

---

## 3. Data Flow
1. **Input Loading**: `TrainingDataLoader` reads `app/ai/features/dataset_splits.npz` and `class_names.json`.
2. **Dimension Validation**: Asserts feature dimensions match `(*, 184, 173)` and reshapes to `(*, 184, 173, 1)`.
3. **Class Balancing**: Analyzes training set distribution and computes balanced loss weights $w_c = \frac{N}{K \cdot n_c}$.
4. **Model Training**: `ModelTrainer` executes Keras training loop on `CNNSoundClassifier`.
5. **Model Checkpointing**: Highest `val_accuracy` checkpoint is saved to `app/ai/models/sound_classifier_best.keras`.
6. **Artifact Generation**: Saves `training_history.json`, `model_metadata.json`, and training curves.
7. **Verification**: Verifies that the saved model can be loaded and predicts a test sample correctly.

---

## 4. Configuration Options (`config.py`)
Configured via `settings.training`:
- `batch_size`: Mini-batch size (default: `4`)
- `epochs`: Total training epochs (default: `50`)
- `learning_rate`: Adam optimizer initial learning rate (default: `0.001`)
- `optimizer`: Optimizer name (default: `"adam"`)
- `loss_function`: Loss metric (default: `"sparse_categorical_crossentropy"`)
- `dropout_rate`: Spatial dropout rate (default: `0.3`)
- `random_seed`: Deterministic seed (default: `42`)
- `early_stopping_patience`: Epochs to wait before early stopping (default: `15`)
- `reduce_lr_patience`: Epochs to wait before learning rate reduction (default: `7`)

---

## 5. Usage Example

```python
from app.ai.training.pipeline import TrainingPipeline

# Run automated training pipeline
pipeline = TrainingPipeline(epochs=50, batch_size=4, random_seed=42)
results = pipeline.run(generate_visualizations=True)

print(f"Status: {results['status']}")
print(f"Best Val Accuracy: {results['best_val_accuracy']:.2%}")
print(f"Best Model: {results['best_model_path']}")
```

---

## 6. Output Files
- `app/ai/models/sound_classifier_best.keras`: Best performing model checkpoint.
- `app/ai/models/sound_classifier_final.keras`: Final epoch model weights.
- `app/ai/models/model_metadata.json`: Model specifications, hyperparameters, and metrics.
- `app/outputs/model_training/training_history.json`: Machine-readable training curves data.
- `app/outputs/model_training/accuracy_curve.png`: Accuracy progression chart.
- `app/outputs/model_training/loss_curve.png`: Loss progression chart.
- `app/outputs/model_training/training_metrics.png`: Dual-panel composite metric chart.

---

## 7. Downstream Integration
- **Phase 6 (Model Evaluation)**: Uses `sound_classifier_best.keras` and `dataset_splits.npz` test split for calculating confusion matrices, precision, recall, F1-scores, and classification latency.
- **Phase 7 (Real-Time Recognition)**: Uses trained model weights inside `SoundInferenceEngine` for real-time sound detection.
