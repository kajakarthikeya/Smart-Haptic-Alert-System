# AI Models Subsystem (`app/ai/models/`)

## 1. Purpose
The **AI Models Subsystem** defines abstract base interfaces, factory registries, and deep learning model architectures for environmental sound classification. It encapsulates model definition, layer connectivity, weight serialization/deserialization, and inference routines.

---

## 2. Architecture & Components

```
app/ai/models/
├── __init__.py           # Package exports (BaseSoundClassifier, CNNSoundClassifier, ModelFactory)
├── base_model.py         # Abstract Base Class BaseSoundClassifier
├── cnn_classifier.py     # CNNSoundClassifier 2D CNN architecture
├── exceptions.py         # Domain model exceptions (ModelSaveError, ModelLoadError, ModelTrainingError)
├── model_factory.py      # ModelFactory registration and instantiation engine
├── model_metadata.json   # Model hyperparameter, architecture, and dataset metadata
├── sound_classifier_best.keras  # Checkpointed best-performing model weights
└── sound_classifier_final.keras # Final epoch trained model weights
```

---

## 3. CNNSoundClassifier Architecture
Specialized for 2D composite acoustic feature matrices `(184, 173, 1)`:
- **Input Shape**: `(184, 173, 1)`
- **Block 1**: Conv2D(32, 3x3) ➔ BatchNorm ➔ ReLU ➔ MaxPool2D(2x2) ➔ Dropout(0.25)
- **Block 2**: Conv2D(64, 3x3) ➔ BatchNorm ➔ ReLU ➔ MaxPool2D(2x2) ➔ Dropout(0.25)
- **Block 3**: Conv2D(128, 3x3) ➔ BatchNorm ➔ ReLU ➔ MaxPool2D(2x2) ➔ Dropout(0.30)
- **Global Pooling**: GlobalAveragePooling2D
- **Dense Head**: Dense(128) ➔ BatchNorm ➔ ReLU ➔ Dropout(0.50)
- **Output Layer**: Dense(5, Softmax) ➔ Probabilities over 5 classes (`ambulance`, `car_horn`, `fire_alarm`, `doorbell`, `dog_bark`)
- **Total Parameters**: 111,237

---

## 4. Usage via ModelFactory

```python
from app.ai.models.model_factory import ModelFactory

# Instantiate registered CNN classifier
classifier = ModelFactory.create_model("cnn")

# Load trained model weights
classifier.load_model("app/ai/models/sound_classifier_best.keras")

# Perform inference on acoustic feature map (184, 173)
label, confidence, probs = classifier.predict(feature_matrix)
print(f"Detected: {label} ({confidence:.2%})")
```
