"""
==============================================================================
FILE: train_stage3.py
DESCRIPTION:
Training pipeline for Stage 3 (The Specialist / Dental Diseases Classification).
This script builds and fine-tunes an EfficientNetB4 architecture to classify
images into 6 distinct dental conditions. It implements advanced techniques
like Class Weighting (for imbalanced data) and 2-Phase Training (Warm-up ->
Fine-Tuning) to maximize clinical precision.
==============================================================================
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# ==============================================================================
# 1. DYNAMIC PATH RESOLUTION
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Points directly to the output of our clean_data.py script
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "stage3_disease")
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "val")

# Save models directly to the centralized models folder
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

PHASE1_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "stage3_efficientnet_phase1.keras")
FINAL_MODEL_PATH = os.path.join(
    MODEL_SAVE_DIR, "stage3_efficientnet_finetuned_best.keras"
)

# ==============================================================================
# 2. DATA PREPARATION & AUGMENTATION
# ==============================================================================
# Note: EfficientNet handles its own pixel rescaling internally!
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

print(f" [INFO] Loading datasets from:\n   -> {DATASET_PATH}")

train_datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
)
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=True,
)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=False,
)

# Compute intelligent class weights to handle data imbalance
print("⚖️ [INFO] Computing balanced class weights...")
classes = train_generator.classes
class_weights_array = compute_class_weight(
    "balanced", classes=np.unique(classes), y=classes
)
CLASS_WEIGHTS = dict(enumerate(class_weights_array))

# ==============================================================================
# 3. ARCHITECTURE CONSTRUCTION
# ==============================================================================
print(" [INFO] Building EfficientNetB4 Architecture...")
base_model = EfficientNetB4(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze for Phase 1

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
outputs = Dense(6, activation="softmax", dtype="float32")(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ==============================================================================
# 4. TRAINING - PHASE 1: WARM-UP
# ==============================================================================
checkpoint1 = ModelCheckpoint(
    PHASE1_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

print("\n [PHASE 1] STARTING WARM-UP TRAINING (Frozen Base)...")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    class_weight=CLASS_WEIGHTS,
    callbacks=[checkpoint1, early_stop],
)

# ==============================================================================
# 5. TRAINING - PHASE 2: FINE-TUNING
# ==============================================================================
print("\n🔓 [PHASE 2] Unfreezing top 100 layers for Fine-Tuning...")
base_model.trainable = True
for layer in base_model.layers[:-100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),  # Extremely low learning rate
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

checkpoint2 = ModelCheckpoint(
    FINAL_MODEL_PATH, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
)

print("\n🚀 [PHASE 2] STARTING DEEP FINE-TUNING...")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    class_weight=CLASS_WEIGHTS,
    callbacks=[checkpoint2, early_stop],
)

print(
    f"\n [SUCCESS] Training Complete! Final production model securely saved at:\n   -> {FINAL_MODEL_PATH}"
)
