"""
==============================================================================
FILE: stage1_train.py
DESCRIPTION:
    Training pipeline for Stage 1 — Security Guard (Binary Classification).
    Trains a MobileNetV2 model to distinguish between valid dental images
    ('Teeth') and invalid/irrelevant images ('Not_Teeth').

    Applies robust data augmentation and saves the best model to:
        models/stage1/stage1_mobilenet.keras

    This path MUST match STAGE1_PATH in deployment/master_pipeline.py.

DATASET STRUCTURE EXPECTED:
    data/stage1_binary/train/
    ├── Teeth/        (dental images)
    └── Not_Teeth/    (non-dental / face images)

USAGE:
    python src/stage1_train.py

AUTHOR:  Eng. Ahmed Ayman — AI & Data Science Engineer
VERSION: 1.1.0  (fix — model save path aligned with master_pipeline.STAGE1_PATH)
==============================================================================
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, callbacks

# ==============================================================================
# 1. DYNAMIC PATH RESOLUTION
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "stage1_binary", "train")

# ⚠️  This path MUST match STAGE1_PATH in deployment/master_pipeline.py
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "stage1")
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "stage1_mobilenet.keras")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

print("=" * 60)
print("🦷  DentMatch AI — Stage 1 Training  v1.1.0")
print("=" * 60)
print(f"   Dataset  : {DATASET_DIR}")
print(f"   Output   : {MODEL_SAVE_PATH}")
print("=" * 60 + "\n")

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

# ==============================================================================
# 3. DATA PIPELINE
# ==============================================================================
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    validation_split=0.2,
)

print(f"⏳ Loading training data from:\n   {DATASET_DIR}\n")

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
)

# ==============================================================================
# 4. MODEL ARCHITECTURE (TRANSFER LEARNING)
# ==============================================================================
print("🧠 Initialising MobileNetV2 architecture...")
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)
base_model.trainable = False  # Freeze pretrained weights

model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),  # Binary classification
    ]
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ==============================================================================
# 5. CALLBACKS
# ==============================================================================
my_callbacks = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        save_best_only=True,
        monitor="val_loss",
        verbose=1,
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]

# ==============================================================================
# 6. TRAINING
# ==============================================================================
print("\n🚀 Starting training...\n")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=my_callbacks,
)

print(f"\n✅ Training complete. Best model saved at:\n   {MODEL_SAVE_PATH}\n")
