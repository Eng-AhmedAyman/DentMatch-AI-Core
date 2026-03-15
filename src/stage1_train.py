"""
==============================================================================
FILE: stage1_train.py
DESCRIPTION:
Training pipeline for Stage 1 (Security Guard / Binary Classification).
This script trains a MobileNetV2 model to distinguish between valid dental
images ('teeth') and invalid/irrelevant images ('not').
It applies robust data augmentation and saves the best model directly to
the 'models' directory for production use.
==============================================================================
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, callbacks

# ==============================================================================
# 1. DYNAMIC PATH RESOLUTION
# ==============================================================================
# Automatically locate the project root (HealthySmile_AI_Core)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Dynamically point to the Training Data and Models folders
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "stage1_binary", "train")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)  # Ensure the models folder exists

# The exact path where the model will be saved
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "stage1_mobilenet.keras")

# ==============================================================================
# 2. CONFIGURATION & PARAMETERS
# ==============================================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

# ==============================================================================
# 3. DATA PIPELINE (IMAGE AUGMENTATION)
# ==============================================================================
# We define an ImageDataGenerator to rescale pixels and add random transformations.
# This makes the model 'tough' and able to recognize teeth from any angle.
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    validation_split=0.2,
)

print(f"⏳ [INFO] Loading training data from:\n   -> {DATASET_DIR}")

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
# Using MobileNetV2 as a base feature extractor.
# It provides high accuracy with low computational cost (perfect for fast inference).
print("🧠 [INFO] Initializing MobileNetV2 architecture...")
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze pre-trained weights

model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),  # Dropout layer to prevent overfitting
        layers.Dense(1, activation="sigmoid"),  # Binary classification
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ==============================================================================
# 5. TRAINING PROCESS
# ==============================================================================
# Callbacks for better model convergence:
# EarlyStopping: halts training if no improvement.
# ModelCheckpoint: saves the best performing version of the model directly to /models.
my_callbacks = [
    callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_loss"),
]

print("🚀 [INFO] Starting training. Please monitor the loss/accuracy metrics...")
history = model.fit(
    train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=my_callbacks
)

print(
    f"\n✅ [SUCCESS] Training completed. Best model securely saved at:\n   -> {MODEL_SAVE_PATH}"
)
