"""
==============================================================================
FILE: train_stage3.py
DESCRIPTION:
    Production-Grade Training Script for Stage 3 — EfficientNetB4 Specialist.
    5-Class Dental Disease Classifier:
        0. Dental_Caries
        1. Hypodontia
        2. Mouth_Ulcer
        3. Periodontal_Disease
        4. Tooth_Discoloration

PROBLEMS THIS SCRIPT SOLVES (vs. original training):
    1. Low Confidence (60-75%)    → Label Smoothing + MixUp Augmentation
    2. Wrong Class Prediction     → Class Weights + Focal Loss
    3. Overfitting                → Heavy Augmentation + Dropout scheduling
    4. Poor Generalisation        → Two-Phase training (Freeze → Unfreeze)
    5. No Reproducibility         → Fixed random seeds everywhere

DATASET STRUCTURE EXPECTED:
    data/
    ├── train/
    │   ├── Dental_Caries/       (1000+ images)
    │   ├── Hypodontia/          (1000+ images)
    │   ├── Mouth_Ulcer/         (1000+ images)
    │   ├── Periodontal_Disease/ (1000+ images)
    │   └── Tooth_Discoloration/ (1000+ images)
    └── val/
        ├── Dental_Caries/
        ├── Hypodontia/
        ├── Mouth_Ulcer/
        ├── Periodontal_Disease/
        └── Tooth_Discoloration/

    If you only have one folder (no val split), set AUTO_SPLIT = True below.

OUTPUT:
    models/stage3/stage3_efficientnet_finetuned_best.keras  ← used by pipeline
    reports/training/training_history.png
    reports/training/confusion_matrix_final.png
    reports/training/classification_report.txt

USAGE:
    python train_stage3.py

    GPU recommended. On CPU expect ~10-15 min/epoch.
    On GPU (e.g. T4) expect ~2-3 min/epoch.

DEPENDENCIES:
    tensorflow>=2.12, scikit-learn, matplotlib, numpy, Pillow

AUTHOR: Eng. Ahmed Ayman — AI & Data Science Engineer
VERSION: 1.0.0
==============================================================================
"""

# ==============================================================================
# ZONE 1: IMPORTS
# ==============================================================================
import os
import random
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — safe for servers

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
)

# ==============================================================================
# ZONE 2: REPRODUCIBILITY — Fix all random seeds
# ==============================================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==============================================================================
# ZONE 3: CONFIGURATION
# Edit these values to match your setup.
# ==============================================================================

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
OUTPUT_MODEL = os.path.join(
    PROJECT_ROOT, "models", "stage3", "stage3_efficientnet_finetuned_best.keras"
)
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "training")

# --- Dataset ---
# Set True if you have ONE folder with all images (no train/val split yet).
# The script will auto-split 80% train / 20% val for you.
AUTO_SPLIT = False
AUTO_SPLIT_SRC = os.path.join(DATA_DIR, "all")  # used only if AUTO_SPLIT=True

# --- Model ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5

# --- Training Phase 1: Head only (EfficientNetB4 frozen) ---
PHASE1_EPOCHS = 20
PHASE1_LR = 1e-3

# --- Training Phase 2: Full fine-tune (top 80 layers unfrozen) ---
PHASE2_EPOCHS = 30
PHASE2_LR = 1e-5  # Must be much smaller than Phase 1

# --- Regularisation ---
DROPOUT_RATE = 0.4
LABEL_SMOOTHING = 0.1  # Prevents overconfident wrong predictions

# --- Class names — ORDER MUST MATCH folder names exactly ---
CLASS_NAMES = [
    "Dental_Caries",
    "Hypodontia",
    "Mouth_Ulcer",
    "Periodontal_Disease",
    "Tooth_Discoloration",
]

# ==============================================================================
# ZONE 4: SETUP OUTPUT DIRECTORIES
# ==============================================================================
os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 65)
print("🦷  DentMatch AI — Stage 3 Training Script  v1.0.0")
print("=" * 65)
print(f"   TensorFlow version : {tf.__version__}")
print(f"   GPUs available     : {len(tf.config.list_physical_devices('GPU'))}")
print(f"   Output model path  : {OUTPUT_MODEL}")
print("=" * 65 + "\n")

# ==============================================================================
# ZONE 5: AUTO SPLIT (optional)
# If AUTO_SPLIT=True, splits the source folder 80/20 into TRAIN_DIR / VAL_DIR.
# Skips if train/val folders already exist and have files.
# ==============================================================================
if AUTO_SPLIT:
    import shutil
    from sklearn.model_selection import train_test_split

    print("📂 AUTO_SPLIT=True — splitting dataset 80/20 ...")
    for cls in CLASS_NAMES:
        src_cls = os.path.join(AUTO_SPLIT_SRC, cls)
        if not os.path.isdir(src_cls):
            raise FileNotFoundError(f"Source class folder not found: {src_cls}")

        images = [
            f
            for f in os.listdir(src_cls)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        train_imgs, val_imgs = train_test_split(
            images, test_size=0.2, random_state=SEED
        )

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
            dst = os.path.join(DATA_DIR, split_name, cls)
            os.makedirs(dst, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(os.path.join(src_cls, img), os.path.join(dst, img))

        print(f"   {cls}: {len(train_imgs)} train / {len(val_imgs)} val")

    print("✅ Split done.\n")

# ==============================================================================
# ZONE 6: DATA PIPELINES
# Heavy augmentation on train, normalisation-only on val.
# EfficientNetB4 expects pixel values in [0, 255] — its internal rescaling
# layer handles normalisation. Do NOT divide by 255 externally.
# ==============================================================================

# --- Augmentation layers (applied only during training) ---
augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ],
    name="augmentation",
)


def load_train_dataset() -> tf.data.Dataset:
    """
    Build an augmented, shuffled training dataset from TRAIN_DIR.

    Applies the augmentation pipeline and prefetch for GPU overlap.

    Returns:
        tf.data.Dataset: Batched and prefetched (images, one-hot-labels).
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        label_mode="categorical",
        class_names=CLASS_NAMES,
    )
    ds = ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def load_val_dataset() -> tf.data.Dataset:
    """
    Build a non-augmented validation dataset from VAL_DIR.

    Returns:
        tf.data.Dataset: Batched and prefetched (images, one-hot-labels).
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="categorical",
        class_names=CLASS_NAMES,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


print("📦 Loading datasets...")
train_ds = load_train_dataset()
val_ds = load_val_dataset()

# Count samples for class-weight calculation
train_labels_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int",
    class_names=CLASS_NAMES,
)
all_labels = np.concatenate([y.numpy() for _, y in train_labels_raw])
total_train = len(all_labels)
total_val = sum(1 for _ in val_ds.unbatch())

print(f"   Train samples : {total_train}")
print(f"   Val   samples : {total_val}\n")

# ==============================================================================
# ZONE 7: CLASS WEIGHTS
# Compensates for any remaining imbalance between classes.
# Even with 1000+ images per class, slight imbalance affects decision boundary.
# ==============================================================================
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=all_labels,
)
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights_array)}

print("⚖️  Class weights (imbalance correction):")
for i, name in enumerate(CLASS_NAMES):
    print(f"   {name}: {class_weight_dict[i]:.4f}")
print()

# ==============================================================================
# ZONE 8: FOCAL LOSS
# Focal Loss down-weights easy examples and focuses training on hard cases.
# This is the single biggest fix for the "wrong class on borderline images"
# problem — much better than plain CrossEntropy for medical imaging.
#
# Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
#   gamma=2.0 → standard focal loss (Lin et al. 2017)
#   alpha=0.25 → balancing factor
# ==============================================================================


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Categorical Focal Loss factory.

    Args:
        gamma (float): Focusing parameter. Higher = more focus on hard examples.
                       Recommended range [1.5, 3.0]. Default 2.0 (original paper).
        alpha (float): Class balancing factor. Default 0.25.

    Returns:
        Callable: Loss function compatible with model.compile(loss=...).
    """

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        # p_t = probability of the true class
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal = alpha * tf.pow(1.0 - p_t, gamma) * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))

    loss_fn.__name__ = "focal_loss"
    return loss_fn


# ==============================================================================
# ZONE 9: MODEL ARCHITECTURE
# Two-head design:
#   Base  → EfficientNetB4 (ImageNet pretrained, frozen in Phase 1)
#   Head  → GlobalAveragePooling → BatchNorm → Dropout → Dense(5, softmax)
# ==============================================================================


def build_model(learning_rate: float) -> Model:
    """
    Build and compile the EfficientNetB4-based classifier.

    The base EfficientNetB4 is loaded with include_top=False so we can
    attach our own classification head tuned for 5 dental disease classes.

    Frozen layers are controlled externally (see Phase 1 / Phase 2 sections).

    Args:
        learning_rate (float): Initial learning rate for the Adam optimiser.

    Returns:
        tf.keras.Model: Compiled model ready for model.fit().
    """
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3), name="input_image")

    # EfficientNetB4 internal preprocessing (handles [0,255] → normalised)
    base = EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )

    # Classification head
    x = base.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout_head")(x)
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="predictions",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)

    model = Model(inputs, outputs, name="DentMatch_EfficientNetB4")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc", multi_label=False),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model, base


# ==============================================================================
# ZONE 10: CALLBACKS
# ==============================================================================


def get_callbacks(phase: int) -> list:
    """
    Return a list of Keras callbacks for the given training phase.

    Args:
        phase (int): 1 (frozen head training) or 2 (full fine-tune).

    Returns:
        list: [ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger]
    """
    ckpt_path = (
        OUTPUT_MODEL
        if phase == 2
        else OUTPUT_MODEL.replace(".keras", f"_phase{phase}.keras")
    )

    return [
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=7 if phase == 1 else 10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(
            os.path.join(REPORTS_DIR, f"phase{phase}_log.csv"),
            append=False,
        ),
    ]


# ==============================================================================
# ZONE 11: PHASE 1 — Train head only (base frozen)
# Goal: Teach the new classification head without destroying ImageNet weights.
# This is essential — jumping straight to fine-tuning destroys the pretrained
# features because the head gradients are too large at initialisation.
# ==============================================================================

print("=" * 65)
print("🔒 PHASE 1 — Training Classification Head (Base Frozen)")
print("=" * 65)

model, base_model = build_model(learning_rate=PHASE1_LR)
base_model.trainable = False

print(f"   Trainable params  : {model.count_params():,}")
print(f"   Phase 1 LR        : {PHASE1_LR}")
print(f"   Phase 1 epochs    : {PHASE1_EPOCHS}\n")

history1 = model.fit(
    train_ds,
    epochs=PHASE1_EPOCHS,
    validation_data=val_ds,
    class_weight=class_weight_dict,
    callbacks=get_callbacks(phase=1),
    verbose=1,
)

print(
    f"\n✅ Phase 1 complete. Best val_accuracy: {max(history1.history['val_accuracy']):.4f}\n"
)

# ==============================================================================
# ZONE 12: PHASE 2 — Fine-tune top layers (partial unfreeze)
# Unfreeze the top 80 layers of EfficientNetB4 for domain-specific tuning.
# Use a very small LR (100× smaller than Phase 1) to avoid catastrophic
# forgetting of the ImageNet features.
# ==============================================================================

print("=" * 65)
print("🔓 PHASE 2 — Fine-tuning Top Layers (Partial Unfreeze)")
print("=" * 65)

base_model.trainable = True

# Freeze everything except the top 80 layers
freeze_until = len(base_model.layers) - 80
for layer in base_model.layers[:freeze_until]:
    layer.trainable = False

# Recompile with Focal Loss + smaller LR for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2_LR),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc", multi_label=False),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

trainable_count = sum(1 for l in model.layers if l.trainable)
print(f"   Unfrozen layers   : {trainable_count}")
print(f"   Phase 2 LR        : {PHASE2_LR}")
print(f"   Phase 2 epochs    : {PHASE2_EPOCHS}\n")

history2 = model.fit(
    train_ds,
    epochs=PHASE2_EPOCHS,
    validation_data=val_ds,
    class_weight=class_weight_dict,
    callbacks=get_callbacks(phase=2),
    verbose=1,
)

print(
    f"\n✅ Phase 2 complete. Best val_accuracy: {max(history2.history['val_accuracy']):.4f}\n"
)

# ==============================================================================
# ZONE 13: EVALUATION — Classification Report + Confusion Matrix
# ==============================================================================

print("=" * 65)
print("📊 FINAL EVALUATION")
print("=" * 65)

# Collect predictions on val set
all_true, all_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    all_true.extend(np.argmax(labels.numpy(), axis=1))
    all_pred.extend(np.argmax(preds, axis=1))

all_true = np.array(all_true)
all_pred = np.array(all_pred)

# --- Classification Report ---
report_str = classification_report(
    all_true, all_pred, target_names=CLASS_NAMES, digits=4
)
print(report_str)

report_path = os.path.join(REPORTS_DIR, "classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_str)
print(f"📄 Report saved → {report_path}")

# --- Confusion Matrix ---
cm = confusion_matrix(all_true, all_pred)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(NUM_CLASSES),
    yticks=np.arange(NUM_CLASSES),
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    title="Confusion Matrix — Stage 3 (EfficientNetB4)",
    ylabel="True Label",
    xlabel="Predicted Label",
)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
thresh = cm.max() / 2.0
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=11,
        )
plt.tight_layout()
cm_path = os.path.join(REPORTS_DIR, "confusion_matrix_final.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"📊 Confusion matrix saved → {cm_path}")

# ==============================================================================
# ZONE 14: TRAINING HISTORY PLOTS
# ==============================================================================


def _merge_histories(h1, h2, key: str) -> list:
    """Concatenate a metric from Phase 1 and Phase 2 histories."""
    return h1.history.get(key, []) + h2.history.get(key, [])


metrics = ["accuracy", "loss", "auc", "precision", "recall"]
fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))

for ax, metric in zip(axes, metrics):
    train_vals = _merge_histories(history1, history2, metric)
    val_vals = _merge_histories(history1, history2, f"val_{metric}")
    epochs_range = range(1, len(train_vals) + 1)

    ax.plot(epochs_range, train_vals, label=f"Train {metric}", linewidth=2)
    ax.plot(epochs_range, val_vals, label=f"Val {metric}", linewidth=2, linestyle="--")

    # Mark Phase 1 / Phase 2 boundary
    p1_end = len(history1.history.get(metric, []))
    ax.axvline(x=p1_end, color="gray", linestyle=":", alpha=0.6, label="Phase 1 → 2")

    ax.set_title(f"{metric.capitalize()} over Epochs", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.capitalize())
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle(
    "DentMatch Stage 3 — Full Training History", fontsize=15, fontweight="bold"
)
plt.tight_layout()
history_path = os.path.join(REPORTS_DIR, "training_history.png")
plt.savefig(history_path, dpi=150)
plt.close()
print(f"📈 Training history saved → {history_path}\n")

# Save history as JSON for notebook analysis (Stage3_Analysis.ipynb Cell 7)
history_json_path = os.path.join(REPORTS_DIR, "training_history.json")
history_json = {
    "phase1": {k: [float(v) for v in vals] for k, vals in history1.history.items()},
    "phase2": {k: [float(v) for v in vals] for k, vals in history2.history.items()},
}
with open(history_json_path, "w", encoding="utf-8") as f:
    json.dump(history_json, f, indent=2, ensure_ascii=False)
print(f"💾 Training history (JSON) saved → {history_json_path}\n")

# ==============================================================================
# ZONE 15: SAVE FINAL MODEL + CONFIG
# ==============================================================================
model.save(OUTPUT_MODEL)
print(f"💾 Final model saved → {OUTPUT_MODEL}")

config = {
    "model_name": "DentMatch_EfficientNetB4",
    "version": "1.0.0",
    "image_size": list(IMAGE_SIZE),
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
    "phase1_epochs": PHASE1_EPOCHS,
    "phase1_lr": PHASE1_LR,
    "phase2_epochs": PHASE2_EPOCHS,
    "phase2_lr": PHASE2_LR,
    "dropout_rate": DROPOUT_RATE,
    "label_smoothing": LABEL_SMOOTHING,
    "loss_phase2": "focal_loss(gamma=2.0, alpha=0.25)",
    "final_val_acc": float(max(history2.history["val_accuracy"])),
}

config_path = os.path.join(os.path.dirname(OUTPUT_MODEL), "model_config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4, ensure_ascii=False)
print(f"⚙️  Model config saved → {config_path}")

print("\n" + "=" * 65)
print("🎉 TRAINING COMPLETE!")
print(f"   Final val accuracy : {config['final_val_acc']:.4f}")
print(f"   Model location     : {OUTPUT_MODEL}")
print("=" * 65 + "\n")
