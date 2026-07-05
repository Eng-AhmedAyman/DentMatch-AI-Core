"""
==============================================================================
FILE: stage3_evaluation.py
VERSION: 2.2.0  — Full evaluation suite
FIXES:
    - plot_gradcam defined BEFORE __main__ block (was causing NameError)
    - t-SNE restored to match original quality (larger dots, clearer clusters,
      rescale=1/255, perplexity=30, better palette matching notebook output)
    - Output filenames now match exactly what app.py's Analytics tab reads
      (previously saved as stage3_*.png, which the dashboard never displayed —
      re-running this script silently produced orphan files instead of
      updating the dashboard's figures).
    - Added plot_predictions_grid(): app.py displays predictions_grid.png but
      no script previously generated it.
GENERATES (all in reports/figures/ — same names app.py's Analytics tab reads):
    reports/figures/confusion_matrix.png
    reports/figures/roc_curves.png
    reports/figures/tsne.png
    reports/figures/confidence_plot.png
    reports/figures/gradcam.png
    reports/figures/predictions_grid.png
USAGE:
    python src/stage3_evaluation.py
==============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.manifold import TSNE

# ==============================================================================
# 1. PATHS
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "stage3", "stage3_efficientnet_finetuned_best.keras"
)
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "stage3_disease", "test")
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


# ==============================================================================
# 2. FOCAL LOSS (compatible with all TF versions)
# ==============================================================================
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0)
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.pow(1.0 - y_pred, gamma)
    return tf.reduce_sum(weight * cross_entropy, axis=-1)


try:
    focal_loss = tf.keras.saving.register_keras_serializable()(focal_loss)
except AttributeError:
    try:
        focal_loss = tf.keras.utils.register_keras_serializable()(focal_loss)
    except Exception:
        pass


# ==============================================================================
# 3. LOAD MODEL & DATA
# ==============================================================================
CLASS_NAMES = [
    "Dental_Caries",
    "Hypodontia",
    "Mouth_Ulcer",
    "Periodontal_Disease",
    "Tooth_Discoloration",
]
NUM_CLASSES = len(CLASS_NAMES)


def load_everything():
    print("=" * 60)
    print("  DentMatch AI — Stage 3 Full Evaluation  v2.1.0")
    print("=" * 60)

    print("\n Loading model...")
    model = tf.keras.models.load_model(
        MODEL_PATH, custom_objects={"focal_loss": focal_loss}
    )
    print("✅ Model loaded.")

    print("\n Loading test data...")
    gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    test_gen = gen.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        batch_size=16,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    NUM_CLASSES = len(CLASS_NAMES)
    print(f"✅ {test_gen.samples} images | {NUM_CLASSES} classes: {CLASS_NAMES}")

    print("\n Running inference...")
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    return model, predictions, y_pred, y_true, CLASS_NAMES, NUM_CLASSES


# ==============================================================================
# 4. CLASSIFICATION REPORT
# ==============================================================================
def plot_classification_report(y_true, y_pred, CLASS_NAMES):
    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


# ==============================================================================
# 5. CONFUSION MATRIX
# ==============================================================================
def plot_confusion_matrix(y_true, y_pred, CLASS_NAMES):
    print("📊 Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        annot_kws={"size": 13, "weight": "bold"},
        linewidths=1,
        linecolor="black",
    )
    plt.title("Stage 3 — Confusion Matrix", fontsize=15, fontweight="bold", pad=20)
    plt.ylabel("Actual Diagnosis", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Diagnosis", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✅ Saved → {path}")


# ==============================================================================
# 6. ROC-AUC
# ==============================================================================
def plot_roc_auc(y_true, predictions, CLASS_NAMES, NUM_CLASSES):
    print("📈 ROC-AUC Curves...")
    y_onehot = np.eye(NUM_CLASSES)[y_true]
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(5 * NUM_CLASSES, 5))
    fig.suptitle("Stage 3 — Per-Class ROC-AUC", fontsize=16, fontweight="bold", y=1.02)
    for i, (name, ax) in enumerate(zip(CLASS_NAMES, axes)):
        fpr, tpr, _ = roc_curve(y_onehot[:, i], predictions[:, i])
        auc = roc_auc_score(y_onehot[:, i], predictions[:, i])
        ax.plot(fpr, tpr, color="royalblue", lw=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.fill_between(fpr, tpr, alpha=0.08, color="royalblue")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.legend(loc="lower right", fontsize=11)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "roc_curves.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    mean_auc = roc_auc_score(y_onehot, predictions, average="macro")
    print(f"✅ Saved → {path}  |  Mean AUC: {mean_auc:.4f}")


# ==============================================================================
# 7. t-SNE  — restored to match original notebook quality
#    Key fixes vs v2.0.0:
#      • rescale=1/255 on feature-extraction generator (was missing → bad features)
#      • perplexity=30  (was 40 → over-smoothed clusters)
#      • s=60, edgecolors="white", lw=0.3  (larger, cleaner dots like original)
#      • "tab10" palette with explicit color list matching original order
#      • figure size 14×10 (was 12×9)
# ==============================================================================
def plot_tsne(model, y_true, CLASS_NAMES):
    print("🧠 t-SNE (extracting features — this takes ~2 min)...")

    # Feature extractor: penultimate layer (before the Dense classifier)
    feature_extractor = tf.keras.Model(
        inputs=model.input, outputs=model.layers[-2].output
    )

    # ⚠️ rescale=1/255 is mandatory — raw pixel values distort the feature space
    gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    feat_gen = gen.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode=None,
        classes=CLASS_NAMES,
        shuffle=False,
    )
    features = feature_extractor.predict(feat_gen, verbose=1)

    print("   Running t-SNE projection...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,  # 30 = tighter, more separated clusters
        max_iter=1000,
        learning_rate="auto",
        init="pca",  # PCA init → more stable, reproducible layout
    )
    reduced = tsne.fit_transform(features)

    # Palette matching the original notebook output
    palette = sns.color_palette("tab10", len(CLASS_NAMES))

    fig, ax = plt.subplots(figsize=(14, 10))
    for i, name in enumerate(CLASS_NAMES):
        mask = y_true == i
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            c=[palette[i]],
            label=name,
            alpha=0.75,
            s=60,  # larger dots — matches original
            edgecolors="white",
            linewidths=0.3,
        )

    ax.set_title(
        "t-SNE Visualization of AI Feature Space (Clinical Clusters)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=13)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=13)
    legend = ax.legend(
        title="True Diagnosis",
        fontsize=11,
        title_fontsize=12,
        markerscale=2,
        framealpha=0.9,
        edgecolor="lightgray",
    )
    ax.grid(True, alpha=0.25, linestyle="--")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "tsne.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved → {path}")


# ==============================================================================
# 8. CONFIDENCE PLOT
# ==============================================================================
def plot_confidence(predictions, y_pred, y_true, CLASS_NAMES):
    print("🩺 Confidence Distribution...")

    confidences = np.max(predictions, axis=1) * 100
    correct = y_pred == y_true

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Stage 3 — AI Confidence & Reliability Analysis",
        fontsize=15,
        fontweight="bold",
    )

    ax = axes[0]
    ax.hist(
        confidences[correct],
        bins=30,
        alpha=0.7,
        color="steelblue",
        label=f"Correct ({correct.sum()})",
        edgecolor="white",
    )
    ax.hist(
        confidences[~correct],
        bins=30,
        alpha=0.7,
        color="tomato",
        label=f"Incorrect ({(~correct).sum()})",
        edgecolor="white",
    )
    ax.axvline(80, color="black", linestyle="--", lw=1.5, label="80% Threshold")
    ax.set_title("Confidence: Correct vs Incorrect", fontsize=13, fontweight="bold")
    ax.set_xlabel("Confidence (%)", fontsize=11)
    ax.set_ylabel("Number of Predictions", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    data_per_class = [confidences[y_true == i] for i in range(len(CLASS_NAMES))]
    bp = ax2.boxplot(data_per_class, patch_artist=True, notch=False)
    palette = sns.color_palette("tab10", len(CLASS_NAMES))
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax2.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=10)
    ax2.set_title("Per-Class Confidence Distribution", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Confidence (%)", fontsize=11)
    ax2.axhline(80, color="black", linestyle="--", lw=1.5, label="80% Threshold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "confidence_plot.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✅ Saved → {path}")


# ==============================================================================
# 9. GRAD-CAM
#    FIX: this function is now defined BEFORE __main__ so Python sees it
# ==============================================================================
def _make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    """Compute Grad-CAM heatmap for the top predicted class."""
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return (
        heatmap.numpy(),
        int(pred_index),
        float(tf.reduce_max(preds) * 100),
    )


def plot_gradcam(model, y_true, CLASS_NAMES):
    import cv2

    print("🔥 Grad-CAM Visualizations...")

    gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    vis_gen = gen.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        batch_size=1,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
        seed=42,
    )

    # Collect one correctly-classified sample per class
    collected = {}
    attempts = 0
    while len(collected) < len(CLASS_NAMES) and attempts < 500:
        img_batch, label_batch = next(vis_gen)
        true_idx = int(np.argmax(label_batch[0]))
        if true_idx not in collected:
            collected[true_idx] = img_batch
        attempts += 1

    n = len(CLASS_NAMES)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    fig.suptitle(
        "Stage 3 — Grad-CAM: What the AI Sees",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )

    for row, class_idx in enumerate(sorted(collected.keys())):
        img_array = collected[class_idx]
        heatmap, pred_idx, confidence = _make_gradcam_heatmap(img_array, model)

        orig = (img_array[0] * 255).astype(np.uint8)
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(orig, 0.6, heatmap_color, 0.4, 0)

        correct = "✅" if pred_idx == class_idx else "❌"

        axes[row, 0].imshow(orig)
        axes[row, 0].set_title(f"Original\nTrue: {CLASS_NAMES[class_idx]}", fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(heatmap_resized, cmap="jet")
        axes[row, 1].set_title("Activation Heatmap", fontsize=10)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(superimposed)
        axes[row, 2].set_title(
            f"{correct} Predicted: {CLASS_NAMES[pred_idx]}\n"
            f"Confidence: {confidence:.1f}%",
            fontsize=10,
        )
        axes[row, 2].axis("off")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "gradcam.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved → {path}")


def plot_predictions_grid(model, CLASS_NAMES, n_samples=16):
    """
    Saves a grid of random test-set predictions (image + true vs. predicted
    label, colour-coded green/red for correct/incorrect) to
    reports/figures/predictions_grid.png.

    Added because app.py's Analytics tab displays this file, but no script
    previously generated it — it existed only as a manually-created artifact.
    """
    print("🖼️  Predictions Grid...")

    gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    vis_gen = gen.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        batch_size=1,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
        seed=7,
    )

    cols = 4
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i in range(n_samples):
        img_batch, label_batch = next(vis_gen)
        true_idx = int(np.argmax(label_batch[0]))
        pred_probs = model.predict(img_batch, verbose=0)[0]
        pred_idx = int(np.argmax(pred_probs))
        confidence = float(pred_probs[pred_idx] * 100)
        correct = pred_idx == true_idx

        axes[i].imshow(img_batch[0])
        axes[i].axis("off")
        color = "green" if correct else "red"
        axes[i].set_title(
            f"True: {CLASS_NAMES[true_idx]}\n"
            f"Pred: {CLASS_NAMES[pred_idx]} ({confidence:.0f}%)",
            fontsize=9,
            color=color,
        )

    for j in range(n_samples, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Stage 3 — Sample Predictions (random test-set batch)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "predictions_grid.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved → {path}")


# ==============================================================================
# MAIN — all functions defined above, no NameError possible
# ==============================================================================
if __name__ == "__main__":
    model, predictions, y_pred, y_true, CLASS_NAMES, NUM_CLASSES = load_everything()

    plot_classification_report(y_true, y_pred, CLASS_NAMES)
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)
    plot_roc_auc(y_true, predictions, CLASS_NAMES, NUM_CLASSES)
    plot_confidence(predictions, y_pred, y_true, CLASS_NAMES)
    plot_tsne(model, y_true, CLASS_NAMES)
    plot_gradcam(model, y_true, CLASS_NAMES)
    plot_predictions_grid(model, CLASS_NAMES)

    print("\n" + "=" * 60)
    print("🎉 All figures saved to:")
    print(f"   {OUT_DIR}")
    print("=" * 60)
