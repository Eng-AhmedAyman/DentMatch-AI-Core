"""
==============================================================================
FILE: explainability.py
DESCRIPTION:
Explainable AI (XAI) Module using Grad-CAM.
Visualizes exactly where the AI is looking to make its clinical diagnosis.
==============================================================================
"""

import numpy as np
import tensorflow as tf
import cv2


def get_gradcam_heatmap(img_array, model):
    try:
        # 1. 🔍 Find the exact last Convolutional Layer (Dynamic & Bulletproof)
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

        if not last_conv_layer_name:
            print("⚠️ XAI Error: No Conv2D layer found.")
            return np.zeros((224, 224), dtype=np.float32)

        # 2. 🧠 THE MAGIC FIX: Bypass Softmax Saturation!
        # Save the original activation, remove it temporarily to get raw gradients, then restore it.
        final_layer = model.layers[-1]
        original_activation = final_layer.activation
        final_layer.activation = None  # Use raw Logits

        # 3. 🏗️ Build Gradient Tracking Model
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output],
        )

        # 4. 📈 Compute Gradients
        with tf.GradientTape() as tape:
            inputs = tf.cast(img_array, tf.float32)
            conv_outputs, predictions = grad_model(inputs)

            # Unpack safely
            if isinstance(predictions, list):
                predictions = predictions[0]
            if isinstance(conv_outputs, list):
                conv_outputs = conv_outputs[0]

            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)

        # ⚠️ CRITICAL: Restore the Softmax activation immediately!
        final_layer.activation = original_activation

        if grads is None:
            print("⚠️ XAI Error: Gradients returned None.")
            return np.zeros((224, 224), dtype=np.float32)

        # 5. 🗺️ Generate Heatmap
        conv_outputs = conv_outputs[0]
        grads = grads[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU (We only care about pixels that POSITIVELY identify the disease)
        heatmap = tf.maximum(heatmap, 0)

        # Normalize safely
        max_val = tf.math.reduce_max(heatmap)
        if max_val == 0:
            return np.zeros((224, 224), dtype=np.float32)

        heatmap = heatmap / max_val
        return np.array(heatmap.numpy(), dtype=np.float32)

    except Exception as e:
        print(f"⚠️ XAI Exception: {e}")
        # Ensure we always restore activation even if an error occurs
        try:
            model.layers[-1].activation = tf.keras.activations.softmax
        except:
            pass
        return np.zeros((224, 224), dtype=np.float32)


def create_gradcam_image(img_path, heatmap):
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if np.max(heatmap) == 0 or np.isnan(heatmap).any():
            return img

        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # 🚀 التعديل الأول: Sharpening (انكماش البقع وتحديد المركز)
        # ضرب الخريطة في أس 2 بيخلي القيم العالية تفضل عالية، والقيم المتوسطة تنزل للأرض
        heatmap_resized = np.power(heatmap_resized, 2)

        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        jet_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)

        # 🚀 التعديل التاني: Strict Alpha Blending (قطع الإشارات الضعيفة)
        alpha = heatmap_resized.copy()
        # رفعنا الحد من 0.2 لـ 0.45 عشان نمسح أي بقع ملهاش لازمة ونسيب التركيز الصافي
        alpha[alpha < 0.45] = 0.0
        alpha = np.expand_dims(alpha, axis=-1)

        superimposed = img * (1.0 - alpha) + jet_heatmap * alpha
        return np.clip(superimposed, 0, 255).astype(np.uint8)

    except Exception as e:
        print(f"⚠️ Image Blend Error: {e}")
        img = cv2.imread(img_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
