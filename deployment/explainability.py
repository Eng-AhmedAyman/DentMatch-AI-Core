"""
==============================================================================
FILE: explainability.py
DESCRIPTION:
    Explainable AI (XAI) module for the DentMatch AI system.
    Implements True Grad-CAM++ with auto-detection of the correct hook layer
    and safe preprocessing auto-matching.

    ROOT CAUSE OF PREVIOUS FAILURES (v5, v6, v7):
    ───────────────────────────────────────────────
    1. WRONG HOOK LAYER:
       All previous versions hooked the last Conv2D layer by type search.
       In EfficientNetB4, that is `top_conv` — which comes BEFORE BatchNorm
       and the final activation. Feature maps at that point are un-normalised
       and un-activated, so gradients are noisy and spatially scattered.
       The correct hook point is `top_activation` (post-BN, post-Swish),
       which gives clean, spatially coherent feature maps.

    2. UNSAFE PREPROCESSING ASSUMPTION:
       v6/v7 always applied efficientnet.preprocess_input (scales to [-1,1]).
       If the model was trained with rescale=1./255 (scales to [0,1]), the
       XAI forward pass runs on a completely different input distribution →
       gradients are wrong → heatmap highlights background anatomy.
       FIX: Detect training preprocessing from model config and match it.

    3. EIGENCAM IS CLASS-AGNOSTIC (v5/v6):
       EigenCAM finds the dominant spatial pattern in feature maps, not the
       class-specific one. For Hypodontia, the dominant pattern is the bright
       gum tissue, not the dark gap. Grad-CAM++ is class-specific.

    4. MISSING compile=False ON MODEL LOAD (v7 / app.py):
       Loading the specialist model without compile=False triggers a
       focal-loss deserialisation error at import time in some environments,
       causing load_cam_model() to raise before a single gradient is computed.
       The caller MUST pass compile=False — see CALLER CONTRACT below.

    SOLUTION — ROBUST GRAD-CAM++ (v8.1):
    ──────────────────────────────────────
    1. SMART LAYER DETECTION: Tries layers in priority order:
         top_activation → top_conv → block7 → last spatial feature map
       Guarantees fully-activated feature maps every time.

    2. PREPROCESSING AUTO-MATCH: Inspects the first six non-input layers to
       detect internal Rescaling/Normalization. Falls back to
       efficientnet.preprocess_input when nothing is detected — correct for
       stock EfficientNetB4 from tf.keras.applications.

    3. TRUE GRAD-CAM++ (Chattopadhay et al. 2018):
       - Score  : log(softmax(logits)[predicted_class])  ← class-specific
       - Weights: second-order alpha per channel
       - Combines class-discriminative regions, not just salient ones.

    4. ADAPTIVE OVERLAY:
       - Percentile-based threshold (top 35 % activation mass).
       - Gaussian smoothing for clean edges.
       - Largest contour only (> 0.3 % image area).

PUBLIC API (drop-in — same signatures as all previous versions):
    get_gradcampp_heatmap(img_array, model) -> np.ndarray  (H×W float32 [0,1])
    overlay_heatmap_on_image(original_pil, heatmap, alpha) -> PIL.Image.Image

CALLER CONTRACT:
    • img_array must be shape (1, 224, 224, 3), dtype float32, range [0, 255].
    • model must be loaded with compile=False to avoid focal-loss errors:
          tf.keras.models.load_model(STAGE3_PATH, compile=False)
    • Both functions are thread-safe (no mutable module-level state).

AUTHOR:  Eng. Ahmed Ayman — AI & Data Science Engineer
VERSION: 8.1.0  (Root-Cause Fix + compile=False contract + full docstrings)
==============================================================================
"""

# ==============================================================================
# ZONE 1: IMPORTS
# ==============================================================================
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# ==============================================================================
# ZONE 2: SMART LAYER DETECTION
# ==============================================================================

# Priority-ordered substrings to try as the CAM hook layer name.
# top_activation is post-BN + post-Swish in EfficientNetB4 → best features.
_HOOK_PRIORITY: list[str] = [
    "top_activation",  # EfficientNetB4 best hook (post-BN, post-Swish)
    "top_conv",  # Fallback: last conv before the classifier head
    "block7",  # Fallback: last major residual block
]


def _find_hook_layer(model: tf.keras.Model) -> str | None:
    """
    Identify the optimal layer to hook for Grad-CAM++, in priority order.

    Strategy
    --------
    1. Try each substring in ``_HOOK_PRIORITY`` against every layer name
       (reversed so the *last* matching layer is preferred).
    2. Fall back to the last layer with a 4-D output shape (H, W, C) and
       spatial extent > 1 — i.e., the last spatial feature-map layer.

    Parameters
    ----------
    model : tf.keras.Model
        The loaded Keras model to inspect.

    Returns
    -------
    str | None
        Name of the chosen hook layer, or ``None`` if nothing suitable was
        found.  When ``None`` is returned, callers should return a zero
        heatmap rather than attempting to continue.
    """
    layer_names = [layer.name for layer in model.layers]

    for priority_name in _HOOK_PRIORITY:
        for name in reversed(layer_names):
            if priority_name in name.lower():
                print(f"   >> [XAI] Hook layer (priority match): {name}")
                return name

    # Fallback: last layer with a spatial (H, W, C) output
    for layer in reversed(model.layers):
        try:
            shape = layer.output_shape
            if isinstance(shape, list):
                shape = shape[0]
            if len(shape) == 4 and shape[1] is not None and shape[1] > 1:
                print(f"   >> [XAI] Hook layer (spatial fallback): {layer.name}")
                return layer.name
        except Exception:
            continue

    return None


# ==============================================================================
# ZONE 3: SAFE PREPROCESSING AUTO-MATCH
# ==============================================================================


def _detect_and_preprocess(img_array: np.ndarray, model: tf.keras.Model) -> tf.Tensor:
    """
    Detect the model's expected input preprocessing and apply it.

    Detection logic (inspects the first six non-input layers)
    ----------------------------------------------------------
    - Internal ``Rescaling`` layer with scale ≈ 1/255 or 1/127.5
      → model normalises internally; forward raw ``[0, 255]``.
    - Internal ``Normalization`` layer
      → model normalises internally; forward raw ``[0, 255]``.
    - No normalisation layer detected
      → apply ``efficientnet.preprocess_input`` (scales to ``[-1, 1]``).
        This is correct for stock ``tf.keras.applications.EfficientNetB4``
        built without a manual rescaling layer.

    Why raw ``[0, 255]`` when an internal layer exists?
    Applying ``preprocess_input`` on top of an internal ``Rescaling(1/255)``
    layer sends the model ``[-1, 1]`` input instead of the expected ``[0, 1]``
    — wrong distribution → wrong gradients → misleading heatmap.

    Parameters
    ----------
    img_array : np.ndarray
        Shape ``(1, 224, 224, 3)``, dtype ``float32``, range ``[0, 255]``.
    model : tf.keras.Model
        The full Keras model (inspected for preprocessing clues only).

    Returns
    -------
    tf.Tensor
        Preprocessed tensor ready for the model forward pass.
    """
    img_f = tf.cast(img_array, tf.float32)

    try:
        for layer in model.layers[:6]:
            lname = layer.name.lower()
            ltype = type(layer).__name__.lower()

            if "rescaling" in lname or "rescaling" in ltype:
                cfg = layer.get_config()
                scale = cfg.get("scale", 1.0)
                if abs(scale - 1.0 / 255.0) < 1e-5:
                    print(
                        "   >> [XAI] Preprocessing: internal 1/255 Rescaling → raw [0,255]"
                    )
                    return img_f
                if abs(scale - 1.0 / 127.5) < 1e-4:
                    print(
                        "   >> [XAI] Preprocessing: internal 1/127.5 Rescaling → raw [0,255]"
                    )
                    return img_f

            if "normalization" in lname or "normalization" in ltype:
                print(
                    "   >> [XAI] Preprocessing: internal Normalization layer → raw [0,255]"
                )
                return img_f

    except Exception as exc:
        print(
            f"   >> [XAI] Preprocessing detection error: {exc} "
            "→ falling back to preprocess_input"
        )

    print("   >> [XAI] Preprocessing: applying efficientnet.preprocess_input")
    return tf.keras.applications.efficientnet.preprocess_input(img_f)


# ==============================================================================
# ZONE 4: CORE GRAD-CAM++ ENGINE
# ==============================================================================


def _run_gradcampp(
    img_tensor: tf.Variable,
    model: tf.keras.Model,
    hook_layer_name: str,
) -> np.ndarray | None:
    """
    Compute the Grad-CAM++ heatmap for the top predicted class.

    Algorithm  (Chattopadhay et al., 2018)
    ----------------------------------------
    1. Build a sub-model: ``input → (conv_output, logits)``.
    2. Inside a ``GradientTape``, compute the class-specific score
       ``log(p_c)`` where ``c = argmax(softmax(logits))``.
    3. Obtain first-order gradients ``g = ∂score / ∂A^k``.
    4. Compute second-order alpha weights per channel::

           α^k = g² / (2·g² + Σ_A · g³ + ε)

    5. Weighted ReLU sum over channels → raw CAM.
    6. Percentile normalisation (clips top 1 % outliers).

    Fallback
    --------
    If the weighted CAM is all-zero (model very uncertain), fall back to the
    mean of the positive feature maps.  This is class-agnostic but at least
    non-trivial.  A second all-zero check after the fallback returns ``None``.

    Parameters
    ----------
    img_tensor : tf.Variable
        Preprocessed input, shape ``(1, H, W, 3)``.
    model : tf.keras.Model
        Full Keras model (must be loaded with ``compile=False``).
    hook_layer_name : str
        Name of the hook layer identified by :func:`_find_hook_layer`.

    Returns
    -------
    np.ndarray | None
        Float32 heatmap in ``[0, 1]``, shape ``(H_c, W_c)``.
        ``None`` if computation failed.
    """
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(hook_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_output, logits = grad_model(img_tensor, training=False)

        probs = tf.nn.softmax(logits, axis=-1)
        top_class = int(tf.argmax(probs[0]))
        confidence = float(probs[0, top_class])
        score = tf.math.log(probs[0, top_class] + 1e-7)

        print(
            f"   >> [XAI] Predicted class: {top_class} | Confidence: {confidence:.3f}"
        )

    grads = tape.gradient(score, conv_output)

    if grads is None:
        print(
            "Warning [XAI] GradientTape returned None — "
            "possibly a non-differentiable path or disconnected graph."
        )
        return None

    grads_val = grads[0]  # (H_c, W_c, K)
    conv_maps = conv_output[0]  # (H_c, W_c, K)

    # Second-order Grad-CAM++ weights
    g2 = grads_val**2
    g3 = grads_val**3
    sum_A = tf.reduce_sum(conv_maps, axis=(0, 1), keepdims=True)  # (1, 1, K)
    alpha = g2 / (2.0 * g2 + sum_A * g3 + 1e-7)  # (H_c, W_c, K)

    weights = tf.reduce_sum(
        alpha * tf.nn.relu(grads_val),
        axis=(0, 1),  # → (K,)
    )

    cam = tf.reduce_sum(
        conv_maps * weights[tf.newaxis, tf.newaxis, :],
        axis=-1,
    )  # (H_c, W_c)
    cam = tf.nn.relu(cam).numpy()

    if cam.max() < 1e-7:
        print("Warning [XAI] Grad-CAM++ zero map → falling back to mean-feature CAM.")
        cam = np.mean(np.maximum(conv_maps.numpy(), 0.0), axis=-1)
        if cam.max() < 1e-7:
            print("Warning [XAI] Mean-feature CAM also zero — cannot generate heatmap.")
            return None

    p99 = np.percentile(cam, 99)
    cam = np.clip(cam, 0.0, p99) / (p99 + 1e-7)

    return cam.astype(np.float32)


# ==============================================================================
# ZONE 5: PUBLIC API — get_gradcampp_heatmap
# ==============================================================================


def get_gradcampp_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
) -> np.ndarray:
    """
    Generate a Grad-CAM++ heatmap for the top predicted class.

    This is the primary public entry point.  It orchestrates layer detection,
    preprocessing auto-match, and the Grad-CAM++ computation, returning a
    zero array on any failure so callers can check ``np.max(heatmap) > 0``.

    Parameters
    ----------
    img_array : np.ndarray
        Shape ``(1, 224, 224, 3)``, dtype ``float32``, pixel range ``[0, 255]``.
        Build from a PIL image with::

            arr = np.expand_dims(keras_image.img_to_array(pil.resize((224,224))), 0)

    model : tf.keras.Model
        EfficientNetB4 specialist model.

        .. important::
            Load with ``compile=False`` to avoid focal-loss deserialisation::

                model = tf.keras.models.load_model(path, compile=False)

    Returns
    -------
    np.ndarray
        Float32 heatmap in ``[0, 1]``, shape ``(224, 224)``.
        All-zero array on failure.
    """
    fallback = np.zeros((224, 224), dtype=np.float32)

    try:
        # Step 1: Find best hook layer
        hook_layer = _find_hook_layer(model)
        if hook_layer is None:
            print(
                "Warning [XAI] No suitable hook layer found — returning zero heatmap."
            )
            return fallback

        # Step 2: Auto-match preprocessing
        preprocessed = _detect_and_preprocess(img_array, model)
        img_tensor = tf.Variable(preprocessed)

        # Step 3: Compute Grad-CAM++
        heatmap = _run_gradcampp(img_tensor, model, hook_layer)

        if heatmap is None:
            print(
                "Warning [XAI] _run_gradcampp returned None — returning zero heatmap."
            )
            return fallback

        print(
            f"   OK [XAI] Grad-CAM++ complete | "
            f"range [{heatmap.min():.3f}, {heatmap.max():.3f}] | "
            f"shape {heatmap.shape}"
        )
        return heatmap

    except Exception as exc:
        print(f"Warning [XAI] get_gradcampp_heatmap exception: {exc}")
        return fallback


# ==============================================================================
# ZONE 6: PUBLIC API — overlay_heatmap_on_image
# ==============================================================================


def overlay_heatmap_on_image(
    original_pil: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.60,
) -> Image.Image:
    """
    Blend a Grad-CAM++ heatmap over the original image with adaptive masking.

    The overlay uses a soft alpha mask derived from a percentile threshold so
    that only the most activated regions are tinted, leaving the rest of the
    image unchanged.  A single contour is drawn around the largest activated
    region for clinical clarity.

    Parameters
    ----------
    original_pil : PIL.Image.Image
        Original full-resolution patient image (RGB mode).
    heatmap : np.ndarray
        Float32 array of shape ``(H, W)``, values in ``[0, 1]``, as returned
        by :func:`get_gradcampp_heatmap`.
    alpha : float, optional
        JET blend strength inside the lesion mask (default ``0.60``).
        ``0.0`` = original only; ``1.0`` = full JET overlay.

    Returns
    -------
    PIL.Image.Image
        RGB composite at the same resolution as *original_pil*.
        Returns *original_pil* unchanged on any failure.

    Notes
    -----
    * A 5-pixel border is zeroed out to suppress convolution-padding artefacts.
    * Activation threshold = 65th percentile (top 35 % mass).
      Relaxed to 50th percentile when the active region < 0.5 % of the image,
      so small lesions (e.g. a missing-tooth gap in Hypodontia) remain visible.
    """
    try:
        img_array = np.array(original_pil.convert("RGB"))
        h, w = img_array.shape[:2]

        if heatmap.max() < 1e-7 or np.isnan(heatmap).any():
            print("Warning [XAI] Degenerate heatmap — returning original image.")
            return original_pil

        # 1. Upsample to full resolution (bicubic)
        hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

        # 2. Zero-out 5-pixel border (convolution-padding artefacts)
        b = 5
        hm[:b, :] = hm[-b:, :] = hm[:, :b] = hm[:, -b:] = 0.0

        # 3. Re-normalise after border suppression
        if hm.max() < 1e-7:
            return original_pil
        hm = hm / hm.max()

        # 4. Gaussian smoothing (11×11 kernel)
        hm = cv2.GaussianBlur(hm, (11, 11), 0)

        # 5. Adaptive percentile threshold — top 35 % activation mass
        thresh = float(np.percentile(hm, 65))
        thresh = float(np.clip(thresh, 0.10, 0.70))
        print(f"   >> [XAI] Overlay threshold (p65): {thresh:.3f}")

        hm_thresh = np.where(hm >= thresh, hm, 0.0)

        # Widen threshold if active region is too small (< 0.5 % of image)
        region_ratio = np.sum(hm_thresh > 0) / (h * w)
        if region_ratio < 0.005:
            thresh = float(np.percentile(hm, 50))
            thresh = float(np.clip(thresh, 0.05, 0.60))
            hm_thresh = np.where(hm >= thresh, hm, 0.0)
            print(f"   >> [XAI] Widened threshold (p50, small region): {thresh:.3f}")

        if hm_thresh.max() < 1e-7:
            return original_pil

        # 6. Build smooth alpha mask
        denom = max(1.0 - thresh, 1e-7)
        mask = np.clip((hm_thresh - thresh) / denom, 0.0, 1.0)
        mask_3d = mask[:, :, np.newaxis]

        # 7. JET colormap applied to the full (non-thresholded) heatmap
        jet_bgr = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
        jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)

        # 8. Blend: JET inside mask, original outside
        img_f = img_array.astype(np.float32)
        blended = (1.0 - alpha) * img_f + alpha * jet_rgb.astype(np.float32)
        composite = img_f * (1.0 - mask_3d) + blended * mask_3d
        result = composite.astype(np.uint8)

        # 9. Draw single clean contour around largest active region > 0.3 % image
        binary = (mask > 0.40).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 0.003 * w * h:
                cv2.drawContours(result, [largest], -1, (255, 255, 255), 2)

        return Image.fromarray(result)

    except Exception as exc:
        print(f"Warning [XAI] overlay_heatmap_on_image exception: {exc}")
        return original_pil
