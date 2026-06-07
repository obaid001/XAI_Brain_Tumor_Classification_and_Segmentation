import cv2, os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
os.makedirs("Report", exist_ok=True)

names = {
    "stem_conv":   "Stem Convolution",

    "enc1_conv2":  "Encoder 1 (Low-level edges)",
    "enc2_conv2":  "Encoder 2 (Textures)",
    "enc3_conv2":  "Encoder 3 (High-level structures)",

    "bridge_conv2": "Bridge (Deep semantics)",

    "dec1_conv2":  "Decoder 1 (Coarse fusion)",
    "dec2_conv2":  "Decoder 2 (Upsampling refinement)",
    "dec3_conv2":  "Decoder 3 (Fine localization)",

    "mask_refine": "Mask Refinement",
    "mask_pred":   "Final Prediction",
}

# -----------------------------------------------------------------------------
# Overlay segmentation mask
# -----------------------------------------------------------------------------
def overlay_mask_on_image(base_img, mask, color=(255, 0, 0), alpha=0.5):
    mask = np.array(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    base = (base_img * 255).astype(np.uint8) if base_img.max() <= 1.0 else base_img.astype(np.uint8)
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    mask = cv2.resize(mask, (base.shape[1], base.shape[0]))
    mask = np.clip(mask, 0.0, 1.0)

    mask_colored = np.zeros_like(base)
    mask_colored[:] = color

    overlay = cv2.addWeighted(base, 1 - alpha, mask_colored, alpha, 0)
    overlay = np.where(mask[..., None] > 0.5, overlay, base)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# -----------------------------------------------------------------------------
# Segmentation Grad-CAM
# -----------------------------------------------------------------------------
def seg_gradcam_for_layer(model, img_array, layer_name):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(layer_name).output,
            model.get_layer("mask_pred").output
        ]
    )

    with tf.GradientTape() as tape:
        feats, mask_out = grad_model(img_array, training=False)
        loss = tf.reduce_mean(mask_out)   # stable segmentation loss

    grads = tape.gradient(loss, feats)
    if grads is None:
        raise ValueError(f"No gradients for layer {layer_name}")

    weights = tf.reduce_mean(grads[0], axis=(0, 1))
    cam = tf.reduce_sum(feats[0] * weights, axis=-1)
    cam = np.maximum(cam, 0)
    cam /= (cam.max() + 1e-8)
    return cam

# -----------------------------------------------------------------------------
# Plot single Grad-CAM
# -----------------------------------------------------------------------------
def plot_layer_gradcam(base_img, cam, layer_name):
    cam = cv2.resize(cam, (base_img.shape[1], base_img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    base = base_img.astype(np.uint8)
    overlay = cv2.addWeighted(base, 0.5, heatmap, 0.5, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    title = names.get(layer_name, layer_name)
    save_path = f"Report/{title}.png"

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title(title)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.show()

    print(f"✅ Saved {save_path}")

# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------
def plot_gradcam(img, model):

    feature_layer_names = [
        "stem_conv",
        "enc1_conv2",
        "enc2_conv2",
        "enc3_conv2",
        "bridge_conv2",
        "dec1_conv2",
        "dec2_conv2",
        "dec3_conv2",
        "mask_refine",
        "mask_pred"
    ]

    layers = [l for l in feature_layer_names if l in [x.name for x in model.layers]]

    img_org = img.copy()
#     img = skull_strip(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    img = np.expand_dims(img, axis=-1)
    Xb = np.expand_dims(img, axis=0)

    # Predict
    mask_pred, cls_pred = model.predict(Xb, verbose=0)
    mask_pred = mask_pred[0, :, :, 0]

    cls_label = int(np.argmax(cls_pred[0]))
    cls_prob = float(np.max(cls_pred[0]))

    overlay = overlay_mask_on_image(img_org, mask_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title(f"Tumor Prediction (Class {cls_label}, p={cls_prob:.2f})")
    plt.axis("off")
    plt.savefig("Report/Predicted Mask Overlay.png", bbox_inches="tight", pad_inches=0.1)
    plt.show()

    print("✅ Saved Predicted Mask Overlay")

    for lname in layers:
        try:
            print(f"🔍 Grad-CAM → {lname}")
            cam = seg_gradcam_for_layer(model, Xb, lname)
            plot_layer_gradcam(img_org, cam, lname)
        except Exception as e:
            print(f"⚠ Skipped {lname}: {e}")
