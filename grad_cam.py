import cv2, os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
os.makedirs("Report", exist_ok=True)
names = {
        "conv2d_88":   "Shallow CNN Features",
        "conv_block_7": "Encoder Block 1 (Low-level)",
        "conv_block_8": "Encoder Block 2 (Texture)",
        "conv_block_9": "Encoder Block 3 (High-level)",
        "conv_block_10": "Encoder Block 4 (Semantic)",
        "conv_block_11": "Decoder Block 1 (Fusion)",
        "conv_block_12": "Decoder Block 2 (Upsampling)",
        "conv_block_13": "Decoder Block 3 (Localization)",
        "mask_pred":    "Final Prediction",
    }

def skull_strip(img):
    img = cv2.resize(img, [224,224])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ----------------------------
    # 1. Binarize (auto threshold)
    # ----------------------------
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Ensure foreground is white
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    # ----------------------------
    # 2. Morphology to FORCE closure
    # ----------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    binary_closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=3
    )

    binary_closed = cv2.dilate(binary_closed, kernel, iterations=2)

    # ----------------------------
    # 3. Fill holes (guarantees solid object)
    # ----------------------------
    filled = binary_closed.copy()
    h, w = filled.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(filled, mask, (0, 0), 255)
    filled = cv2.bitwise_not(filled)
    filled = cv2.bitwise_or(filled, binary_closed)

    # ----------------------------
    # 4. Find ONLY external contours
    # ----------------------------
    contours, _ = cv2.findContours(
        filled,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    assert len(contours) > 0, "No contours found"

    # Largest contour = outer boundary
    outer = max(contours, key=cv2.contourArea)

    # ----------------------------
    # 5. Ensure contour is closed
    # ----------------------------
    if not np.array_equal(outer[0], outer[-1]):
        outer = np.vstack([outer, np.expand_dims(outer[0], 0)])

    # ----------------------------
    # 6. Draw result
    # ----------------------------
    result = img.copy()
#     cv2.drawContours(result, [outer], -1, (0, 0, 255), 2)


    epsilon = 10
    inner_contour = cv2.approxPolyDP(outer, epsilon, True)

    cv2.drawContours(result, [inner_contour], -1, (0, 0, 0), 11)
            
    return result

def overlay_mask_on_image(base_img, mask, color=(255, 0, 0), alpha=0.5):
    """Overlay segmentation mask (binary or soft) on grayscale image."""
    mask = np.array(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    base = (base_img * 255).astype(np.uint8) if base_img.max() <= 1.0 else base_img.astype(np.uint8)
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    elif base.shape[-1] == 1:
        base = cv2.cvtColor(base[..., 0], cv2.COLOR_GRAY2BGR)

    mask = cv2.resize(mask, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = np.clip(mask, 0.0, 1.0)

    mask_colored = np.zeros_like(base, dtype=np.uint8)
    mask_colored[:, :, 0] = color[0]
    mask_colored[:, :, 1] = color[1]
    mask_colored[:, :, 2] = color[2]

    overlay = cv2.addWeighted(base, 1 - alpha, mask_colored, alpha, 0)
    overlay = np.where(mask[..., None] > 0.5, overlay, base)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay


def seg_gradcam_for_layer(model, img_array, layer_name):
    """
    Layer-wise Grad-CAM for segmentation head.
    Backprop from mask_pred to given layer.
    """
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output,
                 model.get_layer("mask_pred").output]
    )

    with tf.GradientTape() as tape:
        feats, mask_out = grad_model(img_array, training=False)
        # scalar loss: mean log probability of predicted mask
        loss = tf.reduce_mean(tf.math.log(mask_out + 1e-6))

    grads = tape.gradient(loss, feats)
    if grads is None:
        raise ValueError(f"Gradients are None for layer {layer_name}")

    # Global average pooling over spatial dims
    weights = tf.reduce_mean(grads[0], axis=(0, 1))         # (C,)
    cam = tf.reduce_sum(feats[0] * weights, axis=-1)        # (H,W)
    cam = np.maximum(cam, 0)
    cam /= (cam.max() + 1e-8)
    return cam


def plot_layer_gradcam(base_img, cam, layer_name):
    # Resize CAM to input size
    cam_resized = cv2.resize(cam, (base_img.shape[1], base_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

    base = (base_img * 255).astype(np.uint8) if base_img.max() <= 1.0 else base_img.astype(np.uint8)
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    elif base.shape[-1] == 1:
        base = cv2.cvtColor(base[..., 0], cv2.COLOR_GRAY2BGR)

    overlayed = cv2.addWeighted(base, 0.5, heatmap, 0.5, 0)
    overlayed = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlayed)
    plt.title(f"{names.get(layer_name, layer_name)}")
    plt.axis("off")
    save_path = f"Report/{names.get(layer_name, layer_name)}.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.show()
    print(f"✅ Saved {save_path}")


def plot_gradcam(img, model, plot = True):
    
    feature_layers = [
    l for l in model.layers
    if hasattr(l, "output_shape")
    and len(l.output_shape) == 4
    and ("block" in l.name or "reshape_back" in l.name)
    ]

    layers = [l.name for l in feature_layers]
    
    for lname in layers:
        names.setdefault(lname, lname)

    if any(l.name == "mask_pred" for l in model.layers):
        layers.append("mask_pred")

    layers = list(dict.fromkeys(layers))
    
    img_org = img.copy()
    img = skull_strip(img)
    # img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
    img = np.expand_dims(img, -1)
    Xb = np.expand_dims(img, 0)
    img = np.squeeze(img)

    # Dual-output model: (mask_pred, cls_pred)
    mask_pred, cls_pred = model.predict(Xb, verbose=0)
    mask_pred = np.array(mask_pred[0, :, :, 0], dtype=np.float32)
    cls_pred = np.array(cls_pred[0], dtype=np.float32)

    cls_label = int(np.argmax(cls_pred))
    cls_prob = float(np.max(cls_pred))

    overlay_img = overlay_mask_on_image(img_org, mask_pred, color=(255, 0, 0), alpha=0.5)
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay_img)
    plt.title(f"Predicted Tumor Region (Class {cls_label}, p={cls_prob:.2f})")
    plt.axis("off")
    plt.savefig("Report/Predicted Mask Overlay.png", bbox_inches="tight", pad_inches=0.1)
    plt.show()
    print("✅ Saved Predicted Mask Overlay")

    grads = []
    for layer_name in layers:
        try:
            print(f"🔍 Computing Seg-Grad-CAM for layer: {layer_name}")
            cam = seg_gradcam_for_layer(model, Xb, layer_name)
            
            if "mhsa" in layer_name:
                im = (cam*255).astype("uint8")
                im = cv2.resize(im, (224, 224))
                grads.append(im)
            
            if plot:
                plot_layer_gradcam(img_org, cam, layer_name)

        except Exception as e:
            print(f"⚠ Skipped {layer_name} due to: {e}")
    return grads