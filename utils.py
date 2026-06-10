import matplotlib.pyplot as plt
import cv2, math
import tensorflow as tf
import numpy as np

def smooth_curve(y, alpha=0.1):
    y = np.array(y)
    smoothed = np.zeros_like(y)
    smoothed[0] = y[0]
    for i in range(1, len(y)):
        smoothed[i] = alpha * y[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed

def plot_history(history):
    train_keys = [k for k in history.keys() if not k.startswith('val_')]

    metric_groups = {}
    for k in train_keys:
        metric_groups.setdefault(k, []).append(k)
        if f'val_{k}' in history:
            metric_groups[k].append(f'val_{k}')

    n_metrics = len(metric_groups)
    cols = 2  
    rows = math.ceil(n_metrics / cols)

    plt.figure(figsize=(6 * cols, 4 * rows))

    for idx, (metric, keys) in enumerate(metric_groups.items(), start=1):
        plt.subplot(rows, cols, idx)
        for k in keys:
#             plt.plot(history[k], label=k)
            plt.plot(
                smooth_curve(history[k], alpha=0.05),
                label=f"{k} (EMA)",
                linewidth=2
            )

        plt.title(metric)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def corrupt_center(img, mask, max_frac=0.10, min_radius=3, ring_radius=15):
    H, W = mask.shape

    fg = mask > 0.5
    mask_area = fg.sum()

    if mask_area == 0:
        return img, mask, 0

    ys, xs = np.where(fg)

    # exact mask center / centroid
    cy = int(np.round(ys.mean()))
    cx = int(np.round(xs.mean()))

    max_radius = int(np.sqrt((max_frac * mask_area) / np.pi))
    max_radius = max(min_radius, max_radius)

    R = np.random.randint(min_radius, max_radius + 1)

    yy, xx = np.ogrid[:H, :W]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * ring_radius + 1, 2 * ring_radius + 1)
    )

    dilated_mask = cv2.dilate(fg.astype(np.uint8), kernel).astype(bool)
    ring_area = dilated_mask & (~fg)

    if ring_area.sum() == 0:
        ring_area = ~fg

    source_pixels = img[ring_area]

    img_new = img.copy()
    mask_new = mask.copy()

    mask_new[circle] = 0

    img_new[circle] = np.random.choice(
        source_pixels,
        size=circle.sum(),
        replace=True
    )

    return img_new, mask_new


import cv2
import numpy as np

def corrupt_center(img, mask, max_frac=0.10, min_radius=3, ring_radius=15):
    H, W = mask.shape[:2]

    fg = mask > 127 if mask.max() > 1 else mask > 0.5
    mask_area = fg.sum()

    if mask_area == 0:
        return img, mask

    ys, xs = np.where(fg)

    cy = int(np.round(ys.mean()))
    cx = int(np.round(xs.mean()))

    max_radius = int(np.sqrt((max_frac * mask_area) / np.pi))
    max_radius = max(min_radius, max_radius)

    R = np.random.randint(min_radius, max_radius + 1)

    yy, xx = np.ogrid[:H, :W]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * ring_radius + 1, 2 * ring_radius + 1)
    )

    dilated = cv2.dilate(fg.astype(np.uint8), kernel).astype(bool)

    # preferred source: just outside the mask
    ring_area = dilated & (~fg)

    # fallback 1: anywhere outside mask
    if ring_area.sum() == 0:
        ring_area = ~fg

    # fallback 2: anywhere outside corrupted circle
    if ring_area.sum() == 0:
        ring_area = ~circle

    # fallback 3: whole image
    if ring_area.sum() == 0:
        ring_area = np.ones((H, W), dtype=bool)

    source_pixels = img[ring_area]

    img_new = img.copy()
    mask_new = mask.copy()

    mask_new[circle] = 0

    if source_pixels.size > 0 and circle.sum() > 0:
        img_new[circle] = np.random.choice(
            source_pixels.reshape(-1),
            size=circle.sum(),
            replace=True
        )

    return img_new, mask_new

def flip(img, mask):
    aug = np.random.choice([np.fliplr, np.flipud])

    return aug(img), aug(mask)

def data_gen(df, batch_size=3):
    while True:
        X, y1, y2 = [], [], []
        sample = df.sample(batch_size, replace = True)
        for i in range(len(sample)):
            img = cv2.imread(sample['image_path'].iloc[i], 0)
            img = cv2.resize(img, [224, 224])
            
            mask = cv2.imread(sample['mask_path'].iloc[i], 0)
            mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)


            aug = np.random.choice([corrupt_center, flip, None])
            if aug:
                img, mask = aug(img, mask)

            img = img/255
            mask = mask/255
            
            l = sample['label'].iloc[i]
            l = tf.keras.utils.to_categorical(l, num_classes=3)
            
            X.append(img)
            y1.append(mask)
            y2.append(l)
            
        X = np.array(X)
        X = np.expand_dims(X, -1)
        y1 = np.array(y1)
        y1 = np.expand_dims(y1, -1)
        y2 = np.array(y2)
        
        yield X, [y1, y2]