import tensorflow as tf
from tensorflow.keras import layers as L, Model
from tensorflow.keras import layers as Layers
import tensorflow.keras.backend as K
import numpy as np
from scipy.ndimage import distance_transform_edt

class SEBlock(L.Layer):
    def __init__(self, filters, ratio=16, name=None):
        super().__init__(name=name)
        self.gap = L.GlobalAveragePooling2D()
        self.fc1 = L.Dense(filters // ratio, activation='relu')
        self.fc2 = L.Dense(filters, activation='sigmoid')
        self.reshape = L.Reshape((1, 1, filters))
    def call(self, x):
        s = self.gap(x)
        s = self.fc1(s)
        s = self.fc2(s)
        s = self.reshape(s)
        return x * s

def drop_block(x, rate=0.2, name=None):
    return L.SpatialDropout2D(rate, name=name)(x)

def mhsa_block(x, num_heads=4, reduction=1, name=None, attn_dropout=0, proj_dropout=0):
    H, W, C = x.shape[1], x.shape[2], x.shape[3]
    assert C % num_heads == 0, f"Channels ({C}) must be divisible by num_heads ({num_heads})."
    q = L.LayerNormalization(name=f'{name}_q_ln')(x)
    q_seq = L.Reshape((H*W, C), name=f'{name}_q_reshape')(q)
    kv = L.AveragePooling2D(pool_size=reduction, strides=reduction, padding='same', name=f'{name}_pool')(x) if reduction > 1 else x
    Hk, Wk = kv.shape[1], kv.shape[2]
    kv = L.LayerNormalization(name=f'{name}_kv_ln')(kv)
    kv_seq = L.Reshape((Hk*Wk, C), name=f'{name}_kv_reshape')(kv)
    attn = L.MultiHeadAttention(
        num_heads=num_heads, key_dim=C // num_heads, dropout=attn_dropout, name=f'{name}_mha'
    )(q_seq, kv_seq, kv_seq)
    x_seq = L.Add(name=f'{name}_res')([q_seq, attn])
    x_seq = L.LayerNormalization(name=f'{name}_post_ln')(x_seq)
    out = L.Reshape((H, W, C), name=f'{name}_reshape_back')(x_seq)
    if proj_dropout and proj_dropout > 0:
        out = L.Dropout(proj_dropout, name=f'{name}_proj_drop')(out)
    return out

class ConvBlock(L.Layer):
    def __init__(self, filters, stride=1, block_name='block', use_se=True, drop_rate=0.2):
        super().__init__(name=block_name)
        filter1, filter2, filter3 = filters
        self.conv1 = L.Conv2D(filter1, 1, strides=stride, use_bias=False, name=f'{block_name}_conv1')
        self.bn1 = L.BatchNormalization(name=f'{block_name}_bn1')
        self.conv2 = L.Conv2D(filter2, 3, padding='same', use_bias=False, name=f'{block_name}_conv2')
        self.bn2 = L.BatchNormalization(name=f'{block_name}_bn2')
        self.conv3 = L.Conv2D(filter3, 1, use_bias=False, name=f'{block_name}_conv3')
        self.bn3 = L.BatchNormalization(name=f'{block_name}_bn3')
        self.relu = L.LeakyReLU(alpha=0.1, name=f'{block_name}_relu')
        self.use_se = use_se
        if use_se: self.se = SEBlock(filter3, ratio=8, name=f'{block_name}_se')
        self.drop = L.SpatialDropout2D(drop_rate, name=f'{block_name}_drop')
        self.shortcut = tf.keras.Sequential(name=f'{block_name}_shortcut')
        if stride != 1 or filter3 != filter1:
            self.shortcut.add(L.Conv2D(filter3, 1, strides=stride, use_bias=False, name=f'{block_name}_sc_conv'))
            self.shortcut.add(L.BatchNormalization(name=f'{block_name}_sc_bn'))
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        if self.use_se:
            x = self.se(x)
        x = self.drop(x, training=training)
        shortcut = self.shortcut(inputs)
        x = L.add([x, shortcut], name=f'{self.name}_add')
        x = self.relu(x)
        return x

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return bce + d_loss

num_heads=4
def build_model(input_shape=(224,224,1), num_classes=3):
    inputs = L.Input(shape=input_shape, name='input')
    # Stem
    x = L.Conv2D(64, 7, strides=2, padding='same', use_bias=False, name='stem_conv')(inputs)  # 112x112
    x = L.BatchNormalization(name='stem_bn')(x)
    x = L.ReLU(name='stem_relu')(x)
    x = L.MaxPooling2D(3, strides=2, padding='same', name='stem_pool')(x)  # 56x56
    # Encoder: Save all relevant skips for U-Net decoder
    x1 = ConvBlock([64, 64, 256], stride=1, block_name='block1')(x)     # 56x56
    x2 = ConvBlock([64, 64, 256], stride=2, block_name='block2')(x1)    # 28x28
    att1 = mhsa_block(x2, num_heads=num_heads, name='b2_mhsa')
    x3 = ConvBlock([64, 64, 256], stride=2, block_name='block3')(att1)  # 14x14
    att2 = mhsa_block(x3, num_heads=num_heads, name='b3_mhsa')
    x4 = ConvBlock([64, 64, 256], stride=2, block_name='block4')(att2)  # 7x7
    att3 = mhsa_block(x4, num_heads=num_heads, name='b4_mhsa')
    feats = L.Add(name='A3')([x4, att3])  # Final encoder features (7x7)
    # Classification head
    x_cls = L.GlobalAveragePooling2D(name='avg_pool')(feats)
    x_cls = L.Dense(256, activation='relu')(x_cls)
    x_cls = L.Dense(128, activation='relu')(x_cls)
    x_cls = L.Dense(64, activation='relu')(x_cls)
    x_cls = L.Dense(32, activation='relu')(x_cls)
    cls = L.Dense(num_classes, activation='softmax', name='classifier')(x_cls)
    
    # Decoder
    # 7x7 -> 14x14
    x = L.UpSampling2D()(feats)
    x = L.Concatenate()([x, att2])  # att2 is 14x14
    x = ConvBlock([64, 64, 128], block_name='up_block1')(x)
    up_att1 = mhsa_block(x, num_heads=num_heads, name='up1_mhsa')
    x = SEBlock(128)(up_att1)
    x = Layers.Add(name='Add1')([x, up_att1])
    x = drop_block(x, 0.15, name='up_drop1')
    
    # 14x14 -> 28x28
    x = L.UpSampling2D()(x)
    x = L.Concatenate()([x, att1])  # att1 is 28x28
    x = ConvBlock([32, 32, 64], block_name='up_block2')(x)
    up_att2 = mhsa_block(x, num_heads=num_heads, name='up2_mhsa')
    x = SEBlock(64)(up_att2)
    x = Layers.Add(name='Add2')([x, up_att2])
    x = drop_block(x, 0.1, name='up_drop2')
    
    # 28x28 -> 56x56
    x = L.UpSampling2D()(x)
    x = L.Concatenate()([x, x1])  # x1 is 56x56
    x = ConvBlock([16, 16, 32], block_name='up_block3')(x)
    up_att3 = mhsa_block(x, num_heads=num_heads, name='up3_mhsa')
    x = SEBlock(32)(up_att3)
    x = Layers.Add(name='Add3')([x, up_att3])
    x = drop_block(x, 0.1, name='up_drop3')
    
    # 56x56 -> 224x224 (up 4x)
    up_att4 = mhsa_block(x, num_heads=num_heads, name='up4_mhsa')
    x = L.UpSampling2D(size=4)(up_att4)
    x = L.Conv2D(16, 3, padding='same', activation='relu', name='mask_refine')(x)
    mask_pred = L.Conv2D(1, 1, activation='sigmoid', name='mask_pred')(x)
    model = Model(inputs, [mask_pred, cls], name='SabreNet')
    return model

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_loss_plus_bce(y_true, y_pred):
    bce_loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice_coeff = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    dice_loss = 1.0 - dice_coeff
    
    return bce_loss + dice_loss

def sef(gt, grad):
    """
    gt, grad: tensors of shape [B, H, W] or [B, H, W, 1]
              binary masks: 0/1
    returns: scalar mean FinalScore over batch
    """

    grad = grad/grad.max()
    grad = grad>0.5

    gt = gt>0.5

    eps = tf.keras.backend.epsilon()

    gt = tf.cast(gt, tf.float32)
    grad = tf.cast(grad, tf.float32)

    if gt.shape.rank == 4:
        gt = tf.squeeze(gt, axis=-1)
        grad = tf.squeeze(grad, axis=-1)

    grad_bin = tf.cast(tf.equal(grad, 1.0), tf.float32)
    gt_bin = tf.cast(tf.equal(gt, 1.0), tf.float32)

    # IoU
    intersection = tf.reduce_sum(grad_bin * gt_bin, axis=[1, 2])
    union = tf.reduce_sum(
        tf.cast(tf.logical_or(grad_bin == 1, gt_bin == 1), tf.float32),
        axis=[1, 2]
    )

    IoU = intersection / (union + eps)

    # Coverage
    gt_area = tf.reduce_sum(gt_bin, axis=[1, 2])
    Coverage = intersection / (gt_area + eps)

    # Distance map equivalent needs scipy normally.
    # TensorFlow has no direct distance_transform_edt.
    # Approximation: compute distance from each grad pixel to nearest gt pixel.
    batch_size = tf.shape(gt_bin)[0]
    H = tf.shape(gt_bin)[1]
    W = tf.shape(gt_bin)[2]

    yy, xx = tf.meshgrid(
        tf.range(H, dtype=tf.float32),
        tf.range(W, dtype=tf.float32),
        indexing="ij"
    )

    coords = tf.stack([yy, xx], axis=-1)  # [H, W, 2]

    def compute_focus(args):
        g, y = args

        gt_points = tf.where(tf.equal(y, 1.0))
        grad_points = tf.where(tf.equal(g, 1.0))

        gt_points = tf.cast(gt_points, tf.float32)
        grad_points = tf.cast(grad_points, tf.float32)

        def no_points():
            return tf.constant(0.0, dtype=tf.float32)

        def has_points():
            # pairwise squared distances: grad points -> gt points
            diff = grad_points[:, None, :] - gt_points[None, :, :]
            dist2 = tf.reduce_sum(tf.square(diff), axis=-1)
            min_dist2 = tf.reduce_min(dist2, axis=1)

            D = tf.reduce_mean(min_dist2)

            ys = gt_points[:, 0]
            xs = gt_points[:, 1]

            w_gt = tf.reduce_max(xs) - tf.reduce_min(xs) + 1.0
            h_gt = tf.reduce_max(ys) - tf.reduce_min(ys) + 1.0
            s_gt = tf.sqrt(w_gt ** 2 + h_gt ** 2)

            D_norm = D / (s_gt ** 2 + eps)
            return 1.0 / (1.0 + D_norm)

        return tf.cond(
            tf.logical_or(tf.equal(tf.shape(gt_points)[0], 0),
                          tf.equal(tf.shape(grad_points)[0], 0)),
            no_points,
            has_points
        )

    Focus = tf.map_fn(
        compute_focus,
        (grad_bin, gt_bin),
        fn_output_signature=tf.float32
    )

    final_score = tf.reduce_mean(
        tf.stack([IoU, Coverage, Focus], axis=1),
        axis=1
    )

    return tf.reduce_mean(final_score)

def sef_numpy(y_true, y_pred):
    eps = np.finfo(np.float32).eps

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    scores = []

    for gt_mask, grad in zip(y_true, y_pred):
        grad = np.squeeze(grad).astype(np.float32)
        gt = np.squeeze(gt_mask).astype(np.float32)

        max_val = grad.max()
        if max_val > 0:
            grad = grad / max_val

        grad = (grad > 0.5).astype(np.float32)
        gt = (gt > 0.5).astype(np.float32)

        intersection = np.logical_and(grad == 1, gt == 1).sum()
        union = np.logical_or(grad == 1, gt == 1).sum()
        iou = intersection / (union + eps)

        gt_area = (gt == 1).sum()
        coverage = intersection / (gt_area + eps)

        if gt_area == 0 or grad.sum() == 0:
            focus = 0.0
        else:
            dist_map = distance_transform_edt(1 - gt)
            d = np.sum(grad * dist_map ** 2) / (np.sum(grad) + eps)

            ys, xs = np.where(gt == 1)
            w_gt = xs.max() - xs.min() + 1
            h_gt = ys.max() - ys.min() + 1
            s_gt = np.sqrt(w_gt ** 2 + h_gt ** 2)

            d_norm = d / (s_gt ** 2 + eps)
            focus = 1.0 / (1.0 + d_norm)

        final_score = np.mean([iou, coverage, focus])
        scores.append(final_score)

    return np.array(np.mean(scores), dtype=np.float32)


class SEFMetric(tf.keras.metrics.Metric):
    def __init__(self, name="sef", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        score = tf.py_function(
            func=sef_numpy,
            inp=[y_true, y_pred],
            Tout=tf.float32
        )
        score.set_shape(())

        self.total.assign_add(score)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / (self.count + tf.keras.backend.epsilon())

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)