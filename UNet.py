import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def conv_block(x, filters, name):
    x = L.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = L.BatchNormalization(name=f"{name}_bn1")(x)
    x = L.ReLU(name=f"{name}_relu1")(x)

    x = L.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = L.BatchNormalization(name=f"{name}_bn2")(x)
    x = L.ReLU(name=f"{name}_relu2")(x)
    return x


def build_unet(input_shape=(224, 224, 1), num_classes=3):
    inputs = L.Input(shape=input_shape, name="input")

    # =========================
    # Stem (match your proposal)
    # =========================
    x = L.Conv2D(64, 7, strides=2, padding="same", use_bias=False, name="stem_conv")(inputs)  # 112×112
    x = L.BatchNormalization(name="stem_bn")(x)
    x = L.ReLU(name="stem_relu")(x)
    x = L.MaxPooling2D(3, strides=2, padding="same", name="stem_pool")(x)  # 56×56

    # =========
    # Encoder
    # =========
    e1 = conv_block(x, 64, "enc1")             # 56×56
    p1 = L.MaxPooling2D()(e1)                  # 28×28

    e2 = conv_block(p1, 128, "enc2")           # 28×28
    p2 = L.MaxPooling2D()(e2)                  # 14×14

    e3 = conv_block(p2, 256, "enc3")           # 14×14
    p3 = L.MaxPooling2D()(e3)                  # 7×7

    # Bridge
    b = conv_block(p3, 512, "bridge")           # 7×7

    # =====================
    # Classification head
    # =====================
    x_cls = L.GlobalAveragePooling2D(name="avg_pool")(b)
    x_cls = L.Dense(256, activation="relu")(x_cls)
    x_cls = L.Dense(128, activation="relu")(x_cls)
    x_cls = L.Dense(64, activation="relu")(x_cls)
    x_cls = L.Dense(32, activation="relu")(x_cls)
    cls = L.Dense(num_classes, activation="softmax", name="classifier")(x_cls)

    # =========
    # Decoder
    # =========
    d1 = L.UpSampling2D()(b)                    # 7 → 14
    d1 = L.Concatenate()([d1, e3])
    d1 = conv_block(d1, 256, "dec1")

    d2 = L.UpSampling2D()(d1)                   # 14 → 28
    d2 = L.Concatenate()([d2, e2])
    d2 = conv_block(d2, 128, "dec2")

    d3 = L.UpSampling2D()(d2)                   # 28 → 56
    d3 = L.Concatenate()([d3, e1])
    d3 = conv_block(d3, 64, "dec3")

    # 56 → 224 (×4 upsample like your proposal)
    d4 = L.UpSampling2D(size=4)(d3)
    d4 = L.Conv2D(16, 3, padding="same", activation="relu", name="mask_refine")(d4)
    mask_pred = L.Conv2D(1, 1, activation="sigmoid", name="mask_pred")(d4)

    model = Model(inputs, [mask_pred, cls], name="UNet_Baseline")
    return model
