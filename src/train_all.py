"""Train both models (custom CNN + ResNet50 transfer learning) on the chest X-ray dataset.
Author: Lakshminarayana Aigal
"""
import json, sys, time
import numpy as np
import cv2, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Conv2D, MaxPool2D, Flatten, Dropout,
                                     BatchNormalization, GlobalAveragePooling2D,
                                     Input)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "chest_xray")
RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")
os.makedirs(RESULTS, exist_ok=True); os.makedirs(MODELS, exist_ok=True)
LABELS = ["NORMAL", "PNEUMONIA"]
IMG = 150

def log(*a):
    print(*a, flush=True)

def load_split(split, channels):
    x, y = [], []
    for i, lbl in enumerate(LABELS):
        d = os.path.join(DATA, split, lbl)
        for f in os.listdir(d):
            if not f.lower().endswith((".jpeg", ".jpg", ".png")):
                continue
            img = cv2.imread(os.path.join(d, f), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG, IMG))
            x.append(img if channels == 1 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
            y.append(i)
    x = np.array(x).reshape(-1, IMG, IMG, channels) / 255.0
    return np.array(x), np.array(y)

def plot_history(hist, name):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist.history["accuracy"], label="train")
    if "val_accuracy" in hist.history: ax[0].plot(hist.history["val_accuracy"], label="val")
    ax[0].set_title(f"{name} accuracy"); ax[0].legend()
    ax[1].plot(hist.history["loss"], label="train")
    if "val_loss" in hist.history: ax[1].plot(hist.history["val_loss"], label="val")
    ax[1].set_title(f"{name} loss"); ax[1].legend()
    fig.savefig(os.path.join(RESULTS, f"{name}_curves.png"), dpi=120)

def save_confmat(y, p, name):
    cm = confusion_matrix(y, p)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", color="red", fontsize=14)
    ax.set_xticks([0, 1], LABELS); ax.set_yticks([0, 1], LABELS)
    ax.set_title(f"{name} confusion matrix")
    fig.savefig(os.path.join(RESULTS, f"{name}_confusion_matrix.png"), dpi=120)
    return cm

metrics_all = {}

# ---------------- Model 1: Custom CNN ----------------
log("=" * 30, "Loading data for custom CNN (grayscale)")
t0 = time.time()
x_train, y_train = load_split("train", 1)
x_val, y_val = load_split("val", 1)
x_test, y_test = load_split("test", 1)
log(f"loaded in {time.time()-t0:.0f}s: train {x_train.shape}, val {x_val.shape}, test {x_test.shape}")

model = Sequential([
    Conv2D(32, (3, 3), strides=1, padding="same", activation="relu", input_shape=(IMG, IMG, 1)),
    BatchNormalization(), MaxPool2D((2, 2), strides=2, padding="same"),
    Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"),
    Dropout(0.1), BatchNormalization(), MaxPool2D((2, 2), strides=2, padding="same"),
    Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"),
    BatchNormalization(), MaxPool2D((2, 2), strides=2, padding="same"),
    Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"),
    Dropout(0.2), BatchNormalization(), MaxPool2D((2, 2), strides=2, padding="same"),
    Conv2D(256, (3, 3), strides=1, padding="same", activation="relu"),
    Dropout(0.2), BatchNormalization(), MaxPool2D((2, 2), strides=2, padding="same"),
    Flatten(),
    Dense(128, activation="relu"), Dropout(0.2),
    Dense(1, activation="sigmoid"),
])
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
log(model.summary())

datagen = ImageDataGenerator(
    rotation_range=15, zoom_range=0.1, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.1, horizontal_flip=True)
lr_red = ReduceLROnPlateau(monitor="accuracy", patience=2, factor=0.3, min_lr=1e-6, verbose=1)

log("=" * 30, "TRAINING custom CNN (15 epochs)")
h1 = model.fit(datagen.flow(x_train, y_train, batch_size=32),
               epochs=15, callbacks=[lr_red],
               validation_data=(x_val, y_val), verbose=2)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
pred1 = (model.predict(x_test, verbose=0) > 0.5).astype("int32").reshape(-1)
report1 = classification_report(y_test, pred1, target_names=LABELS)
log("CNN TEST:", f"acc={acc:.4f} loss={loss:.4f}\n", report1)
plot_history(h1, "custom_cnn")
cm1 = save_confmat(y_test, pred1, "custom_cnn")
model.save(os.path.join(MODELS, "laksh_custom_cnn.keras"))
metrics_all["custom_cnn"] = {"test_accuracy": float(acc), "test_loss": float(loss),
                             "report": report1, "confusion_matrix": cm1.tolist()}

# ---------------- Model 2: ResNet50 transfer learning ----------------
log("=" * 30, "Loading data for ResNet (RGB)")
x_train3, y_train3 = load_split("train", 3)
x_val3, y_val3 = load_split("val", 3)
x_test3, y_test3 = load_split("test", 3)
log(f"train {x_train3.shape}, val {x_val3.shape}, test {x_test3.shape}")

base = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG, IMG, 3))
base.trainable = False  # phase 1: freeze entire backbone

inputs = Input(shape=(IMG, IMG, 3))
x = base(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)
rmodel = Model(inputs, outputs)
rmodel.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])

cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_train3), y=y_train3)
cw = dict(enumerate(cw))
log("class weights:", cw)

dg = ImageDataGenerator(rotation_range=15, zoom_range=0.1,
                        width_shift_range=0.1, height_shift_range=0.1,
                        horizontal_flip=True)
lr2 = ReduceLROnPlateau(monitor="val_accuracy", patience=3, factor=0.5, min_lr=1e-7, verbose=1)
es = EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1)

log("=" * 30, "TRAINING ResNet50 phase 1: frozen backbone, head only (5 epochs)")
rmodel.fit(dg.flow(x_train3, y_train3, batch_size=32),
           epochs=5, validation_data=(x_val3, y_val3),
           class_weight=cw, verbose=2)

# phase 2: unfreeze top layers, gentle fine-tune
base.trainable = True
for layer in base.layers[:-10]:
    layer.trainable = False
rmodel.compile(optimizer=keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])
log("=" * 30, "TRAINING ResNet50 phase 2: fine-tune top 10 layers (up to 15 epochs, early stopping)")
h2 = rmodel.fit(dg.flow(x_train3, y_train3, batch_size=32),
                epochs=15, validation_data=(x_val3, y_val3),
                class_weight=cw, callbacks=[lr2, es], verbose=2)

loss2, acc2 = rmodel.evaluate(x_test3, y_test3, verbose=0)
pred2 = (rmodel.predict(x_test3, verbose=0) > 0.5).astype("int32").reshape(-1)
report2 = classification_report(y_test3, pred2, target_names=LABELS)
log("RESNET TEST:", f"acc={acc2:.4f} loss={loss2:.4f}\n", report2)
plot_history(h2, "resnet_transfer")
cm2 = save_confmat(y_test3, pred2, "resnet_transfer")
rmodel.save(os.path.join(MODELS, "laksh_resnet_transfer.keras"))
metrics_all["resnet_transfer"] = {"test_accuracy": float(acc2), "test_loss": float(loss2),
                                  "report": report2, "confusion_matrix": cm2.tolist()}

with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
    json.dump(metrics_all, f, indent=2)
log("=" * 30, "ALL DONE")
