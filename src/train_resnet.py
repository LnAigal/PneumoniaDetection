"""Retrain ResNet50 transfer learning with correct preprocessing.
Author: Lakshminarayana Aigal
"""
import json, time, os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "chest_xray")
RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")
LABELS = ["NORMAL", "PNEUMONIA"]
IMG = 150

def log(*a): print(*a, flush=True)

def load_raw(split):
    x, y = [], []
    for i, lbl in enumerate(LABELS):
        d = os.path.join(DATA, split, lbl)
        for f in os.listdir(d):
            if not f.lower().endswith((".jpeg", ".jpg", ".png")):
                continue
            img = cv2.imread(os.path.join(d, f), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG, IMG))
            x.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
            y.append(i)
    return np.array(x).astype("float32"), np.array(y)

def plot_history(hist, name):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist.history["accuracy"], label="train")
    ax[0].plot(hist.history["val_accuracy"], label="val")
    ax[0].set_title(f"{name} accuracy"); ax[0].legend()
    ax[1].plot(hist.history["loss"], label="train")
    ax[1].plot(hist.history["val_loss"], label="val")
    ax[1].set_title(f"{name} loss"); ax[1].legend()
    fig.savefig(os.path.join(RESULTS, f"{name}_curves.png"), dpi=120)

x_train, y_train = load_raw("train")
x_val, y_val = load_raw("val")
x_test, y_test = load_raw("test")
# ResNet-specific preprocessing (BGR + mean centering)
x_train_p, x_val_p, x_test_p = preprocess_input(x_train.copy()), preprocess_input(x_val.copy()), preprocess_input(x_test.copy())
log(f"train {x_train_p.shape}, val {x_val_p.shape}, test {x_test_p.shape}")

base = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG, IMG, 3))
base.trainable = False

inputs = Input(shape=(IMG, IMG, 3))
x = base(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)
model = Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])

cw = dict(enumerate(class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)))
log("class weights:", cw)

dg = ImageDataGenerator(rotation_range=15, zoom_range=0.1, width_shift_range=0.1,
                        height_shift_range=0.1, horizontal_flip=True)
lr2 = ReduceLROnPlateau(monitor="val_accuracy", patience=3, factor=0.5, min_lr=1e-7, verbose=1)
es = EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1)

log("=" * 30, "PHASE 1: frozen backbone, head only (5 epochs)")
model.fit(dg.flow(x_train_p, y_train, batch_size=32), epochs=5,
          validation_data=(x_val_p, y_val), class_weight=cw, verbose=2)

base.trainable = True
for layer in base.layers[:-10]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])
log("=" * 30, "PHASE 2: fine-tune top 10 layers (up to 15 epochs)")
h2 = model.fit(dg.flow(x_train_p, y_train, batch_size=32), epochs=15,
               validation_data=(x_val_p, y_val), class_weight=cw,
               callbacks=[lr2, es], verbose=2)

loss, acc = model.evaluate(x_test_p, y_test, verbose=0)
pred = (model.predict(x_test_p, verbose=0) > 0.5).astype("int32").reshape(-1)
report = classification_report(y_test, pred, target_names=LABELS)
log("RESNET TEST:", f"acc={acc:.4f} loss={loss:.4f}\n", report)
plot_history(h2, "resnet_transfer")
cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", color="red", fontsize=14)
ax.set_xticks([0, 1], LABELS); ax.set_yticks([0, 1], LABELS)
ax.set_title("resnet_transfer confusion matrix")
fig.savefig(os.path.join(RESULTS, "resnet_transfer_confusion_matrix.png"), dpi=120)
model.save(os.path.join(MODELS, "laksh_resnet_transfer.keras"))

# update metrics.json, keeping the CNN entry
mfile = os.path.join(RESULTS, "metrics.json")
metrics = json.load(open(mfile)) if os.path.exists(mfile) else {}
metrics["resnet_transfer"] = {"test_accuracy": float(acc), "test_loss": float(loss),
                              "report": report, "confusion_matrix": cm.tolist()}
json.dump(metrics, open(mfile, "w"), indent=2)
log("DONE")
