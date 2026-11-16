# Pneumonia Detection from Chest X-Rays

**Course:** Neural Networks & Deep Learning (NNDL) — Open-Ended Project
**Author:** Lakshminarayana Aigal

## Problem Statement

Pneumonia is a lung infection that inflames the air sacs, which may fill with fluid or pus. Chest X-rays are the most common diagnostic tool, but interpreting them requires an experienced radiologist. This project builds deep learning models to automatically classify a chest X-ray as **NORMAL** or **PNEUMONIA**, and compares a **custom (normal) CNN** trained from scratch against a **transfer learning** approach using a pretrained backbone (ResNet).

## Dataset

- **Source:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kaggle, by Paul Mooney & Daniel Kermany
- **Size:** 5,863 chest X-ray images (JPEG) from pediatric patients aged 1–5
- **Classes:** 2 — `NORMAL`, `PNEUMONIA`
- **Predefined splits:**

| Split | Images |
|---|---|
| Train | 5,216 |
| Validation | 16 |
| Test | 624 |

> Note: the official validation split is tiny (16 images), so models are also evaluated with additional validation monitoring during training.

## Methodology

### 1. Custom CNN (`notebooks/laksh_cnn_chest_xray.ipynb`)
- Built from scratch with Keras/TensorFlow
- Stacked `Conv2D → MaxPooling` blocks with increasing filters (32 → 64 → 128), followed by `Flatten`, `Dropout`, and a `Dense` output with `sigmoid`
- Preprocessing: resizing to 180×180, scaling to [0,1], grayscale

### 2. Transfer Learning (`notebooks/laksh_resnet_transfer_learning.ipynb`)
- Backbone: pretrained **ResNet** (ImageNet weights), convolutional base frozen
- Custom classifier head (GlobalAveragePooling → Dropout → Dense)
- Compares convergence speed and accuracy vs. the from-scratch CNN

### Evaluation
- Accuracy, loss curves, confusion matrix, precision/recall
- Side-by-side comparison of both models on the same test set

## Repository Structure

```
├── README.md
├── notebooks/
│   ├── laksh_cnn_chest_xray.ipynb          # Custom CNN from scratch
│   └── laksh_resnet_transfer_learning.ipynb # Transfer learning model
├── src/
│   ├── train_all.py       # End-to-end training script (both models)
│   └── train_resnet.py    # ResNet two-phase fine-tuning script
├── data/          # Kaggle dataset (downloaded via Kaggle API, not committed)
├── models/        # Saved model weights
├── results/       # Plots, confusion matrices, metrics, training logs
└── .gitignore
```

## Setup

```bash
pip install kaggle tensorflow matplotlib scikit-learn
# place your Kaggle API key, then:
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia --unzip -p data/
```

Then run the notebooks in order (CNN first, then transfer learning).

## Results

| Model | Test Accuracy | Precision (NORMAL / PNEUMONIA) | Recall (NORMAL / PNEUMONIA) |
|---|---|---|---|
| Custom CNN (from scratch) | **91.03%** | 0.95 / 0.89 | 0.80 / 0.97 |
| Transfer Learning (ResNet50) | **92.15%** | 0.92 / 0.92 | 0.86 / 0.96 |

Both models were evaluated on the same held-out test set of 624 images. Accuracy/loss curves and confusion matrices are in `results/`. The transfer learning model slightly outperforms the from-scratch CNN and required far fewer training epochs to converge — demonstrating the value of pretrained ImageNet features.

## Learning Outcomes

- Difference between training a CNN from scratch vs. fine-tuning pretrained features
- Handling class imbalance and small validation splits
- Image preprocessing pipelines for medical imaging
- Evaluation beyond raw accuracy (precision/recall matters for medical diagnosis)

## References

- Kaggle dataset: paultimothymooney/chest-xray-pneumonia
- Kermany et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell (2018)
- Rajpurkar et al., *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*, arXiv (2017)
- He et al., *Deep Residual Learning for Image Recognition*, arXiv (2015)
