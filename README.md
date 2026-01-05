# 🍓 Strawberry Ripeness Classification

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-94.5%25-success)
![AUC](https://img.shields.io/badge/AUC--ROC-0.99-success)

Automated classification of strawberry ripeness stages for industrial production. Compares **unsupervised clustering (K-Means)** vs **deep learning (ResNet18)**.

<p align="center">
  <img src="assets/clustering_samples.png" alt="Strawberry Ripeness Stages" width="800"/>
</p>

---

## 🎯 Problem

Classify strawberries into **4 ripeness stages**:

| Class | Stage | Color |
|:-----:|-------|-------|
| 0 | Unripe | 🟢 Green |
| 1 | Semi-ripe | 🟡 Yellow-green |
| 2 | Almost ripe | 🟠 Orange |
| 3 | Ripe | 🔴 Red |

---

## 🔬 Methods

| Approach | Method | Accuracy |
|----------|--------|:--------:|
| **Unsupervised** | K-Means + Color Descriptors | 67.9% |
| **Supervised** | ResNet18 (fine-tuned) | **94.5%** ✅ |

---

## 📊 Results

### K-Means Clustering

Tested 6 configurations (3 color spaces × 2 descriptors):

| Model | Silhouette ↑ | Accuracy ↑ | ARI ↑ |
|-------|:------------:|:----------:|:-----:|
| RGB + histogram2d | 0.567 | 0.650 | **0.434** |
| RGB + mean | 0.394 | 0.511 | 0.314 |
| HSV + histogram2d | 0.403 | 0.487 | 0.181 |
| HSV + mean | 0.439 | 0.584 | 0.231 |
| Lab + histogram2d | **0.895** | 0.312 | -0.003 |
| **Lab + mean** | 0.466 | **0.679** | 0.404 |

**Best:** Lab + mean (67.9% accuracy)

---

### CNN Classification (ResNet18)

<p align="center">
  <img src="assets/training_curves.png" alt="Training Curves" width="600"/>
</p>

| Metric | Value |
|--------|:-----:|
| **Accuracy** | 94.51% |
| **F1-Score** | 0.95 |
| **AUC-ROC** | 0.99 |

<p align="center">
  <img src="assets/roc_curve.png" alt="ROC Curve" width="400"/>
  <img src="assets/confusion_matrix.png" alt="Confusion Matrix" width="350"/>
</p>

---

## 🚀 Usage

```bash
# Clone
git clone https://github.com/Noemie21/Strawberry-Ripeness.git
cd Strawberry-Ripeness
pip install -r requirements.txt

# K-Means clustering
python run_clustering.py --color_space Lab --descriptor mean

# CNN training
python train.py --model resnet18 --epochs 30
```

---

## 📁 Structure

```
Strawberry-Ripeness/
├── classifier.py         # K-Means clustering
├── dataset.py            # PyTorch Dataset
├── model.py              # CNN architectures
├── train.py              # Training script
├── run_clustering.py     # Clustering experiments
├── utils.py              # Utilities
└── assets/               # Result visualizations
```

> 📦 Dataset (~4000 images) not included due to size.

---

## 🎯 Key Takeaways

- **K-Means** achieves ~68% accuracy without labels — good for prototyping
- **ResNet18** achieves **94.5% accuracy** with AUC = 0.99
- Lab color space works best for unsupervised clustering
- Most errors occur between adjacent ripeness stages

---


## 📄 License

MIT License
