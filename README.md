# Benchmarking-Models-for-Breast-Cancer-Detection-Using-Mammography-Images

## Overview

This project presents a comprehensive benchmarking framework for Breast tumor detection using Mammography images. It evaluates multiple deep learning architectures — including classical convolutional neural networks (CNNs) and hybrid quantum-classical models — to analyze their performance, robustness, and scalability. I used the same data for training/validating/testing for all models, to keep it fair and compare the performance of the models only. So what I was looking for was an answer to this question: we have CNNs ready at our disposal, QNNs are a new trend, is it worth it to go for hybrid QNN-CNN ? Do they give us any advantages ?

The goal of this project is to provide a reproducible and extensible pipeline for comparing state-of-the-art models in medical image classification.

### Whole project in a glance:

* Step 1 : Trainin well known CNNs pretrained on ImageNet (VGG16/19, DenseNet121/201, etc.)
* Step 2 : Using the trained models in Step 1 for feature extraction then on top of that having quantum layers based on transfer learning technique (PennyLane + JAX)

---

## Objectives

* Benchmarking widely-used CNN architectures for tumor detection
* Explore hybrid Quantum Neural Network (QNN) + CNN models
* Evaluate models using robust metrics beyond accuracy
* Provide a reproducible and modular experimentation pipeline

---

## Models Evaluated

### Classical Models

* VGG16 / VGG19
* ResNet50V2
* DenseNet121 / DenseNet201
* EfficientNetB0
* MobileNetV2
* InceptionV3
* Xception

### Hybrid Models

* CNN feature extractor + Quantum layer (PennyLane + JAX)
* Variable number of qubits (2, 4, 6, 8, 12, 16) with depth 2

---

## Evaluation Metrics

Each model is evaluated using:

* Accuracy
* Precision
* Recall (Sensitivity)
* Specificity
* F1-score
* ROC-AUC
* Confusion Matrix

---

## Dataset

* [VinDr-Mammo](https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png)
* Binary classification: **tumor / no tumor**
* I considered BI-RADS 4 and BI-RADS5 as cancer and BI-RADS 1 and BI-RADS 2 as no cancer
* BI-RADS 3 is ambiguous whether cancer is present or not, so I disregarded it totally

> ⚠️ Due to dataset licensing and privacy constraints, the data is not included in this repository.

---

## Reproducibility

### 1. Clone the repository

```bash
git clone https://github.com/mousavi-hn/Benchmarking-Models-for-Breast-Cancer-Detection-Using-Mammography-Images.git
cd Benchmarking-Models-for-Breast-Cancer-Detection-Using-Mammography-Images
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare dataset

* Download dataset from the provided source
* Place images in:

```
data/Mammography_Images/
    ├── yes/
    ├── no/
```

### 4. Run training

```bash
python scripts/train_hybrid.py
```

---

## Project Structure

```
project/
│
├── CNN/                 
├── hybrid CNN + QNN/                    
├── results/             # Outputs (models, plots, reports)
```

### Sub-project Structure

```
sub-project/
│
├── scripts/        # Entry-point scripts
├── src/            # Core modules (data, models, training)
    ├── configs
    ├── data/
    ├── models/
    ├── train/
    ├── evaluate/

```

---

## Results

Results are automatically saved as:

* CSV summaries
* JSON reports
* Training plots (accuracy & loss curves)

Example output:

```
results/
├── saved_models/
├── plots/
├── reports/
└── hybrid_benchmark_summary.csv
```

---

## Discussion (most favourite part!)

Here I have shared the summary results, top 9 models, in a table sorted by low to high false negatives.

| model_name | n_qubits | fn | fp | tn | tp | accuracy | precision | recall_sensitivity | specificity | f1_score | roc_auc | q_depth | training_time_sec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DenseNet201 | 4 | 36 | 65 | 60 | 77 | 0.5756 | 0.5423 | 0.6814 | 0.4800 | 0.6039 | 0.6183 | 2 | 1629 |
| ResNet50V2 | 4 | 37 | 46 | 79 | 76 | 0.6513 | 0.6230 | 0.6726 | 0.6320 | 0.6468 | 0.7069 | 2 | 1816 |
| MobileNetV2 | 12 | 40 | 64 | 61 | 73 | 0.5630 | 0.5328 | 0.6460 | 0.4880 | 0.5840 | 0.6183 | 2 | 1666 |
| DenseNet121 | NaN | 42 | 55 | 70 | 71 | 0.5924 | 0.5634 | 0.6283 | 0.56 | 0.5941 | 0.6392 | Nan | 1835 | 
| VGG16 | 8 | 43 | 46 | 79 | 70 | 0.6261 | 0.6034 | 0.6195 | 0.6320 | 0.6114 | 0.6859 | 2 | 3166 |
| EfficientNetB0 | 2 | 43 | 60 | 65 | 70 | 0.5672 | 0.5385 | 0.6195 | 0.5200 | 0.5761 | 0.6181 | 2 | 2328 |
| ResNet50V2 | 8 | 45 | 50 | 75 | 68 | 0.6008 | 0.5763 | 0.6018 | 0.6000 | 0.5887 | 0.6554 | 2 | 1692 |
| Xception | 8 | 45 | 55 | 70 | 68 | 0.5798 | 0.5528 | 0.6018 | 0.5600 | 0.5763 | 0.5933 | 2 | 6123 |
| VGG19 | 8 | 46 | 50 | 75 | 67 | 0.5966 | 0.5726 | 0.5929 | 0.6000 | 0.5826 | 0.6288 | 2 | 4519 |


### Some important points about the table:
* In top 9 models we have 8 hybrids and 1 classical CNN
* DenseNet201 with 4 qubits and depth 2 shows a roughly 14.3% decrease in number of false negatives vs DenseNet121 which had the best performance among classical CNNs
* The above two points shows clearly that performance increases by adding a quantum layer to our classical CNNs
* I have provided the full tables with all the variants in TABLES.md, you find it in the main page of repository

## Key Contributions

* Unified benchmarking of multiple CNN architectures
* Integration of hybrid quantum-classical models
* Modular and reproducible ML pipeline
* Evaluation using medically relevant metrics

---

## Future Work

* Tumor segmentation (e.g., U-Net architectures)
* Model explainability (Grad-CAM, saliency maps)
* Hyperparameter optimization
* Clinical dataset validation

---

## References

* Deep Learning for Medical Image Analysis
* Quantum Machine Learning frameworks (PennyLane, JAX)
* Transfer learning in medical imaging

---

## Acknowledgments

This project is developed as part of ongoing research and study in machine learning and medical imaging.

---

## Pretrained models can be found here:
https://drive.google.com/drive/folders/1aWNdRHJqgsHlvnzahH66naxPWpQ-hzuQ?usp=sharing

* Due to high volume of the trained models, I have shared them in a google drive link! Please feel free to have a look and use the models for your work, in that case I would be glad if you please let me know about it, yet there are no licensing restrictions here, all is my independent work!

---

## Contact

For questions or collaboration:

* GitHub: https://github.com/mousavi-hn
* Email: mousavi.hn@gmail.com


