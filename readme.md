# Metal Casting Defect Detection with Transfer Learning (MobileNetV3)

This repository contains a complete, reproducible pipeline for **automated visual inspection of metal castings** using **transfer learning with MobileNetV3** and a dedicated **QA Inspector** desktop tool for manual inspection.

The project is built around the public *casting product* dataset from Kaggle (binary classification of casting surfaces into **defective** vs **non-defective** parts) and focuses on:

- transparent training & evaluation of a transfer learning model,  
- detailed QA of model quality and calibration,  
- a small GUI app for interactive inspection of test images.

---

## 1. Project Overview

- **Task:** Binary image classification of casting surfaces  
  - `def_front` – defective part  
  - `ok_front` – non-defective part  
- **Model:** `mobilenet_v3_small` pretrained on ImageNet, used as a **frozen feature extractor** with a custom 2-class classifier head.  
- **Framework:** PyTorch  
- **Extras:**
  - Rich console logs and summary tables (train / val / test)  
  - Calibration metrics (Brier score, ECE, MCE, Expected Cost)  
  - Tkinter-based **QA Inspector** app to inspect single images or entire test folders.

---

## 2. Team & Roles

The project was developed as a team effort:

- **Jakub Wiktor Michalski** – Project Manager / AI Engineer  
- **Mikołaj Skrocki** – Tech Lead / Software Engineer  
- **Piotr Żuryński** – AI/ML Engineer / QA & Release Owner  

## 3. Dataset

The project uses the *Real-life industrial dataset of casting product* from Kaggle:

- **Source:** `https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product`
- **Original structure:** `train/` and `test/` folders with two subfolders:
  - `def_front/`
  - `ok_front/`

For training and QA in this project, the dataset is reorganized into:

```text
casting_data_fixed4/
    train/
        def_front/
        ok_front/
    val/
        def_front/
        ok_front/
    test/
        def_front/
        ok_front/
