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

---

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
```

---

## 4. Environment & Installation

### 4.1. Prerequisites

Before you start, you should have:

- **Python 3.12.10** (3.10+ also works)
- **Git**
- (Optional but recommended) **CUDA-capable GPU** with recent NVIDIA drivers

> Install PyTorch according to your OS / CUDA version from the official PyTorch website.

### 4.2. Create and activate a virtual environment

It is recommended to create a virtual environment to manage project dependencies.

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 4.3. Install Python dependencies

Install the required packages using the provided `requirements.txt` file:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Training the MobileNetV3 model

The main training entry point is `model.py`.

### 5.1. Basic training command

```bash
python model.py --data casting_data --model mobilenet --aug-level standard --lr 2e-4
```

### 5.2. Training outputs and artifacts

Each training run creates a timestamped subfolder under `runs_final_v4/`, for example:

```text
runs_final_v4/
└── mobilenet_adamw_standard_dr0.5_20251122_154442/
    ├── best_model.pt
    ├── hparams.json
    ├── history.csv
    ├── history.jsonl
    ├── train_samples.png
    ├── loss_curves.png
    ├── f1_curves.png
    ├── learning_curves.png
    ├── confusion_matrices.png
    ├── pr_curve.png
    ├── roc_curve.png
    ├── reliability.png
    └── top-errors.png
```

---

## 6. Evaluation results

After training, the script automatically reloads the best MobileNetV3 checkpoint (selected on validation macro-F1 with a tuned threshold) and evaluates it on the held-out test split.

On the cleaned **casting_data_fixed4** dataset the final run achieves:

- **Training set:** accuracy ≈ **0.98**, macro-F1 ≈ **0.98**, recall ≈ **0.98**, precision ≈ **0.99**.
- **Validation set:** accuracy ≈ **0.97**, macro-F1 ≈ **0.97**, recall ≈ **0.98**, precision ≈ **0.97**.
- **Test set:** accuracy ≈ **0.96**, macro-F1 ≈ **0.96**, recall ≈ **0.97**, precision ≈ **0.97**.

This corresponds to an overall error rate of about **2%** on train/validation and **4%** on the test set.

The script also prints a compact "quality metrics" table (Train / Valid / Test) with additional metrics such as Balanced Accuracy, MCC, AUC-PR, AUC-ROC, Brier score, ECE/MCE and Expected Cost, which can be inspected directly in the console output if needed.

---

## 7. QA Inspector GUI

In addition to the training script, the repository includes a desktop application **`qa_inspector.py`** that allows manual inspection of the trained model on individual images or whole folders.

### 7.1. Launching the inspector

With your virtual environment activated and dependencies installed:

```bash
python qa_inspector.py
```

### 7.2. QA Inspector — running the compiled .exe

If you don't want to install Python, you can run QA Inspector as a standalone Windows executable.

**Building the executable:**

```bash
pyinstaller --onefile --windowed --name "QA_Inspector" prototype_final.py --add-data ".FINAL/model_full.pt;.FINAL" --add-data "qa_inspector_settings.json;."
```

After building, your `dist/` folder should look like:

```text
dist/
├── QA_Inspector.exe
└── qa_inspector_settings.json
```

**Run:**

1. Go to: `dist/`
2. Double-click: `QA_Inspector.exe`

### 7.3. Using QA Inspector

#### 1. Load model

- Click **Browse…** and select a model file:
  - TorchScript (`*.torchscript.pt`, `*.pt`)
  - Full model saved with `torch.save(model, ...)` (`*.pt`, `*.pth`)
  - State_dict alone is not supported — export full model/TorchScript.

#### 2. Set classes

- Enter exact class names, comma-separated, e.g.: `def_front, ok_front`
- Use **Swap classes** if the order is reversed.

#### 3. Choose Positive class + Threshold

- Pick which class the threshold applies to (usually `def_front`).
- Move the **Threshold** slider to trade missed defects (FN) vs false alarms (FP).

#### 4. Match preprocessing to training

- **Channels:** RGB (3ch) for ImageNet backbones, grayscale (1ch) if trained that way.
- **Size:** usually 224.
- **Normalization:**
  - RGB preset: ImageNet norm
  - Gray preset: 0.5 / 0.5
- Optional **CenterCrop** to mirror training.

#### 5. Single image mode

- Click **Open image…**
- Click **Predict**

#### 6. Batch mode

- Click **Load Folder…** and point to a root with subfolders per class:

  ```text
  test/
  ├── def_front/
  └── ok_front/
  ```

- Navigate with **Previous** / **Next** / **Random**, then **Predict**.
- The status line shows GT → MATCH/MISMATCH for quick scanning.

#### Outputs shown

- Predicted label (OK/DEF)
- Confidence (max probability)
- p(OK) and p(DEF)
- Latency (ms)
- Current threshold + class mapping

#### Settings persistence

- App auto-saves your last settings (size, mean/std, threshold, device, etc.) into `qa_inspector_settings.json`.
- Delete this file if you want a clean reset.

![QA Inspector Interface](images/qainspector.png)
*Figure 7.3a: QA Inspector interface showing prediction results*

---

## 8. Recommendations

Based on our experiments and error analysis, here are practical recommendations for using this model:

- **Use it as decision support.** The model is reliable, but we still recommend a human-in-the-loop setup, especially for borderline cases.
- **Keep capture conditions consistent.** The model works best when the camera angle and lighting match the training setup. Strong glare or deep shadows increase errors.
- **Watch low-contrast rims.** Tiny defects on low-contrast areas are the main source of missed defects, so these cases deserve extra attention.
- **Tune the threshold for your goal.** If the priority is to catch every defect, lower the threshold (higher recall). If the priority is fewer false alarms, raise it.
- **Re-check calibration after changes.** If camera, lighting, surface finish, or part type changes, we should re-validate performance and possibly re-tune the threshold.

---

## 9. Next Steps

If we continued this project, these are the most natural next directions:

1. **Small production-style validation set.** Collect a small batch of real shop-floor images to confirm robustness under real lighting and handling.
2. **Improve lighting / reduce glare at capture.** Even a simple diffuser or better angle control could reduce reflection-based false positives.
3. **Optional light fine-tuning.** Right now we use MobileNetV3 as a frozen feature extractor. If needed, we could unfreeze the last block(s) and fine-tune with a very low LR, then compare fairly again.
4. **More targeted data for edge cases.** Add more examples of low-contrast defects and shiny rims, because these dominate the remaining errors.
5. **Extend QA Inspector with quick reporting.** For example: export a CSV summary for a batch run (counts, FP/FN list), keeping the same simple workflow.