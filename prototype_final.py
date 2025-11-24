#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Any

import numpy as np
from PIL import Image, ImageOps, ImageTk, ImageGrab
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import random

import torch
import torch.nn as nn


# =========================
# App configuration
# =========================

APP_TITLE = "QA Inspector — Metal Castings (v3)"
APP_SETTINGS = Path("qa_inspector_settings.json")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_CLASSES = ["ok_front", "def_front"]
DEFAULT_POSITIVE = "def_front"
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]


@dataclass
class UIState:
    model_file: str = ""
    classes: List[str] = tuple(DEFAULT_CLASSES)
    positive: str = DEFAULT_POSITIVE
    channels: str = "RGB (3ch)"
    input_size: int = 224
    threshold: float = 0.50
    mean: str = "0.5"
    std: str = "0.5"
    centercrop: bool = True
    device: str = "auto"  # auto/cpu/cuda


def load_settings() -> UIState:
    try:
        if APP_SETTINGS.exists():
            data = json.loads(APP_SETTINGS.read_text(encoding="utf-8"))
            return UIState(**data)
    except Exception:
        pass
    return UIState()


def save_settings(state: UIState):
    try:
        APP_SETTINGS.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    except Exception:
        pass


# =========================
# Model loading
# =========================

ERR_STATEDICT = (
    "The selected file looks like a 'state_dict' (weights only).\n"
    "Save the full model object instead:\n"
    "    torch.save(model, 'model_full.pt')\n"
    "or export TorchScript:\n"
    "    scripted = torch.jit.trace(model, example)\n"
    "    scripted.save('model.torchscript.pt')"
)


def load_model_generic(path: Path, device: torch.device) -> Tuple[Optional[nn.Module], Optional[str]]:
    try:
        m = torch.jit.load(str(path), map_location=device)
        m.eval()
        return m, None
    except Exception:
        pass

    try:
        try:
            from torchvision.models.mobilenetv3 import MobileNetV3
            from torch.serialization import add_safe_globals  # type: ignore
            add_safe_globals({MobileNetV3})
        except Exception:
            pass

        m = torch.load(str(path), map_location=device, weights_only=False)
        if isinstance(m, dict):
            return None, ERR_STATEDICT
        if not isinstance(m, nn.Module):
            return None, ERR_STATEDICT
        m.eval()
        return m, None
    except Exception as e:
        return None, f"Failed to load model:\n{e}"


# =========================
# Preprocessing / prediction
# =========================

def parse_floats_maybe_list(text: str, channels: int) -> List[float]:
    parts = [p.strip() for p in str(text).split(",") if p.strip() != ""]
    if len(parts) == 0:
        return [0.5] * channels
    if len(parts) == 1:
        v = float(parts[0])
        return [v] * channels
    vals = [float(p) for p in parts]
    if len(vals) < channels:
        vals += [vals[-1]] * (channels - len(vals))
    return vals[:channels]


def preprocess_pil(
    pil_img: Image.Image,
    size: int,
    rgb: bool,
    mean_txt: str,
    std_txt: str,
    center_crop: bool
) -> torch.Tensor:
    img = pil_img.convert("RGB") if rgb else ImageOps.grayscale(pil_img)

    if center_crop:
        w, h = img.size
        if min(w, h) == 0:
            raise ValueError("Empty image.")
        scale = size / min(w, h)
        img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
        w2, h2 = img.size
        left = (w2 - size) // 2
        top = (h2 - size) // 2
        img = img.crop((left, top, left + size, top + size))
    else:
        img = img.resize((size, size), Image.BILINEAR)

    arr = np.array(img).astype(np.float32) / np.float32(255.0)

    if rgb:
        C = 3
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
    else:
        C = 1
        if arr.ndim == 3:
            arr = arr[..., 0:1]
        else:
            arr = arr[..., None]

    mean = np.asarray(parse_floats_maybe_list(mean_txt, C), dtype=np.float32)
    std = np.asarray(parse_floats_maybe_list(std_txt, C), dtype=np.float32)

    arr = (arr - mean) / std
    arr = np.transpose(arr.astype(np.float32), (2, 0, 1))  # HWC -> CHW
    x = torch.from_numpy(arr).unsqueeze(0)  # [1,C,H,W]
    return x


def forward_probs(model: nn.Module, x: torch.Tensor, device: torch.device) -> Tuple[np.ndarray, float]:
    x = x.to(device, non_blocking=True, dtype=torch.float32)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    if out.ndim != 2 or out.shape[0] != 1 or out.shape[1] not in (1, 2):
        raise RuntimeError(f"Unsupported model output shape: {tuple(out.shape)}")

    if out.shape[1] == 1:
        p_def = torch.sigmoid(out[0, 0]).item()
        p_ok = 1.0 - p_def
    else:
        sm = torch.softmax(out, dim=1)[0]
        p_ok = sm[0].item()
        p_def = sm[1].item()

    return np.array([p_ok, p_def], dtype=np.float32), dt_ms


def decide(
    pred_ok: float,
    pred_def: float,
    classes: List[str],
    positive: str,
    thr: float
) -> str:
    c0 = classes[0].strip().lower()
    if {"ok_front", "def_front"} <= set(map(str.lower, classes)):
        if c0 == "def_front":
            pred_ok, pred_def = pred_def, pred_ok

    if positive.lower() == "def_front":
        return "def_front" if pred_def >= thr else "ok_front"
    else:
        return "ok_front" if pred_ok >= thr else "def_front"


# =========================
# GUI (tkinter)
# =========================

class App(tk.Tk):
    def _toggle_norm_fields(self):
        for w in (self.norm_adv_frame, self.norm_simple_frame, self.gray_norm_frame):
            w.grid_forget()

        rgb = self.channels_var.get().startswith("RGB")

        if self.adv_norm_var.get():
            if rgb:
                self.norm_adv_frame.grid(row=5, column=3, columnspan=3, sticky="we")
            else:
                self.gray_norm_frame.grid(row=5, column=3, columnspan=3, sticky="we")
        else:
            if rgb:
                self.norm_simple_frame.grid(row=5, column=3, columnspan=3, sticky="we")
            else:
                self.gray_norm_frame.grid(row=5, column=3, columnspan=3, sticky="we")

    def _imagenet_norm(self):
        if self.channels_var.get().startswith("RGB"):
            self.mean_r.set("0.485")
            self.mean_g.set("0.456")
            self.mean_b.set("0.406")
            self.std_r.set("0.229")
            self.std_g.set("0.224")
            self.std_b.set("0.225")
            self._update_status("ImageNet mean/std set.", color="#455a64")
        else:
            self.gray_mean.set("0.5")
            self.gray_std.set("0.5")
            self._update_status("Gray mean/std set to 0.5/0.5.", color="#455a64")
        self._toggle_norm_fields()

    def _gray05_norm(self):
        self.gray_mean.set("0.5")
        self.gray_std.set("0.5")
        self._update_status("Gray mean/std set to 0.5/0.5.", color="#455a64")
        self._toggle_norm_fields()

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.state = load_settings()
        self.model: Optional[nn.Module] = None
        self.device = torch.device("cpu")
        self.current_image: Optional[Image.Image] = None
        self.preview_imgtk: Optional[ImageTk.PhotoImage] = None
        self.fullscreen_state = False

        self.batch_files: List[Path] = []
        self.batch_labels: List[str] = []
        self.batch_index = -1

        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure(".", font=("Segoe UI", 12))
        self.style.configure("TButton", padding=8)
        self.style.configure("TLabel", padding=2)

        self.pred_var = tk.StringVar(value="—")
        self.conf_var = tk.StringVar(value="—")
        self.pok_var = tk.StringVar(value="—")
        self.pdef_var = tk.StringVar(value="—")
        self.lat_var = tk.StringVar(value="—")
        self.img_path = tk.StringVar(value="")
        self.model_file = tk.StringVar()
        self.classes_var = tk.StringVar()
        self.positive_var = tk.StringVar()
        self.channels_var = tk.StringVar()
        self.size_var = tk.IntVar()
        self.thr_var = tk.DoubleVar()
        self.center_var = tk.BooleanVar()
        self.dev_var = tk.StringVar()
        self.thr_val_var = tk.StringVar()
        self.adv_norm_var = tk.BooleanVar(value=False)
        self.mean_r = tk.StringVar()
        self.mean_g = tk.StringVar()
        self.mean_b = tk.StringVar()
        self.std_r = tk.StringVar()
        self.std_g = tk.StringVar()
        self.std_b = tk.StringVar()
        self.gray_mean = tk.StringVar()
        self.gray_std = tk.StringVar()
        self.batch_dir = tk.StringVar(value="")
        self.batch_status_var = tk.StringVar(value="—")
        self.true_label_var = tk.StringVar(value="—")

        self._build_ui()
        self._load_state_into_ui()
        self._update_status("Ready. Press F11 for fullscreen.", color="#2e7d32")

        self.bind("<F11>", self._toggle_fullscreen)
        self.bind("<Escape>", self._end_fullscreen)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        ttk.Label(left, text="Image", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        self.canvas = tk.Canvas(left, bg="#202020", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        btnbar = ttk.Frame(left)
        btnbar.pack(fill="x", pady=(8, 0))
        ttk.Button(btnbar, text="Open image…", command=self._open_image).pack(side="left")
        ttk.Entry(btnbar, textvariable=self.img_path).pack(side="left", fill="x", expand=True, padx=(8, 0))

        ttk.Label(right, text="Model & Settings", font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, columnspan=6, sticky="w"
        )

        right.columnconfigure((1, 3, 5), weight=2)
        right.columnconfigure((0, 2, 4), weight=1)

        ttk.Label(right, text="Model file").grid(row=1, column=0, sticky="w")
        ttk.Entry(right, textvariable=self.model_file).grid(row=1, column=1, columnspan=4, sticky="we")
        ttk.Button(right, text="Browse", command=self._browse_model).grid(row=1, column=5, sticky="we", padx=(6, 0))

        ttk.Label(right, text="Classes (comma)").grid(row=2, column=0, sticky="w")
        ttk.Entry(right, textvariable=self.classes_var).grid(row=2, column=1, columnspan=5, sticky="we")

        ttk.Label(right, text="Positive class").grid(row=3, column=0, sticky="w")
        ttk.Combobox(
            right, textvariable=self.positive_var, values=["ok_front", "def_front"],
            state="readonly", width=12
        ).grid(row=3, column=1, sticky="w", padx=(0, 6))

        ttk.Label(right, text="Channels").grid(row=3, column=2, sticky="e")
        self.channels_combo = ttk.Combobox(
            right, textvariable=self.channels_var,
            values=["RGB (3ch)", "grayscale (1ch)"], state="readonly", width=14
        )
        self.channels_combo.grid(row=3, column=3, sticky="w")
        self.channels_combo.bind("<<ComboboxSelected>>", lambda e: self._toggle_norm_fields())

        ttk.Label(right, text="Size").grid(row=3, column=4, sticky="e")
        ttk.Spinbox(
            right, from_=128, to=512, increment=32, textvariable=self.size_var, width=6
        ).grid(row=3, column=5, sticky="w")

        thr_frame = ttk.Frame(right)
        thr_frame.grid(row=4, column=0, columnspan=6, sticky="we", pady=(4, 0))
        ttk.Label(thr_frame, text="Threshold:").pack(side="left")
        ttk.Scale(
            thr_frame, from_=0.0, to=1.0, variable=self.thr_var, length=200
        ).pack(side="left", fill="x", expand=True, padx=6)
        self.thr_var.trace_add("write", lambda *args: self.thr_val_var.set(f"{self.thr_var.get():.2f}"))
        ttk.Label(thr_frame, textvariable=self.thr_val_var).pack(side="left")

        ttk.Label(right, text="Normalize").grid(row=5, column=0, sticky="w")
        adv_chk = ttk.Checkbutton(
            right, text="Advanced normalization", variable=self.adv_norm_var,
            command=self._toggle_norm_fields
        )
        adv_chk.grid(row=5, column=1, sticky="w", columnspan=2)

        self.norm_simple_frame = ttk.Frame(right)
        self.norm_simple_frame.grid(row=5, column=3, columnspan=3, sticky="we")
        ttk.Button(
            self.norm_simple_frame, text="ImageNet norm", command=self._imagenet_norm, width=15
        ).pack(side="left", padx=(0, 6), fill="x", expand=True)
        ttk.Button(
            self.norm_simple_frame, text="0.5/0.5 (gray)", command=self._gray05_norm, width=15
        ).pack(side="left", fill="x", expand=True)

        self.norm_adv_frame = ttk.Frame(right)
        ttk.Label(self.norm_adv_frame, text="Mean").grid(row=0, column=0, sticky="e", padx=(0, 6))
        ttk.Entry(self.norm_adv_frame, textvariable=self.mean_r, width=7).grid(row=0, column=1, sticky="w")
        ttk.Entry(self.norm_adv_frame, textvariable=self.mean_g, width=7).grid(row=0, column=2, sticky="w", padx=(6, 0))
        ttk.Entry(self.norm_adv_frame, textvariable=self.mean_b, width=7).grid(row=0, column=3, sticky="w", padx=(6, 0))
        ttk.Label(self.norm_adv_frame, text="Std").grid(row=1, column=0, sticky="e", padx=(0, 6))
        ttk.Entry(self.norm_adv_frame, textvariable=self.std_r, width=7).grid(row=1, column=1, sticky="w")
        ttk.Entry(self.norm_adv_frame, textvariable=self.std_g, width=7).grid(row=1, column=2, sticky="w", padx=(6, 0))
        ttk.Entry(self.norm_adv_frame, textvariable=self.std_b, width=7).grid(row=1, column=3, sticky="w", padx=(6, 0))

        self.gray_norm_frame = ttk.Frame(right)
        ttk.Label(self.gray_norm_frame, text="Mean/Std").grid(row=0, column=0, sticky="e", padx=(0, 6))
        ttk.Entry(self.gray_norm_frame, textvariable=self.gray_mean, width=7).grid(row=0, column=1, sticky="w")
        ttk.Entry(self.gray_norm_frame, textvariable=self.gray_std, width=7).grid(row=0, column=2, sticky="w", padx=(6, 0))

        ttk.Label(right, text="Pre-proc. options").grid(row=6, column=0, columnspan=6, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            right, text="CenterCrop (like training)", variable=self.center_var
        ).grid(row=7, column=0, columnspan=2, sticky="w")
        ttk.Button(right, text="Swap classes", command=self._swap_classes).grid(
            row=7, column=4, columnspan=2, sticky="we", padx=(6, 0)
        )

        ttk.Label(right, text="Device").grid(row=8, column=0, sticky="w")
        ttk.Combobox(
            right, textvariable=self.dev_var, values=["auto", "cpu", "cuda"],
            state="readonly", width=10
        ).grid(row=8, column=1, sticky="w")

        btn_row = ttk.Frame(right)
        btn_row.grid(row=8, column=3, columnspan=3, sticky="we")
        btn_row.columnconfigure((0, 1, 2), weight=1)

        ttk.Button(btn_row, text="Load model", command=self._load_model).grid(row=0, column=0, sticky="we", padx=(0, 6))
        ttk.Button(btn_row, text="Predict", command=self._predict).grid(row=0, column=1, sticky="we", padx=(0, 6))
        ttk.Button(btn_row, text="Save screenshot", command=self._save_screenshot).grid(row=0, column=2, sticky="we")

        batch_frame = ttk.LabelFrame(right, text="Batch Inspection", padding=10)
        batch_frame.grid(row=9, column=0, columnspan=6, sticky="we", pady=(10, 0))
        batch_frame.columnconfigure((0, 2, 4, 6), weight=1)
        batch_frame.columnconfigure((1, 3, 5), weight=0)

        ttk.Button(
            batch_frame, text="Load Folder...", command=self._load_batch_folder
        ).grid(row=0, column=0, sticky="we", padx=(0, 6))
        ttk.Entry(
            batch_frame, textvariable=self.batch_dir, state="readonly"
        ).grid(row=0, column=1, columnspan=4, sticky="we", padx=(0, 6))
        ttk.Label(batch_frame, textvariable=self.batch_status_var).grid(row=0, column=5, columnspan=2, sticky="e")

        ttk.Button(
            batch_frame, text="< Previous", command=lambda: self._navigate_batch(-1)
        ).grid(row=1, column=0, columnspan=2, sticky="we", pady=(6, 0), padx=(0, 6))

        ttk.Button(
            batch_frame, text="Random", command=self._navigate_random
        ).grid(row=1, column=2, sticky="we", pady=(6, 0), padx=(0, 6))

        ttk.Label(
            batch_frame, text="Ground Truth:", font=("Segoe UI", 12, "bold")
        ).grid(row=1, column=3, sticky="e", pady=(6, 0))

        tk.Label(
            batch_frame, textvariable=self.true_label_var, font=("Segoe UI", 14, "bold"),
            fg="#006064", bg=self.cget("bg"), anchor="w"
        ).grid(row=1, column=4, sticky="w", pady=(6, 0), padx=(6, 6))

        ttk.Button(
            batch_frame, text="Next >", command=lambda: self._navigate_batch(1)
        ).grid(row=1, column=5, columnspan=2, sticky="we", pady=(6, 0))

        sep = ttk.Separator(right, orient="horizontal")
        sep.grid(row=10, column=0, columnspan=6, sticky="we", pady=(10, 4))

        bg_color = self.cget("bg")

        ttk.Label(right, text="Prediction:").grid(row=11, column=0, sticky="w")
        tk.Label(
            right, textvariable=self.pred_var, font=("Segoe UI", 16, "bold"),
            fg="#1976d2", bg=bg_color, anchor="w"
        ).grid(row=11, column=1, sticky="w", columnspan=5)

        ttk.Label(right, text="Confidence:").grid(row=12, column=0, sticky="w")
        ttk.Label(right, textvariable=self.conf_var).grid(row=12, column=1, sticky="w")
        ttk.Label(right, text="p(DEF):").grid(row=13, column=0, sticky="w")
        ttk.Label(right, textvariable=self.pok_var).grid(row=13, column=1, sticky="w")
        ttk.Label(right, text="p(OK):").grid(row=14, column=0, sticky="w")
        ttk.Label(right, textvariable=self.pdef_var).grid(row=14, column=1, sticky="w")
        ttk.Label(right, text="Latency [ms]:").grid(row=15, column=0, sticky="w")
        ttk.Label(right, textvariable=self.lat_var).grid(row=15, column=1, sticky="w")

        self.status = tk.Label(
            self, text="—", anchor="w", padx=8, pady=6,
            fg="#ffffff", bg="#303a46", font=("Segoe UI", 12)
        )
        self.status.pack(fill="x", side="bottom")

    def _load_state_into_ui(self):
        st = self.state
        self.model_file.set(st.model_file)
        self.classes_var.set(", ".join(st.classes))
        self.positive_var.set(st.positive)
        self.channels_var.set(st.channels)
        self.size_var.set(st.input_size)
        self.thr_var.set(st.threshold)
        self.center_var.set(st.centercrop)
        self.dev_var.set(st.device)
        self.thr_val_var.set(f"{st.threshold:.2f}")

        self._set_norm_from_state(st)
        self._toggle_norm_fields()

    def _grab_state(self) -> UIState:
        classes = [p.strip() for p in self.classes_var.get().split(",") if p.strip()]
        if not classes:
            classes = DEFAULT_CLASSES

        if self.channels_var.get().startswith("RGB"):
            mean_txt = f"{self.mean_r.get()}, {self.mean_g.get()}, {self.mean_b.get()}"
            std_txt = f"{self.std_r.get()}, {self.std_g.get()}, {self.std_b.get()}"
        else:
            mean_txt = self.gray_mean.get()
            std_txt = self.gray_std.get()

        return UIState(
            model_file=self.model_file.get(),
            classes=classes,
            positive=self.positive_var.get(),
            channels=self.channels_var.get(),
            input_size=int(self.size_var.get()),
            threshold=float(self.thr_var.get()),
            mean=mean_txt,
            std=std_txt,
            centercrop=bool(self.center_var.get()),
            device=self.dev_var.get(),
        )

    def _load_batch_folder(self):
        folder_path = filedialog.askdirectory(title="Select Root Folder with Test Images")
        if not folder_path:
            return

        path = Path(folder_path)
        self.batch_dir.set(str(path))
        self.batch_files = []
        self.batch_labels = []

        valid_classes = set(map(str.lower, [c.strip() for c in self.classes_var.get().split(",")]))

        for ext in IMAGE_EXTENSIONS:
            for file_path in path.glob(f"**/{ext}"):
                parent_name = file_path.parent.name.lower()
                if parent_name in valid_classes:
                    self.batch_files.append(file_path)
                    self.batch_labels.append(parent_name)

        if not self.batch_files:
            self._update_status(f"No images found in class subfolders under {path.name}.", color="#b71c1c")
            self.batch_index = -1
            self.batch_status_var.set("No files")
            self.true_label_var.set("—")
            return

        self.batch_index = 0
        self._show_batch_image()
        self._update_status(
            f"Loaded {len(self.batch_files)} images. Use Next / Previous / Random.",
            color="#2e7d32",
        )

    def _navigate_batch(self, direction: int):
        if not self.batch_files:
            self._update_status("Load a batch folder first.", color="#ff9800")
            return

        new_index = self.batch_index + direction
        num_files = len(self.batch_files)

        if 0 <= new_index < num_files:
            self.batch_index = new_index
            self._show_batch_image()
        else:
            self._update_status(
                f"Reached the {'start' if new_index < 0 else 'end'} of the list.",
                color="#455a64",
            )

    def _navigate_random(self):
        if not self.batch_files or len(self.batch_files) <= 1:
            self._update_status("At least two files are needed for random navigation.", color="#ff9800")
            return

        num_files = len(self.batch_files)
        new_index = random.randrange(0, num_files)
        if new_index == self.batch_index and num_files > 1:
            new_index = (new_index + 1) % num_files

        self.batch_index = new_index
        self._show_batch_image()

    def _show_batch_image(self):
        if self.batch_index == -1 or not self.batch_files:
            return

        file_path = self.batch_files[self.batch_index]
        true_label = self.batch_labels[self.batch_index]

        try:
            img = Image.open(file_path)
            self.current_image = img
            self._on_canvas_resize(None)
            self.img_path.set(str(file_path))

            self.pred_var.set("—")
            self.conf_var.set("—")
            self.pok_var.set("—")
            self.pdef_var.set("—")
            self.lat_var.set("—")

            self.batch_status_var.set(f"{self.batch_index + 1}/{len(self.batch_files)}")
            self.true_label_var.set(true_label.upper())
            self._update_status(
                f"Loaded: {file_path.name}. Ground truth: {true_label.upper()}. Click Predict.",
                color="#006064",
            )

        except Exception as e:
            self.current_image = None
            self._update_status(f"Failed to load file: {file_path.name}: {e}", color="#b71c1c")
            if self.batch_index < len(self.batch_files) - 1:
                self.batch_index += 1
                self.after(50, self._show_batch_image)

    def _toggle_fullscreen(self, event: Any = None):
        self.fullscreen_state = not self.fullscreen_state
        self.attributes("-fullscreen", self.fullscreen_state)

        if self.fullscreen_state:
            self.status.pack_forget()
        else:
            self.status.pack(fill="x", side="bottom")

        if self.current_image:
            self.after(100, lambda: self._on_canvas_resize(None))

    def _end_fullscreen(self, event: Any = None):
        self.fullscreen_state = False
        self.attributes("-fullscreen", False)
        self.status.pack(fill="x", side="bottom")
        if self.current_image:
            self.after(100, lambda: self._on_canvas_resize(None))

    def _browse_model(self):
        p = filedialog.askopenfilename(
            title="Select model",
            filetypes=[("Model files", "*.pt;*.pth;*.torchscript.pt"), ("All", "*.*")],
        )
        if p:
            self.model_file.set(p)

    def _open_image(self):
        p = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"), ("All", "*.*")],
        )
        if not p:
            return

        self.batch_index = -1
        self.batch_files = []
        self.batch_dir.set("")
        self.batch_status_var.set("—")
        self.true_label_var.set("—")

        self.img_path.set(p)
        try:
            img = Image.open(p)
            self.current_image = img
            self._on_canvas_resize(None)
            self._update_status("Image loaded. Ready to predict.", color="#2e7d32")
        except Exception as e:
            self.current_image = None
            self._update_status(f"Image load error: {e}", color="#b71c1c")
            messagebox.showerror("Error", f"Cannot load image:\n{e}")

    def _on_canvas_resize(self, event: Any):
        if not self.current_image:
            return

        w_canvas = self.canvas.winfo_width()
        h_canvas = self.canvas.winfo_height()

        if w_canvas < 10 or h_canvas < 10:
            return

        self._show_preview(self.current_image, target_width=w_canvas, target_height=h_canvas)

    def _show_preview(self, pil_img: Image.Image, target_width: int, target_height: int):
        w, h = pil_img.size
        scale = min(target_width / max(1, w), target_height / max(1, h))

        if scale != 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            pil_img_resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
        else:
            pil_img_resized = pil_img

        self.preview_imgtk = ImageTk.PhotoImage(pil_img_resized)

        x_center = target_width // 2
        y_center = target_height // 2

        self.canvas.delete("all")
        self.canvas.create_image(x_center, y_center, image=self.preview_imgtk, anchor="center")

    def _swap_classes(self):
        parts = [p.strip() for p in self.classes_var.get().split(",") if p.strip()]
        if len(parts) != 2:
            messagebox.showerror("Error", "Classes must contain exactly 2 names, e.g. 'ok_front, def_front'.")
            return

        parts = [parts[1], parts[0]]
        self.classes_var.set(", ".join(parts))

        pos = self.positive_var.get().strip()
        self.positive_var.set(parts[0] if pos == parts[1] else parts[1])
        self._update_status(
            "Class order swapped. Update positive class if needed.",
            color="#455a64",
        )

    def _decide_device(self, choice: str) -> torch.device:
        c = (choice or "auto").lower()
        if c == "cpu":
            return torch.device("cpu")

        if c in ("cuda", "auto"):
            if torch.cuda.is_available():
                return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    def _load_model(self):
        st = self._grab_state()
        if not st.model_file:
            messagebox.showerror("Error", "Select a model file first.")
            return

        self.device = self._decide_device(st.device)
        self._update_status(f"Loading model on {self.device}…", color="#006064")
        self.update_idletasks()

        try:
            m, err = load_model_generic(Path(st.model_file), self.device)
            if m is None:
                self._update_status("Model load failed.", color="#b71c1c")
                messagebox.showerror("Model load error", err or "Failed to load model.")
                return
            self.model = m.to(self.device)
            self._update_status(f"Model loaded on {self.device}.", color="#2e7d32")
        except Exception as e:
            self._update_status(f"Load error: {e}", color="#b71c1c")
            messagebox.showerror("Error", f"An error occurred:\n{e}\n\n{traceback.format_exc()}")

    def _predict(self):
        if self.model is None:
            messagebox.showerror("Error", "Load a model first.")
            return
        if self.current_image is None:
            messagebox.showerror("Error", "Load an image first.")
            return

        st = self._grab_state()
        try:
            rgb = st.channels.startswith("RGB")

            if rgb:
                mean_txt = f"{self.mean_r.get()}, {self.mean_g.get()}, {self.mean_b.get()}"
                std_txt = f"{self.std_r.get()}, {self.std_g.get()}, {self.std_b.get()}"
            else:
                mean_txt = self.gray_mean.get()
                std_txt = self.gray_std.get()

            x = preprocess_pil(self.current_image, st.input_size, rgb, mean_txt, std_txt, st.centercrop)
            (p_ok, p_def), dt = forward_probs(self.model, x, self.device)
            label = decide(p_ok, p_def, st.classes, st.positive, st.threshold)

            self.pred_var.set(label.upper())
            self.conf_var.set(f"{max(p_ok, p_def):.3f}")
            self.pok_var.set(f"{p_ok:.3f}")
            self.pdef_var.set(f"{p_def:.3f}")
            self.lat_var.set(f"{dt:.1f}")

            if label.lower().startswith("def"):
                status_text = f"DEFECT (thr={st.threshold:.2f}) — pOK={p_ok:.3f}, pDEF={p_def:.3f}"
                status_color = "#b71c1c"
            else:
                status_text = f"OK (thr={st.threshold:.2f}) — pOK={p_ok:.3f}, pDEF={p_def:.3f}"
                status_color = "#2e7d32"

            if self.true_label_var.get() != "—":
                true_label = self.true_label_var.get().upper()
                match = "MATCH" if true_label == label.upper() else "MISMATCH"
                status_text += f" [GT: {true_label} -> {match}]"

            self._update_status(status_text, color=status_color)

        except Exception as e:
            self._update_status(f"Prediction error: {e}", color="#b71c1c")
            messagebox.showerror("Error", f"An error occurred:\n{e}\n\n{traceback.format_exc()}")

    def _save_screenshot(self):
        try:
            self.update_idletasks()
            x = self.winfo_rootx()
            y = self.winfo_rooty()
            w = x + self.winfo_width()
            h = y + self.winfo_height()
            im = ImageGrab.grab(bbox=(x, y, w, h))

            out = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG", "*.png")],
                title="Save screenshot",
            )
            if out:
                im.save(out, "PNG")
                self._update_status(f"Screenshot saved -> {out}", color="#455a64")
        except Exception as e:
            self._update_status(f"Screenshot error: {e}", color="#b71c1c")
            messagebox.showerror("Error", f"Failed to save screenshot:\n{e}")

    def _update_status(self, text: str, color: str = "#303a46"):
        self.status.configure(text=text, bg=color)

    def _on_close(self):
        save_settings(self._grab_state())
        self.destroy()

    def _parse_list_norm(
        self,
        txt: str,
        channels: int,
        fallback_r: float = 0.5,
        fallback_g: float = 0.5,
        fallback_b: float = 0.5,
    ):
        try:
            parts = [float(p.strip()) for p in str(txt).split(",") if p.strip() != ""]
            if channels == 3:
                if len(parts) == 0:
                    return [fallback_r, fallback_g, fallback_b]
                if len(parts) == 1:
                    return [parts[0]] * 3
                return parts[:3]
            else:
                if len(parts) == 0:
                    return [fallback_r]
                return [parts[0]]
        except Exception:
            if channels == 3:
                return [fallback_r, fallback_g, fallback_b]
            return [fallback_r]

    def _set_norm_from_state(self, st: UIState):
        rgb = (st.channels or "RGB (3ch)").startswith("RGB")

        m_vals = self._parse_list_norm(
            st.mean, 3 if rgb else 1,
            IMAGENET_MEAN[0], IMAGENET_MEAN[1], IMAGENET_MEAN[2]
        )
        s_vals = self._parse_list_norm(
            st.std, 3 if rgb else 1,
            IMAGENET_STD[0], IMAGENET_STD[1], IMAGENET_STD[2]
        )

        if rgb:
            self.mean_r.set(f"{m_vals[0]:.3f}")
            self.mean_g.set(f"{m_vals[1]:.3f}")
            self.mean_b.set(f"{m_vals[2]:.3f}")
            self.std_r.set(f"{s_vals[0]:.3f}")
            self.std_g.set(f"{s_vals[1]:.3f}")
            self.std_b.set(f"{s_vals[2]:.3f}")
            self.gray_mean.set("0.5")
            self.gray_std.set("0.5")
        else:
            self.gray_mean.set(f"{m_vals[0]:.3f}")
            self.gray_std.set(f"{s_vals[0]:.3f}")
            self.mean_r.set(f"{IMAGENET_MEAN[0]:.3f}")
            self.mean_g.set(f"{IMAGENET_MEAN[1]:.3f}")
            self.mean_b.set(f"{IMAGENET_MEAN[2]:.3f}")
            self.std_r.set(f"{IMAGENET_STD[0]:.3f}")
            self.std_g.set(f"{IMAGENET_STD[1]:.3f}")
            self.std_b.set(f"{IMAGENET_STD[2]:.3f}")


# =========================
# Entrypoint
# =========================

def main():
    app = App()
    app.minsize(1000, 700)
    app.mainloop()


if __name__ == "__main__":
    main()
