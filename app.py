#!/usr/bin/env python3
"""
Neural Network Fine-Tuning Studio v3.1
PySide6 · Numba · CuPy 12.1+ · PyTorch (AMP Enabled)

Changelog v3.1:
  - FIX: PyTorch model no longer applies Softmax before CrossEntropyLoss
  - FIX: Numba network now stores actual gradients for visualization
  - FIX: Complete main() function (was truncated)
  - FIX: Thread cleanup on window close
  - FIX: ROC multiclass one-vs-rest
  - FEAT: Learning rate schedule plot (dual-axis)
  - FEAT: Gradient norms history plot
  - FEAT: Best-model checkpoint & restore on early stop
  - FEAT: Export training history to CSV
  - FEAT: Batch progress logged to console
  - FEAT: Normalized confusion matrix option
  - QUALITY: Readable formatting, type hints, docstrings
"""

import sys
import os
import time
import json
import csv
import numpy as np
from pathlib import Path
from collections import OrderedDict

# ── PySide6 ──────────────────────────────────────────────────
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QLabel, QLineEdit, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QProgressBar,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QCheckBox, QStatusBar, QFrame, QGridLayout,
    QSizePolicy, QMessageBox, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QColor, QAction

# ── Matplotlib ───────────────────────────────────────────────
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ── Numba ────────────────────────────────────────────────────
from numba import jit, prange, cuda, float32, float64
import numba as nb

# ── CuPy ─────────────────────────────────────────────────────
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = np

# ── PyTorch ──────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as torch_optim
    from torch.utils.data import TensorDataset, DataLoader
    from torch.cuda.amp import autocast, GradScaler
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# ── Sklearn ──────────────────────────────────────────────────
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.decomposition import PCA


# ╔════════════════════════════════════════════════════════════╗
# ║  NUMBA JIT-COMPILED ACCELERATION FUNCTIONS                ║
# ╚════════════════════════════════════════════════════════════╝

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def numba_clip_gradients(grads: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip gradient vector by global L2 norm (in-place)."""
    norm_sq = 0.0
    for i in prange(grads.shape[0]):
        norm_sq += grads[i] * grads[i]
    norm = np.sqrt(norm_sq)
    if norm > max_norm:
        scale = max_norm / norm
        for i in prange(grads.shape[0]):
            grads[i] *= scale
    return grads


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def numba_compute_accuracy(preds: np.ndarray, labels: np.ndarray, n: int) -> float:
    """Compute classification accuracy from predicted & true label arrays."""
    correct = 0
    for i in prange(n):
        if preds[i] == labels[i]:
            correct += 1
    return float(correct) / float(n)


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def numba_standardize(X: np.ndarray, means: np.ndarray, stds: np.ndarray, out: np.ndarray):
    """Standardize columns of X in-place → out (zero mean, unit variance)."""
    for i in prange(X.shape[0]):
        for j in range(X.shape[1]):
            out[i, j] = (X[i, j] - means[j]) / (stds[j] + 1e-8)


if cuda.is_available():
    @cuda.jit
    def cuda_elementwise_mul(a, b, c, n):
        idx = cuda.grid(1)
        if idx < n:
            c[idx] = a[idx] * b[idx]


# ╔════════════════════════════════════════════════════════════╗
# ║  CUPY-POWERED NEURAL NETWORK                              ║
# ╚════════════════════════════════════════════════════════════╝

class Layer:
    """Single dense layer with activation, Adam optimizer state, and gradient storage."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", use_gpu: bool = True):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.frozen = False

        xp = cp if self.use_gpu else np
        # He init for ReLU, Xavier for others
        std = np.sqrt(2.0 / in_dim) if activation == "relu" else np.sqrt(1.0 / in_dim)
        self.W = xp.random.randn(in_dim, out_dim).astype(np.float32) * std
        self.b = xp.zeros((1, out_dim), dtype=np.float32)

        # Adam optimizer moments
        self.mW = xp.zeros_like(self.W)
        self.vW = xp.zeros_like(self.W)
        self.mb = xp.zeros_like(self.b)
        self.vb = xp.zeros_like(self.b)

        # Forward pass cache
        self.input = None
        self.z = None
        self.a = None

        # ── NEW: Gradient storage for visualization ──
        self.dW = xp.zeros_like(self.W)
        self.db = xp.zeros_like(self.b)

    def _act(self, z):
        xp = cp if self.use_gpu else np
        if self.activation == "relu":
            return xp.maximum(0, z)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + xp.exp(-xp.clip(z, -500, 500)))
        if self.activation == "tanh":
            return xp.tanh(z)
        if self.activation == "softmax":
            e = xp.exp(z - xp.max(z, axis=1, keepdims=True))
            return e / xp.sum(e, axis=1, keepdims=True)
        return z  # linear

    def _act_deriv(self, z, a):
        if self.activation == "relu":
            return (z > 0).astype(np.float32)
        if self.activation == "sigmoid":
            return a * (1.0 - a)
        if self.activation == "tanh":
            return 1.0 - a * a
        return cp.ones_like(z) if self.use_gpu else np.ones_like(z)

    def forward(self, x):
        xp = cp if self.use_gpu else np
        self.input = x
        self.z = xp.dot(x, self.W) + self.b
        self.a = self._act(self.z)
        return self.a

    def update(self, dW, db, lr, optimizer, t, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0):
        """Apply gradient update with optional Adam/AdamW + gradient clipping."""
        if self.frozen:
            return
        xp = cp if self.use_gpu else np

        # ── NEW: Store gradients BEFORE clipping for visualization ──
        self.dW = dW.copy()
        self.db = db.copy()

        # L2 regularization (for non-AdamW)
        if wd > 0 and optimizer != "adamw":
            dW = dW + wd * self.W

        # Gradient clipping via Numba
        flat = dW.flatten()
        arr = cp.asnumpy(flat) if self.use_gpu else flat
        arr = numba_clip_gradients(arr, 5.0)
        dW = (cp.asarray(arr) if self.use_gpu else arr).reshape(dW.shape)

        if optimizer in ("adam", "adamw"):
            # Adam moment updates
            self.mW = beta1 * self.mW + (1 - beta1) * dW
            self.vW = beta2 * self.vW + (1 - beta2) * dW ** 2
            mW_hat = self.mW / (1 - beta1 ** t)
            vW_hat = self.vW / (1 - beta2 ** t)
            uW = lr * mW_hat / (xp.sqrt(vW_hat) + eps)
            if optimizer == "adamw":
                uW = uW + lr * wd * self.W  # decoupled weight decay

            self.mb = beta1 * self.mb + (1 - beta1) * db
            self.vb = beta2 * self.vb + (1 - beta2) * db ** 2
            mb_hat = self.mb / (1 - beta1 ** t)
            vb_hat = self.vb / (1 - beta2 ** t)
            ub = lr * mb_hat / (xp.sqrt(vb_hat) + eps)
            if optimizer == "adamw":
                ub = ub + lr * wd * self.b

            self.W -= uW
            self.b -= ub
        else:
            # Vanilla SGD
            self.W -= lr * dW
            self.b -= lr * db

    def get_np(self):
        """Return (weights, bias) as numpy arrays."""
        return (
            cp.asnumpy(self.W) if self.use_gpu else self.W.copy(),
            cp.asnumpy(self.b) if self.use_gpu else self.b.copy(),
        )

    def set_np(self, w, b):
        """Set weights & bias from numpy arrays."""
        self.W = cp.asarray(w) if self.use_gpu else w.copy()
        self.b = cp.asarray(b) if self.use_gpu else b.copy()

    def get_grad_norm(self) -> float:
        """Return L2 norm of stored weight gradients (for visualization)."""
        xp = cp if self.use_gpu else np
        g = self.dW
        return float(xp.sqrt(xp.sum(g * g)))


class NeuralNetworkNumba:
    """Custom neural network using CuPy (GPU) or NumPy (CPU) with Numba-accelerated ops."""

    def __init__(self, use_gpu: bool = True):
        self.layers: list[Layer] = []
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.xp = cp if self.use_gpu else np

    def add_layer(self, in_dim: int, out_dim: int, activation: str = "relu"):
        self.layers.append(Layer(in_dim, out_dim, activation, self.use_gpu))

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def loss(self, pred, y, kind: str = "cross_entropy") -> float:
        xp = self.xp
        m = pred.shape[0]
        eps = 1e-12
        if kind == "cross_entropy":
            clipped = xp.clip(pred, eps, 1 - eps)
            if y.ndim == 1:
                ll = -xp.log(clipped[xp.arange(m), y.astype(int)])
            else:
                ll = -xp.sum(y * xp.log(clipped), axis=1)
            return float(xp.mean(ll))
        return float(xp.mean((pred - y) ** 2))

    def backward(self, pred, y, lr, optimizer: str = "adam", t: int = 1, wd: float = 0.0, loss: str = "cross_entropy"):
        """Backpropagation through all layers."""
        xp = self.xp
        m = pred.shape[0]
        delta = pred.copy()

        if loss == "cross_entropy":
            if y.ndim == 1:
                delta[xp.arange(m), y.astype(int)] -= 1.0
            else:
                delta -= y
            delta /= m
        else:
            delta = 2.0 * (pred - y) / m

        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            dW = xp.dot(layer.input.T, delta)
            db = xp.sum(delta, axis=0, keepdims=True)
            layer.update(dW, db, lr, optimizer, t, wd=wd)
            if i > 0:
                prev = self.layers[i - 1]
                delta = xp.dot(delta, layer.W.T) * prev._act_deriv(prev.z, prev.a)

    def accuracy(self, pred, y) -> float:
        p = cp.asnumpy(pred) if self.use_gpu else pred
        t = cp.asnumpy(y) if (self.use_gpu and isinstance(y, cp.ndarray)) else y
        pred_labels = np.argmax(p, axis=1)
        true_labels = t.astype(int) if t.ndim == 1 else np.argmax(t, axis=1)
        return numba_compute_accuracy(pred_labels, true_labels, len(pred_labels))

    def get_gradient_norms(self) -> list[float]:
        """Return per-layer gradient L2 norms (from stored dW)."""
        norms = []
        for l in self.layers:
            if not l.frozen:
                norms.append(l.get_grad_norm())
            else:
                norms.append(0.0)
        return norms

    def freeze_up_to(self, idx: int):
        for i, l in enumerate(self.layers):
            l.frozen = i < idx

    def unfreeze_all(self):
        for l in self.layers:
            l.frozen = False

    def get_state(self) -> list[dict]:
        """Serialize all layer weights/biases for checkpointing."""
        state = []
        for l in self.layers:
            w, b = l.get_np()
            state.append(dict(W=w.tolist(), b=b.tolist()))
        return state

    def set_state(self, state: list[dict]):
        """Restore from checkpoint state."""
        for l, s in zip(self.layers, state):
            l.set_np(np.array(s["W"], np.float32), np.array(s["b"], np.float32))

    @property
    def total_params(self) -> int:
        return sum(l.in_dim * l.out_dim + l.out_dim for l in self.layers)

    @property
    def trainable_params(self) -> int:
        return sum(l.in_dim * l.out_dim + l.out_dim for l in self.layers if not l.frozen)

    def save(self, path: str):
        d = dict(layers=[
            dict(in_dim=l.in_dim, out_dim=l.out_dim, act=l.activation, frozen=l.frozen,
                 W=l.get_np()[0].tolist(), b=l.get_np()[1].tolist())
            for l in self.layers
        ])
        with open(path, "w") as f:
            json.dump(d, f)

    def load(self, path: str):
        with open(path) as f:
            d = json.load(f)
        self.layers = []
        for ld in d["layers"]:
            l = Layer(ld["in_dim"], ld["out_dim"], ld["act"], self.use_gpu)
            l.set_np(np.array(ld["W"], np.float32), np.array(ld["b"], np.float32))
            l.frozen = ld.get("frozen", False)
            self.layers.append(l)


# ╔════════════════════════════════════════════════════════════╗
# ║  PYTORCH NEURAL NETWORK                                   ║
# ╚════════════════════════════════════════════════════════════╝

class PyTorchNet(nn.Module):
    """
    PyTorch feed-forward network.

    IMPORTANT: Does NOT include Softmax in forward().
    CrossEntropyLoss expects raw logits. Softmax is applied
    only at inference when probabilities are needed.
    """

    def __init__(self, architecture, n_features, n_classes):
        super().__init__()
        layers = []
        in_d = n_features
        for out_d, act in architecture:
            layers.append(nn.Linear(in_d, out_d))
            if act == "relu":
                layers.append(nn.ReLU())
            elif act == "sigmoid":
                layers.append(nn.Sigmoid())
            elif act == "tanh":
                layers.append(nn.Tanh())
            # NOTE: Softmax is NOT added here — CrossEntropyLoss handles it
            in_d = out_d
        if in_d != n_classes:
            layers.append(nn.Linear(in_d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # Returns raw logits

    def predict_proba(self, x):
        """Apply softmax to get probabilities (for ROC, etc.)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def get_gradient_norms(self) -> list[float]:
        norms = []
        for l in self.net:
            if isinstance(l, nn.Linear) and l.weight.requires_grad:
                if l.weight.grad is not None:
                    norms.append(l.weight.grad.norm().item())
                else:
                    norms.append(0.0)
        return norms


# ╔════════════════════════════════════════════════════════════╗
# ║  DATASETS & STATS                                         ║
# ╚════════════════════════════════════════════════════════════╝

class DatasetManager:
    _LOADERS = {
        "Iris": load_iris,
        "Wine": load_wine,
        "Breast Cancer": load_breast_cancer,
        "Digits (8×8)": load_digits,
    }

    @staticmethod
    def load(name: str, test_size: float = 0.2, val_size: float = 0.1, seed: int = 42) -> dict:
        data = DatasetManager._LOADERS[name]()
        X, y = data.data.astype(np.float32), data.target.astype(np.int32)

        # Standardize using Numba-accelerated function
        means, stds = np.mean(X, axis=0), np.std(X, axis=0)
        X_std = np.empty_like(X)
        numba_standardize(X, means, stds, X_std)
        X = X_std

        X_tv, X_te, y_tv, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        rel = val_size / (1 - test_size)
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_tv, y_tv, test_size=rel, random_state=seed, stratify=y_tv
        )

        return dict(
            X_train=X_tr, y_train=y_tr,
            X_val=X_va, y_val=y_va,
            X_test=X_te, y_test=y_te,
            n_features=X.shape[1],
            n_classes=len(np.unique(y)),
            feature_names=list(data.feature_names) if hasattr(data, 'feature_names') else [],
            target_names=list(data.target_names) if hasattr(data, 'target_names') else [],
        )


# ╔════════════════════════════════════════════════════════════╗
# ║  TRAINING WORKER (QThread)                                ║
# ╚════════════════════════════════════════════════════════════╝

class TrainingWorker(QThread):
    epoch_done = Signal(int, float, float, float, float, float, float, float, list)
    batch_done = Signal(int, int, float)
    finished_ = Signal(dict)
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, model, ds, cfg, backend: str = "numba"):
        super().__init__()
        self.model = model
        self.ds = ds
        self.cfg = cfg
        self.backend = backend
        self._run = True
        self._pause = False

    def stop(self):
        self._run = False

    def pause(self):
        self._pause = True

    def resume(self):
        self._pause = False

    def run(self):
        try:
            epochs = self.cfg["epochs"]
            bs = self.cfg["batch_size"]
            lr0 = self.cfg["lr"]
            opt = self.cfg["optimizer"]
            wd = self.cfg["weight_decay"]
            sched = self.cfg.get("lr_sched", "constant")
            decay = self.cfg.get("lr_decay", 0.95)
            warmup = self.cfg.get("warmup", 0)
            patience = self.cfg.get("patience", 0)
            use_amp = self.cfg.get("use_amp", False)
            hist = dict(
                train_loss=[], val_loss=[], train_acc=[], val_acc=[],
                lr=[], grad_norms=[]
            )

            if self.backend == "pytorch":
                self._run_pytorch(epochs, bs, lr0, opt, wd, sched, decay, warmup, patience, use_amp, hist)
            else:
                self._run_numba(epochs, bs, lr0, opt, wd, sched, decay, warmup, patience, hist)
        except Exception as exc:
            self.error.emit(str(exc))

    def _check_state(self) -> bool:
        if not self._run:
            return False
        while self._pause:
            time.sleep(0.05)
            if not self._run:
                return False
        return True

    def _get_lr(self, lr0: float, ep: int, warmup: int, sched: str, decay: float, epochs: int) -> float:
        if warmup > 0 and ep <= warmup:
            return lr0 * ep / warmup
        if sched == "step":
            return lr0 * decay ** (ep // 10)
        if sched == "exponential":
            return lr0 * decay ** ep
        if sched == "cosine":
            return lr0 * 0.5 * (1 + np.cos(np.pi * ep / epochs))
        return lr0

    # ── Numba / CuPy training loop ───────────────────────────
    def _run_numba(self, epochs, bs, lr0, opt, wd, sched, decay, warmup, patience, hist):
        xp = self.model.xp
        X_tr = xp.asarray(self.ds["X_train"])
        y_tr = xp.asarray(self.ds["y_train"])
        X_va = xp.asarray(self.ds["X_val"])
        y_va = xp.asarray(self.ds["y_val"])
        n = X_tr.shape[0]
        nb_ = max(1, (n + bs - 1) // bs)
        best_vl = float('inf')
        wait = 0
        t = 0
        best_state = None  # ── NEW: checkpoint for early stopping

        for ep in range(1, epochs + 1):
            if not self._check_state():
                return
            start = time.time()
            lr = self._get_lr(lr0, ep, warmup, sched, decay, epochs)
            perm = xp.random.permutation(n)
            Xs, ys = X_tr[perm], y_tr[perm]

            for bi in range(nb_):
                if not self._check_state():
                    return
                s = bi * bs
                e = min(bi * bs + bs, n)
                t += 1
                pred = self.model.forward(Xs[s:e])
                bloss = self.model.loss(pred, ys[s:e])
                self.model.backward(pred, ys[s:e], lr, opt, t, wd=wd)
                self.batch_done.emit(ep, bi + 1, bloss)
                self.progress.emit(int(((ep - 1) * nb_ + bi + 1) / (epochs * nb_) * 100))

            dt = time.time() - start
            ptr = self.model.forward(X_tr)
            pva = self.model.forward(X_va)
            tl = self.model.loss(ptr, y_tr)
            vl = self.model.loss(pva, y_va)
            ta = self.model.accuracy(ptr, y_tr)
            va = self.model.accuracy(pva, y_va)
            g_norms = self.model.get_gradient_norms()

            hist["train_loss"].append(tl)
            hist["val_loss"].append(vl)
            hist["train_acc"].append(ta)
            hist["val_acc"].append(va)
            hist["lr"].append(lr)
            hist["grad_norms"].append(g_norms)
            self.epoch_done.emit(ep, tl, vl, ta, va, lr, dt, n / dt, g_norms)

            # Early stopping with best-model checkpoint
            if patience > 0:
                if vl < best_vl:
                    best_vl = vl
                    wait = 0
                    best_state = self.model.get_state()  # ── NEW: save best
                else:
                    wait += 1
                if wait >= patience:
                    if best_state is not None:
                        self.model.set_state(best_state)  # ── NEW: restore best
                    break

        self._finalize(hist)

    # ── PyTorch training loop ────────────────────────────────
    def _run_pytorch(self, epochs, bs, lr0, opt, wd, sched, decay, warmup, patience, use_amp, hist):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        X_tr = torch.tensor(self.ds["X_train"])
        y_tr = torch.tensor(self.ds["y_train"]).long()
        X_va = torch.tensor(self.ds["X_val"])
        y_va = torch.tensor(self.ds["y_val"]).long()
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True)

        # Optimizer selection
        if opt in ("adam", "adamw"):
            optimizer = torch_optim.Adam(self.model.parameters(), lr=lr0, weight_decay=wd)
        else:
            optimizer = torch_optim.SGD(self.model.parameters(), lr=lr0, weight_decay=wd)

        # CrossEntropyLoss expects raw logits (no softmax in model)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler(enabled=use_amp and torch.cuda.is_available())
        best_vl = float('inf')
        wait = 0
        n = len(X_tr)
        best_state = None  # ── NEW: checkpoint

        for ep in range(1, epochs + 1):
            if not self._check_state():
                return
            start = time.time()
            lr = self._get_lr(lr0, ep, warmup, sched, decay, epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            self.model.train()

            for bi, (xb, yb) in enumerate(train_loader):
                if not self._check_state():
                    return
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                with autocast(enabled=use_amp and torch.cuda.is_available()):
                    out = self.model(xb)
                    loss = criterion(out, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                self.batch_done.emit(ep, bi + 1, loss.item())

            dt = time.time() - start
            self.model.eval()

            with torch.no_grad():
                out_tr = self.model(X_tr.to(device))
                tl = criterion(out_tr, y_tr.to(device)).item()
                ta = (out_tr.argmax(1) == y_tr.to(device)).float().mean().item()

                out_va = self.model(X_va.to(device))
                vl = criterion(out_va, y_va.to(device)).item()
                va = (out_va.argmax(1) == y_va.to(device)).float().mean().item()

            g_norms = self.model.get_gradient_norms()
            hist["train_loss"].append(tl)
            hist["val_loss"].append(vl)
            hist["train_acc"].append(ta)
            hist["val_acc"].append(va)
            hist["lr"].append(lr)
            hist["grad_norms"].append(g_norms)
            self.epoch_done.emit(ep, tl, vl, ta, va, lr, dt, n / dt, g_norms)
            self.progress.emit(int(ep / epochs * 100))

            # Early stopping with best-model checkpoint
            if patience > 0:
                if vl < best_vl:
                    best_vl = vl
                    wait = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    wait += 1
                if wait >= patience:
                    if best_state is not None:
                        self.model.load_state_dict(best_state)
                    break

        self._finalize(hist)

    def _finalize(self, hist: dict):
        if not self._run:
            return
        if self.backend == "pytorch":
            self.model.eval()
            device = next(self.model.parameters()).device
            X_te = torch.tensor(self.ds["X_test"]).to(device)
            y_te = torch.tensor(self.ds["y_test"]).long().to(device)
            with torch.no_grad():
                pte = self.model(X_te)
                hist["test_loss"] = nn.CrossEntropyLoss()(pte, y_te).item()
                hist["test_acc"] = (pte.argmax(1) == y_te).float().mean().item()
                hist["test_probs"] = torch.softmax(pte, dim=1).cpu().numpy()
        else:
            xp = self.model.xp
            X_te = xp.asarray(self.ds["X_test"])
            y_te = xp.asarray(self.ds["y_test"])
            pte = self.model.forward(X_te)
            hist["test_loss"] = self.model.loss(pte, y_te)
            hist["test_acc"] = self.model.accuracy(pte, y_te)
            probs = cp.asnumpy(pte) if self.model.use_gpu else pte
            hist["test_probs"] = probs
        self.finished_.emit(hist)


# ╔════════════════════════════════════════════════════════════╗
# ║  MATPLOTLIB CANVAS WIDGETS                                ║
# ╚════════════════════════════════════════════════════════════╝

BG = "#1e1e2e"
FG = "#cdd6f4"
GRID = "#45475a"


class _Canvas(FigureCanvas):
    """Base canvas with dark theme styling."""

    def __init__(self):
        self.fig = Figure(figsize=(5, 3.5), dpi=100, facecolor=BG)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._style()

    def _style(self):
        self.ax.set_facecolor(BG)
        self.ax.tick_params(colors=FG, labelsize=8)
        for sp in self.ax.spines.values():
            sp.set_color(GRID)
        self.ax.grid(True, alpha=0.15, color=FG)


class TrainCanvas(FigureCanvas):
    """Training curves with dual-axis for LR schedule."""

    def __init__(self):
        self.fig = Figure(figsize=(6, 4), dpi=100, facecolor=BG)
        super().__init__(self.fig)
        self.ax_loss = self.fig.add_subplot(111)
        self.ax_lr = self.ax_loss.twinx()  # ── NEW: second y-axis for LR
        self._style()

    def _style(self):
        for ax in [self.ax_loss, self.ax_lr]:
            ax.set_facecolor(BG)
            ax.tick_params(colors=FG, labelsize=8)
            for sp in ax.spines.values():
                sp.set_color(GRID)
        self.ax_loss.grid(True, alpha=0.15, color=FG)

    def update_plot(self, tl, vl, ta, va, lr_list=None):
        self.ax_loss.clear()
        self.ax_lr.clear()
        self._style()

        self.ax_loss.set_xlabel("Epoch", color=FG, fontsize=9)
        self.ax_loss.set_ylabel("Loss / Accuracy", color=FG, fontsize=9)
        self.ax_loss.set_title("Training Curves", color=FG, fontsize=10, weight="bold")

        if tl:
            self.ax_loss.plot(tl, color="#f38ba8", lw=1.4, label="Train Loss")
        if vl:
            self.ax_loss.plot(vl, color="#89b4fa", lw=1.4, label="Val Loss")
        if ta:
            self.ax_loss.plot(ta, color="#fab387", lw=1.4, ls="--", label="Train Acc")
        if va:
            self.ax_loss.plot(va, color="#74c7ec", lw=1.4, ls="--", label="Val Acc")

        # ── NEW: LR schedule on right axis ──
        if lr_list and len(lr_list) > 0:
            self.ax_lr.plot(lr_list, color="#a6e3a1", lw=1.2, ls=":", label="LR")
            self.ax_lr.set_ylabel("Learning Rate", color="#a6e3a1", fontsize=9)
            self.ax_lr.tick_params(axis='y', colors="#a6e3a1", labelsize=7)

        # Merge legends from both axes
        lines1, labels1 = self.ax_loss.get_legend_handles_labels()
        lines2, labels2 = self.ax_lr.get_legend_handles_labels()
        if lines1 or lines2:
            self.ax_loss.legend(
                lines1 + lines2, labels1 + labels2,
                facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=7, loc="upper right"
            )

        self.fig.tight_layout()
        self.draw()


class VizCanvas(_Canvas):
    """Multi-purpose visualization canvas."""

    def update_weights(self, model, backend):
        self.ax.clear()
        self._style()
        self.ax.set_title("Weight Distribution", color=FG, fontsize=10, weight="bold")
        if not model:
            return
        if backend == "pytorch":
            for i, p in enumerate(model.parameters()):
                if p.dim() >= 2:
                    self.ax.hist(p.detach().cpu().numpy().flatten(), bins=50, alpha=0.6, label=f"L{i}")
        else:
            for i, l in enumerate(model.layers):
                w, _ = l.get_np()
                self.ax.hist(w.flatten(), bins=50, alpha=0.6, label=f"L{i}")
        self.ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
        self.fig.tight_layout()
        self.draw()

    def update_roc(self, y_true, y_prob, n_classes):
        self.ax.clear()
        self._style()
        self.ax.set_title("ROC Curve", color=FG, fontsize=10, weight="bold")

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            self.ax.plot(fpr, tpr, color="#a6e3a1", lw=2, label=f'AUC={roc_auc:.2f}')
        else:
            for i in range(n_classes):
                # ── FIX: proper one-vs-rest binary labels ──
                y_binary = (y_true == i).astype(int)
                fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                self.ax.plot(fpr, tpr, lw=1.5, label=f'C{i} (AUC={roc_auc:.2f})')

        self.ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls='--')
        self.ax.set_xlabel("FPR", color=FG)
        self.ax.set_ylabel("TPR", color=FG)
        self.ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=7)
        self.fig.tight_layout()
        self.draw()

    def update_grads(self, g_norms_history):
        """Plot gradient norms across layers for the last epoch AND over time."""
        self.ax.clear()
        self._style()

        if not g_norms_history:
            return

        # Bar chart of latest epoch gradient norms
        latest = g_norms_history[-1]
        x = range(len(latest))
        self.ax.bar(x, latest, color="#f9e2af", alpha=0.8)
        self.ax.set_title("Gradient L2 Norms per Layer (latest epoch)", color=FG, fontsize=10, weight="bold")
        self.ax.set_xlabel("Layer Index", color=FG)
        self.ax.set_ylabel("Grad L2 Norm", color=FG)
        self.fig.tight_layout()
        self.draw()

    def update_grads_history(self, g_norms_history):
        """Plot gradient norm evolution over epochs for each layer."""
        self.ax.clear()
        self._style()
        self.ax.set_title("Gradient Norms Over Time", color=FG, fontsize=10, weight="bold")

        if not g_norms_history:
            return

        n_layers = len(g_norms_history[0])
        colors = ["#f38ba8", "#89b4fa", "#a6e3a1", "#fab387", "#f9e2af", "#74c7ec", "#cba6f7", "#94e2d5"]
        for li in range(n_layers):
            vals = [epoch_norms[li] for epoch_norms in g_norms_history if li < len(epoch_norms)]
            c = colors[li % len(colors)]
            self.ax.plot(vals, color=c, lw=1.2, label=f"Layer {li}")

        self.ax.set_xlabel("Epoch", color=FG)
        self.ax.set_ylabel("Grad L2 Norm", color=FG)
        self.ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=7)
        self.fig.tight_layout()
        self.draw()

    def update_decision(self, X, y):
        self.ax.clear()
        self._style()
        self.ax.set_title("Data PCA (2D)", color=FG, fontsize=10, weight="bold")
        X_2d = PCA(n_components=2).fit_transform(X) if X.shape[1] > 2 else X
        scatter = self.ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="viridis", edgecolors='w', s=40, alpha=0.8)
        self.ax.set_xlabel("PC 1", color=FG)
        self.ax.set_ylabel("PC 2", color=FG)
        self.fig.tight_layout()
        self.draw()


# ╔════════════════════════════════════════════════════════════╗
# ║  STYLED HELPERS                                           ║
# ╚════════════════════════════════════════════════════════════╝

def _btn(text: str, color: str = "#89b4fa") -> QPushButton:
    b = QPushButton(text)
    b.setStyleSheet(f"""
        QPushButton {{
            background:{color}; color:#1e1e2e; border:none; border-radius:6px;
            padding:8px 18px; font-weight:600; font-size:13px;
        }}
        QPushButton:hover {{background:{color}cc;}}
        QPushButton:disabled {{background:#585b70; color:#6c7086;}}
    """)
    return b


def _group(title: str) -> QGroupBox:
    g = QGroupBox(title)
    g.setStyleSheet(f"""
        QGroupBox {{
            color:{FG}; border:1px solid {GRID}; border-radius:8px;
            margin-top:12px; padding-top:16px; font-weight:600; font-size:12px;
        }}
        QGroupBox::title {{
            subcontrol-origin:margin; left:12px; padding:0 6px;
        }}
    """)
    return g


# ╔════════════════════════════════════════════════════════════╗
# ║  MAIN WINDOW                                              ║
# ╚════════════════════════════════════════════════════════════╝

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "⚡ Neural Fine-Tuning Studio v3.1  —  PySide6 · Numba · CuPy · PyTorch"
        )
        self.resize(1360, 860)
        self._apply_global_style()

        # State
        self.model_numba = None
        self.model_pytorch = None
        self.dataset = None
        self.worker = None
        self.active_backend = "numba"
        self.hist = dict(
            train_loss=[], val_loss=[], train_acc=[], val_acc=[],
            lr=[], grad_norms=[]
        )

        # Build UI
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self._build_data_tab()
        self._build_model_tab()
        self._build_train_tab()
        self._build_viz_tab()
        self._build_eval_tab()
        self._build_console_tab()

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

    # ── NEW: Clean up worker thread on close ──
    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        event.accept()

    def _apply_global_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{background:{BG}; color:{FG};}}
            QTabWidget::pane {{border:1px solid {GRID}; border-radius:6px;}}
            QTabBar::tab {{
                background:#313244; color:{FG}; padding:10px 22px;
                border-top-left-radius:8px; border-top-right-radius:8px;
                margin-right:2px; font-weight:600; font-size:12px;
            }}
            QTabBar::tab:selected {{background:{BG}; border-bottom:2px solid #89b4fa;}}
            QLabel {{color:{FG}; font-size:12px;}}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background:#313244; color:{FG}; border:1px solid {GRID};
                border-radius:5px; padding:5px 8px; font-size:12px;
            }}
            QComboBox::drop-down {{border:none;}}
            QComboBox QAbstractItemView {{
                background:#313244; color:{FG}; selection-background-color:#89b4fa;
            }}
            QTextEdit {{
                background:#181825; color:#a6e3a1; border:1px solid {GRID};
                border-radius:5px; font-family:'Consolas','Courier New',monospace;
                font-size:11px;
            }}
            QProgressBar {{
                background:#313244; border:none; border-radius:4px;
                text-align:center; color:{FG}; font-size:11px; height:18px;
            }}
            QProgressBar::chunk {{background:#89b4fa; border-radius:4px;}}
            QTableWidget {{
                background:#181825; color:{FG}; gridline-color:{GRID};
                border:1px solid {GRID}; font-size:11px;
            }}
            QHeaderView::section {{
                background:#313244; color:{FG}; border:1px solid {GRID};
                padding:4px; font-weight:600; font-size:11px;
            }}
            QCheckBox {{color:{FG}; font-size:12px;}}
            QCheckBox::indicator {{
                width:16px; height:16px; border-radius:4px;
                border:1px solid {GRID}; background:#313244;
            }}
            QCheckBox::indicator:checked {{background:#89b4fa;}}
            QScrollArea {{border:none;}}
        """)

    # ══════════════════════════════════════════════════════════
    #  TAB 1 — DATA
    # ══════════════════════════════════════════════════════════
    def _build_data_tab(self):
        page = QWidget()
        lay = QHBoxLayout(page)

        # Left panel
        left = QVBoxLayout()

        g1 = _group("Dataset Source")
        g1l = QVBoxLayout(g1)
        self.ds_combo = QComboBox()
        self.ds_combo.addItems(["Iris", "Wine", "Breast Cancer", "Digits (8×8)"])
        g1l.addWidget(QLabel("Built-in dataset:"))
        g1l.addWidget(self.ds_combo)

        g2 = _group("Split Ratios")
        g2l = QGridLayout(g2)
        self.spin_test = QDoubleSpinBox()
        self.spin_test.setRange(0.05, 0.5)
        self.spin_test.setValue(0.2)
        self.spin_test.setSingleStep(0.05)
        self.spin_val = QDoubleSpinBox()
        self.spin_val.setRange(0.05, 0.5)
        self.spin_val.setValue(0.1)
        self.spin_val.setSingleStep(0.05)
        g2l.addWidget(QLabel("Test split:"), 0, 0)
        g2l.addWidget(self.spin_test, 0, 1)
        g2l.addWidget(QLabel("Val split:"), 1, 0)
        g2l.addWidget(self.spin_val, 1, 1)

        load_btn = _btn("📂  Load Dataset", "#89b4fa")
        load_btn.clicked.connect(self._load_dataset)

        left.addWidget(g1)
        left.addWidget(g2)
        left.addWidget(load_btn)
        left.addStretch()
        lay.addLayout(left, 1)

        # Right panel
        right = QVBoxLayout()
        g3 = _group("Dataset Info")
        g3l = QVBoxLayout(g3)
        self.ds_info = QTextEdit()
        self.ds_info.setReadOnly(True)
        g3l.addWidget(self.ds_info)
        right.addWidget(g3)

        g4 = _group("Preview (first 10 rows)")
        g4l = QVBoxLayout(g4)
        self.ds_table = QTableWidget()
        self.ds_table.setEditTriggers(QTableWidget.NoEditTriggers)
        g4l.addWidget(self.ds_table)
        right.addWidget(g4)
        lay.addLayout(right, 2)

        self.tabs.addTab(page, "📊 Data")

    def _load_dataset(self):
        try:
            self.dataset = DatasetManager.load(
                self.ds_combo.currentText(),
                test_size=self.spin_test.value(),
                val_size=self.spin_val.value(),
            )
            ds = self.dataset
            self.ds_info.setPlainText(
                f"Features: {ds['n_features']}\n"
                f"Classes: {ds['n_classes']}\n"
                f"Train: {len(ds['X_train'])}\n"
                f"Val: {len(ds['X_val'])}\n"
                f"Test: {len(ds['X_test'])}\n"
                f"Targets: {ds['target_names']}"
            )
            X = ds["X_train"]
            cols = ds["feature_names"] or [f"f{i}" for i in range(ds["n_features"])]
            self.ds_table.setColumnCount(len(cols) + 1)
            self.ds_table.setHorizontalHeaderLabels(list(cols) + ["target"])
            self.ds_table.setRowCount(min(10, len(X)))
            for i in range(min(10, len(X))):
                for j in range(ds["n_features"]):
                    self.ds_table.setItem(i, j, QTableWidgetItem(f"{X[i, j]:.3f}"))
                self.ds_table.setItem(i, ds["n_features"], QTableWidgetItem(str(ds["y_train"][i])))
            self.ds_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            self.status.showMessage(
                f"Dataset loaded — {len(ds['X_train'])} train / "
                f"{len(ds['X_val'])} val / {len(ds['X_test'])} test"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ══════════════════════════════════════════════════════════
    #  TAB 2 — MODEL
    # ══════════════════════════════════════════════════════════
    def _build_model_tab(self):
        page = QWidget()
        lay = QHBoxLayout(page)

        # Left panel
        left = QVBoxLayout()

        g0 = _group("Backend Engine")
        g0l = QVBoxLayout(g0)
        self.cb_backend = QComboBox()
        self.cb_backend.addItems(["Numba + CuPy", "PyTorch"])
        if not PYTORCH_AVAILABLE:
            self.cb_backend.model().item(1).setEnabled(False)
        g0l.addWidget(self.cb_backend)

        g1 = _group("Architecture (out_dim,activation)")
        g1l = QVBoxLayout(g1)
        self.arch_text = QTextEdit()
        self.arch_text.setFixedHeight(120)
        self.arch_text.setPlaceholderText("e.g.\n64,relu\n32,relu\n3,softmax")
        g1l.addWidget(self.arch_text)

        auto_btn = _btn("✨ Auto-Suggest Architecture", "#cba6f7")
        auto_btn.clicked.connect(self._auto_arch)
        g1l.addWidget(auto_btn)

        build_btn = _btn("🔨  Build Model", "#89b4fa")
        build_btn.clicked.connect(self._build_model)
        g1l.addWidget(build_btn)

        g2 = _group("Model I/O & Fine-Tuning")
        g2l = QGridLayout(g2)
        save_btn = _btn("💾 Save", "#a6e3a1")
        load_btn = _btn("📁 Load", "#f9e2af")
        save_btn.clicked.connect(self._save_model)
        load_btn.clicked.connect(self._load_model)
        g2l.addWidget(save_btn, 0, 0)
        g2l.addWidget(load_btn, 0, 1)

        self.freeze_spin = QSpinBox()
        self.freeze_spin.setRange(0, 20)
        self.freeze_spin.setValue(0)
        freeze_btn = _btn("❄ Freeze 0..N-1", "#74c7ec")
        freeze_btn.clicked.connect(self._apply_freeze)
        unfreeze_btn = _btn("🔓 Unfreeze All", "#cba6f7")
        unfreeze_btn.clicked.connect(self._unfreeze_all)
        g2l.addWidget(QLabel("Freeze up to:"), 1, 0)
        g2l.addWidget(self.freeze_spin, 1, 1)
        g2l.addWidget(freeze_btn, 2, 0)
        g2l.addWidget(unfreeze_btn, 2, 1)

        left.addWidget(g0)
        left.addWidget(g1)
        left.addWidget(g2)
        left.addStretch()
        lay.addLayout(left, 1)

        # Right panel
        right = QVBoxLayout()
        g3 = _group("Model Summary")
        g3l = QVBoxLayout(g3)
        self.layer_table = QTableWidget()
        self.layer_table.setColumnCount(7)
        self.layer_table.setHorizontalHeaderLabels(
            ["#", "In", "Out", "Act", "Frozen", "Params", "W std"]
        )
        self.layer_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        g3l.addWidget(self.layer_table)
        right.addWidget(g3)
        self.model_info_label = QLabel("No model built yet.")
        self.model_info_label.setStyleSheet("font-size:13px; font-weight:600; color:#89b4fa;")
        right.addWidget(self.model_info_label)
        lay.addLayout(right, 2)

        self.tabs.addTab(page, "🧠 Model")

    def _auto_arch(self):
        if not self.dataset:
            QMessageBox.warning(self, "Warning", "Load dataset first.")
            return
        nf = self.dataset["n_features"]
        nc = self.dataset["n_classes"]
        h1 = max(32, nf * 2)
        h2 = max(16, nf)
        self.arch_text.setPlainText(f"{h1},relu\n{h2},relu\n{nc},softmax")

    def _build_model(self):
        if not self.dataset:
            QMessageBox.warning(self, "Warning", "Load a dataset first.")
            return
        try:
            lines = self.arch_text.toPlainText().strip().splitlines()
            if not lines:
                return
            self.active_backend = "pytorch" if self.cb_backend.currentText() == "PyTorch" else "numba"
            arch = [(int(l.split(",")[0]), l.split(",")[1].strip()) for l in lines]
            in_dim = self.dataset["n_features"]

            if self.active_backend == "pytorch":
                # NOTE: Softmax NOT in model — CrossEntropyLoss expects logits
                self.model_pytorch = PyTorchNet(arch, in_dim, self.dataset["n_classes"])
                self._refresh_table_torch()
            else:
                self.model_numba = NeuralNetworkNumba(use_gpu=CUDA_AVAILABLE)
                for out_d, act in arch:
                    self.model_numba.add_layer(in_dim, out_d, act)
                    in_dim = out_d
                if self.model_numba.layers[-1].out_dim != self.dataset["n_classes"]:
                    self.model_numba.add_layer(in_dim, self.dataset["n_classes"], "softmax")
                self._refresh_table_numba()
            self.status.showMessage(f"Model built — Backend: {self.active_backend}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _refresh_table_numba(self):
        m = self.model_numba
        self.layer_table.setRowCount(len(m.layers))
        for i, l in enumerate(m.layers):
            w, _ = l.get_np()
            vals = [
                str(i), str(l.in_dim), str(l.out_dim), l.activation,
                "❄" if l.frozen else "—",
                f"{l.in_dim * l.out_dim + l.out_dim:,}",
                f"{np.std(w):.4f}",
            ]
            for j, v in enumerate(vals):
                self.layer_table.setItem(i, j, QTableWidgetItem(v))
        self.model_info_label.setText(
            f"Params: {m.total_params:,} (Trainable: {m.trainable_params:,})  |  "
            f"GPU: {'✅' if m.use_gpu else '❌'}"
        )

    def _refresh_table_torch(self):
        m = self.model_pytorch
        layers = [l for l in m.net if isinstance(l, nn.Linear)]
        self.layer_table.setRowCount(len(layers))
        total_p = sum(p.numel() for p in m.parameters())
        train_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
        for i, l in enumerate(layers):
            vals = [
                str(i), str(l.in_features), str(l.out_features), "—",
                "—" if l.weight.requires_grad else "❄",
                f"{sum(p.numel() for p in l.parameters()):,}",
                f"{l.weight.std().item():.4f}",
            ]
            for j, v in enumerate(vals):
                self.layer_table.setItem(i, j, QTableWidgetItem(v))
        dev = next(m.parameters()).device
        self.model_info_label.setText(
            f"Params: {total_p:,} (Trainable: {train_p:,})  |  Dev: {dev}"
        )

    def _save_model(self):
        if self.active_backend == "pytorch" and self.model_pytorch:
            p, _ = QFileDialog.getSaveFileName(self, "Save PyTorch", "model.pt", "PT (*.pt)")
            if p:
                torch.save(self.model_pytorch.state_dict(), p)
                self.status.showMessage(f"Saved → {p}")
        elif self.active_backend == "numba" and self.model_numba:
            p, _ = QFileDialog.getSaveFileName(self, "Save Numba", "model.json", "JSON (*.json)")
            if p:
                self.model_numba.save(p)
                self.status.showMessage(f"Saved → {p}")

    def _load_model(self):
        if self.active_backend == "pytorch" and self.model_pytorch:
            p, _ = QFileDialog.getOpenFileName(self, "Load PyTorch", "", "PT (*.pt)")
            if p:
                self.model_pytorch.load_state_dict(torch.load(p, weights_only=True))
                self._refresh_table_torch()
                self.status.showMessage(f"Loaded ← {p}")
        elif self.active_backend == "numba" and self.model_numba:
            p, _ = QFileDialog.getOpenFileName(self, "Load Numba", "", "JSON (*.json)")
            if p:
                self.model_numba.load(p)
                self._refresh_table_numba()
                self.status.showMessage(f"Loaded ← {p}")

    def _apply_freeze(self):
        idx = self.freeze_spin.value()
        if self.active_backend == "numba" and self.model_numba:
            self.model_numba.freeze_up_to(idx)
            self._refresh_table_numba()
        elif self.active_backend == "pytorch" and self.model_pytorch:
            linear_layers = [l for l in self.model_pytorch.net if isinstance(l, nn.Linear)]
            for i, l in enumerate(linear_layers[:idx]):
                l.weight.requires_grad = False
                l.bias.requires_grad = False
            self._refresh_table_torch()
        self.status.showMessage(f"Frozen layers 0–{idx - 1}")

    def _unfreeze_all(self):
        if self.active_backend == "numba" and self.model_numba:
            self.model_numba.unfreeze_all()
            self._refresh_table_numba()
        elif self.active_backend == "pytorch" and self.model_pytorch:
            for p in self.model_pytorch.parameters():
                p.requires_grad = True
            self._refresh_table_torch()

    # ══════════════════════════════════════════════════════════
    #  TAB 3 — TRAINING
    # ══════════════════════════════════════════════════════════
    def _build_train_tab(self):
        page = QWidget()
        lay = QHBoxLayout(page)

        # Left panel — controls
        left = QVBoxLayout()

        g1 = _group("Hyperparameters")
        gl = QGridLayout(g1)

        # ── FIX: replaced walrus operators with explicit assignments ──
        self.sp_epochs = QSpinBox()
        self.sp_batch = QSpinBox()
        self.sp_lr = QDoubleSpinBox()
        self.sp_wd = QDoubleSpinBox()
        self.sp_decay = QDoubleSpinBox()
        self.sp_warmup = QSpinBox()
        self.sp_patience = QSpinBox()

        self.sp_epochs.setRange(1, 10000)
        self.sp_epochs.setValue(100)
        self.sp_batch.setRange(1, 4096)
        self.sp_batch.setValue(32)
        self.sp_lr.setRange(1e-6, 10)
        self.sp_lr.setValue(0.001)
        self.sp_lr.setDecimals(6)
        self.sp_lr.setSingleStep(0.0001)
        self.sp_wd.setRange(0, 1)
        self.sp_wd.setValue(0.0)
        self.sp_wd.setDecimals(4)
        self.sp_wd.setSingleStep(0.001)
        self.sp_decay.setRange(0.5, 1.0)
        self.sp_decay.setValue(0.95)
        self.sp_decay.setDecimals(3)
        self.sp_warmup.setRange(0, 100)
        self.sp_patience.setRange(0, 100)
        self.sp_patience.setValue(5)

        r = 0
        for lbl, w in [
            ("Epochs:", self.sp_epochs),
            ("Batch size:", self.sp_batch),
            ("Learning rate:", self.sp_lr),
            ("Weight decay:", self.sp_wd),
            ("LR decay:", self.sp_decay),
            ("Warmup epochs:", self.sp_warmup),
            ("Early Stop Patience:", self.sp_patience),
        ]:
            gl.addWidget(QLabel(lbl), r, 0)
            gl.addWidget(w, r, 1)
            r += 1

        gl.addWidget(QLabel("Optimizer:"), r, 0)
        self.cb_opt = QComboBox()
        self.cb_opt.addItems(["adam", "adamw", "sgd"])
        gl.addWidget(self.cb_opt, r, 1)
        r += 1

        gl.addWidget(QLabel("LR schedule:"), r, 0)
        self.cb_sched = QComboBox()
        self.cb_sched.addItems(["constant", "step", "exponential", "cosine"])
        gl.addWidget(self.cb_sched, r, 1)
        r += 1

        self.cb_amp = QCheckBox("Use Mixed Precision (AMP)")
        if not (PYTORCH_AVAILABLE and torch.cuda.is_available()):
            self.cb_amp.setEnabled(False)
        gl.addWidget(self.cb_amp, r, 0, 1, 2)

        # Control buttons
        btn_row = QHBoxLayout()
        self.btn_train = _btn("▶  Start Training", "#a6e3a1")
        self.btn_pause = _btn("⏸  Pause", "#f9e2af")
        self.btn_stop = _btn("⏹  Stop", "#f38ba8")
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_train.clicked.connect(self._start_training)
        self.btn_pause.clicked.connect(self._pause_training)
        self.btn_stop.clicked.connect(self._stop_training)
        btn_row.addWidget(self.btn_train)
        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(self.btn_stop)

        left.addWidget(g1)
        left.addLayout(btn_row)

        # Stats panel
        g2 = _group("Stats")
        g2l = QVBoxLayout(g2)
        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        self.lbl_epoch = QLabel("Epoch: —")
        self.lbl_loss = QLabel("Loss: —")
        self.lbl_acc = QLabel("Acc: —")
        self.lbl_lr = QLabel("LR: —")
        self.lbl_time = QLabel("Time: —")
        for w in [self.pbar, self.lbl_epoch, self.lbl_loss, self.lbl_acc, self.lbl_lr, self.lbl_time]:
            g2l.addWidget(w)

        # ── NEW: Export button ──
        export_btn = _btn("📤  Export History (CSV)", "#cba6f7")
        export_btn.clicked.connect(self._export_history)
        g2l.addWidget(export_btn)

        left.addWidget(g2)
        left.addStretch()
        lay.addLayout(left, 1)

        # Right panel — training plot
        right = QVBoxLayout()
        self.train_canvas = TrainCanvas()
        right.addWidget(self.train_canvas)
        lay.addLayout(right, 2)

        self.tabs.addTab(page, "🏋 Training")

    def _start_training(self):
        model = self.model_pytorch if self.active_backend == "pytorch" else self.model_numba
        if not model or not self.dataset:
            QMessageBox.warning(self, "Warning", "Need model + dataset.")
            return
        self.hist = dict(
            train_loss=[], val_loss=[], train_acc=[], val_acc=[],
            lr=[], grad_norms=[]
        )
        cfg = dict(
            epochs=self.sp_epochs.value(),
            batch_size=self.sp_batch.value(),
            lr=self.sp_lr.value(),
            optimizer=self.cb_opt.currentText(),
            weight_decay=self.sp_wd.value(),
            lr_sched=self.cb_sched.currentText(),
            lr_decay=self.sp_decay.value(),
            warmup=self.sp_warmup.value(),
            patience=self.sp_patience.value(),
            loss="cross_entropy",
            use_amp=self.cb_amp.isChecked(),
        )
        self.worker = TrainingWorker(model, self.dataset, cfg, self.active_backend)
        self.worker.epoch_done.connect(self._on_epoch)
        self.worker.batch_done.connect(self._on_batch)
        self.worker.finished_.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.progress.connect(self.pbar.setValue)
        self.worker.start()
        self.btn_train.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)

    def _pause_training(self):
        if self.worker:
            if self.btn_pause.text().startswith("⏸"):
                self.worker.pause()
                self.btn_pause.setText("▶  Resume")
            else:
                self.worker.resume()
                self.btn_pause.setText("⏸  Pause")

    def _stop_training(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(3000)
        self._reset_btns()

    def _reset_btns(self):
        self.btn_train.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸  Pause")
        self.btn_stop.setEnabled(False)

    def _on_epoch(self, ep, tl, vl, ta, va, lr, dt, throughput, g_norms):
        self.hist["train_loss"].append(tl)
        self.hist["val_loss"].append(vl)
        self.hist["train_acc"].append(ta)
        self.hist["val_acc"].append(va)
        self.hist["lr"].append(lr)
        self.hist["grad_norms"].append(g_norms)
        self.lbl_epoch.setText(f"Epoch: {ep}")
        self.lbl_loss.setText(f"Train Loss: {tl:.4f}  |  Val Loss: {vl:.4f}")
        self.lbl_acc.setText(f"Train Acc: {ta:.4f}  |  Val Acc: {va:.4f}")
        self.lbl_lr.setText(f"LR: {lr:.6f}")
        self.lbl_time.setText(f"Time: {dt:.3f}s  |  Speed: {throughput:.0f} samp/s")
        # ── FIX: pass lr_list to canvas ──
        self.train_canvas.update_plot(
            self.hist["train_loss"], self.hist["val_loss"],
            self.hist["train_acc"], self.hist["val_acc"],
            lr_list=self.hist["lr"],
        )
        self._log(
            f"[Ep {ep}] loss={tl:.4f}/{vl:.4f}  acc={ta:.4f}/{va:.4f}  "
            f"lr={lr:.6f}  time={dt:.3f}s  speed={throughput:.0f} samp/s"
        )

    def _on_batch(self, ep, bi, bl):
        """Log batch progress every 10 batches."""
        if bi % 10 == 0:
            self._log(f"  [Ep {ep} | Batch {bi}] loss={bl:.4f}")

    def _on_finished(self, h):
        self._reset_btns()
        ta = h.get("test_acc", 0)
        tl = h.get("test_loss", 0)
        self.status.showMessage(f"Done — Test acc: {ta:.4f}  Test loss: {tl:.4f}")
        self._log(f"\n✅ Done! Test Acc: {ta:.4f}  Test Loss: {tl:.4f}")

    def _on_error(self, msg):
        self._reset_btns()
        QMessageBox.critical(self, "Error", msg)

    def _export_history(self):
        """Export training history to CSV."""
        if not self.hist["train_loss"]:
            QMessageBox.warning(self, "Warning", "No training history to export.")
            return
        p, _ = QFileDialog.getSaveFileName(self, "Export History", "history.csv", "CSV (*.csv)")
        if not p:
            return
        with open(p, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr"])
            for i in range(len(self.hist["train_loss"])):
                writer.writerow([
                    i + 1,
                    f"{self.hist['train_loss'][i]:.6f}",
                    f"{self.hist['val_loss'][i]:.6f}",
                    f"{self.hist['train_acc'][i]:.6f}",
                    f"{self.hist['val_acc'][i]:.6f}",
                    f"{self.hist['lr'][i]:.8f}",
                ])
        self.status.showMessage(f"History exported → {p}")
        self._log(f"📤 History exported to {p}")

    # ══════════════════════════════════════════════════════════
    #  TAB 4 — VISUALIZATION
    # ══════════════════════════════════════════════════════════
    def _build_viz_tab(self):
        page = QWidget()
        lay = QVBoxLayout(page)

        ctrl = QHBoxLayout()
        self.viz_combo = QComboBox()
        self.viz_combo.addItems([
            "Weight Histograms",
            "ROC Curve",
            "Gradient Norms (latest)",
            "Gradient Norms (history)",
            "PCA Scatter",
        ])
        gen_btn = _btn("Generate Plot", "#89b4fa")
        gen_btn.clicked.connect(self._gen_viz)
        ctrl.addWidget(QLabel("Select Plot:"))
        ctrl.addWidget(self.viz_combo)
        ctrl.addWidget(gen_btn)
        ctrl.addStretch()
        lay.addLayout(ctrl)

        self.viz_canvas = VizCanvas()
        lay.addWidget(self.viz_canvas)

        self.tabs.addTab(page, "👁 Visualization")

    def _get_preds(self):
        """Get test predictions + probabilities."""
        model = self.model_pytorch if self.active_backend == "pytorch" else self.model_numba
        if not model or not self.dataset:
            return None, None, None
        X_te, y_te = self.dataset["X_test"], self.dataset["y_test"]

        if self.active_backend == "pytorch":
            model.eval()
            device = next(model.parameters()).device
            with torch.no_grad():
                # Use predict_proba for proper softmax
                probs = model.predict_proba(torch.tensor(X_te).to(device)).cpu().numpy()
        else:
            xp = model.xp
            probs = model.forward(xp.asarray(X_te))
            probs = cp.asnumpy(probs) if model.use_gpu else probs

        return probs, y_te, X_te

    def _gen_viz(self):
        idx = self.viz_combo.currentIndex()
        model = self.model_pytorch if self.active_backend == "pytorch" else self.model_numba

        if idx == 0 and model:
            self.viz_canvas.update_weights(model, self.active_backend)
        elif idx == 1:
            probs, y_te, _ = self._get_preds()
            if probs is not None:
                self.viz_canvas.update_roc(y_te, probs, self.dataset["n_classes"])
        elif idx == 2 and self.hist["grad_norms"]:
            self.viz_canvas.update_grads(self.hist["grad_norms"])
        elif idx == 3 and self.hist["grad_norms"]:
            self.viz_canvas.update_grads_history(self.hist["grad_norms"])
        elif idx == 4:
            _, y_te, X_te = self._get_preds()
            if X_te is not None:
                self.viz_canvas.update_decision(X_te, y_te)

    # ══════════════════════════════════════════════════════════
    #  TAB 5 — EVALUATION
    # ══════════════════════════════════════════════════════════
    def _build_eval_tab(self):
        page = QWidget()
        lay = QVBoxLayout(page)

        btn_row = QHBoxLayout()
        btn = _btn("📊  Full Evaluation on Test Set", "#89b4fa")
        btn.clicked.connect(self._evaluate)
        btn_row.addWidget(btn)

        # ── NEW: Normalized confusion matrix toggle ──
        self.cb_norm_cm = QCheckBox("Normalize Confusion Matrix")
        btn_row.addWidget(self.cb_norm_cm)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        splitter = QSplitter(Qt.Horizontal)
        self.eval_canvas = _Canvas()
        splitter.addWidget(self.eval_canvas)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(5)
        self.stats_table.setHorizontalHeaderLabels(
            ["Class", "Precision", "Recall", "F1-Score", "Support"]
        )
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        splitter.addWidget(self.stats_table)
        lay.addWidget(splitter, 1)

        self.eval_label = QLabel("Run evaluation after training.")
        self.eval_label.setStyleSheet("font-size:14px; font-weight:600; color:#a6e3a1;")
        lay.addWidget(self.eval_label)

        self.tabs.addTab(page, "📈 Evaluate")

    def _evaluate(self):
        probs, y_te, _ = self._get_preds()
        if probs is None:
            return
        nc = self.dataset["n_classes"]
        pl = np.argmax(probs, axis=1)

        # Confusion matrix
        cm = np.zeros((nc, nc), dtype=int)
        for a, b in zip(pl, y_te):
            cm[b, a] += 1

        ax = self.eval_canvas.ax
        ax.clear()
        self.eval_canvas._style()

        # ── NEW: optional normalization ──
        display_cm = cm.astype(float)
        if self.cb_norm_cm.isChecked():
            row_sums = display_cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # avoid division by zero
            display_cm = display_cm / row_sums

        ax.imshow(display_cm, cmap="Blues")
        ax.set_xlabel("Predicted", color=FG)
        ax.set_ylabel("True", color=FG)
        title_suffix = " (normalized)" if self.cb_norm_cm.isChecked() else ""
        ax.set_title(f"Confusion Matrix{title_suffix}", color=FG, weight="bold")

        for i in range(nc):
            for j in range(nc):
                val = display_cm[i, j]
                text = f"{val:.2f}" if self.cb_norm_cm.isChecked() else str(int(val))
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color)

        self.eval_canvas.fig.tight_layout()
        self.eval_canvas.draw()

        # Classification report
        tn = self.dataset.get("target_names", [str(i) for i in range(nc)])
        report = classification_report(y_te, pl, target_names=tn, output_dict=True)
        self.stats_table.setRowCount(len(tn) + 1)
        for i, name in enumerate(tn):
            m = report[name]
            for j, v in enumerate([
                name,
                f"{m['precision']:.3f}",
                f"{m['recall']:.3f}",
                f"{m['f1-score']:.3f}",
                str(int(m['support'])),
            ]):
                self.stats_table.setItem(i, j, QTableWidgetItem(v))
        m = report['macro avg']
        for j, v in enumerate([
            "Macro Avg",
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",
            f"{m['f1-score']:.3f}",
            str(int(m['support'])),
        ]):
            self.stats_table.setItem(len(tn), j, QTableWidgetItem(v))

        acc = (pl == y_te).mean()
        self.eval_label.setText(f"Overall Test Accuracy: {acc:.4f}")

    # ══════════════════════════════════════════════════════════
    #  TAB 6 — CONSOLE
    # ══════════════════════════════════════════════════════════
    def _build_console_tab(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        lay.addWidget(self.console)

        btn_row = QHBoxLayout()
        clear_btn = _btn("Clear", "#585b70")
        clear_btn.clicked.connect(self.console.clear)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        info = (
            f"⚡ Neural Fine-Tuning Studio v3.1\n"
            f"{'─' * 50}\n"
            f"CuPy GPU   : {'✅' if CUDA_AVAILABLE else '❌'}\n"
            f"Numba CUDA : {'✅' if cuda.is_available() else '❌'}\n"
            f"PyTorch    : {'✅' if PYTORCH_AVAILABLE else '❌'}\n"
            f"PyTorch GPU: {'✅' if PYTORCH_AVAILABLE and torch.cuda.is_available() else '❌'}\n"
            f"{'─' * 50}\n"
        )
        self.console.setPlainText(info)
        self.tabs.addTab(page, "🖥 Console")

    def _log(self, msg: str):
        self.console.append(msg)
        self.console.verticalScrollBar().setValue(
            self.console.verticalScrollBar().maximum()
        )


# ╔════════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                              ║
# ╚════════════════════════════════════════════════════════════╝

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    win = MainWindow()
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()