#!/usr/bin/env python3
"""Train the InferCore demo CNN on MNIST and export all model artifacts.

Produces (under models/):
  mnist_cnn.json       graph description consumed by the native runtime
  mnist_fp32.weights   ICW1 fp32 weights
  mnist_int8.weights   ICQ1 int8 weights (per-output-channel symmetric PTQ)
  mnist_cnn.onnx       same network exported to ONNX (for the ORT backend)
  samples.bin          10 x (int64 label + 784 float32 pixels) for tests/demo

Everything is deterministic (fixed seeds) so artifacts are reproducible.
"""
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MNIST = MODELS / "mnist"
SEED = 1337
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- mnist io

def load_idx_images(path):
    data = path.read_bytes()
    magic, n, rows, cols = struct.unpack(">IIII", data[:16])
    assert magic == 0x803
    return np.frombuffer(data[16:], dtype=np.uint8).reshape(n, rows, cols)

def load_idx_labels(path):
    data = path.read_bytes()
    magic, n = struct.unpack(">II", data[:8])
    assert magic == 0x801
    return np.frombuffer(data[8:], dtype=np.uint8)

X_train_full = load_idx_images(MNIST / "train-images-idx3-ubyte").astype(np.float32) / 255.0
y_train_full = load_idx_labels(MNIST / "train-labels-idx1-ubyte")
X_test = load_idx_images(MNIST / "t10k-images-idx3-ubyte").astype(np.float32) / 255.0
y_test = load_idx_labels(MNIST / "t10k-labels-idx1-ubyte")

# ---------------------------------------------------------------- layers

def im2col(x, k):  # x: (N,C,H,W), stride 1, pad 0 -> (N*outH*outW, C*k*k)
    N, C, H, W = x.shape
    outH, outW = H - k + 1, W - k + 1
    cols = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))
    # cols: (N, C, outH, outW, k, k) -> (N, outH, outW, C, k, k) -> (N*outH*outW, C*k*k)
    cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(N * outH * outW, C * k * k)
    return cols, (N, C, H, W, outH, outW)

def col2im(dcol, meta, k):
    N, C, H, W, outH, outW = meta
    dcols = dcol.reshape(N, outH, outW, C, k, k).transpose(0, 3, 1, 2, 4, 5)
    dx = np.zeros((N, C, H, W), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            dx[:, :, i:i + outH, j:j + outW] += dcols[:, :, :, :, i, j]
    return dx

class Conv:
    def __init__(self, cin, cout, k=3):
        self.k = k
        limit = np.sqrt(2.0 / (cin * k * k))
        self.w = (rng.standard_normal((cout, cin, k, k)) * limit).astype(np.float32)
        self.b = np.zeros(cout, dtype=np.float32)

    def forward(self, x):
        col, meta = im2col(x, self.k)
        N, C, H, W, outH, outW = meta
        out = col @ self.w.reshape(self.w.shape[0], -1).T + self.b
        self.cache = (col, meta)
        return out.reshape(N, outH, outW, -1).transpose(0, 3, 1, 2)

    def backward(self, dout):
        col, meta = self.cache
        N, C, H, W, outH, outW = meta
        d = dout.transpose(0, 2, 3, 1).reshape(-1, self.w.shape[0])
        self.dw = (d.T @ col).reshape(self.w.shape)
        self.db = d.sum(axis=0)
        dcol = d @ self.w.reshape(self.w.shape[0], -1)
        return col2im(dcol, meta, self.k)

class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask
    def backward(self, dout):
        return dout * self.mask

class MaxPool:
    def forward(self, x):
        N, C, H, W = x.shape
        H2, W2 = H - H % 2, W - W % 2
        self.meta = x.shape
        xr = x[:, :, :H2, :W2].reshape(N, C, H2 // 2, 2, W2 // 2, 2)
        out = xr.max(axis=(3, 5))
        self.cache = (x, out)
        return out
    def backward(self, dout):
        x, out = self.cache
        N, C, H, W = self.meta
        H2, W2 = H - H % 2, W - W % 2
        dx = np.zeros_like(x)
        xr = x[:, :, :H2, :W2].reshape(N, C, H2 // 2, 2, W2 // 2, 2)
        mask = xr == out[:, :, :, None, :, None]
        dxr = mask * (dout[:, :, :, None, :, None])
        dx[:, :, :H2, :W2] = dxr.reshape(N, C, H2, W2)
        return dx

class Flatten:
    def forward(self, x):
        self.meta = x.shape
        return x.reshape(x.shape[0], -1)
    def backward(self, dout):
        return dout.reshape(self.meta)

class Linear:
    def __init__(self, cin, cout):
        limit = np.sqrt(2.0 / cin)
        self.w = (rng.standard_normal((cout, cin)) * limit).astype(np.float32)
        self.b = np.zeros(cout, dtype=np.float32)
    def forward(self, x):
        self.x = x
        return x @ self.w.T + self.b
    def backward(self, dout):
        self.dw = dout.T @ self.x
        self.db = dout.sum(axis=0)
        return dout @ self.w

# ---------------------------------------------------------------- model

layers = [Conv(1, 8), ReLU(), MaxPool(), Conv(8, 16), ReLU(), MaxPool(), Flatten(), Linear(16 * 5 * 5, 10)]

def forward(x, train=False):
    for l in layers:
        x = l.forward(x)
    return x

def softmax_ce(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    dlogits = p
    dlogits[np.arange(n), y] -= 1
    return loss, dlogits / n

def predict(x):
    return forward(x).argmax(axis=1)

# ---------------------------------------------------------------- train

N_TRAIN = 12000
EPOCHS = 6
BATCH = 64
LR = 0.01

idx = rng.permutation(len(X_train_full))[:N_TRAIN]
X_train = X_train_full[idx][:, None, :, :]
y_train = y_train_full[idx]

params = [l for l in layers if hasattr(l, "w")]
# Adam state
m = [{k: np.zeros_like(getattr(l, k)) for k in ("w", "b")} for l in params]
v = [{k: np.zeros_like(getattr(l, k)) for k in ("w", "b")} for l in params]
t = 0
t0 = time.time()
for epoch in range(EPOCHS):
    order = rng.permutation(len(X_train))
    tot_loss, nb = 0.0, 0
    for s in range(0, len(order), BATCH):
        bidx = order[s:s + BATCH]
        xb, yb = X_train[bidx], y_train[bidx]
        logits = forward(xb)
        loss, d = softmax_ce(logits, yb)
        for l in reversed(layers):
            d = l.backward(d)
        t += 1
        for i, l in enumerate(params):
            for k in ("w", "b"):
                g = getattr(l, "d" + k)
                m[i][k] = 0.9 * m[i][k] + 0.1 * g
                v[i][k] = 0.999 * v[i][k] + 0.001 * g * g
                mh = m[i][k] / (1 - 0.9 ** t)
                vh = v[i][k] / (1 - 0.999 ** t)
                setattr(l, k, getattr(l, k) - LR * mh / (np.sqrt(vh) + 1e-8))
        tot_loss += loss
        nb += 1
    # eval
    correct = sum((predict(X_test[i:i + 1000, None]) == y_test[i:i + 1000]).sum()
                  for i in range(0, 10000, 1000))
    acc = 100.0 * correct / 10000.0
    print(f"epoch {epoch+1}/{EPOCHS} loss {tot_loss/nb:.4f} test_acc {acc:.2f}% "
          f"({time.time()-t0:.0f}s)", flush=True)

fp32_acc = 100.0 * sum((predict(X_test[i:i + 1000, None]) == y_test[i:i + 1000]).sum()
                       for i in range(0, 10000, 1000)) / 10000.0
print(f"final fp32 test accuracy: {fp32_acc:.2f}%")

conv1, conv2, fc = layers[0], layers[3], layers[7]

# ---------------------------------------------------------------- save ICW1

def write_icw1(path, tensors):
    with open(path, "wb") as f:
        f.write(b"ICW1")
        f.write(struct.pack("<I", len(tensors)))
        for name, arr in tensors:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<q", d))
            f.write(struct.pack("<I", 0))  # dtype f32
            f.write(arr.tobytes())

write_icw1(MODELS / "mnist_fp32.weights", [
    ("conv1.w", conv1.w), ("conv1.b", conv1.b),
    ("conv2.w", conv2.w), ("conv2.b", conv2.b),
    ("fc1.w", fc.w), ("fc1.b", fc.b),
])

# ---------------------------------------------------------------- quantize

def quant_per_channel(w):
    flat = w.reshape(w.shape[0], -1)
    scales = np.maximum(np.abs(flat).max(axis=1) / 127.0, 1e-12).astype(np.float32)
    q = np.clip(np.round(flat / scales[:, None]), -127, 127).astype(np.int8)
    return q.reshape(w.shape), scales

def write_icq1(path, tensors):
    with open(path, "wb") as f:
        f.write(b"ICQ1")
        f.write(struct.pack("<I", len(tensors)))
        for name, payload in tensors:
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            if isinstance(payload, tuple):  # (int8 array, scales)
                arr, scales = payload
                f.write(struct.pack("<I", arr.ndim))
                for d in arr.shape:
                    f.write(struct.pack("<q", d))
                f.write(struct.pack("<I", 1))  # dtype s8
                f.write(struct.pack("<I", len(scales)))
                f.write(np.ascontiguousarray(scales, dtype=np.float32).tobytes())
                f.write(np.ascontiguousarray(arr, dtype=np.int8).tobytes())
            else:
                arr = np.ascontiguousarray(payload, dtype=np.float32)
                f.write(struct.pack("<I", arr.ndim))
                for d in arr.shape:
                    f.write(struct.pack("<q", d))
                f.write(struct.pack("<I", 0))
                f.write(arr.tobytes())

q1 = quant_per_channel(conv1.w)
q2 = quant_per_channel(conv2.w)
qf = quant_per_channel(fc.w)
write_icq1(MODELS / "mnist_int8.weights", [
    ("conv1.w", q1), ("conv1.b", conv1.b),
    ("conv2.w", q2), ("conv2.b", conv2.b),
    ("fc1.w", qf), ("fc1.b", fc.b),
])

# ---------------------------------------------------------------- graph json

graph = {
    "name": "mnist-cnn",
    "input_name": "input",
    "input_shape": [1, 1, 28, 28],
    "output_layer": "softmax",
    "layers": [
        {"name": "conv1", "type": "Conv2D", "stride": 1, "pad": 0,
         "weights": "conv1.w", "bias": "conv1.b"},
        {"name": "relu1", "type": "ReLU"},
        {"name": "pool1", "type": "MaxPool", "kernel": 2, "stride": 2},
        {"name": "conv2", "type": "Conv2D", "stride": 1, "pad": 0,
         "weights": "conv2.w", "bias": "conv2.b"},
        {"name": "relu2", "type": "ReLU"},
        {"name": "pool2", "type": "MaxPool", "kernel": 2, "stride": 2},
        {"name": "flatten", "type": "Flatten"},
        {"name": "fc1", "type": "Gemm", "weights": "fc1.w", "bias": "fc1.b"},
        {"name": "softmax", "type": "Softmax"},
    ],
}
(MODELS / "mnist_cnn.json").write_text(json.dumps(graph, indent=2) + "\n")

# ---------------------------------------------------------------- onnx export

def export_onnx():
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    inits = []
    for name, arr in [("conv1.w", conv1.w), ("conv1.b", conv1.b),
                      ("conv2.w", conv2.w), ("conv2.b", conv2.b),
                      ("fc1.w", fc.w), ("fc1.b", fc.b)]:
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=np.float32), name))

    nodes = [
        helper.make_node("Conv", ["input", "conv1.w", "conv1.b"], ["conv1"],
                         kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0]),
        helper.make_node("Relu", ["conv1"], ["relu1"]),
        helper.make_node("MaxPool", ["relu1"], ["pool1"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Conv", ["pool1", "conv2.w", "conv2.b"], ["conv2"],
                         kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0]),
        helper.make_node("Relu", ["conv2"], ["relu2"]),
        helper.make_node("MaxPool", ["relu2"], ["pool2"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Flatten", ["pool2"], ["flatten"], axis=1),
        helper.make_node("Gemm", ["flatten", "fc1.w", "fc1.b"], ["fc1"], transB=1),
        helper.make_node("Softmax", ["fc1"], ["output"], axis=1),
    ]
    g = helper.make_graph(
        nodes, "mnist-cnn",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 28, 28])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])],
        inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, MODELS / "mnist_cnn.onnx")

export_onnx()

# ---------------------------------------------------------------- samples.bin

with open(MODELS / "samples.bin", "wb") as f:
    for i in range(10):
        f.write(struct.pack("<q", int(y_test[i])))
        f.write(np.ascontiguousarray(X_test[i].reshape(-1), dtype=np.float32).tobytes())

# ---------------------------------------------------------------- ort sanity check

import onnxruntime as ort
sess = ort.InferenceSession(str(MODELS / "mnist_cnn.onnx"))
def softmax_np(logits):
    z = logits - logits.max()
    p = np.exp(z)
    return p / p.sum()

diffs = []
for i in range(20):
    x = X_test[i][None, None]
    ref = softmax_np(forward(x)[0])  # forward() returns logits; ONNX emits probabilities
    got = sess.run(["output"], {"input": x})[0][0]
    diffs.append(np.abs(ref - got).max())
print(f"numpy vs onnxruntime max abs diff (20 inputs): {max(diffs):.3e}")

sizes = {p.name: p.stat().st_size for p in MODELS.glob("*") if p.is_file()}
print("artifacts:", json.dumps(sizes, indent=None))

# single-sample file for the dlopen demo (label + 784 floats)
with open(MODELS / "sample_demo.bin", "wb") as f:
    f.write(struct.pack("<q", int(y_test[0])))
    f.write(np.ascontiguousarray(X_test[0].reshape(-1), dtype=np.float32).tobytes())
print("sample_demo.bin label:", int(y_test[0]))
