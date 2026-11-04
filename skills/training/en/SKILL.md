---
name: training-skills
description: |
  A skill for training the PowerPredictor Linear Model using Python + MLX.
  Trains a linear regression model using Huber Loss + Adam that takes DINOv3 Embeddings (768 dimensions) as input and outputs a battle power score, then exports safetensors to be integrated into the Swift app.
  Defines the end-to-end workflow covering dataset management (CSV), model training, evaluation, and saving.
---

# PowerPredictor Training Skill

## Overview

This skill summarizes the procedures, code, and data conventions for training the PowerPredictor Linear Model using **Python + MLX**.

```
  CSV (Embedding + power)
         │
         ▼
  Python + MLX (Training)
         │
         ▼
  power_predictor.safetensors
         │
         ▼
  Integration into Swift App (Inference only)
```

> **Important**: Training is performed using Python + MLX, while the Swift side runs inference only.
> The model is exported as `power_predictor.safetensors` and placed in the iOS app.

---

## 1. Prerequisites & Environment

### Required Tools
- **uv** (Package manager) — Preferred
- Python >= 3.13
- Apple Silicon Mac (MLX requires Apple Silicon)

### Setup

```bash
# Set up environment with uv (using pyproject.toml)
uv sync

# Or install packages individually
uv pip install mlx>=0.32.0 pandas numpy safetensors
```

### Key Dependencies in pyproject.toml

| Package | Version | Purpose |
|---|---|---|
| `mlx` | >=0.32.0 | ML framework for Apple Silicon |
| `pandas` | 2.3.3 | CSV manipulation |
| `numpy` | 1.26.2 | Numerical computation |
| `safetensors` | 0.8.0 | Model weights storage format |
| `transformers` | >=5.4.0 | Loading DINOv3 model |

---

## 2. Dataset Conventions

### 2-1. CSV File Naming Convention

```
embeddings_yyyymmdd_num.csv
```

Example: `embeddings_20260814_1.csv`

### 2-2. CSV Column Definitions

| Column Name | Type | Description |
|---|---|---|
| `filename` | str | Original image file name |
| `person_id` | int | Person ID (see table below) |
| `power` | float | **Target variable**: Battle power score (1–99) |
| `body_direction` | str | Face direction: `front` / `left` / `right` / `back` |
| `take` | str | Framing: `full` (80%+ of full body visible) / `half` |
| `distance` | str | Subject distance: `near` / `medium` / `far` |
| `posture` | str | Posture: `standing` / `sitting` / `lying` / `moving` |
| `gender` | str | Gender: `male` / `female` |
| `e0` to `e767` | float | DINOv3 CLS Token Embedding (768 dimensions) |

### 2-4. Design Principles for `power`

- The score is an integer from **1 to 99** (displayed as an integer rounded in the app).
- Records with `power == 99` are skipped (treated as outliers).
- Records where `power` is empty are also skipped.

---

## 3. Model Architecture

### Linear Model (Version 1)

```python
class PowerPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 1)

    def __call__(self, x):
        return self.linear(x)
```

- Input: DINOv3 CLS Token Embedding (768 dimensions)
- Output: Battle power score (scalar value)
- **MLP is not required. As an initial design policy, a Linear model alone is used.**

---

## 4. Training Script

Reference implementation: [`scouter_train_maecount.py`](resources/scouter_train_maecount.py)

### 4-1. Data Loading

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import csv

X_list, y_list = [], []

with open('embeddings_all_5.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not row['power']:
            continue
        try:
            power = float(row['power'])
        except ValueError:
            continue
        if power == 99:
            continue

        emb = [float(row[f'e{i}']) for i in range(768)]
        X_list.append(emb)
        y_list.append([power])

X = mx.array(X_list)
y = mx.array(y_list)
```

### 4-2. Train/Validation Split (8:2)

```python
mx.random.seed(42)
n = X.shape[0]
indices = mx.random.permutation(n)
split = int(n * 0.8)

X_train, y_train = X[indices[:split]], y[indices[:split]]
X_val,   y_val   = X[indices[split:]], y[indices[split:]]
```

### 4-3. Loss Function & Optimizer

```python
def loss_fn(model, x, y):
    predictions = model(x)
    return nn.losses.huber_loss(predictions, y, delta=1.0, reduction="mean")

optimizer = optim.Adam(learning_rate=0.001)
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
```

- **Huber Loss** (delta=1.0): Regression loss robust to outliers. Adopted because it achieved the lowest Validation Loss among several tested loss functions.
- **Adam** (lr=0.001): Default recommendation.

### 4-4. Training Loop

```python
epochs = 200

for epoch in range(epochs):
    loss, grads = loss_and_grad_fn(model, X_train, y_train)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    if (epoch + 1) % 20 == 0:
        val_loss = loss_fn(model, X_val, y_val)
        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
```

### 4-5. Evaluation (MAE)

```python
predictions = model(X_val)
mx.eval(predictions)

absolute_errors = mx.abs(predictions - y_val)
mae = mx.mean(absolute_errors)
print(f"Validation MAE: {mae.item():.4f}")
```

#### Detailed MAE per Power Value

```python
pred_list   = predictions.tolist()
target_list = y_val.tolist()
power_results = {}

for pred, target in zip(pred_list, target_list):
    tv = target[0]
    power_results.setdefault(tv, []).append(abs(pred[0] - tv))

for power in sorted(power_results):
    errors = power_results[power]
    print(f"Power {int(power):2d} : Samples={len(errors):3d}, MAE={sum(errors)/len(errors):.4f}")
```

### 4-6. Saving the Model

```python
model.save_weights("power_predictor.safetensors")
print("Model saved successfully.")
```

---

## 5. Training Execution Steps (for Agents)

1. **Prepare CSV**
   - Verify that `embeddings_yyyymmdd_num.csv` exists in `resources/`.
   - Ensure the records have the `power` column filled (empty rows are automatically skipped).

2. **Environment Setup**
   ```bash
   cd skills/training/resources
   uv sync
   ```

3. **Run Training Script**
   ```bash
   uv run python scouter_train_maecount.py
   ```

4. **Verify Output**
   - Check that `power_predictor.safetensors` has been generated.
   - Verify the Validation MAE (target: 4.0 or lower is desirable).

5. **Deploy to iOS App**
   - Copy the generated `power_predictor.safetensors` to the `Models/linear` directory in the Xcode project.

---

## 6. Data Annotation Support

Detailed instructions are provided in `resources/linear-dataset/SKILL.md`.

### Automatic Completion of Metadata

The image attributes `body_direction`, `take`, `distance`, `posture`, and `gender` can be automatically populated using a VLM (Vision Language Model).

**Model with verified usage**: `LFM2.5-VL-450M-MLX-4bit`
- URL: https://huggingface.co/LiquidAI/LFM2.5-VL-450M-MLX-4bit
- Lightweight VLM compatible with Apple Silicon / MLX
- Can be used locally via lm-studio

> **Note**: Requesting completion of multiple fields simultaneously may result in errors.
> Specify and complete one attribute at a time.

---

## 7. Hyperparameter Tuning Guide

| Parameter | Default | Adjustment Guideline |
|---|---|---|
| `learning_rate` | 0.001 | If MAE does not decrease, reduce to 0.0001 |
| `epochs` | 200 | If Val Loss has not converged, increase |
| `delta` (Huber) | 1.0 | If there are many outliers, increase to 2.0–5.0 |
| Train/Val Ratio | 8:2 | If data is limited, try 9:1 |

---

## 8. Troubleshooting

| Symptom | Cause | Solution |
|---|---|---|
| `MAE` does not improve | Insufficient or biased data | Increase CSV record count; balance sample counts across `person_id` |
| Many `power == 99` | Outlier exclusion design | Skipping is the expected behavior and working as intended |
| `e{i}` column not found | Mismatched embedding column names | Check CSV header and ensure names match `e0` through `e767` |
| Safetensors loading error (Swift) | MLX and Swift type mismatch | Verify Linear weight/bias shapes (`[1, 768]`, `[1]`) |
| MLX does not run | Non-Apple Silicon hardware | MLX is exclusive to Apple Silicon. Only M-series Macs are supported |

---

## 9. Related Files

| File | Content |
|---|---|
| [`scouter_train_maecount.py`](../resources/scouter_train_maecount.py) | Main training script |
| [`pyproject.toml`](../resources/pyproject.toml) | Python project dependencies |
| [`embeddings_20260814_1.csv`](../resources/embeddings_20260814_1.csv) | Sample dataset |
| [`SKILL.md`](../resources/linear-dataset/en/SKILL.md) | Data annotation conventions (linear-dataset skill) |
