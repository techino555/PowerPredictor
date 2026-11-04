---
name: training-skills
description: |
  PowerPredictor の Linear Model を Python + MLX で訓練するためのスキル。
  DINOv3 Embedding（768次元）を入力とし、戦闘力スコアを出力する Linear 回帰モデルを
  Huber Loss + Adam で学習し、Swiftアプリへ組み込む safetensors を出力する。
  データセット管理（CSV）・モデル訓練・評価・保存まで一連の手順を定義する。
---

# PowerPredictor 訓練スキル

## 概要

このスキルは **Python + MLX** を用いて PowerPredictor の Linear Model を訓練するための手順・コード・データ規約をまとめたものです。

```
  CSV（Embedding + power）
         │
         ▼
  Python + MLX（訓練）
         │
         ▼
  power_predictor.safetensors
         │
         ▼
  Swiftアプリへ組み込み（推論専用）
```

> **重要**: 学習は Python + MLX で行い、Swift 側では推論のみ実行する。
> モデルは `power_predictor.safetensors` として出力し、iOS アプリに配置する。

---

## 1. 前提環境

### 必要ツール
- **uv**（パッケージマネージャー）— 優先使用
- Python >= 3.13
- Apple Silicon Mac（MLX は Apple Silicon 専用）

### セットアップ

```bash
# uvで環境構築（pyproject.toml を使用）
uv sync

# または個別インストール
uv pip install mlx>=0.32.0 pandas numpy safetensors
```

### pyproject.toml の主要依存パッケージ

| パッケージ | バージョン | 用途 |
|---|---|---|
| `mlx` | >=0.32.0 | Apple Silicon 向け ML フレームワーク |
| `pandas` | 2.3.3 | CSV 操作 |
| `numpy` | 1.26.2 | 数値計算 |
| `safetensors` | 0.8.0 | モデル重み保存フォーマット |
| `transformers` | >=5.4.0 | DINOv3 モデル読み込み |

---

## 2. データセット規約

### 2-1. CSVファイル命名規則

```
embeddings_yyyymmdd_num.csv
```

例：`embeddings_20260814_1.csv`

### 2-2. CSVカラム定義

| カラム名 | 型 | 内容 |
|---|---|---|
| `filename` | str | 元画像ファイル名 |
| `person_id` | int | 人物ID（下表参照） |
| `power` | float | **目的変数**：戦闘力スコア（1〜99） |
| `body_direction` | str | 顔の向き：`front` / `left` / `right` / `back` |
| `take` | str | 撮影範囲：`full`（全身8割以上） / `half` |
| `distance` | str | 距離感：`near` / `medium` / `far` |
| `posture` | str | 姿勢：`standing` / `sitting` / `lying` / `moving` |
| `gender` | str | 性別：`male` / `female` |
| `e0` 〜 `e767` | float | DINOv3 CLS Token Embedding（768次元） |

### 2-4. power の設計思想

- スコアは **1〜99** の整数（アプリには整数で表示。四捨五入）
- `power == 99` はスキップ（外れ値扱い）
- `power` が空のレコードもスキップ

---

## 3. モデルアーキテクチャ

### Linear Model（Version 1）

```python
class PowerPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 1)

    def __call__(self, x):
        return self.linear(x)
```

- 入力：DINOv3 CLS Token Embedding（768次元）
- 出力：戦闘力スコア（スカラー値）
- **MLP は不要。初期の設計方針としてLinearのみで完結する**

---

## 4. 訓練スクリプト

参考実装：[`scouter_train_maecount.py`](resources/scouter_train_maecount.py)

### 4-1. データ読み込み

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

### 4-2. Train/Validation 分割（8:2）

```python
mx.random.seed(42)
n = X.shape[0]
indices = mx.random.permutation(n)
split = int(n * 0.8)

X_train, y_train = X[indices[:split]], y[indices[:split]]
X_val,   y_val   = X[indices[split:]], y[indices[split:]]
```

### 4-3. 損失関数・Optimizer

```python
def loss_fn(model, x, y):
    predictions = model(x)
    return nn.losses.huber_loss(predictions, y, delta=1.0, reduction="mean")

optimizer = optim.Adam(learning_rate=0.001)
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
```

- **Huber Loss**（delta=1.0）：外れ値に頑健な回帰損失。他にもいくつか損失関数をテストしたがValidation Lossが一番低かったため採用。
- **Adam**（lr=0.001）：デフォルト推奨

### 4-4. 訓練ループ

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

### 4-5. 評価（MAE）

```python
predictions = model(X_val)
mx.eval(predictions)

absolute_errors = mx.abs(predictions - y_val)
mae = mx.mean(absolute_errors)
print(f"Validation MAE: {mae.item():.4f}")
```

#### Power ごとの詳細 MAE

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

### 4-6. モデル保存

```python
model.save_weights("power_predictor.safetensors")
print("モデルを保存しました。")
```

---

## 5. 訓練実行手順（エージェント向け）

1. **CSV の準備**
   - `embeddings_yyyymmdd_num.csv` が `resources/` に存在するか確認
   - `power` カラムが埋まっているレコードを確認（空行は自動スキップ）

2. **環境セットアップ**
   ```bash
   cd skills/training/resources
   uv sync
   ```

3. **訓練スクリプト実行**
   ```bash
   uv run python scouter_train_maecount.py
   ```

4. **出力確認**
   - `power_predictor.safetensors` が生成されたか確認
   - Validation MAE を確認（目安：4.0 以下が望ましい）

5. **iOS アプリへ配置**
   - 生成した `power_predictor.safetensors` を Xcode プロジェクトの `Models/linear` フォルダへコピー

---

## 6. データアノテーション支援

`resources/linear-dataset/SKILL.md`に詳細を記載。

### メタ情報の自動補完

画像の `body_direction` / `take` / `distance` / `posture` / `gender` カラムをVLM（Vision Language Model）で自動補完できる。

**活用経験のあるモデル**：`LFM2.5-VL-450M-MLX-4bit`
- URL: https://huggingface.co/LiquidAI/LFM2.5-VL-450M-MLX-4bit
- Apple Silicon / MLX 対応の軽量 VLM
- ローカルで lm-studio から利用可能

> **注意**: 複数の値を一度に補完しようとすると誤動作することがある。
> 項目を1つずつ指定して補完させること。

---

## 7. ハイパーパラメータ調整ガイド

| パラメータ | デフォルト | 変更の目安 |
|---|---|---|
| `learning_rate` | 0.001 | MAE が下がらない場合は 0.0001 に減らす |
| `epochs` | 200 | Val Loss が収束していなければ増やす |
| `delta`（Huber） | 1.0 | 外れ値が多い場合は 2.0〜5.0 に増やす |
| train/val 比率 | 8:2 | データが少ない場合は 9:1 を試す |

---

## 8. トラブルシューティング

| 症状 | 原因 | 対処 |
|---|---|---|
| `MAE` が改善しない | データが少ない・偏っている | CSVのレコード数を増やす、各person_idのサンプル数を均等にする |
| `power == 99` が多い | 外れ値の除外設計 | スキップは正しい動作。意図通り |
| `e{i}` カラムが見つからない | Embedding 列名が異なる | CSVヘッダーを確認し `e0`〜`e767` になっているか確認 |
| safetensors の読み込みエラー（Swift） | MLXとSwiftの型不一致 | Linearのweight/bias shapeを確認（`[1, 768]`, `[1]`） |
| MLXが動かない | Apple Silicon 以外 | MLXはApple Silicon専用。MシリーズのMacのみ対応 |

---

## 9. 関連ファイル

| ファイル | 内容 |
|---|---|
| [`scouter_train_maecount.py`](../resources/scouter_train_maecount.py) | 訓練スクリプト本体 |
| [`pyproject.toml`](../resources/pyproject.toml) | Python プロジェクト依存関係 |
| [`embeddings_20260814_1.csv`](../resources/embeddings_20260814_1.csv) | サンプルデータセット |
| [`resources/SKILL.md`](../resources/linear-dataset/ja-JP/SKILL.md) | データアノテーション規約（linear-dataset スキル） |