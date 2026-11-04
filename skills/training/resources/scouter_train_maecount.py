import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import csv

# データの読み込み
X_list = []
y_list = []

with open('embeddings_20260814_1.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # powerが空または無効な場合はスキップ
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

# MLX配列に変換
X = mx.array(X_list)
y = mx.array(y_list)

# データ数
n = X.shape[0]

# シード固定
mx.random.seed(42)

# インデックスをシャッフル
indices = mx.random.permutation(n)

# 8:2に分割
split = int(n * 0.8)

train_idx = indices[:split]
val_idx = indices[split:]

X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]

print(f"Samples: {n}")
print(f"Train: {X_train.shape[0]}")
print(f"Validation: {X_val.shape[0]}")

# モデルの定義
class PowerPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 1)

    def __call__(self, x):
        return self.linear(x)

model = PowerPredictor()
mx.eval(model.parameters())

# Huber Loss
def loss_fn(model, x, y):
    predictions = model(x)
    return nn.losses.huber_loss(
        predictions,
        y,
        delta=1.0,
        reduction="mean"
    )


# Optimizer
learning_rate = 0.001
optimizer = optim.Adam(learning_rate=learning_rate)

# 勾配計算
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# 訓練
epochs = 200

for epoch in range(epochs):
    loss, grads = loss_and_grad_fn(model, X_train, y_train)

    optimizer.update(model, grads)

    mx.eval(model.parameters(), optimizer.state)

    if (epoch + 1) % 20 == 0:
        val_loss = loss_fn(model, X_val, y_val)

        print(
            f"Epoch {epoch+1}, "
            f"Train Loss: {loss.item():.4f}, "
            f"Val Loss: {val_loss.item():.4f}"
        )


# ============================================================
# Validation評価
# ============================================================

predictions = model(X_val)

# MLXの計算を実行
mx.eval(predictions)

print("\n--- Validation Result ---")

print("Prediction:", predictions[:5])
print("Target:", y_val[:5])
print("Validation max power:", mx.max(y_val))


# ============================================================
# 全ValidationデータのMAE
# ============================================================

absolute_errors = mx.abs(predictions - y_val)

mae = mx.mean(absolute_errors)

print(f"Validation MAE: {mae.item():.4f}")


# ============================================================
# Powerごとの評価
# ============================================================

# Pythonのリストに変換
pred_list = predictions.tolist()
target_list = y_val.tolist()

# Powerごとに誤差を集計
power_results = {}

for pred, target in zip(pred_list, target_list):

    pred_value = pred[0]
    target_value = target[0]

    if target_value not in power_results:
        power_results[target_value] = []

    power_results[target_value].append(
        abs(pred_value - target_value)
    )


print("\n--- PowerごとのValidation MAE ---")

for power in sorted(power_results.keys()):

    errors = power_results[power]

    count = len(errors)
    mae = sum(errors) / count

    print(
        f"Power {int(power):2d} : "
        f"Samples={count:3d}, "
        f"MAE={mae:.4f}"
    )


# ============================================================
# モデル保存
# ============================================================

model.save_weights(
    f"power_predictor.safetensors"
)

print("\nモデルを保存しました。")
