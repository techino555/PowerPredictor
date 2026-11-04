---
name: feature extraction
description: >
  学習用データセット作成スキル。
  macOS用Swiftアプリ「PowerPredictor-feature-ext」を使用し、人物画像フォルダを入力としてDINOv3による768次元Embeddingを一括抽出し、CSVを出力する。
  PowerPredictorのLinear Model 学習前に必ず実行する。
---

大幅に変える予定。


# Feature Extraction スキル

## 概要

PowerPredictor の学習データを作るための前処理パイプライン。  
macOS アプリ（`PowerPredictor_feature_ext`）を Xcode でビルドして起動し、  
人物画像が入ったフォルダを選択するだけで **DINOv3 の CLS Token（768次元）** を一括抽出して CSV に保存する。

```
人物画像フォルダ
      │
      ▼
PowerPredictor_feature_ext（macOS アプリ）
      │
      ├── Vision Framework（PersonDetector → PersonSegmenter）
      │       └─ 人物検出・背景除去
      ├── ImagePreprocessor
      │       └─ NSImage → MLXArray
      └── DINOv3Encoder
              └─ CLS Token 抽出（768次元）
      │
      ▼
embeddings_YYYYMMDD_HHmmss.csv
（filename, person_id, power, body_direction, take, distance, posture, gender, e0…e767）
```

---

## ファイル構成

```
source/PowerPredictor_feature_ext/
├── PowerPredictor_feature_extApp.swift # @main エントリポイント
├── ContentView.swift          # UI + TrainingViewModel（処理フロー制御）
├── EmbeddingExtractor.swift   # 抽出パイプライン統合クラス
├── ImagePreprocessor.swift    # NSImage → MLXArray 変換
├── Vision/
│   ├── VisionPipeline.swift   # Vision処理全体の制御
│   ├── PersonDetector.swift   # 人物バウンディングボックス取得
│   └── PersonSegmenter.swift  # 人物マスク生成・背景色処理
└── Dataset/
    └── CSVExporter.swift      # Embedding → CSV 出力
```

---

## 使い方（手順）

### Step 1: Xcode でビルド・起動

```
source/PowerPredictor.xcodeproj を Xcode で開く
  ↓
スキームを「PowerPredictor_feature_ext」に設定
  ↓
対象デバイスを「My Mac」に設定
  ↓
⌘R でビルド＆実行
```

> **依存ライブラリ**: `MLX`、`MLXDINOv3` が Swift Package として解決されること。  
> DINOv3 のモデルファイル（`dinov3-vitb16-mlx/`）をプロジェクト内の所定パスに配置しておくこと。

---

### Step 2: 画像フォルダを選択

- 「**フォルダを選択**」ボタンをクリック
- jpg / jpeg / png を含むフォルダを選択（サブフォルダも再帰的にスキャン）
- 選択後、画像枚数がステータスに表示される

---

### Step 3: Embedding 生成

- 「**Embedding 生成**」ボタンをクリック
- 以下の処理が画像ごとに順次実行される：

| ステップ | 処理クラス | 内容 |
|---|---|---|
| 1 | `PersonDetector` | `VNDetectHumanRectanglesRequest` で人物検出 |
| 2 | `PersonSegmentProcessor` | `VNGeneratePersonSegmentationRequest` でマスク生成 |
| 3 | `ImagePreprocessor` | `NSImage → MLXArray` に変換（`MLXDINOv3.ImageProcessor` 使用） |
| 4 | `DINOv3Encoder` | CLS Token（768次元）を抽出 |

- 人物が検出されなかった画像はスキップ（ログに記録）
- 進捗バーでリアルタイムに確認可能

---

### Step 4: CSV を保存

- 「**CSV を保存**」ボタンをクリック
- `~/Documents/embeddings_YYYYMMDD_HHmmss.csv` として保存される
- 保存後、Finder で自動的にファイルが選択表示される

---

## 出力 CSV フォーマット

| カラム名 | 型 | 説明 |
|---|---|---|
| `filename` | String | 入力フォルダ相対パス |
| `person_id` | String | 人物ID（空欄。手動で後から記入） |
| `power` | String | 戦闘力ラベル（空欄。手動で後から記入） |
| `body_direction` | String | 体の向き（空欄。手動で後から記入） |
| `take` | String | 写っている人物写真が全身か半身か（空欄） |
| `distance` | String | 撮影距離（空欄） |
| `posture` | String | 姿勢（空欄） |
| `gender` | String | 性別（空欄） |
| `e0` … `e767` | Float | DINOv3 CLS Token（768次元） |

> **重要**: Embedding 生成後、`power` 列に戦闘力スコアを手動で記入してから学習に使用する。

---

## 内部処理の詳細

### EmbeddingExtractor（抽出パイプライン）

```swift
// NSImage → CGImage → VisionPipeline → ImagePreprocessor → DINOv3Encoder → CLS Token
func extract(from image: NSImage) async throws -> MLXArray? {
    let (_, _, completeImage) = try await visionPipeline.process(image: image)
    guard let personImage = completeImage else { return nil }
    let input = try preprocessor.preprocess(personImage)
    return encoder.encode(input)
}
```

### VisionPipeline の出力

`process(image:)` は 3つの値を返す：

| 戻り値 | 型 | 用途 |
|---|---|---|
| `boundingBox` | `CGRect?` | 人物の位置（正規化座標） |
| `analysisImage` | `NSImage?` | 解析中エフェクト画像（人物：黄土色 / 背景：黄色） |
| `completeImage` | `NSImage?` | DINOv3 への入力画像（背景：黄色フィルター適用済み） |

> Embedding 抽出には `completeImage`（背景黄色フィルター適用済み）を使用する。

---

## よくあるエラーと対処

| エラー | 原因 | 対処 |
|---|---|---|
| `画像の読み込みに失敗` | 破損ファイル / 非対応フォーマット | jpg/jpeg/png のみ対応。ファイルを確認する |
| `Embedding抽出に失敗（人物未検出）` | Vision が人物を検出できなかった | 人物が小さすぎる・横向き・正面でない可能性がある |
| `DINOv3モデルの読み込みに失敗` | モデルファイルが見つからない | `dinov3-vitb16-mlx/` の配置パスを確認する |
| CSV が空 | 全画像でスキップが発生 | 人物が明確に写っている画像を使う |

---

## 次のステップ

CSV が生成されたら、以下のスキルへ続く：

```
feature extraction（このスキル）
      ↓ embeddings_YYYYMMDD_HHmmss.csv
      ↓ power 列を手動で記入
training（Python + MLX で Linear Model 学習。trainingフォルダ参照）
      ↓ power_predictor.safetensors
inference（iOSアプリへ組み込み → 戦闘力表示）
```
