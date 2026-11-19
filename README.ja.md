# PowerPredictor

このプロジェクトはMLX Swiftおよび[NOTICE](NOTICE) に記載のパッケージを使用して画像から特徴量を抽出し、対象の「戦闘力」を算出するiOSアプリです。日本のアニメ※『ドラゴンボールZ』のスカウターへのオマージュです。

※
©バードスタジオ／集英社・フジテレビ・東映アニメーション

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Swift](https://img.shields.io/badge/Swift-6.0-F05138.svg?style=flat-square&logo=swift&logoColor=white)](https://swift.org)
[![iOS](https://img.shields.io/badge/iOS-17.0+-000000.svg?style=flat-square&logo=apple&logoColor=white)](https://developer.apple.com/ios/)
[![Platform](https://img.shields.io/badge/Platform-iOS%20%7C%20macOS-lightgrey.svg?style=flat-square)]()
[![MLX Swift](https://img.shields.io/badge/MLX-Swift-FF4F00.svg?style=flat-square)](https://github.com/ml-explore/mlx-swift)

## デモ

iPhoneアプリをインストール後に起動。起動するとカメラの画面になりScanボタンを押せば起動する。戦闘力表示は人に向ける必要がある。測定後にRESETを押して元に戻る。これを繰り返すだけ。

**[ScanDemo1](demo/PowerPredictor_demo1.mp4)**

**[ScanDemo2](demo/PowerPredictor_demo2.mp4)**


## アーキテクチャ

コードベースのアーキテクチャ

```

source/PowerPredictor/

├── App
│   └── PowerPredictor.swift
│        # @main エントリポイント
│ 
├── Audio
│   └── SoundPlayer.swift
│        # 効果音再生
│
├── Camera
│   ├── CameraManager.swift
│   │      # AVCaptureSession管理、フレーム取得
│   │
│   └── ImagePreprocessor.swift
│          # UIImage/CVPixelBufferをMLX入力テンソルへ変換
│
├── DINOv3
│      └── DINOv3Encoder.swift
│             # DINOv3読み込み
│             # Embedding(768次元)抽出
│
├── Embedding
│      └── EmbeddingPipeline.swift
│             # 前処理
│             # 人物切り抜き
│             # DINOv3Encoder呼び出し
│             # Embedding生成
│
└── Models
│      ├── dinov3-vitb16-mlx/
│      │      # DINOv3モデル
│      │      # config.json
│      │      # model.safetensors
│      │
│      └── mylinear/
│             └── power_predictor.safetensors
│                    # Predictor学習済み重み
│                    # （Linearならconfig不要）
│
├── Predictor
│      └── LinearPredictor.swift
│             # Embedding → 戦闘力スコア推論
│
└── Resources
│    └── Assets.xcassets
│    │      # アイコン、画像、色、フォントなど
│    └── Sounds
│           # 効果音ファイル
│           # nc116296_scouter.wav（ニコニ・コモンズ nc116296）
│
├── Stores
│   └── ScouterStore.swift
│          # アプリ全体の推論フロー管理
│          # Vision→Embedding→Predictor→UI更新
│
├── ViewModels
│   └── ScanViewModel.swift
│          # UIイベントの制御
│          # SCAN開始、Store呼び出し、状態更新
│          # ScanState enum: idle / detecting / analyzing / complete
│
├── Views
│   ├── ContentView.swift
│   │      # メインUI（カメラプレビュー + スコア表示 + SCANボタン）
│   │
│   ├── ScanOverlayView.swift
│   │      # 人物検出枠、LOCK ON、SCANアニメーションなどを描画
│   │      # 人物を検出した後の三角HUDの演出用
│   │      # 背景をopacity0.6くらいの黄色に、解析中の人物を黄土色に濃くする演出用
│   │      # 内部View構成:
│   │      #   ScanEffectView   … スキャンアニメーション演出
│   │      #   HUDOverlayView   … HUD全体オーバーレイ
│   │      #   TriangleMarker   … 四隅の三角マーカー描画
│   │      #   TriangleDirection … 三角の向きを表すenum
│   │
│   └── PowerView.swift
│          # 戦闘力・コメント・ゲージなど結果表示UI
│
├── Vision
    ├── VisionPipeline.swift
    │      # Vision Framework全体の実行
    │      # PersonDetector → PoseValidator → PersonSegmentProcessor の制御
    │      # VisionPipelineError enum: measurementRejected / noPersonDetected
    │
    ├── PersonDetector.swift
    │      # 人物検出（Bounding Box取得）
    │
    ├── PoseValidator.swift
    │      # 無制限の検出を防ぐための姿勢判定
    │      # 検出部位: 鼻・左右肩・左右腰
    │      # 信頼度閾値: 肩・顔=0.5、腰=0.3
    │
    ├── FrameValidator.swift
    │      # 無制限の検出を防ぐための姿勢判定
    │      # 検出部位: 鼻・左右肩・左右腰
    │      # 信頼度閾値: 肩・顔=0.5、腰=0.3
    │
    ├── MeasurementResult.swift
    │      # 測定可否の結果構造体 (isValid: Bool, reason: String)
    │
    └── PersonSegmenter.swift
           # 人物マスク生成・背景除去
           # 実クラス名: PersonSegmentProcessor
           # 解析中: 人物→黄土色、背景→黄色
           # 完了時: 全体を黄色フィルター適用
```

それぞれのコード構成の狙いについて主要なものは下記のドキュメント、SKILLを参照ください。

### カメラから人物検出、背景除去、検出判定

**[PoseDetector](docs/ja-JP/posedetector.md)**

### 戦闘力の学習元となる特徴量の抽出
**[FeatureExtract](docs/ja-JP/feature-extraction.md)**

特徴量の抽出方法

**[FeatureExtract Skills](skills/feature-extraction/ja-JP/SKILL.md)**

### 戦闘力の学習
**[Training](docs/ja-JP/powertraining.md)**

学習方法

**[Training Skills](skills/training/ja-JP/SKILL.md)**

### アプリのUIと状態遷移について

**[UI and ScanState](docs/ja-JP/uiandscanstate.md)**


## 導入
このプロダクトは本リポジトリをクローンしただけでは活用することができず、いくつか3rdParty製品の規約に同意の上、インストール必須の資源があります。導入の方法はこちらのSKILLを参照ください。

**[How to Build app](skills/build-to-iOS/ja-JP/SKILL.md)**

## ドキュメント

**[FAQ](docs/ja-JP/faq.md)** についてこちら


## 問い合わせ

お問い合わせ、要望はこちらへ : 

[GitHub issues](https://github.com/techino555/PowerPredictor/issues). and [X](https://x.com/hidenobu_is).


## 感謝

まだ開発、訓練は継続しますのでよろしければこちらで支援をいただきますと幸いです。

[Buy Me a Coffee](https://buymeacoffee.com/techino555). 

有料コンテンツは一切ありません。今はMITであり、これからもMITであり続けます。　

## ライブラリについて

App：Swift。Package Dependencies

```swift
eventsource 1.6.0
Jinja 2.5.2
mlx-swift 0.32.3
mlx-swift-lm 3.32.3
mlxdinov3 2.0.0
swift-asn1 1.7.4
swift-collections 1.6.0
swift-crypto 4.5.2
swift-huggingface 0.13.0
swift-numerics 1.1.1
swift-syntax 603.0.2
swift-transformers 1.3.3
yyjson 0.12.0
```

Training：Python。Project.toml

```toml
[project]
name = "PowerPredictor-training"
version = "0.1.0"
description = "PowerPredictor model training with MLX and dataset utilities"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "mlx>=0.32.0",
    "safetensors==0.8.0",
    "numpy>=1.26.2",
    "pandas>=2.3.3",
    "datasets==5.0.1",
    "pillow>=12.3.0"
]
```

## License

MITです。こちらを確認ください： [LICENSE](LICENSE)

3rdパーティのコードや製品もあり、それらのライセンスはNOTICEに記載の通り、依然として3rdパーティの側に帰属します： [NOTICE](NOTICE) 