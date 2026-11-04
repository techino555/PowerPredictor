---
name: build-to-iOS
description: 
  PowerPredictorアプリをiPhone実機にビルドするための手順ガイド。
  環境準備からDINOv3モデルの取得・変換、Xcodeビルド、実機インストールまでをカバーする。
---

# PowerPredictor — iPhone実機ビルド手順

> **概要**
> PowerPredictorはiPhoneカメラで人物を捉え、DINOv3（Meta製Vision Transformer）と
> 学習済みLinear Predictorで「戦闘力スコア」をリアルタイム表示するスカウター風iOSアプリです。
> MLX Swift を使い Apple Silicon 上で推論します。

---

## 前提条件チェックリスト

以下がすべて揃っていることを確認してから作業を開始してください。

| 項目 | 必要なもの |
|------|-----------|
| Mac | Apple Silicon (M1 以降) + macOS Sequoia 以降推奨 |
| Xcode | 16.0 以降（App Store または Developer サイトから） |
| Apple ID | 無料 Apple ID でも実機ビルド可（7日間証明書） |
| iPhone | iOS 17 以降、USB ケーブルで Mac に接続 |
| Git | リポジトリのクローンに必要 |
| uv | Python環境管理 (MLXDINOv3 の重み変換スクリプトに使用) |

---

## Step 1 — リポジトリのクローン

```bash
git clone https://github.com/techino555/PowerPredictor.git
cd PowerPredictor
```

クローン後のディレクトリ構成を確認します。

```
PowerPredictor/
├── NOTICE                  # ← 必ず読むこと（ライセンス・許諾情報）
├── LICENSE
├── README.ja.md
├── source/                 # Xcodeプロジェクト本体
├── skills/
│   └── buildtoiOS/         # 本スキル
├── docs/
└── ...
```

---

## Step 2 — 事前許諾が必要なリソースの取得

> **⚠️ 重要 — NOTICEファイルを必ず確認してください**
> `NOTICE` ファイルには、本アプリが利用するサードパーティ製品のライセンスと
> 許諾要件が記載されています。ビルド前に必ず目を通し、各許諾手続きを完了してください。

### 2-1. DINOv3 モデル重みの取得（Meta への申請が必要）

DINOv3はMeta Platforms, Inc.の独自ライセンス下で公開されています。
**モデル重みを取得するには、Metaの公式フォームから申請が必要です。**

1. 以下のURLにアクセスしてアクセス申請フォームを送信します。
   ```
   https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
   ```
2. 申請後、メールで承認通知が届きます（通常数時間〜数日）。
3. 承認されたら、`dinov2_vitb14` の重みファイルをダウンロードします。

> **帰属表示の義務**
> DINOv3 License の条件として、再配布・派生物に `"Built with DINOv3"` の
> 帰属表示を含める必要があります。

### 2-2. スカウター効果音の取得（ニコニ・コモンズ）

効果音はニコニ・コモンズからダウンロードします。商用利用・改変などの制限は
作者ページの利用規約を確認してください。

1. 以下のURLにアクセスします。
   ```
   https://commons.nicovideo.jp/works/nc116296
   ```
2. ページの利用規約を確認・同意します。
3. 音声ファイル（.wav）をダウンロードします。
4. ダウンロードしたファイルをプロジェクトの以下のパスに配置します。
   ```
   Resources/Sounds/
   ```

> **ニコニコ関連プラットフォームに投稿する場合**
> `nc116296` を親作品として子作品登録（親作品登録）が必要です。

---

## Step 3 — MLXDINOv3の取得、モデル重みファイルの生成

PowerPredictorは [MLXDINOv3](https://github.com/vincentamato/MLXDINOv3)（MIT License）を
使用して、Apple Silicon 上で DINOv3 推論を行います。MLXDINOv3のツールで生成するDINOv3のPyTorch重みをMLX/safetensors形式に変換します。

### 3-1. Python 環境のセットアップ（uv を使用）

```bash
# uv がない場合はインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# 変換スクリプト用の仮想環境を作成
uv venv .venv
source .venv/bin/activate

# 必要パッケージをインストール。training/resources配下にあるpyproject.tomlを配置し、実行
uv sync
```

### 3-2. 重みファイルの変換

MLXDINOv3 リポジトリのスクリプトを使い、PyTorch 形式 → safetensors 形式へ変換します。

```bash
# MLXDINOv3 をクローン（別ディレクトリ推奨）
git clone https://github.com/vincentamato/MLXDINOv3.git /tmp/MLXDINOv3

# 変換スクリプトの実行（MLXDINOv3 のドキュメントに従って実施）
python /tmp/MLXDINOv3/convert.py \
  --input  /path/to/downloaded/dinov2_vitb14.pth \
  --output ./source/Models/dinov3-vitb16-mlx/
```

変換後、以下の2ファイルが生成されていることを確認します。

```
source/Models/dinov3-vitb16-mlx/
├── config.json
└── model.safetensors
```

> **変換スクリプトの詳細は MLXDINOv3 リポジトリの README を参照してください。**
> バージョンによってコマンドが異なる場合があります。

### 3-3. power_predictor.safetensors の配置

学習済みの戦闘力推論モデルは以下に配置します（リポジトリに同梱またはtraining skillで生成）。

```
Models/power_predictor.safetensors
```

配置後のModelsディレクトリ構成：

```
Models/
├── dinov3-vitb16-mlx/
│   ├── config.json
│   └── model.safetensors
├── linear/
└── power_predictor.safetensors
```

---

## Step 4 — Xcode でのビルド設定

### 4-1. Xcode でプロジェクトを開く

```bash
open source/PowerPredictor.xcodeproj
```

またはFinderから`source/`ディレクトリ内の`.xcodeproj`ファイルをダブルクリックします。

### 4-2. Swift Package の依存関係を解決する

Xcodeが開いたら、SPM（Swift Package Manager）が自動的に依存パッケージを取得します。
ネットワーク接続を確認し、以下のパッケージが取得されるまで待ちます。

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| mlx-swift-lm | 3.31.4 | MLX Swift (mlx-swift も包含) |
| MLXDINOv3 | 2.0.0 | DINOv3 推論 |
| swift-transformers | 1.3.3 | HuggingFace 互換トランスフォーマー |
| swift-huggingface | 0.9.0 | HuggingFace Hub アクセス |

> Xcode メニュー → `File` → `Packages` → `Resolve Package Versions` で手動解決もできます。

### 4-3. Signing & Capabilities の設定

1. プロジェクトナビゲータで `PowerPredictor` プロジェクトをクリック
2. `Signing & Capabilities` タブを選択
3. **Team** で自分の Apple ID を選択（無料 Apple ID で可）
4. **Bundle Identifier** を一意の値に変更（例: `com.yourname.PowerPredictor`）
5. `Automatically manage signing` にチェックが入っていることを確認

### 4-4. カメラ権限の確認

`Info.plist` に以下のキーが含まれていることを確認します（通常はリポジトリ内に含まれています）。

```xml
<key>NSCameraUsageDescription</key>
<string>カメラを使って人物の戦闘力を測定します。</string>
```

### 4-5. 必要なファイルのTarget Membershipの追加

下記3つはAppでもTrainingでも活用するため、両方のTargetでMemberShipに含まれている必要があります。

- PowerPredictor/DINOv3/DINOv3Encoder.swift
- PowerPredictor/Models/dinov3-vitb16-mlx/config.json
- PowerPredictor/Models/dinov3-vitb16-mlx/model.safetensors

確認方法

1. Xcodeで上記ファイルを選択
2. 右上のFile Inspectorを開く
3. Target Membership を確認
4. PowerPredictorとPowerPredictor_feature_extのターゲットにチェックが入っているか確認


---

## Step 5 — iPhone実機へのビルド・インストール

### 5-1. iPhoneをMacに接続

1. LightningまたはUSB-CケーブルでiPhoneをMacに接続します。
2. iPhoneの画面に「このコンピュータを信頼しますか？」と表示されたら **「信頼」** をタップします。

### 5-2. ビルドターゲットを iPhone に変更

Xcode上部のデバイス選択ドロップダウンから、接続したiPhoneを選択します。

```
[PowerPredictor] > [あなたのiPhone名]
```

### 5-3. ビルドを実行

`⌘ + R` を押すか、メニュー `Product` → `Run` を選択します。

初回ビルドは依存パッケージのコンパイルが含まれるため **10〜20分程度** かかる場合があります。

### 5-4. iPhone 上での開発者アプリの信頼設定

無料Apple IDを使用している場合、iPhone側で信頼設定が必要です。

1. iPhone の `設定` → `一般` → `VPNとデバイス管理` に移動します。
2. 自分の Apple ID が表示されているデベロッパ App をタップします。
3. **「[Apple ID]を信頼」** をタップします。

---

## Step 6 — インストールの確認

iPhone のホーム画面に **PowerPredictor** アプリのアイコンが表示されていれば成功です。

アプリを起動し、以下を確認してください。

- [ ] カメラが起動し、プレビューが表示される
- [ ] 人物を映すと検出枠（HUD）が表示される
- [ ] `SCAN` ボタンをタップするとスキャン音が鳴る
- [ ] 戦闘力スコアが画面に表示される

---

## トラブルシューティング

### ビルドエラー: `model.safetensors not found`

`source/Models/dinov3-vitb16-mlx/model.safetensors` が存在するか確認してください。
Step 3 のモデル変換が完了していない可能性があります。

### ビルドエラー: `Signing certificate ... has expired`

無料Apple ID の証明書は7日で失効します。Xcodeから再度 `Product` → `Run` を実行して
証明書を更新してください。

### iPhone に転送されない: `Unable to install...`

`設定` → `一般` → `VPNとデバイス管理` でデベロッパアプリの信頼設定が完了しているか
確認してください（Step 5-4）。

### SPM が解決できない: `network error`

インターネット接続を確認し、`File` → `Packages` → `Resolve Package Versions` を再実行します。

---

## 参考

| リソース | URL |
|---------|-----|
| PowerPredictor GitHub | https://github.com/techino555/PowerPredictor/ |
| DINOv3 申請フォーム | https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ |
| MLXDINOv3 | https://github.com/vincentamato/MLXDINOv3 |
| 効果音 (nc116296) | https://commons.nicovideo.jp/works/nc116296 |
| MLX Swift | https://github.com/ml-explore/mlx-swift |
| Xcodeダウンロード | https://developer.apple.com/xcode/ |
