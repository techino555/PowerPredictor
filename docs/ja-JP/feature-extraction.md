# VLMからVision Encoderへの転換

当初の設計構想では、超軽量なビジョン言語モデル（VLM）に画像を入力し、プロンプトによって直接数値を算出させる方法を検討していた。しかし、VLMは本質的に「次のトークン（単語）を予測する」言語モデルであり、テキスト表現に引っ張られて出力がブレやすく、毎回完全に一貫した数値を安定して出力させることは極めて困難だった。

そこで色々調査をしたところ、「画像から視覚的特徴を凝縮した数値ベクトル（Image Embedding）を抽出し、そのベクトルを小さな回帰モデル（MLPや線形層）に入力してスコアを計算する」というアーキテクチャに辿り着いた。そしてImage Embeddingを抽出するためにはVision Encoderが必要ということも知った。人間に例えるなら、Vision Encoderは被写体をくまなく観察する「カメラマン」であり、抽出されるEmbeddingは「撮影された人物の特徴メモ（多次元数値）」だ。そして後段のモデルが、そのメモを評価して戦闘力を判定する「審査員」の役割を果たすものとなる形だ。

## Vision Encoderモデルの選定

Vision Encoderが必要、ということがわかり採用可能なモデルを調査・比較した。

**Vision Encoder候補の比較**

| モデル名 | Swift対応 | Core ML対応 | 総合評価 |
| :--- | :--- | :--- | :--- |
| CLIP | 3/5 | 3/5 | 良好 |
| SigLIP | 4/5 | 4/5 | 良好 |
| DINOv3 | 5/5 | 5/5 | 最良（採用） |
| AIMv2 | 2/5 | 2/5 | 発展途上 |

この中でCLIPやSigLIPは画像とテキストの対比学習によって優れた大域的意味表現を持つ特徴がある。本家のスカウターに近づけるために必要なのは「筋肉のカットや体型の厚み」など実際の人間的な特徴の微細な差異を識別する能力と考えた。その点、自己教師あり学習で訓練されたVision TransformerであるMeta社の「DINOv3（ViT-B/16）」は、テクスチャや物体の形状、パーツごとの局所的な解像度（Dense性能）において他のモデルよりも優位である、と知った。そこで本アプリではDINOv3を採用することにした。なお、もう一つAIMv2というAppleのML開発チームが推進しているVision Encoderがあることを知ったが、モデルサイズが大きく今回の用途にはそぐわないと判断した。

## モデルのロードと重み抽出の試行錯誤

### モデルのロードと特徴量の確認

基本に立ち返り、PythonとPyTorch環境下でHugging Faceに公開されている公式のDINOv3モデル（facebook/dinov3-vitb16-pretrain-lvd1689m）をローカルにロードし、下記のサンプルコードでCLSトークン（768次元の画像特徴量）が期待通りに抽出できるかを検証することからスタートした。下記が端末でPythonで実行し、成功した際の出力結果である。

```python
# PythonでのDINOv3 Embedding抽出の事前検証
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
model.eval()

image = Image.open("sample.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.inference_mode():
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0]

print("CLS Embedding shape:", cls_embedding.shape) # (1, 768)
```

**結果出力例**

```text
(mlx-image-test) hide@MacBookAir mlx-image-test % uv run main.py
preprocessor_config.json: 100%|█████████████████████████████████████████████████████████████████████| 585/585 [00:00<00:00, 1.22MB/s]
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████| 744/744 [00:00<00:00, 1.27MB/s]
model.safetensors: downloading bytes: ███████████████████████████████████████████████████████████████████████████|  325MB, 7.30MB/s
model.safetensors: reconstructing file: 100%|███████████████████████████████████████████████████████████|  343MB /  343MB, 24.2MB/s
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████| 211/211 [00:00<00:00, 53445.14it/s]
CLS Embedding shape: torch.Size([1, 768])
tensor([[ 5.3363e-01, -1.6388e+00, -9.6853e-01, -4.5372e-02,  4.7354e-01,
         -6.8991e-01,  9.4870e-01,  2.3846e-01,  1.5063e+00, -2.8736e-01,
          8.0710e-01,  2.7081e-01, -4.7755e-01,  1.3042e+00,  3.2303e-01,
          2.1479e-02,  3.9907e-02,  7.8938e-01, -9.5227e-02, -7.9796e-04,
         -7.0539e-01, -7.6839e-01, -5.1912e-01, -1.7844e+00, -2.2501e-01,
         -2.9161e-01,  1.4622e-01,  3.3855e-01, -5.7057e-02, -1.1111e-01,
          6.4662e-01,  9.1380e-01, -1.0130e+00,  1.4300e+00,  1.0324e+00,
・
・
(mlx-image-test) hide@MacBookAir mlx-image-test %
```

この5.3363e-01、などと出力された数値が特徴量である。これが全部で768次元を表す、768列出力されることがわかった。

### このモデルをSwiftで実行可能か

モデルはロードできたものの、このモデルでMLX Swiftで動くものはあるのか、というのが疑問だったが幸いMLX SwiftでDINOv3のモデルの重みを抽出可能なツールを発見した。GitHubのvincentamato氏によるリポジトリのMLXDINOv3である。このツールがないと人物の特徴から戦闘力を推定する特徴量が作れないので結果的にはスカウターに必須のアプリである。

MLXDINOv3リポジトリは、SwiftでDINOv3を動かすためのプログラム本体は提供しているものの、学習済みの「重み（Weights）」データ自体は含まれていなかった。そのため、Meta社が公開しているPyTorch形式の重みを、MLXが解釈できる形式へと自力で変換する必要があった。これについてもこのリポジトリはクリアしており、具体的にはリポジトリに付属していた変換スクリプトを用いて、Hugging Face上のベースモデルをMLX向けに変換する方法を提供していたのでそれを実行した。

**MLX形式への重み変換の実行**

```bash
./mlx-run.sh Convert facebook/dinov3-vitb16-pretrain-lvd1689m 
./Models/dinov3-vitb16-mlx
```

ビルド時にいくつかの警告は出力されたものの、無事に210個の重みテンソルがMLX形式に変換され、モデルデータが出力された。ちなみにこのような出力がされた。

**重み変換の実行ログ**

```text
・
・
たくさんの警告
・
・
Converting facebook/dinov3-vitb16-pretrain-lvd1689m to MLX format
Downloading model from Hugging Face...
Loading PyTorch weights...
Loaded 211 weight tensors from PyTorch model
Model type: dinov3_vit
Converting weights...
Converted 210 weight tensors to MLX format
Saving config to /Users/hide/Desktop/gitclone/PowerPredictor/develop-plan/resources/
MLXDINOv3/Models/dinov3-vitb16-mlx/config.json
Saving weights to /Users/hide/Desktop/gitclone/PowerPredictor/develop-plan/resources/
MLXDINOv3/Models/dinov3-vitb16-mlx/model.safetensors
Model saved successfully to ./Models/dinov3-vitb16-mlx
Conversion complete!
hide@MacBookAir MLXDINOv3 %
```

次に、生成した重みをMac上で一時的に読み込み、テスト画像を入力して動作確認を行った。ここで最も重要だったのは、「DINOv3の推論により、本当に768次元のEmbeddingがSwift上で得られるのか」という点である。

**動作確認の出力結果**

```text
[1, 768]
[1, 1]
array([[0.182126]], dtype=float32) 
```

実行結果には期待通りに[1, 768]が出力された。さらに、その768次元のベクトルを受け取り、最終的な戦闘力として1つの数値に変換するLinear（全結合層）へと繋いだところ、[1, 1]の配列としてスコアが算出されることまで確認できた。要するに上記は「何も学習していない状態だと戦闘力が0.182126で表示される状態」である。

これを学習を積み重ね、人の特徴により戦闘力が算出されるようになる、という流れができることをこの結果から確信した。「画像をDINOv3に入力して768次元の特徴量を抽出し、それをLinearに渡して1つのスコアを出す」というスカウターの最小構成をSwiftのみで動かせたことは、本アプリ開発のまさに「肝」となるブレイクスルーであった。調べてみるとiOSでも動くことがわかり、スカウターアプリが出来る、と確信したので本格的にアプリ開発に取り掛かることにした。

この実証成功を受けて、最終的にはPyTorchの重みをMLXが直接解釈できるsafetensors形式へとコンバートし、iOSアプリのバンドル内へ組み込む方針を確立した。

### モデルのロードと特徴量出力のコード

DINOv3モデルのロードと推論を担うDINOv3Encoderの実装は以下のようになった。

```swift
import Foundation
import MLX
import MLXDINOv3

final class DINOv3Encoder {
    private let model: DinoVisionTransformer

    init() throws {
        // バンドル内のリソースパスから重みと設定をロード
        guard let modelPath = Bundle.main.resourcePath else {
            throw NSError(domain: "ModelPathError", code: -1)
        }
        
        model = try DinoVisionTransformer.loadPretrained(from: modelPath)
    }

    /// MLXArrayテンソルを受け取り、768次元のEmbeddingベクトルを出力
    func encode(_ input: MLXArray) -> MLXArray {
        let output = model(input)
        // [CLS] トークンの特徴量を取り出す
        return output.clsToken
    }
}
```

## EmbeddingPipelineの設計

全体の推論フローは、Vision Frameworkによる人物特定とDINOv3による特徴量抽出がパイプラインとして結合された形で設計した。

```text
カメラ画像
  ↓
Vision Framework（PersonDetector / PersonSegmenter）
  ↓
人物検出 & 前景切り抜き（cropPerson）
  ↓
人物だけの画像
  ↓
ImagePreprocessor（正規化）
  ↓
DINOv3（DinoVisionTransformer）
  ↓
CLS Token（768次元Embedding）
  ↓
Predictor（Linear）
  ↓
戦闘力スコア
```

クラス設計においても単一責任の原則（SRP）を徹底し、以下のように責任を分割してEmbeddingPipelineとしてまとめ上げた。

```swift
import Foundation
import UIKit
import MLX

final class EmbeddingPipeline {
    private let preprocessor = ImagePreprocessor()
    private let encoder: DINOv3Encoder

    init() throws {
        self.encoder = try DINOv3Encoder()
    }

    /// 切り抜かれた人物画像から768次元のEmbeddingを生成
    func process(personImage: UIImage) throws -> MLXArray {
        // 1. テンソルへの前処理（正規化）
        let inputTensor = preprocessor.preprocess(personImage)
        
        // 2. DINOv3による推論実行（CLS Token抽出）
        let embedding = encoder.encode(inputTensor)
        
        return embedding
    }
}
```

こうして、カメラ映像から切り出した人物領域を、MLX Swift上で扱える768次元の特徴量ベクトルへと瞬時に変換する基盤が整った。

【参考】

AIMv2 : https://github.com/apple-aiml-research/ml-aim

DINOv3 : https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m

MLXDINOv3 : https://github.com/vincentamato/MLXDINOv3
