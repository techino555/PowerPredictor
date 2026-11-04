# Transition from VLM to Vision Encoder

Initial design concepts explored feeding images into an ultra-lightweight Vision Language Model (VLM) and having it output a numerical score directly via prompting. However, because a VLM is fundamentally a language model that predicts the "next token (word)", its outputs fluctuated heavily based on textual phrasing, making it difficult to produce stable, completely consistent numerical scores.

Further research pointed to an architecture where "visual features are condensed into numerical vectors (Image Embeddings) and fed into a small regression model (MLP or linear layer) to calculate scores." We learned that extracting Image Embeddings requires a Vision Encoder. To use an analogy: the Vision Encoder acts as a "photographer" observing the subject closely, the extracted embedding is a "feature memo (multi-dimensional numbers)", and the downstream model acts as a "judge" evaluating that memo to determine battle power.

## Selecting a Vision Encoder Model

Recognizing the need for a Vision Encoder, candidate models were surveyed and compared:

**Comparison of Vision Encoder Candidates**

| Model | Swift Support | Core ML Support | Overall Evaluation |
| :--- | :--- | :--- | :--- |
| CLIP | 3/5 | 3/5 | Good |
| SigLIP | 4/5 | 4/5 | Good |
| DINOv3 | 5/5 | 5/5 | Best (Adopted) |
| AIMv2 | 2/5 | 2/5 | Developing |

CLIP and SigLIP feature strong global semantic representations from contrastive image-text training. However, reproducing the scouter required identifying fine-grained anatomical differences in real humans, such as muscle definition and body thickness. Meta's "DINOv3 (ViT-B/16)", a Vision Transformer trained via self-supervised learning, proved superior in texture, geometric shape, and local resolution (dense performance). Therefore, DINOv3 was selected. AIMv2, a Vision Encoder promoted by Apple's ML research team, was also evaluated, but its larger model size was deemed unsuitable for this use case.

## Trial and Error with Model Loading and Weight Extraction

### Loading the Model and Verifying Features

Starting from fundamentals, we verified whether CLS tokens (768-dimensional visual features) could be extracted as expected by locally loading the official DINOv3 model from Hugging Face (`facebook/dinov3-vitb16-pretrain-lvd1689m`) in a Python / PyTorch environment:

```python
# Validation of DINOv3 Embedding Extraction in Python
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

**Sample Output**

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
...
(mlx-image-test) hide@MacBookAir mlx-image-test %
```

The output values like `5.3363e-01` are the features, outputting across 768 columns representing 768 dimensions.

### Can This Model Run in Swift?

While the model loaded successfully, we needed to know if it could run with MLX Swift. Fortunately, we discovered a tool that can extract DINOv3 model weights for MLX Swift: the `MLXDINOv3` repository by vincentamato on GitHub. Without this tool, features estimating battle power from human characteristics could not be produced, making it essential for the scouter.

While the `MLXDINOv3` repository provides the code to run DINOv3 in Swift, it does not include pre-trained weight files. Thus, Meta's PyTorch weights had to be converted into an MLX-compatible format. The repository provided conversion scripts to transform the base model from Hugging Face for MLX:

**Running Weight Conversion to MLX Format**

```bash
./mlx-run.sh Convert facebook/dinov3-vitb16-pretrain-lvd1689m 
./Models/dinov3-vitb16-mlx
```

Despite warnings during build, 210 weight tensors were successfully converted into MLX format and output:

**Weight Conversion Execution Log**

```text
...
Many warnings
...
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

Next, the generated weights were loaded temporarily on Mac and tested with sample images. The critical question was: "Can 768-dimensional embeddings genuinely be obtained in Swift through DINOv3 inference?"

**Verification Output**

```text
[1, 768]
[1, 1]
array([[0.182126]], dtype=float32) 
```

The output confirmed `[1, 768]` as expected. Furthermore, connecting that 768-dimensional vector into a Linear layer that converts it into a single battle power scalar confirmed a `[1, 1]` score array. In essence, the above represents "battle power displayed as 0.182126 in an untrained state."

This confirmed the pipeline's viability: through repeated training, battle power could be derived from human features. Running the minimal scouter setup—inputting an image to DINOv3, extracting 768-dimensional features, and passing them to Linear to output a score—entirely in Swift was the core breakthrough. Discovering that it could also run on iOS gave us the confidence to embark on full app development.

Following this validation, we established the approach of converting PyTorch weights into safetensors format directly readable by MLX and bundling them in the iOS app.

### Code for Model Loading and Feature Output

Implementation of `DINOv3Encoder` handling model loading and inference:

```swift
import Foundation
import MLX
import MLXDINOv3

final class DINOv3Encoder {
    private let model: DinoVisionTransformer

    init() throws {
        // Load weights and config from app bundle resource path
        guard let modelPath = Bundle.main.resourcePath else {
            throw NSError(domain: "ModelPathError", code: -1)
        }
        
        model = try DinoVisionTransformer.loadPretrained(from: modelPath)
    }

    /// Receives MLXArray tensor and outputs 768-dimensional Embedding vector
    func encode(_ input: MLXArray) -> MLXArray {
        let output = model(input)
        // Extract [CLS] token features
        return output.clsToken
    }
}
```

## EmbeddingPipeline Architecture

The overall inference pipeline combines person localization via Vision Framework and feature extraction via DINOv3:

```text
Camera Image
  ↓
Vision Framework (PersonDetector / PersonSegmenter)
  ↓
Person Detection & Foreground Cropping (cropPerson)
  ↓
Person-Only Image
  ↓
ImagePreprocessor (Normalization)
  ↓
DINOv3 (DinoVisionTransformer)
  ↓
CLS Token (768-dimensional Embedding)
  ↓
Predictor (Linear)
  ↓
Battle Power Score
```

Adhering strictly to Single Responsibility Principle (SRP), classes were organized as `EmbeddingPipeline`:

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

    /// Generates 768-dimensional embedding from cropped person image
    func process(personImage: UIImage) throws -> MLXArray {
        // 1. Preprocess into tensor (normalization)
        let inputTensor = preprocessor.preprocess(personImage)
        
        // 2. Run DINOv3 inference (extract CLS token)
        let embedding = encoder.encode(inputTensor)
        
        return embedding
    }
}
```

This established the infrastructure to instantly transform human regions captured from camera frames into 768-dimensional feature vectors usable in MLX Swift.

[References]

AIMv2: https://github.com/apple-aiml-research/ml-aim

DINOv3: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m

MLXDINOv3: https://github.com/vincentamato/MLXDINOv3
