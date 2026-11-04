# Model Training (Establishing the Battle Power Model)

This document explains the training procedures and underlying design philosophy for the model that computes accurate battle power scores from 768-dimensional DINOv3 Embeddings.

## Training Purpose and Division of Responsibilities

When starting to study machine learning through this project, there was an initial question: "Is a model program code or a data file?" The conclusion reached was that the model structure (the mathematical formula converting a 768-dimensional input into a 1-dimensional scalar) is "code", while the weight parameters optimized through training are "data (safetensors file)".

With this understanding established, development responsibilities were clearly divided across environments:

 * Feature Extraction (Mac): Extract embeddings in batch from DINOv3 using Swift and save them as a CSV dataset.
 * Model Training (Mac): Perform supervised training of a linear model using Python and MLX.
 * In-App Inference (iPhone/iPad): Load the trained weights (safetensors) into the Swift app and run high-speed inference.

We considered whether training should also be done in Swift or on a Mac using MLX. Investigating embedding datasets revealed that typical public datasets rarely distribute embeddings directly, as they cannot predict which foundation model users will employ. However, when fixing DINOv3 as the feature extractor as in this app, training on pre-extracted embeddings rather than raw images is highly effective. Therefore, we chose MLX on a local Mac for training—which completes in seconds—and reserved Swift for inference (battle power display) on iPhone/iPad.

## Feature Extraction during Training vs. Inference

### Standard Object Detection vs. Feature Extraction in This App

In Vision Transformer models like DINOv3, the output CLS (Classification) token essentially functions as a representative feature vector (embedding) for the entire image. In typical applications, characteristics of the entire input photo are aggregated into this single vector. In short, features of the whole photograph are normally extracted.

**[Training Photo](../../resources/origin.png)**

However, in this scouter app, to prevent inference results from being skewed by background scenery, we incorporated a major enhancement into the feature extraction pipeline. Specifically, Apple's Vision Framework analyzes the image beforehand and crops the person region before feeding it into DINOv3, as illustrated below:

```text
Photo
 ↓
Person Cropping
 ↓
DINOv3
 ↓
CLS Token
 ↓
Linear Regression Model
 ↓
Battle Power
```

Through this custom pipeline, high-purity embeddings focusing specifically on the human subject rather than the entire scene could be extracted—essentially extracting embeddings from within the red bounding frame.

### The Major Pitfall: Preprocessing Mismatch between Training and Inference

During the training process, an essential and hard-learned lesson was the "mismatch between training image preprocessing and inference image preprocessing."

Initially, collected person images were directly fed into DINOv3 to generate embeddings for the CSV dataset, as shown in the diagram.

**[Diagram showing detection area](../../resources/mantracking_old.png)**

However, when weights trained on this CSV were integrated into the iOS app and tested on a physical device, the inferred battle power deviated drastically from expectations.

Investigating the cause revealed that, to emulate the original scouter, the iOS app applied a visual effect before inference: "light yellow background with an ochre-tinted person." A sensitive Vision Transformer like DINOv3 is highly reactive to shifts in color tone and contrast. Because the model was trained on "raw photos" while the actual runtime code supplied "ochre-tinted photos", the vectors shifted in feature space, failing to produce accurate scores. This demonstrated just how sensitive such classification/detection foundation models are.

To resolve this discrepancy, the Mac feature extraction tool (Swift) was refactored to generate embeddings from images processed with the exact same tinting and cropping filters as the iOS device. Aligning preprocessing conditions completely yielded stable battle power predictions on physical devices. We applied this processing to photos, generated embeddings from this state, and utilized them for training.

**[Analyzed image during app runtime](../../resources/mantracking.png)**

## Training Image Data and Embedding Extraction App

For training data, person images with diverse body types and postures (full body and upper body) were collected from open datasets such as Hugging Face and Kaggle, as well as personal photos.

To extract features under conditions identical to the iOS app, a native macOS "Feature Extraction Tool" was built in Swift. It follows the scouter app's processing flow up to CLS token extraction, allowing users to select images and export their embeddings to a CSV file. Training proceeded using this CSV containing embeddings, IDs, and labeled battle power values. While computer vision tasks often train directly on images, this approach proved far more efficient when using a fixed feature extractor.

### Porting Code to macOS and Dataset Operations

To extract embedding data for training, the `VisionPipeline` and `ImagePreprocessor` created for the iOS app were ported to macOS. Key modifications included:

 * Eliminating UIKit dependencies: Replacing `UIImage` with `NSImage`.
 * Unifying segmentation: Applying identical scouter image processing before tensor conversion.
 * Implementing `CSVExporter`: Outputting 768-dimensional vectors as `embedding.csv` to match with labeled `power.csv`.

On the Python side, flexible data handling was conducted, such as backfilling specific CSV columns to enrich the dataset. Furthermore, by capturing 10 photos of the same individual across varied angles and distances, we verified how stable the cosine similarity of CLS tokens remained with and without Vision Framework cropping, demonstrating that cropping was indispensable for feature stabilization.

## Linear Regression Training

Now for the training details. Using the collected feature CSV, the battle power prediction model was trained.

The model equation is simple:

```text
y = x · W + b
```

Here, `x` is the 768-dimensional embedding vector from DINOv3, `W` represents the 768 trainable weight parameters, `b` is bias (offset), and `y` is estimated battle power (configured in the range 1–99).

### Choice of Loss Function: MSE Loss vs. Huber Loss

Selecting the loss function directly influences model behavior. Mean Squared Error (MSE Loss), standard for regression tasks, was adopted first.

While MSE Loss is simple to calculate and differentiate, it has a significant flaw: squaring errors means a single severe outlier can disproportionately skew the entire model.

Our initial dataset was skewed, dominated by average builds with very few high-power athletic builds. With MSE, large errors on those rare high-power samples dominated penalties, sacrificing prediction accuracy across intermediate ranges.

This led us to evaluate Huber Loss. Huber Loss is a hybrid function that applies quadratic penalties like MSE for small errors and switches to linear penalties like MAE (Mean Absolute Error) for large errors. Because its behavior shifts at a delta threshold, it handles outliers gracefully: "this sample is indeed off, but will not pull the entire model toward it."

Comparing both on the same dataset showed visibly improved predictions. Under MSE, a subject labeled Power 5 frequently produced wide misses like 2.5–3.2. Switching to Huber Loss converged predictions for the same person to 4.2–5.4, close to ground truth. Huber Loss was therefore adopted as standard.

### Training Loop Implementation

Below is the complete MLX training loop. Python's MLX framework offers a NumPy-like API while providing fast gradient computations taking advantage of Apple Silicon unified memory.

**train_loop**

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

model = nn.Linear(768, 1)
learning_rate = 0.001
optimizer = optim.Adam(learning_rate=learning_rate)

def huber_loss(pred, target, delta=1.0):
    diff = mx.abs(pred - target)
    return mx.mean(
        mx.where(diff < delta,
                 0.5 * mx.square(diff),
                 delta * (diff - 0.5 * delta))
    )

def loss_fn(model, x, y):
    pred = model(x)
    return huber_loss(pred, y)

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

for epoch in range(100):
    loss, grads = loss_and_grad_fn(model, X_train, y_train)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
```

`nn.value_and_grad` computes loss and gradients simultaneously in a single forward/backward pass. Adam, commonly used for Transformer training, was selected as the optimizer, and L2 regularization via weight decay suppressed overfitting. Training 1,000 samples for 200 epochs on a MacBook Air (M1, 8GB) completed in just a few seconds.

### Impact of Learning Rate: Differences across Values

Among hyperparameters, Learning Rate has the greatest influence on convergence behavior. In our experiments, comparing learning rates 0.01 and 0.001 was informative.

With learning rate 0.01, Train Loss dropped rapidly to 0.03 within 100 epochs. However, Validation Loss bottomed out around epoch 30 at 0.94 and subsequently climbed, ultimately reaching 1.14. With learning rate 0.001, Train Loss decreased moderately to 0.27 at 100 epochs, but Validation Loss stabilized at 0.83.

**Comparison of Training Results by Learning Rate**

| Learning Rate | Train Loss (100 epochs) | Val Loss (100 epochs) |
| :--- | :--- | :--- |
| 0.01 | 0.03 | 1.14 |
| 0.001 | 0.27 | 0.83 |

This provided an important lesson: lowering Train Loss is not the goal. Train Loss merely indicates how well the model explains training data; the ultimate goal is how well it predicts unseen photos—namely, low Validation Loss. Extremely low Train Loss coupled with high Validation Loss is typical overfitting.

### Combating Overfitting: Finding the Optimal Epoch Count

Increasing epochs to improve fit eventually leads Validation Loss to hit bottom and start rising. Recognizing this turning point is crucial.

Below is an actual training run with an expanded 1,234-sample dataset:

```text
Epoch  80, Train Loss: 1.1623, Val Loss: 1.6088
Epoch  90, Train Loss: 1.1092, Val Loss: 1.5982
Epoch 100, Train Loss: 1.0649, Val Loss: 1.5912
Epoch 110, Train Loss: 1.0272, Val Loss: 1.5840
Epoch 120, Train Loss: 0.9949, Val Loss: 1.5799
Epoch 130, Train Loss: 0.9675, Val Loss: 1.5783
Epoch 140, Train Loss: 0.9440, Val Loss: 1.5783
Epoch 150, Train Loss: 0.9236, Val Loss: 1.5788
```

While Train Loss consistently decreased, Validation Loss bottomed out around epochs 130–140 and started rising slightly after 150. The epoch where Validation Loss is minimized represents the point of highest generalization performance. The weights at this Best Epoch are saved as `power_predictor.safetensors`.

In an experiment pushing epochs up to 1,000, overfitting risks became even more apparent. While Train Loss dropped to 30.9, Validation Loss failed to decrease, climbing steadily and generating negative predictions (-7.4, -76.9, etc.). More epochs are not necessarily better; proper early stopping is essential.

### Dataset Development: Improve Data before Complexifying Models

Across multiple iterations, the most effective improvement was not making the model complex (such as switching to an MLP), but improving dataset quality and quantity.

Initially, training was conducted only with personal photos, producing promising results with Power 5 predictions between 4.2 and 5.7. However, expanding the dataset with public samples resulted in a Validation MAE of 1.99, showing room for improvement.

We introduced MAE (Mean Absolute Error) analysis across individual Power brackets:

```text
Power  4 : Samples= 66, MAE=1.4842
Power  5 : Samples= 50, MAE=1.5099
Power  6 : Samples= 38, MAE=1.8787
Power 10 : Samples= 19, MAE=3.0647
Power 15 : Samples=  4, MAE=5.1857
```

This clearly showed that brackets with fewer samples exhibited larger MAE. Data imbalance, rather than model capacity, was throttling prediction accuracy.

This realization shifted our data strategy from "simply increasing volume" to "targeted collection for underrepresented Power ranges." Furthermore, intentionally varying capture conditions for the same individual (profile, distances, arm positions, mirror shots, holding smartphone, slight crouch, different lighting) increased diversity and bolstered generalization.

### Initial Physical Device Validation: Progress and New Challenges

Deploying trained weights onto the iOS app and testing on physical devices revealed:

 * Under conditions similar to full-body training photos, scores fluctuated around Power 5.
 * With only half the body visible, scores dropped to around Power 2.5.
 * Measuring via a mirror while holding an iPad resulted in scores in the low Power 4 range.

Device testing demonstrated that full-body framing ratio and shooting conditions significantly influence scores. Because DINOv3 embeddings capture general visual features across the entire image, orientation, pose, and framing ratio cause score fluctuations even on person-only images.

This reinforced the necessity of person cropping via Vision Framework described earlier. Removing background and normalizing person bounds before generating embeddings proved to be a prerequisite for consistent battle power prediction.

### Exporting safetensors and App Integration

Weights from the epoch with minimal Validation Loss are exported as `power_predictor.safetensors`, integrated into iOS, and executed via MLX Swift. The file size is only a few kilobytes, containing 768 weight parameters and 1 bias term (769 floating-point numbers in total).

Copying this lightweight safetensors file into the iOS bundle completes the scouter's intelligence, fusing DINOv3's visual understanding with customized scoring. The training cycle—data collection → embedding extraction → MLX training → safetensors export → app deployment—is lightweight enough to iterate repeatedly on a single MacBook Air, which is a major advantage for indie development.
