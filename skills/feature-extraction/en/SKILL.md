---
name: feature extraction
description: >
  A skill for creating training datasets.
  Uses the macOS Swift app "PowerPredictor-feature-ext" to batch-extract 768-dimensional DINOv3 embeddings from a folder of person images and export them to a CSV.
  Must be executed before training the PowerPredictor Linear Model.
---

Major changes planned.


# Feature Extraction Skill

## Overview

A preprocessing pipeline for generating training data for PowerPredictor.  
Simply build and launch the macOS app (`PowerPredictor_feature_ext`) in Xcode, select a folder containing person images, and it will batch-extract **DINOv3 CLS Tokens (768 dimensions)** and save them to a CSV file.

```
Folder of Person Images
      │
      ▼
PowerPredictor_feature_ext (macOS App)
      │
      ├── Vision Framework (PersonDetector → PersonSegmenter)
      │       └─ Person detection & background removal
      ├── ImagePreprocessor
      │       └─ NSImage → MLXArray
      └── DINOv3Encoder
              └─ CLS Token extraction (768 dimensions)
      │
      ▼
embeddings_YYYYMMDD_HHmmss.csv
(filename, person_id, power, body_direction, take, distance, posture, gender, e0…e767)
```

---

## File Structure

```
source/PowerPredictor_feature_ext/
├── PowerPredictor_feature_extApp.swift # @main entry point
├── ContentView.swift          # UI + TrainingViewModel (processing flow control)
├── EmbeddingExtractor.swift   # Extraction pipeline integration class
├── ImagePreprocessor.swift    # NSImage → MLXArray conversion
├── Vision/
│   ├── VisionPipeline.swift   # Overall Vision processing control
│   ├── PersonDetector.swift   # Person bounding box detection
│   └── PersonSegmenter.swift  # Person mask generation & background coloring
└── Dataset/
    └── CSVExporter.swift      # Embedding → CSV export
```

---

## Usage (Step-by-Step)

### Step 1: Build & Run in Xcode

```
Open source/PowerPredictor.xcodeproj in Xcode
  ↓
Set scheme to "PowerPredictor_feature_ext"
  ↓
Set destination device to "My Mac"
  ↓
Press ⌘R to build & run
```

> **Dependencies**: Ensure `MLX` and `MLXDINOv3` are resolved as Swift Packages.  
> Place the DINOv3 model files (`dinov3-vitb16-mlx/`) in the specified path within the project.

---

### Step 2: Select Image Folder

- Click the "**Select Folder**" button.
- Choose a folder containing jpg / jpeg / png images (subfolders are scanned recursively).
- Once selected, the image count will be displayed in the status area.

---

### Step 3: Generate Embeddings

- Click the "**Generate Embeddings**" button.
- The following steps run sequentially for each image:

| Step | Processing Class | Description |
|---|---|---|
| 1 | `PersonDetector` | Detects persons using `VNDetectHumanRectanglesRequest` |
| 2 | `PersonSegmentProcessor` | Generates person masks using `VNGeneratePersonSegmentationRequest` |
| 3 | `ImagePreprocessor` | Converts `NSImage → MLXArray` (using `MLXDINOv3.ImageProcessor`) |
| 4 | `DINOv3Encoder` | Extracts CLS Token (768 dimensions) |

- Images where no person is detected are skipped (logged).
- Progress can be tracked in real-time via the progress bar.

---

### Step 4: Save CSV

- Click the "**Save CSV**" button.
- Saved to `~/Documents/embeddings_YYYYMMDD_HHmmss.csv`.
- After saving, the file is automatically selected and highlighted in Finder.

---

## Output CSV Format

| Column Name | Type | Description |
|---|---|---|
| `filename` | String | Relative path from the input folder |
| `person_id` | String | Person ID (blank; fill in manually later) |
| `power` | String | Battle power label (blank; fill in manually later) |
| `body_direction` | String | Body direction (blank; fill in manually later) |
| `take` | String | full or half (blank) |
| `distance` | String | Shooting distance (blank) |
| `posture` | String | Posture (blank) |
| `gender` | String | Gender (blank) |
| `e0` … `e767` | Float | DINOv3 CLS Token (768 dimensions) |

> **Important**: After generating embeddings, manually enter battle power scores into the `power` column before using it for training.

---

## Internal Processing Details

### EmbeddingExtractor (Extraction Pipeline)

```swift
// NSImage → CGImage → VisionPipeline → ImagePreprocessor → DINOv3Encoder → CLS Token
func extract(from image: NSImage) async throws -> MLXArray? {
    let (_, _, completeImage) = try await visionPipeline.process(image: image)
    guard let personImage = completeImage else { return nil }
    let input = try preprocessor.preprocess(personImage)
    return encoder.encode(input)
}
```

### VisionPipeline Output

`process(image:)` returns three values:

| Return Value | Type | Purpose |
|---|---|---|
| `boundingBox` | `CGRect?` | Normalized bounding box coordinates of the person |
| `analysisImage` | `NSImage?` | Image with analyzing effect (person: ochre / background: yellow) |
| `completeImage` | `NSImage?` | Input image for DINOv3 (with background yellow filter applied) |

> `completeImage` (with the yellow background filter applied) is used for embedding extraction.

---

## Common Errors & Troubleshooting

| Error | Cause | Solution |
|---|---|---|
| `Failed to load image` | Corrupted file / Unsupported format | Only jpg/jpeg/png are supported. Verify the image file. |
| `Embedding extraction failed (No person detected)` | Vision could not detect a person | The person might be too small, side-facing, or non-frontal. |
| `Failed to load DINOv3 model` | Model files not found | Verify the deployment path of `dinov3-vitb16-mlx/`. |
| CSV is empty | All images were skipped | Use images where persons are clearly visible. |

---

## Next Steps

Once the CSV is generated, proceed to the following skills:

```
feature extraction (this skill)
      ↓ embeddings_YYYYMMDD_HHmmss.csv
      ↓ Manually populate the power column
training (Train the Linear Model with Python + MLX; see the training folder)
      ↓ power_predictor.safetensors
inference (Integrate into iOS app → Display battle power)
```
