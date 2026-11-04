---
name: build-to-iOS
description: 
  A step-by-step guide for building the PowerPredictor app onto a physical iPhone.
  Covers everything from environment preparation, acquiring and converting DINOv3 models, Xcode build configuration, to device installation.
---

# PowerPredictor — Building to a Physical iPhone

> **Overview**
> PowerPredictor is a scouter-style iOS app that captures people with the iPhone camera and displays a real-time "battle power score" using DINOv3 (Vision Transformer by Meta) and a trained Linear Predictor.
> It performs inference on Apple Silicon using MLX Swift.

---

## Prerequisites Checklist

Ensure you have all of the following before starting.

| Item | Requirement |
|------|-----------|
| Mac | Apple Silicon (M1 or later) + macOS Sequoia or later recommended |
| Xcode | 16.0 or later (from App Store or Developer website) |
| Apple ID | A free Apple ID can be used for physical device builds (7-day certificate) |
| iPhone | iOS 17 or later, connected to Mac via USB cable |
| Git | Required for cloning the repository |
| uv | Python environment manager (used for MLXDINOv3 weight conversion scripts) |

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/techino555/PowerPredictor.git
cd PowerPredictor
```

Verify the directory structure after cloning:

```
PowerPredictor/
├── NOTICE                  # ← Must read (license & permission information)
├── LICENSE
├── README.ja.md
├── source/                 # Main Xcode project
├── skills/
│   └── buildtoiOS/         # This skill
├── docs/
└── ...
```

---

## Step 2 — Acquire Resources Requiring Prior Permission

> **⚠️ Important — Please check the NOTICE file**
> The `NOTICE` file describes the third-party licenses and permission requirements used by this application.
> Make sure to review it and complete each permission process before building.

### 2-1. Acquire DINOv3 Model Weights (Application to Meta required)

DINOv3 is released under Meta Platforms, Inc.'s proprietary license.
**To obtain the model weights, you must apply via Meta's official form.**

1. Visit the following URL and submit the access request form:
   ```
   https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
   ```
2. After applying, you will receive an approval notification by email (usually within a few hours to several days).
3. Once approved, download the `dinov2_vitb14` weight file.

> **Attribution Requirement**
> As a condition of the DINOv3 License, you must include the attribution `"Built with DINOv3"` in redistributions and derivative works.

### 2-2. Acquire Scouter Sound Effects (Niconi Commons)

Download the sound effect from Niconi Commons. Check the author's terms of use for restrictions on commercial use, modification, etc.

1. Visit the following URL:
   ```
   https://commons.nicovideo.jp/works/nc116296
   ```
2. Read and agree to the terms of use on the page.
3. Download the audio file (.wav).
4. Place the downloaded file in the following project path:
   ```
   Resources/Sounds/
   ```

> **When posting to Niconico-related platforms**
> You must register `nc116296` as a parent work (tree registration).

---

## Step 3 — Obtain MLXDINOv3 & Generate Model Weight Files

PowerPredictor uses [MLXDINOv3](https://github.com/vincentamato/MLXDINOv3) (MIT License) to perform DINOv3 inference on Apple Silicon.
Convert the PyTorch weights of DINOv3 into MLX/safetensors format using the MLXDINOv3 tools.

### 3-1. Python Environment Setup (using uv)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment for conversion script
uv venv .venv
source .venv/bin/activate

# Install required packages. Place pyproject.toml located under training/resources and run:
uv sync
```

### 3-2. Convert Weight Files

Convert from PyTorch format to safetensors format using the script from the MLXDINOv3 repository.

```bash
# Clone MLXDINOv3 (recommended in a separate directory)
git clone https://github.com/vincentamato/MLXDINOv3.git /tmp/MLXDINOv3

# Run conversion script (follow MLXDINOv3 documentation)
python /tmp/MLXDINOv3/convert.py \
  --input  /path/to/downloaded/dinov2_vitb14.pth \
  --output ./source/Models/dinov3-vitb16-mlx/
```

After conversion, verify that the following two files are generated:

```
source/Models/dinov3-vitb16-mlx/
├── config.json
└── model.safetensors
```

> **For conversion script details, refer to the README in the MLXDINOv3 repository.**
> Commands may vary depending on the version.

### 3-3. Place power_predictor.safetensors

Place the trained battle power prediction model in the following directory (bundled in the repository or generated via the training skill):

```
Models/power_predictor.safetensors
```

Models directory structure after placement:

```
Models/
├── dinov3-vitb16-mlx/
│   ├── config.json
│   └── model.safetensors
├── linear/
└── power_predictor.safetensors
```

---

## Step 4 — Build Configuration in Xcode

### 4-1. Open Project in Xcode

```bash
open source/PowerPredictor.xcodeproj
```

Or double-click the `.xcodeproj` file inside the `source/` directory from Finder.

### 4-2. Resolve Swift Package Dependencies

Once Xcode opens, SPM (Swift Package Manager) will automatically resolve dependency packages.
Ensure your network connection is active and wait until the following packages are fetched:

| Package | Version | Purpose |
|-----------|-----------|------|
| mlx-swift-lm | 3.31.4 | MLX Swift (includes mlx-swift) |
| MLXDINOv3 | 2.0.0 | DINOv3 Inference |
| swift-transformers | 1.3.3 | HuggingFace-compatible Transformers |
| swift-huggingface | 0.9.0 | HuggingFace Hub Access |

> You can also resolve manually via Xcode menu: `File` → `Packages` → `Resolve Package Versions`.

### 4-3. Signing & Capabilities Configuration

1. In the Project Navigator, select the `PowerPredictor` project.
2. Select the `Signing & Capabilities` tab.
3. In **Team**, select your Apple ID (a free Apple ID is sufficient).
4. Change **Bundle Identifier** to a unique value (e.g., `com.yourname.PowerPredictor`).
5. Ensure `Automatically manage signing` is checked.

### 4-4. Verify Camera Permissions

Verify that the following key is included in `Info.plist` (normally already present in the repository):

```xml
<key>NSCameraUsageDescription</key>
<string>Uses camera to measure battle power of people.</string>
```

### 4-5. Add Target Membership for Required Files

The following three files are used in both the App and Training, so they must be included in the Target Membership for both targets:

- PowerPredictor/DINOv3/DINOv3Encoder.swift
- PowerPredictor/Models/dinov3-vitb16-mlx/config.json
- PowerPredictor/Models/dinov3-vitb16-mlx/model.safetensors

How to verify:

1. Select each of the above files in Xcode.
2. Open the File Inspector on the right panel.
3. Check the Target Membership section.
4. Verify that checkboxes for both `PowerPredictor` and `PowerPredictor_feature_ext` are selected.

---

## Step 5 — Build & Install to Physical iPhone

### 5-1. Connect iPhone to Mac

1. Connect your iPhone to your Mac using a Lightning or USB-C cable.
2. When prompted "Trust This Computer?" on the iPhone screen, tap **"Trust"**.

### 5-2. Change Build Target to iPhone

Select your connected iPhone from the device selection dropdown at the top of Xcode:

```
[PowerPredictor] > [Your iPhone Name]
```

### 5-3. Run Build

Press `⌘ + R` or select menu `Product` → `Run`.

The initial build may take **10 to 20 minutes** as it includes compiling dependencies.

### 5-4. Trust Developer App on iPhone

If using a free Apple ID, you must trust the developer certificate on the iPhone:

1. On iPhone, navigate to `Settings` → `General` → `VPN & Device Management`.
2. Tap the Developer App showing your Apple ID.
3. Tap **"Trust [Apple ID]"**.

---

## Step 6 — Verify Installation

If the **PowerPredictor** app icon appears on your iPhone home screen, the installation was successful.

Launch the app and verify the following:

- [ ] Camera activates and preview is displayed
- [ ] Detecting frame (HUD) appears when pointing at a person
- [ ] Scan sound plays when tapping the `SCAN` button
- [ ] Battle power score is displayed on screen

---

## Troubleshooting

### Build error: `model.safetensors not found`

Check if `source/Models/dinov3-vitb16-mlx/model.safetensors` exists.
Model conversion in Step 3 might not have completed.

### Build error: `Signing certificate ... has expired`

Free Apple ID certificates expire after 7 days. Re-run `Product` → `Run` from Xcode to renew the certificate.

### Not installing to iPhone: `Unable to install...`

Verify that you have trusted the developer app in `Settings` → `General` → `VPN & Device Management` (Step 5-4).

### SPM cannot resolve dependencies: `network error`

Check your internet connection and re-run `File` → `Packages` → `Resolve Package Versions`.

---

## References

| Resource | URL |
|---------|-----|
| PowerPredictor GitHub | https://github.com/techino555/PowerPredictor/ |
| DINOv3 Application Form | https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ |
| MLXDINOv3 | https://github.com/vincentamato/MLXDINOv3 |
| Sound Effect (nc116296) | https://commons.nicovideo.jp/works/nc116296 |
| MLX Swift | https://github.com/ml-explore/mlx-swift |
| Xcode Download | https://developer.apple.com/xcode/ |
