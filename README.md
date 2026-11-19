# PowerPredictor

This project is an iOS application that extracts feature representations from images using MLX Swift and packages listed in [NOTICE](NOTICE) to calculate the subject's "Power Level" (Combat Strength / 戦闘力). It is an homage to the Scouter from the Japanese anime series *Dragon Ball Z*\*.

\*
©BIRD STUDIO/SHUEISHA, FUJI TV, TOEI ANIMATION

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Swift](https://img.shields.io/badge/Swift-6.0-F05138.svg?style=flat-square&logo=swift&logoColor=white)](https://swift.org)
[![iOS](https://img.shields.io/badge/iOS-17.0+-000000.svg?style=flat-square&logo=apple&logoColor=white)](https://developer.apple.com/ios/)
[![Platform](https://img.shields.io/badge/Platform-iOS%20%7C%20macOS-lightgrey.svg?style=flat-square)]()
[![MLX Swift](https://img.shields.io/badge/MLX-Swift-FF4F00.svg?style=flat-square)](https://github.com/ml-explore/mlx-swift)

## Demo

Launch the app after installing it on an iPhone. Upon launching, the camera screen opens; press the Scan button to start scanning. The camera needs to be pointed at a person to display their Power Level. Press RESET after measurement to return to the initial state. Simply repeat this process.

**[ScanDemo1](demo/PowerPredictor_demo1.mp4)**

**[ScanDemo2](demo/PowerPredictor_demo2.mp4)**


## Architecture

Codebase Architecture

```
source/PowerPredictor/

├── App
│   └── PowerPredictor.swift
│        # @main entry point
│ 
├── Audio
│   └── SoundPlayer.swift
│        # Sound effect playback
│
├── Camera
│   ├── CameraManager.swift
│   │      # AVCaptureSession management, frame capture
│   │
│   └── ImagePreprocessor.swift
│          # Converts UIImage/CVPixelBuffer to MLX input tensor
│
├── DINOv3
│      └── DINOv3Encoder.swift
│             # DINOv3 loading
│             # Extracts 768-dimensional embeddings
│
├── Embedding
│      └── EmbeddingPipeline.swift
│             # Preprocessing
│             # Person cropping
│             # Calls DINOv3Encoder
│             # Generates embeddings
│
└── Models
│      ├── dinov3-vitb16-mlx/
│      │      # DINOv3 model
│      │      # config.json
│      │      # model.safetensors
│      │
│      └── mylinear/
│             └── power_predictor.safetensors
│                    # Trained predictor weights
│                    # (No config required for Linear)
│
├── Predictor
│      └── LinearPredictor.swift
│             # Embedding → Power Level score inference
│
└── Resources
│    └── Assets.xcassets
│    │      # Icons, images, colors, fonts, etc.
│    └── Sounds
│           # Sound effect files
│           # nc116296_scouter.wav (Niconi Commons nc116296)
│
├── Stores
│   └── ScouterStore.swift
│          # App-wide inference flow management
│          # Vision → Embedding → Predictor → UI update
│
├── ViewModels
│   └── ScanViewModel.swift
│          # Controls UI events
│          # Starts SCAN, invokes Store, updates state
│          # ScanState enum: idle / detecting / analyzing / complete
│
├── Views
│   ├── ContentView.swift
│   │      # Main UI (Camera preview + score display + SCAN button)
│   │
│   ├── ScanOverlayView.swift
│   │      # Draws person bounding box, LOCK ON, SCAN animations, etc.
│   │      # Triangular HUD visuals after detecting a person
│   │      # Visual effects: yellow background with ~0.6 opacity, darkening detected person in ochre during analysis
│   │      # Internal View structure:
│   │      #   ScanEffectView   … Scan animation effect
│   │      #   HUDOverlayView   … Overall HUD overlay
│   │      #   TriangleMarker   … Draws triangular markers at four corners
│   │      #   TriangleDirection … Enum representing triangle orientation
│   │
│   └── PowerView.swift
│          # Result display UI for Power Level, comments, gauge, etc.
│
├── Vision
    ├── VisionPipeline.swift
    │      # Executes the Vision Framework pipeline
    │      # Controls PersonDetector → PoseValidator → PersonSegmentProcessor
    │      # VisionPipelineError enum: measurementRejected / noPersonDetected
    │
    ├── PersonDetector.swift
    │      # Person detection (Bounding Box extraction)
    │
    ├── PoseValidator.swift
    │      # Pose validation to prevent false / unrestricted detections
    │      # Detected keypoints: nose, left/right shoulders, left/right hips
    │      # Confidence thresholds: shoulders/face = 0.5, hips = 0.3
    │
    ├── FrameValidator.swift
    │      # Frame validation to prevent false / unrestricted detections
    │      # Detected keypoints: nose, left/right shoulders, left/right hips
    │      # Confidence thresholds: shoulders/face = 0.5, hips = 0.3
    │
    ├── MeasurementResult.swift
    │      # Measurement eligibility result struct (isValid: Bool, reason: String)
    │
    └── PersonSegmenter.swift
           # Person mask generation and background removal
           # Actual class name: PersonSegmentProcessor
           # During analysis: person → ochre, background → yellow
           # Upon completion: applies yellow filter to the entire screen
```

For details on the design goals of each major component, please refer to the following documents and SKILLs:

### Person Detection, Background Removal, and Pose Validation from Camera

**[PoseDetector](docs/en/posedetector.md)**

### Feature Extraction for Power Level Learning
**[FeatureExtract](docs/en/feature-extraction.md)**

Feature extraction methodology:

**[FeatureExtract Skills](skills/feature-extraction/en/SKILL.md)**

### Power Level Training
**[Training](docs/en/powertraining.md)**

Training methodology:

**[Training Skills](skills/training/en/SKILL.md)**

### App UI and State Transitions

**[UI and ScanState](docs/en/uiandscanstate.md)**


## Getting Started / Setup
This product cannot be utilized simply by cloning this repository; there are required resources that must be installed after agreeing to the terms and conditions of several third-party products. Please refer to this SKILL for installation instructions:

**[How to Build app](skills/build-to-iOS/en/SKILL.md)**

## Documentation

For **[FAQ](docs/en/faq.md)**, see here.


## Contact

For inquiries and feature requests:

[GitHub issues](https://github.com/techino555/PowerPredictor/issues) and [X](https://x.com/hidenobu_is).


## Acknowledgments / Support

Development and model training are ongoing. If you'd like to support this project, donations are greatly appreciated:

[Buy Me a Coffee](https://buymeacoffee.com/techino555). 

There is no paid or gated content. It is open source under the MIT license and will continue to be MIT.


## Dependencies / Libraries

App: Swift. Package Dependencies

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

Training: Python. Project.toml

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

Licensed under MIT. Please see: [LICENSE](LICENSE)

There are also third-party code and assets; their licenses remain with their respective third-party owners as described in NOTICE: [NOTICE](NOTICE)
