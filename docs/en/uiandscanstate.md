# Running the Battle Power Model, Output, and UI Architecture in the iOS App

The previous chapter detailed building the pipeline to extract 768-dimensional embeddings from person images using DINOv3.
This chapter covers integrating the lightweight regression model that calculates battle power scores from these features, as well as UI implementation and state management embodying the scouter experience.

## Scouter UI Architecture and State Management

To recreate the authentic scouter experience, we reproduced the sequence: "acquire target, lock on reticle, perform high-speed internal scan calculations, and finalize the numerical score."

### State Transitions via ScanState

To govern this experience reliably, strict state transitions (`ScanState`) were defined in `ScanViewModel`:

 * `idle`: Default standby state. Displays only the camera preview and "SCAN" button.
 * `detecting`: Target acquisition state. Target markers (triangular HUD corners) appear, highlighting the person in ochre to emulate the original scouter.
 * `analyzing`: Active analysis state. Progress bar appears while battle power digits cycle rapidly at random (counting animation).
 * `complete`: Measurement complete state. Screen is tinted pale yellow, and the finalized battle power score appears boldly at the top.

### Synchronizing Sound, Processing, and Animation

To reproduce the moment the user taps SCAN, sound effect playback, visual animations, and background inference were tightly synchronized. Minor offsets are coordinated in code. The flow proceeds as follows (sound effect sourced from here[^scouter]; refer to citation for permission requirements):

```text
SCAN Button Tapped
    ↓
Start Sound Playback (rapid scanning beeps)
    ↓
Transition to .analyzing state (Show ProgressView & start rapid digit animation)
    ↓
Execute DINOv3 and LinearPredictor inference in background
    ↓
Task.sleep maintains dramatic ~3-second wait
    ↓
Scanning sound ends & completion chime plays
    ↓
Transition to .complete state & display finalized battle power score
```

Within `ScouterStore`, state transitions and pacing are implemented as follows, introducing deliberate wait time to craft the authentic scanning sensation:

```swift
// Synchronized flow within ScouterStore
viewModel.setAnalyzing()
soundPlayer.playScanningSound()

// Wait ~1-3 seconds for visual presentation while running background inference
let score = try await runInferencePipeline(image: targetImage)
try await Task.sleep(nanoseconds: 1_000_000_000)

soundPlayer.stopScanningSound()
soundPlayer.playCompleteSound()
viewModel.finishScan(score: score)
```

### UI Components Powering the Scouter Experience

The UI is built with SwiftUI Views, separated into the following hierarchy:

 1. `HUDOverlayView`: Renders screen border frames and four triangular corner reticles enclosing the subject. The bottom triangle is stamped with "SET" in fidelity to the original.
 2. `ScanOverlayView`: Uses `ZStack` to layer processed camera preview and HUD according to state.
 3. `PowerView`: Displays battle power score, "FIGHT" label, and evaluation gauge.

## Resolving UI Freezes and Flickering

During implementation, several technical hurdles emerged regarding SwiftUI and asynchronous inference integration.

### Offloading to Background Threads

In early implementations, tapping SCAN caused the UI to hitch momentarily, preventing smooth playback of the rapid-cycling number animation. This occurred because DINOv3 inference and model weight memory allocation blocked the main thread. To solve this, the inference pipeline in `ScouterStore` was refactored using `Task.detached` to run on a background thread. This allowed the analyzing count-up animation to run smoothly without dropping main-thread frame rates.

### Preventing Display Flickering via Opacity

When transitioning from `detecting` to `analyzing`, clearing text color (`.clear`) on `PowerView` caused preceding numbers to flicker momentarily due to SwiftUI rendering cycles. Instead of toggling individual text colors, setting `PowerView`'s overall opacity to `.opacity(0)` in `detecting` hides it completely from the render tree, switching to `.opacity(1)` the moment `analyzing` begins. This eliminated visual artifacts and achieved clean transition timing.

Through iterative verification and refinement, PowerPredictor—a scouter-style app on iPhone—was completed.

[References]

Niconi Commons: https://commons.nicovideo.jp/works/nc116296
