# Memory-Efficient Image Capture

In typical camera apps, after triggering the shutter, image data is commonly saved to a file such as JPEG and then reloaded into memory as a UIImage. However, for this app performing real-time inference, writing images to disk was not considered desirable, as accumulated image data would require tedious cleanup.

Therefore, we utilized the `cgImageRepresentation()` method to extract a CGImage directly in memory from the `AVCapturePhoto` received via `AVCapturePhotoCaptureDelegate`. By eliminating unnecessary encoding and file I/O and passing the pixel buffer loaded in memory directly to subsequent steps, we realized a low-latency, memory-efficient pipeline.

# Person Detection and Segmentation

Applying Apple's machine learning framework, Vision Framework, to camera frames allows locating and extracting human silhouettes. This app primarily relies on two requests:

 1. Human Rectangles Detection (`PersonDetector`): Determines whether a person is in the frame and quickly computes a bounding box (`CGRect`).
 2. Person Segmentation (`PersonSegmenter`): Accurately segments the human silhouette (mask) at pixel level, separating background and human subject.

# PersonDetector Implementation

For person detection, we use `VNDetectHumanRectanglesRequest` from the Vision Framework. This bounding box is utilized to determine the positioning of the lock-on HUD marker on screen, described later. The HUD marker is the visual effect where triangles enclose the subject to illustrate tracking, adopted to recreate the look of the original scouter.

# Trial and Error with Scouter Visual Effects

Using `VNGeneratePersonSegmentationRequest`, the person mask is obtained, and Core Image (one of Apple's built-in frameworks) applies the characteristic scouter color filtering. Initially, a flat ochre color layer (`ocherFg`) was simply blended over the segmented mask. However, doing so flattened facial shadows, clothing wrinkles, and all three-dimensional light/dark contours, resulting in an unnatural, flat silhouette. Lowering opacity only made contrast blurry.
Instead of flat color filling, we adopted a method that generates a tinted image (`coloredFg`) by shifting the hue while preserving the original image's luminance and shading, composited with a light-yellow background via the person mask. This allowed creating a futuristic analysis screen highlighted in smooth ochre while crisply preserving facial contours and muscular textures.

# Controls to Prevent Unbounded Detections

Running detection and measurement unconditionally when a person is too far, clipped at screen edges, or in an incomplete posture causes false detections and unnecessary processing load. To prevent this, the app introduces a two-stage validation process using `FrameValidator` and `PoseValidator`, only accepting states suitable for measurement.

# Integration with VisionPipeline

`VisionPipeline` bundles these detection and image processing steps into a single cohesive pipeline. Combining Apple's native AVFoundation, Vision, and Core Image enabled high-speed human tracking and faithful reproduction of the scouter's visual style without introducing third-party dependencies.
