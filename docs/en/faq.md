# FAQ
## What are the verified operating versions for iOS/macOS?
PowerPredictor : iOS 26
PowerPredictor_feature_ext : macOS 26

These versions have been tested and verified. They have also been tested on the latest versions:

PowerPredictor : iOS 27
PowerPredictor_feature_ext : macOS 27

## Will an Android version be released?

There are currently no plans. Both Vision Framework and MLX Swift are Apple proprietary technologies, and supporting Android would require adopting completely different stacks. While Android possesses equivalent technologies, they do not originate from the same ecosystem, which presents disadvantages in compatibility and integration. However, community contributions toward developing an Android version are welcome.

## Can it be released/run on iPad?

Yes, it is possible. It has been built and tested on a physical iPad Pro 11-inch (2nd generation).

## Are photos saved after measurement?

No, they are not saved. We prioritize keeping it strictly as a measurement app.

## The Power score fluctuates heavily.

- Insufficient amount of training data
- Due to the nature of DINOv3's CLS Token, clothing, posture, and environmental factors are also reflected in the extracted features even for the same person, causing fluctuations. Training sample code and training methodology are publicly documented, so please feel free to experiment and improve the model yourself.
