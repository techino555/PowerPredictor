import Foundation
import MLX
import MLXDINOv3

final class DINOv3Encoder {

    private let model: DinoVisionTransformer

    init() throws {

        let modelPath = Bundle.main.resourcePath!

        print(modelPath)

        model = try DinoVisionTransformer.loadPretrained(
            from: modelPath
        )
    }

    /// モデルの推論実行
    func inference(_ input: MLXArray) -> DinoOutput {
        return model(input)
    }

    /// MLXArrayテンソルを受け取り、768次元のEmbeddingベクトルを返す
    func encode(_ input: MLXArray) -> MLXArray {
        let output = inference(input)
        return output.clsToken
    }
}
