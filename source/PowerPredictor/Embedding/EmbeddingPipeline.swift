import Foundation
import UIKit
import MLX

final class EmbeddingPipeline {
    let preprocessor = ImagePreprocessor()
    let encoder: DINOv3Encoder

    init() throws {
        self.encoder = try DINOv3Encoder()
    }

    /// 前処理、推論のパイプラインを実行してEmbeddingを返す
    func encode(personImage: UIImage) throws -> MLXArray {
        // 1. MLXArrayに前処理
        let input = try preprocessor.preprocess(personImage)
        
        // 2. 推論を実行して特徴量(Embedding)を抽出
        return encoder.encode(input)
    }
}
