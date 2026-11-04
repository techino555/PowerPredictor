import Foundation
import AppKit
import MLX

/// 画像からDINOv3 Embedding(768次元)を抽出するパイプライン
///
/// 処理フロー:
/// NSImage → CGImage → VisionPipeline(人物切り出し) → ImagePreprocessor(MLXArray変換) → DINOv3Encoder → CLS Token
final class EmbeddingExtractor {

    private let visionPipeline = VisionPipeline()
    private let preprocessor = ImagePreprocessor()
    private let encoder: DINOv3Encoder

    init() throws {
        self.encoder = try DINOv3Encoder()
    }

    /// 画像から人物のEmbedding(768次元)を抽出する
    /// - Parameter image: 入力画像（NSImage）
    /// - Returns: 768次元のMLXArray。人物が検出されない場合はnil
    func extract(from image: NSImage) async throws -> MLXArray? {
        // 1. VisionPipelineで人物を切り出す
        let (_, _, completeImage) = try await visionPipeline.process(image: image)
        
        guard let personImage = completeImage else {
            print("人物の切り出しに失敗しました")
            return nil
        }

        // 2. ImagePreprocessorでMLXArrayに変換
        let input = try preprocessor.preprocess(personImage)

        // 3. DINOv3Encoderで768次元のEmbedding(CLS Token)を抽出
        return encoder.encode(input)
    }
}