import Foundation
import AppKit
import Vision

class VisionPipeline {
    let detector = PersonDetector()
    let segmenter = PersonSegmentProcessor()

    /// 画像を解析して、人物のバウンディングボックスと解析中画像、完了用画像を返す
    func process(image: NSImage) async throws -> (CGRect?, NSImage?, NSImage?) {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return (nil, nil, nil)
        }
        
        // 1. 人物検出 (Step 2)
        let boundingBox = try await detector.detect(cgImage: cgImage)
        
        // 2. 人物のセグメンテーション（エフェクト適用画像の生成）
        let analysisImage = try await segmenter.generateAnalysisImage(from: cgImage)
        let completeImage = segmenter.generateCompleteImage(from: cgImage)
        
        return (boundingBox, analysisImage, completeImage)
    }
}
