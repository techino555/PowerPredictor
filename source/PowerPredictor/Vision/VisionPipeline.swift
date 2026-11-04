import Foundation
import UIKit
import Vision

enum VisionPipelineError: Error, LocalizedError {
    case measurementRejected(String)
    case noPersonDetected
    
    var errorDescription: String? {
        switch self {
        case .measurementRejected(let reason):
            return "Measurement rejected: \(reason)"
        case .noPersonDetected:
            return "No person detected"
        }
    }
}

class VisionPipeline {
    let detector = PersonDetector()
    let frameValidator = FrameValidator()
    let poseValidator = PoseValidator()
    let segmenter = PersonSegmentProcessor()
    
    /// バリデーションのみ実行（PersonDetector + FrameValidator + PoseValidator）
    /// 全てクリアしたらバウンディングボックスを返す。NGならエラーをスローする。
    func validate(image: UIImage) async throws -> CGRect {
        guard let cgImage = image.cgImage else {
            throw VisionPipelineError.noPersonDetected
        }
        
        // 1. 人物検出
        guard let boundingBox = try await detector.detect(cgImage: cgImage) else {
            throw VisionPipelineError.noPersonDetected
        }
        
        // 2. フレーム検証 (人物の大きさ、位置など)
        let frameResult = frameValidator.evaluate(boundingBox: boundingBox)
        guard frameResult.isValid else {
            print("[Scouter] rejected: \(frameResult.reason)")
            throw VisionPipelineError.measurementRejected(frameResult.reason)
        }
        
        // 3. ポーズ検証 (頭、肩、腰の検出)
        let poseResult = try await poseValidator.evaluate(cgImage: cgImage)
        guard poseResult.isValid else {
            print("[Scouter] rejected: \(poseResult.reason)")
            throw VisionPipelineError.measurementRejected(poseResult.reason)
        }
        
        print("[Scouter] Measurement accepted")
        return boundingBox
    }
    
    /// セグメンテーションのみ実行（バリデーション済みの前提で呼ぶ）
    /// 解析中エフェクト画像と完了用画像を返す
    func segment(image: UIImage) async throws -> (UIImage?, UIImage?) {
        guard let cgImage = image.cgImage else { return (nil, nil) }
        
        let analysisImage = try await segmenter.generateAnalysisImage(from: cgImage)
        let completeImage = segmenter.generateCompleteImage(from: cgImage)
        
        return (analysisImage, completeImage)
    }
}
