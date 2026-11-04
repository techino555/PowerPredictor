import Foundation
import UIKit
import Vision
import CoreImage

class PersonDetector {
    /// DetectHumanRectanglesRequestを使用して人物を検出する
    func detect(cgImage: CGImage) async throws -> CGRect? {
        return try await withCheckedThrowingContinuation { continuation in
            let request = VNDetectHumanRectanglesRequest { request, error in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }
                
                guard let results = request.results as? [VNHumanObservation], let firstPerson = results.first else {
                    continuation.resume(returning: nil)
                    return
                }
                
                // 正規化されたバウンディングボックス(0.0 ~ 1.0)を返す
                continuation.resume(returning: firstPerson.boundingBox)
            }
            
            // より高速な検出のために必要なら revision を指定する
            request.upperBodyOnly = false
            
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            Task.detached {
                do {
                    try handler.perform([request])
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}
