import Foundation
import CoreGraphics

class FrameValidator {
    func evaluate(boundingBox: CGRect) -> MeasurementResult {
        // area = width * height (正規化座標)
        let areaRatio = boundingBox.width * boundingBox.height
        let heightRatio = boundingBox.height
        let center = CGPoint(x: boundingBox.midX, y: boundingBox.midY)
        
        // Visionフレームの原点は左下
        let leftMargin = boundingBox.minX
        let rightMargin = 1.0 - boundingBox.maxX
        
        print("""
        [FrameValidator]
        Person count : 1
        BoundingBox  : x=\(String(format: "%.2f", boundingBox.origin.x)) y=\(String(format: "%.2f", boundingBox.origin.y)) w=\(String(format: "%.2f", boundingBox.size.width)) h=\(String(format: "%.2f", boundingBox.size.height))
        Center       : x=\(String(format: "%.2f", center.x)) y=\(String(format: "%.2f", center.y))
        Area ratio   : \(String(format: "%.1f", areaRatio * 100))%
        Height ratio : \(String(format: "%.1f", heightRatio * 100))%
        Left margin  : \(String(format: "%.1f", leftMargin * 100))%
        Right margin : \(String(format: "%.1f", rightMargin * 100))%
        """)
        
        // 人物のサイズが小さすぎる場合
        if heightRatio < 0.35 {
            return MeasurementResult(isValid: false, reason: "Body size: \(Int(heightRatio * 100))% < 35%")
        }
        
        // 画面の端すぎる場合をチェック (必要に応じて閾値を調整)
        if center.x < 0.2 || center.x > 0.8 {
            return MeasurementResult(isValid: false, reason: "Person is too far from center.")
        }
        
        print("[FrameValidator] Result: ACCEPT")
        return MeasurementResult(isValid: true, reason: "ACCEPT")
    }
}
