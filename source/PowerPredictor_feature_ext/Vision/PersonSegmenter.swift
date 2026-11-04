import Foundation
import AppKit
import Vision
import CoreImage
import CoreImage.CIFilterBuiltins

class PersonSegmentProcessor {
    private let context = CIContext()

    /// 人物のマスクを取得する
    func segment(cgImage: CGImage) async throws -> CIImage? {
        return try await withCheckedThrowingContinuation { continuation in
            let request = VNGeneratePersonSegmentationRequest { request, error in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }
                
                guard let result = request.results?.first as? VNPixelBufferObservation else {
                    continuation.resume(returning: nil)
                    return
                }
                
                let pixelBuffer = result.pixelBuffer
                let maskImage = CIImage(cvPixelBuffer: pixelBuffer)
                continuation.resume(returning: maskImage)
            }
            
            request.qualityLevel = .fast
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
    
    /// 背景を薄い黄色にする
    func applyYellowBackground(to original: CIImage) -> CIImage? {
        let yellowBgColor = CIColor(red: 0.9, green: 0.8, blue: 0.2, alpha: 0.6) // 薄い黄色
        guard let bgGenerator = CIFilter(name: "CIConstantColorGenerator") else { return nil }
        bgGenerator.setValue(yellowBgColor, forKey: kCIInputColorKey)
        guard let yellowBg = bgGenerator.outputImage?.cropped(to: original.extent) else { return nil }
        
        let blend = CIFilter.multiplyBlendMode()
        blend.inputImage = yellowBg
        blend.backgroundImage = original
        return blend.outputImage
    }
    
    /// 人物部分を黄土色（Brown/Ocher）にする
    func applyBrownForeground(to background: CIImage, original: CIImage, mask: CIImage) -> CIImage? {
        // マスクのサイズをオリジナル画像に合わせる
        let scaledMask = mask.transformed(by: CGAffineTransform(scaleX: original.extent.width / mask.extent.width, y: original.extent.height / mask.extent.height))
        
        // 元画像を黄土色に着色（陰影は保持）
        let ocherColor = CIColor(
            red: 0.78,
            green: 0.40,
            blue: 0.24,
            alpha: 0.65
        )

        let fgGenerator = CIFilter(name: "CIConstantColorGenerator")!
        fgGenerator.setValue(ocherColor, forKey: kCIInputColorKey)

        let ocherImage = fgGenerator.outputImage!.cropped(to: original.extent)

        let sourceOver = CIFilter.sourceOverCompositing()

        sourceOver.inputImage = ocherImage
        sourceOver.backgroundImage = original

        guard let mixedImage = sourceOver.outputImage else {
            return nil
        }

        // マスクを使って、人物部分のみ黄土色を適用し、背景は指定の背景画像を使う
        let blend = CIFilter.blendWithMask()

        blend.inputImage = mixedImage
        blend.backgroundImage = background
        blend.maskImage = scaledMask

        return blend.outputImage
    }
    
    /// 解析中の画像（人物：黄土色、背景：黄色）を生成する
    func generateAnalysisImage(from cgImage: CGImage) async throws -> NSImage? {
        let original = CIImage(cgImage: cgImage)
        
        // 1. セグメント（マスク取得）
        guard let mask = try await segment(cgImage: cgImage) else { return nil }
        
        // 2. 背景を黄色にする
        guard let yellowBg = applyYellowBackground(to: original) else { return nil }
        
        // 3. 人物を黄土色にする
        guard let finalResult = applyBrownForeground(to: yellowBg, original: original, mask: mask) else { return nil }
        
        if let outputCGImage = context.createCGImage(finalResult, from: finalResult.extent) {
            return NSImage(
                cgImage: outputCGImage,
                size: NSSize(width: outputCGImage.width, height: outputCGImage.height)
            )
        }
        return nil
    }
    
    /// 完了時の画像（全体が黄色）を生成する
    func generateCompleteImage(from cgImage: CGImage) -> NSImage? {
        let original = CIImage(cgImage: cgImage)
        if let yellowBg = applyYellowBackground(to: original),
           let outputCGImage = context.createCGImage(yellowBg, from: yellowBg.extent) {
            return NSImage(
                cgImage: outputCGImage,
                size: NSSize(width: outputCGImage.width, height: outputCGImage.height)
            )
        }
        return nil
    }
}
