import Foundation
import CoreGraphics
import Vision

/// 人物のポーズ（姿勢）を検証するクラス
/// 半身以上が画面に収まっているか、各部位（頭、肩、腰）が検出可能かを判定します。
class PoseValidator {
    
    /// 与えられた画像から人物のポーズを評価し、測定可能かどうかを判定します
    /// - Parameter cgImage: 評価対象の画像
    /// - Returns: 検証結果（isValidと却下理由）
    func evaluate(cgImage: CGImage) async throws -> MeasurementResult {
        return try await withCheckedThrowingContinuation { continuation in
            // 人体のポーズ（関節位置）を検出するリクエスト
            let request = VNDetectHumanBodyPoseRequest { request, error in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }
                
                // 検出結果から最初の人物を取得。誰も検出されなかった場合はNG
                guard let results = request.results as? [VNHumanBodyPoseObservation], let pose = results.first else {
                    continuation.resume(returning: MeasurementResult(isValid: false, reason: "人物のポーズが検出されませんでした"))
                    return
                }
                
                // 検出されたポーズ情報を元に詳細な検証を行う
                let result = self.validateObservation(pose)
                continuation.resume(returning: result)
            }
            
            // リクエストを非同期で実行
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
    
    /// 検出されたポーズ情報を元に、必要な部位がすべて揃っているか確認します
    private func validateObservation(_ pose: VNHumanBodyPoseObservation) -> MeasurementResult {
        do {
            // 各関節（鼻・両肩・両腰）の座標と信頼度（confidence）を取得
            let nose = try pose.recognizedPoint(.nose)
            let leftShoulder = try pose.recognizedPoint(.leftShoulder)
            let rightShoulder = try pose.recognizedPoint(.rightShoulder)
            let leftHip = try pose.recognizedPoint(.leftHip)
            let rightHip = try pose.recognizedPoint(.rightHip)
            
            // 開発時の調整用として、各関節の位置と信頼度をコンソールに出力
            print("""
            [PoseValidator]
            Nose          : (\(String(format: "%.2f", nose.location.x)), \(String(format: "%.2f", nose.location.y))) confidence=\(String(format: "%.2f", nose.confidence))
            LeftShoulder  : (\(String(format: "%.2f", leftShoulder.location.x)), \(String(format: "%.2f", leftShoulder.location.y))) confidence=\(String(format: "%.2f", leftShoulder.confidence))
            RightShoulder : (\(String(format: "%.2f", rightShoulder.location.x)), \(String(format: "%.2f", rightShoulder.location.y))) confidence=\(String(format: "%.2f", rightShoulder.confidence))
            LeftHip       : (\(String(format: "%.2f", leftHip.location.x)), \(String(format: "%.2f", leftHip.location.y))) confidence=\(String(format: "%.2f", leftHip.confidence))
            RightHip      : (\(String(format: "%.2f", rightHip.location.x)), \(String(format: "%.2f", rightHip.location.y))) confidence=\(String(format: "%.2f", rightHip.confidence))
            """)
            
            // 関節を正しく検出したとみなす信頼度の閾値（0.0 ~ 1.0）
            let threshold: Float = 0.5
            
            // いずれかの部位が閾値を下回る場合は、半身が正しく映っていないと判断してNGを返す
            if nose.confidence < threshold {
                return MeasurementResult(isValid: false, reason: "頭（顔）が正しく検出されませんでした")
            }
            if leftShoulder.confidence < threshold {
                return MeasurementResult(isValid: false, reason: "左肩が正しく検出されませんでした")
            }
            if rightShoulder.confidence < threshold {
                return MeasurementResult(isValid: false, reason: "右肩が正しく検出されませんでした")
            }
            
            // 腰は条件を緩める (0.3)
            let hipThreshold: Float = 0.3
            if leftHip.confidence < hipThreshold {
                return MeasurementResult(isValid: false, reason: "左腰が正しく検出されませんでした")
            }
            if rightHip.confidence < hipThreshold {
                return MeasurementResult(isValid: false, reason: "右腰が正しく検出されませんでした")
            }
            
            // 全ての部位が基準を満たせばOK
            print("[PoseValidator] Result: ACCEPT")
            return MeasurementResult(isValid: true, reason: "ACCEPT")
            
        } catch {
            // 関節情報の取得自体に失敗した場合
            return MeasurementResult(isValid: false, reason: "関節情報の取得に失敗しました: \(error.localizedDescription)")
        }
    }
}