import Foundation
import UIKit
import MLX
internal import Combine

/// エンコーダー・予測モデルを統括し、UIに状態を公開するStore
@MainActor
final class ScouterStore: ObservableObject {
    @Published var powerScore: Float? = nil
    @Published var explanation: String = ""
    @Published var isProcessing: Bool = false

    private let visionPipeline = VisionPipeline()
    private var embeddingPipeline: EmbeddingPipeline?
    private var predictor: LinearPredictor?
    private var isModelLoaded: Bool = false

    func setupModel() {
        do {
            embeddingPipeline = try EmbeddingPipeline()

            let p = LinearPredictor()
            try p.loadWeights()
            predictor = p

            isModelLoaded = true
            print("ScouterStore: モデルセットアップ完了")
        } catch {
            print("ScouterStore: モデルのロードに失敗しました: \(error)")
            isModelLoaded = false
        }
    }

    /// カメラ等で撮影した写真を受け取って戦闘力を算出する
    func processImage(_ image: UIImage, viewModel: ScanViewModel) async {
        let startTime = ContinuousClock.now
        isProcessing = true
        self.powerScore = nil

        defer { isProcessing = false }

        // ── Step 1: バリデーション（トラッキング確認） ──
        // PersonDetector + FrameValidator + PoseValidator を実行
        // ここをクリアしないとスキャンを開始しない
        let boundingBox: CGRect
        do {
            boundingBox = try await visionPipeline.validate(image: image)
        } catch {
            print("[Scouter] バリデーション失敗: \(error.localizedDescription)")
            // バリデーション失敗時はUIを遷移させずにそのまま返す
            return
        }

        // ── Step 2: バリデーション通過 → スキャン開始 ──
        viewModel.startScan()
        SoundPlayer.shared.playScan()

        if !isModelLoaded {
            do {
                let (pipeline, predictor) = try await Task.detached {

                    let pipeline = try EmbeddingPipeline()

                    let predictor = LinearPredictor()
                    try predictor.loadWeights()

                    return (pipeline, predictor)
                }.value
                self.embeddingPipeline = pipeline
                self.predictor = predictor
                self.isModelLoaded = true
            } catch {
                self.explanation = "モデルのロードに失敗しました"
                viewModel.reset()
                return
            }
        }

        guard let pipeline = embeddingPipeline, let predictor = predictor else {
            self.explanation = "モデルの準備ができていません"
            viewModel.reset()
            return
        }

        do {
            // 3. セグメンテーション（エフェクト画像生成）
            let (analysisImage, completeImage) = try await visionPipeline.segment(image: image)

            viewModel.detectedPersonRect = boundingBox
            viewModel.effectImage = analysisImage
            viewModel.completeImage = completeImage

            // 「人物を特定した瞬間(detecting)」を少し見せるためのウェイト (0.5秒)
            try await Task.sleep(nanoseconds: 500_000_000)
            
            // 💡ここで解析中ステートに変更
            viewModel.setAnalyzing()

            // DINOv3に入力する画像 (解析用エフェクト画像がなければオリジナル)
            let personImage = analysisImage ?? image

            // 4. DINOv3とPowerPredictorをバックグラウンドで実行しUIのフリーズを防ぐ
            let scoreValue = try await Task.detached {
                let embedding = try pipeline.encode(personImage: personImage)
                let scoreArray = predictor.predict(embedding)
                return scoreArray.item(Float.self)
            }.value
            
            self.powerScore = scoreValue

            let elapsed = startTime.duration(to: .now)
            print("処理時間:", elapsed)
            print("FIGHT \(scoreValue)")
            
            // 解析中を演出するためのウェイト (最低でも少し見せるため)
            try await Task.sleep(nanoseconds: 1_600_000_000)
            
            // 5. 解析完了ステートに変更
            viewModel.finishScan() // -> .complete
            
        } catch {
            print("解析中にエラーが発生しました: \(error)")
            self.explanation = "測定不能"
            viewModel.reset()
        }
    }
}
