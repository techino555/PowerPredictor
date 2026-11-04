import Foundation
import MLX
import MLXNN

final class LinearPredictor: Module {

    @ModuleInfo
    var linear: Linear

    nonisolated override init() {
        self.linear = Linear(768, 1)
        super.init()
    }

    /// 学習済み重みを読み込む
    func loadWeights() throws {

        print("Resource Path:", Bundle.main.resourcePath!)

        let root = Bundle.main.resourcePath!
        let files = try FileManager.default.subpathsOfDirectory(atPath: root)

        for file in files {
            if file.contains("power") || file.contains("safetensors") {
                print(file)
            }
        }
        
        guard let url = Bundle.main.url(
            forResource: "power_predictor",
            withExtension: "safetensors"
        ) else {
            fatalError("power_predictor_epoch200_3.safetensors が見つかりません")
        }

        let arrays = try MLX.loadArrays(url: url)
        let parameters = ModuleParameters.unflattened(arrays)

        try update(parameters: parameters, verify: .all)
    }

    func predict(_ embedding: MLXArray) -> MLXArray {
        return linear(embedding)
    }
}
