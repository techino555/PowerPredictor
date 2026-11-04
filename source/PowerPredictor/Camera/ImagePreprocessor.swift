import Foundation
import UIKit
import MLXDINOv3

#if canImport(MLX)
import MLX
#else
// MLXがない環境でもビルドが通るようにするためのダミー
public struct MLXArray {
    var data: [Float]
    public init(data: [Float]) {
        self.data = data
    }
}
#endif

final class ImagePreprocessor {
    
    private let processor = ImageProcessor()
    
    /// 画像の前処理を行い、MLXArrayテンソルに変換する
    func preprocess(_ image: UIImage) throws -> MLXArray {
        return try processor(image)
    }
}
