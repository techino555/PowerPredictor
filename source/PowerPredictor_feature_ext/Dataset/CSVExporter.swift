import Foundation
import MLX

/// EmbeddingデータをCSV形式で保存する
struct CSVExporter {

    /// Embeddingの次元数
    static let embeddingDimension = 768

    /*
     既存の列名と型の対応表 (Embedding列 e0~ を除く):
     1. filename: String
     2. person_id: String
     3. power: String
     4. body_direction: String
     5. take: String
     6. distance: String
     7. posture: String
     8. gender: String
     */
    
    /// CSVのヘッダーを生成する
    static func generateHeader() -> String {
        let embeddingColumns = (0..<embeddingDimension).map { "e\($0)" }.joined(separator: ",")
        return "filename,person_id,power,body_direction,take,distance,posture,gender,\(embeddingColumns)"
    }

    /// 1行分のCSVデータを生成する
    static func generateRow(
        filename: String,
        embedding: MLXArray,
        personId: String = "",
        power: String = "",
        bodyDirection: String = "",
        take: String = "",
        distance: String = "",
        posture: String = "",
        gender: String = ""
    ) -> String {
        // MLXArrayからFloat配列に変換
        let values = embedding.asArray(Float.self)
        let embeddingString = values.map { String($0) }.joined(separator: ",")
        return "\(filename),\(personId),\(power),\(bodyDirection),\(take),\(distance),\(posture),\(gender),\(embeddingString)"
    }

    /// 複数のEmbeddingをCSVファイルとして保存する
    /// - Parameters:
    ///   - entries: (ファイル名, Embedding)のタプル配列
    ///   - url: 保存先のファイルURL
    static func export(
        entries: [(filename: String, embedding: MLXArray)],
        to url: URL
    ) throws {
        var lines: [String] = []

        // ヘッダー行
        lines.append(generateHeader())

        // データ行（powerカラムは空欄）
        for entry in entries {
            let row = generateRow(filename: entry.filename, embedding: entry.embedding)
            lines.append(row)
        }

        let csv = lines.joined(separator: "\n")
        try csv.write(to: url, atomically: true, encoding: .utf8)
    }
}