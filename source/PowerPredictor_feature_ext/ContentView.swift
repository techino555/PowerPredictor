//
//  ContentView.swift
//  PowerPredictor_feature_ext
//
//  Created by hidepo on 2026/07/26.
//
import SwiftUI
import AppKit
import MLX

/// Embedding生成の状態管理
@Observable
class TrainingViewModel {
    var selectedFolderURL: URL?
    var statusMessage: String = "フォルダを選択してください"
    var isProcessing: Bool = false
    var progress: Double = 0.0
    var totalImages: Int = 0
    var processedImages: Int = 0
    var skippedImages: Int = 0
    var results: [(filename: String, embedding: MLXArray)] = []
    var savedPath: String?

    /// フォルダ内の画像ファイルを取得する
    func imageFiles(in folderURL: URL) -> [URL] {
        let extensions = ["jpg", "jpeg", "png"]
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: folderURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }
        
        var imageURLs: [URL] = []
        for case let fileURL as URL in enumerator {
            if extensions.contains(fileURL.pathExtension.lowercased()) {
                imageURLs.append(fileURL)
            }
        }
        return imageURLs.sorted { $0.path < $1.path }
    }

    /// フォルダ内の画像を処理してEmbeddingを抽出する
    func processImages() async {
        guard let folderURL = selectedFolderURL else { return }
        isProcessing = true
        results = []
        progress = 0.0
        skippedImages = 0
        savedPath = nil
        let imageURLs = imageFiles(in: folderURL)
        totalImages = imageURLs.count
        processedImages = 0
        if imageURLs.isEmpty {
            statusMessage = "画像ファイルが見つかりませんでした"
            isProcessing = false
            return
        }

        statusMessage = "DINOv3モデルを読み込み中..."
        do {
            let extractor = try EmbeddingExtractor()
            for url in imageURLs {
                let filename = url.path.hasPrefix(folderURL.path) ? String(url.path.dropFirst(folderURL.path.count + 1)) : url.lastPathComponent
                statusMessage = "処理中: \(filename) (\(processedImages + 1)/\(totalImages))"
                guard let image = NSImage(contentsOf: url) else {
                    print("画像の読み込みに失敗: \(filename)")
                    skippedImages += 1
                    processedImages += 1
                    progress = Double(processedImages) / Double(totalImages)
                    continue
                }
                if let embedding = try await extractor.extract(from: image) {
                    results.append((filename: filename, embedding: embedding))
                } else {
                    print("Embedding抽出に失敗（人物未検出）: \(filename)")
                    skippedImages += 1
                }
                processedImages += 1
                progress = Double(processedImages) / Double(totalImages)
            }
            statusMessage = "完了: \(results.count)/\(totalImages)枚を処理（スキップ: \(skippedImages)枚）"
        } catch {
            statusMessage = "エラー: \(error.localizedDescription)"
        }
        isProcessing = false
    }

    /// CSVをDocumentsディレクトリに保存する
    func saveCSV() {
        guard !results.isEmpty else { return }
        // アプリのDocumentsディレクトリに保存
        guard let documentsURL = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first else {
            statusMessage = "Documentsディレクトリが見つかりません"
            return
        }

        // タイムスタンプ付きファイル名
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = formatter.string(from: Date())
        let filename = "embeddings_\(timestamp).csv"
        let csvURL = documentsURL.appendingPathComponent(filename)
        do {
            try CSVExporter.export(entries: results, to: csvURL)
            savedPath = csvURL.path
            statusMessage = "CSVを保存しました: \(csvURL.path)"
            // Finderで表示
            NSWorkspace.shared.activateFileViewerSelecting([csvURL])
        } catch {
            statusMessage = "CSV保存エラー: \(error.localizedDescription)"
        }
    }
}

struct ContentView: View {
    @State private var viewModel = TrainingViewModel()
    var body: some View {
        VStack(spacing: 20) {
            // タイトル
            Text("DINOv3 Embedding Generator")
                .font(.title)
                .fontWeight(.bold)
            
            // フォルダ選択
            HStack {
                Image(systemName: "folder")
                    .foregroundColor(.secondary)
                Text(viewModel.selectedFolderURL?.path ?? "未選択")
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Button("フォルダを選択") {
                    selectFolder()
                }
            }
            .padding(10)
            .background(Color.gray.opacity(0.1))
            .cornerRadius(8)
            
            // 進捗表示
            if viewModel.isProcessing {
                VStack(spacing: 8) {
                    ProgressView(value: viewModel.progress) {
                        Text(viewModel.statusMessage)
                            .font(.caption)
                    }
                    Text("\(viewModel.processedImages)/\(viewModel.totalImages)枚")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            } else {
                // ステータス
                Text(viewModel.statusMessage)
                    .font(.callout)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            
            // 結果サマリー
            if !viewModel.results.isEmpty {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Text("抽出済み: \(viewModel.results.count)件のEmbedding (768次元)")
                        .font(.callout)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            // 保存先パス表示
            if let path = viewModel.savedPath {
                HStack {
                    Image(systemName: "doc.text")
                        .foregroundColor(.blue)
                    Text(path)
                        .font(.caption)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            
            Spacer()
            // アクションボタン
            HStack(spacing: 16) {
                Button {
                    Task {
                        await viewModel.processImages()
                    }
                } label: {
                    Label("Embedding生成", systemImage: "wand.and.stars")
                }
                .disabled(viewModel.selectedFolderURL == nil || viewModel.isProcessing)
                .buttonStyle(.borderedProminent)
                Button {
                    viewModel.saveCSV()
                } label: {
                    Label("CSVを保存", systemImage: "square.and.arrow.down")
                }
                .disabled(viewModel.results.isEmpty || viewModel.isProcessing)
                .buttonStyle(.bordered)
            }
        }
        .padding(30)
        .frame(minWidth: 500, minHeight: 350)
    }
    
    
    private func selectFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.title = "画像フォルダを選択"
        if panel.runModal() == .OK {
            viewModel.selectedFolderURL = panel.url
            if let url = panel.url {
                let count = viewModel.imageFiles(in: url).count
                viewModel.statusMessage = "選択済み: \(count)枚の画像が見つかりました"
                viewModel.results = []
                viewModel.savedPath = nil
            }
        }
    }
}
