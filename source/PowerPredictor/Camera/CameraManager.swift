import Foundation
import AVFoundation
import CoreImage
import UIKit
internal import Combine

class CameraManager: NSObject, ObservableObject {
    @Published var currentImage: UIImage?
    
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let photoOutput = AVCapturePhotoOutput()
    
    private let context = CIContext()
    
    override init() {
        super.init()
        setupCamera()
    }
    
    private func setupCamera() {
        // カメラ設定（高画質）
        captureSession.sessionPreset = .photo
        
        // 背面カメラの取得
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else {
            print("カメラへのアクセスに失敗しました。")
            return
        }
        
        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        }
        
        // ビデオ出力の設定
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        videoOutput.alwaysDiscardsLateVideoFrames = true
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
            updateVideoRotation()
        }
        
        // 写真出力の設定（SCANボタン用）
        if captureSession.canAddOutput(photoOutput) {
            captureSession.addOutput(photoOutput)
        }
        
        // バックグラウンドスレッドでセッションを開始
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession.startRunning()
        }
        
        NotificationCenter.default.addObserver(self, selector: #selector(deviceOrientationDidChange), name: UIDevice.orientationDidChangeNotification, object: nil)
    }
    
    @objc private func deviceOrientationDidChange() {
        updateVideoRotation()
    }
    
    private func updateVideoRotation() {
        guard let connection = videoOutput.connection(with: .video) else { return }
        
        DispatchQueue.main.async {
            let orientation = UIDevice.current.orientation
            let angle: CGFloat
            
            switch orientation {
            case .portrait:
                angle = 90
            case .portraitUpsideDown:
                angle = 270
            case .landscapeLeft: // デバイスの左側が下の時
                angle = 0
            case .landscapeRight: // デバイスの右側が下の時
                angle = 180
            default:
                // 不明な場合や上向き下向きの場合は現在の設定を維持するかデフォルトに
                angle = 90
            }
            
            if connection.isVideoRotationAngleSupported(angle) {
                connection.videoRotationAngle = angle
            }
        }
    }
    
    func capturePhoto() {
        // シミュレータ環境など、カメラの接続がない場合はクラッシュを防ぐためにガード
        guard let connection = photoOutput.connection(with: .video), connection.isActive else {
            print("エラー: カメラの接続がありません。シミュレータでは実機のカメラ機能は使用できません。")
            return
        }
        
        let settings = AVCapturePhotoSettings()
        // 必要に応じてフラッシュ設定などをここに追加
        photoOutput.capturePhoto(with: settings, delegate: self)
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // ピクセルバッファからCGImageを生成
        let ciImage = CIImage(cvPixelBuffer: imageBuffer)
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        
        // UIImageを生成（向きは要件に合わせて.upなどを指定）
        let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .up)
        
        // メインスレッドでUIを更新
        DispatchQueue.main.async {
            self.currentImage = uiImage
        }
    }
}

// MARK: - AVCapturePhotoCaptureDelegate
extension CameraManager: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let error = error {
            print("写真のキャプチャエラー: \(error.localizedDescription)")
            return
        }
        
        guard let imageData = photo.fileDataRepresentation(),
              let image = UIImage(data: imageData) else {
            return
        }
        
        // ここで撮影された1枚の写真をVLMの推論に渡す想定です。もしかして使わないか。
        print("写真の撮影が成功しました: \(image.size)")
    }
}
