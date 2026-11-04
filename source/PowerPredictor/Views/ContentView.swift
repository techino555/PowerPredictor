import SwiftUI

struct ContentView: View {
    @StateObject private var cameraManager = CameraManager()
    @StateObject private var store = ScouterStore()
    @StateObject private var scanViewModel = ScanViewModel()
    
    var body: some View {
        // ZStackのサイズがImageのscaledToFillによって画面外まで広がらないようにする
        GeometryReader { geometry in
            ZStack {
                // 背景レイヤー (カメラプレビューまたはエフェクト画像)
                Group {
                    if scanViewModel.state == .idle {
                        if let uiImage = cameraManager.currentImage {
                            Image(uiImage: uiImage)
                                .resizable()
                                .scaledToFill()
                                .frame(width: geometry.size.width, height: geometry.size.height)
                                .clipped()
                                .ignoresSafeArea()
                        } else {
                            Color.black.ignoresSafeArea()
                            Text("カメラを起動中...")
                                .foregroundColor(.white)
                        }
                    } else {
                        let displayImage = (scanViewModel.state == .complete) ? scanViewModel.completeImage : scanViewModel.effectImage
                        if let img = displayImage ?? cameraManager.currentImage {
                            Image(uiImage: img)
                                .resizable()
                                .scaledToFill()
                                .frame(width: geometry.size.width, height: geometry.size.height)
                                .clipped()
                                .ignoresSafeArea()
                        } else {
                            Color.black.ignoresSafeArea()
                        }
                    }
                }
                
                // HUDオーバーレイ (detecting, analyzing, completeで表示)
                if scanViewModel.state != .idle {
                    ScanOverlayView(
                        state: scanViewModel.state,
                        boundingBox: scanViewModel.detectedPersonRect,
                        effectImage: nil // 画像描画は背景レイヤーで行うため不要
                    )
                    .frame(width: geometry.size.width, height: geometry.size.height)
                }
                
                // 全体UIレイヤー (Safe Area内に収める)
                VStack {
                    // 上部UI (Power Score)
                    if scanViewModel.state != .idle {
                        PowerView(
                            powerScore: store.powerScore ?? 0,
                            state: scanViewModel.state
                        )
                    }
                    
                    Spacer()
                    
                    // 下部UI (SCANボタン・リセットボタン)
                    if scanViewModel.state == .idle {
                        Button(action: {
                            guard let image = cameraManager.currentImage else { return }
                            Task {
                                await store.processImage(image, viewModel: scanViewModel)
                            }
                        }) {
                            VStack {
                                Text("SCAN")
                                    .font(.headline)
                                    .foregroundColor(.yellow)
                                Circle()
                                    .strokeBorder(Color.yellow, lineWidth: 4)
                                    .background(Circle().fill(Color.yellow.opacity(0.3)))
                                    .frame(width: 80, height: 80)
                            }
                        }
                        .disabled(store.isProcessing)
                        .padding(.bottom, 20)
                    } else if scanViewModel.state == .complete {
                        Button(action: {
                            scanViewModel.reset()
                        }) {
                            Text("RESET")
                                .font(.headline)
                                .foregroundColor(.white)
                                .padding()
                                .background(Color.gray.opacity(0.7))
                                .cornerRadius(12)
                        }
                        .padding(.bottom, 20)
                    }
                }
                .padding(.top, 20)
                .frame(width: geometry.size.width, height: geometry.size.height)
            }
        }
    }
}

#Preview {
    ContentView()
}
