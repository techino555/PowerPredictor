import SwiftUI

struct ScanOverlayView: View {
    let state: ScanState
    let boundingBox: CGRect?
    let effectImage: UIImage?
    
    var body: some View {
        ZStack {
            // 背景や人物を色付けしたエフェクト画像
            ScanEffectView(effectImage: effectImage)
            
            // L字フレームやターゲットマーカー
            if state != .idle {
                HUDOverlayView(boundingBox: boundingBox)
            }
        }
    }
}

struct ScanEffectView: View {
    let effectImage: UIImage?
    
    var body: some View {
        Group {
            if let uiImage = effectImage {
                Image(uiImage: uiImage)
                    .resizable()
                    .scaledToFill()
                    .ignoresSafeArea()
            } else {
                Color.clear
            }
        }
    }
}

struct HUDOverlayView: View {
    let boundingBox: CGRect?
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                let framePadding: CGFloat = 20
                let frameSize: CGFloat = 40
                let lineWidth: CGFloat = 4
                
                Path { path in
                    // 左上
                    path.move(to: CGPoint(x: framePadding, y: framePadding + frameSize))
                    path.addLine(to: CGPoint(x: framePadding, y: framePadding))
                    path.addLine(to: CGPoint(x: framePadding + frameSize, y: framePadding))
                    
                    // 右上
                    path.move(to: CGPoint(x: geometry.size.width - framePadding - frameSize, y: framePadding))
                    path.addLine(to: CGPoint(x: geometry.size.width - framePadding, y: framePadding))
                    path.addLine(to: CGPoint(x: geometry.size.width - framePadding, y: framePadding + frameSize))
                    
                    // 左下
                    path.move(to: CGPoint(x: framePadding, y: geometry.size.height - framePadding - frameSize))
                    path.addLine(to: CGPoint(x: framePadding, y: geometry.size.height - framePadding))
                    path.addLine(to: CGPoint(x: framePadding + frameSize, y: geometry.size.height - framePadding))
                    
                    // 右下
                    path.move(to: CGPoint(x: geometry.size.width - framePadding - frameSize, y: geometry.size.height - framePadding))
                    path.addLine(to: CGPoint(x: geometry.size.width - framePadding, y: geometry.size.height - framePadding))
                    path.addLine(to: CGPoint(x: geometry.size.width - framePadding, y: geometry.size.height - framePadding - frameSize))
                }
                .stroke(Color.yellow, lineWidth: lineWidth)
                
                if let rect = boundingBox {
                    let screenRect = CGRect(
                        x: rect.minX * geometry.size.width,
                        y: (1 - rect.maxY) * geometry.size.height,
                        width: rect.width * geometry.size.width,
                        height: rect.height * geometry.size.height
                    )
                    
                    let markerOffset = max(35, screenRect.width * 0.08)
                    let verticalOffset = screenRect.height * 0.12
                    
                    TriangleMarker(direction: .down)
                        .frame(width: 45, height: 45)
                        .position(x: screenRect.midX, y: screenRect.minY - markerOffset - verticalOffset)
                    
                    ZStack {
                        TriangleMarker(direction: .up)
                            .frame(width: 45, height: 45)
                            
                        Text("SET")
                            .font(.system(size: 15, weight: .bold))
                            .foregroundColor(.black)
                            .offset(y: 8)
                    }
                    .position(
                        x: screenRect.midX,
                        y: screenRect.maxY + markerOffset - verticalOffset
                    )
                    
                    TriangleMarker(direction: .right)
                        .frame(width: 45, height: 45)
                        .position(x: screenRect.minX - markerOffset, y: screenRect.midY - verticalOffset)
                    
                    TriangleMarker(direction: .left)
                        .frame(width: 45, height: 45)
                        .position(x: screenRect.maxX + markerOffset, y: screenRect.midY - verticalOffset)
                }
            }
        }
    }
}

enum TriangleDirection {
    case up, down, left, right
}

struct TriangleMarker: View {
    let direction: TriangleDirection
    
    var body: some View {
        GeometryReader { geometry in
            Path { path in
                let w = geometry.size.width
                let h = geometry.size.height
                
                switch direction {
                case .up:
                    path.move(to: CGPoint(x: w/2, y: 0))
                    path.addLine(to: CGPoint(x: w, y: h))
                    path.addLine(to: CGPoint(x: 0, y: h))
                case .down:
                    path.move(to: CGPoint(x: 0, y: 0))
                    path.addLine(to: CGPoint(x: w, y: 0))
                    path.addLine(to: CGPoint(x: w/2, y: h))
                case .left:
                    path.move(to: CGPoint(x: w, y: 0))
                    path.addLine(to: CGPoint(x: w, y: h))
                    path.addLine(to: CGPoint(x: 0, y: h/2))
                case .right:
                    path.move(to: CGPoint(x: 0, y: 0))
                    path.addLine(to: CGPoint(x: w, y: h/2))
                    path.addLine(to: CGPoint(x: 0, y: h))
                }
                path.closeSubpath()
            }
            .stroke(Color.yellow, lineWidth: 3)
            .background(
                Path { path in
                    let w = geometry.size.width
                    let h = geometry.size.height
                    switch direction {
                    case .up: path.move(to: CGPoint(x: w/2, y: 0)); path.addLine(to: CGPoint(x: w, y: h)); path.addLine(to: CGPoint(x: 0, y: h))
                    case .down: path.move(to: CGPoint(x: 0, y: 0)); path.addLine(to: CGPoint(x: w, y: 0)); path.addLine(to: CGPoint(x: w/2, y: h))
                    case .left: path.move(to: CGPoint(x: w, y: 0)); path.addLine(to: CGPoint(x: w, y: h)); path.addLine(to: CGPoint(x: 0, y: h/2))
                    case .right: path.move(to: CGPoint(x: 0, y: 0)); path.addLine(to: CGPoint(x: w, y: h/2)); path.addLine(to: CGPoint(x: 0, y: h))
                    }
                    path.closeSubpath()
                }.fill(Color.yellow.opacity(0.7))
            )
        }
    }
}
