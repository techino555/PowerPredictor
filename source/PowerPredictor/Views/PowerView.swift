import SwiftUI
internal import Combine

struct PowerView: View {
    let powerScore: Float
    let state: ScanState
    
    @State private var scanValue = Int.random(in: 500...9999)

    private let timer = Timer.publish(
        every: 0.02,
        on: .main,
        in: .common
    ).autoconnect()
    
    var body: some View {
        VStack(spacing: 4) {
            Text("FIGHT")
                .font(.system(size: 70, weight: .black, design: .monospaced))
                .foregroundColor(.yellow)

            if state == .analyzing {
                Text("\(scanValue)")
                    .font(.system(size: 84, weight: .black, design: .monospaced))
                    .foregroundColor(.yellow)
                    .onReceive(timer) { _ in
                        scanValue = Int.random(in: 500...9999)
                    }
            } else {
                Text("\(Int(round(powerScore)))")
                    .font(.system(size: 84, weight: .black, design: .monospaced))
                    .foregroundColor(.yellow)
            }
            
            Spacer()
        }
        .opacity(state == .detecting ? 0 : 1)
        .padding(.top, 60)
        // 背景はなし（画面上部に浮いているように見せる）
    }
}
