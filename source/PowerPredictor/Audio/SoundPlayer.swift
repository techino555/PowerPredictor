import Foundation
import AVFoundation

final class SoundPlayer {

    static let shared = SoundPlayer()

    private var player: AVAudioPlayer?

    func playScan() {
        guard let url = Bundle.main.url(forResource: "nc116296_scouter", withExtension: "wav") else {
            print("wavファイルが見つかりません")
            return
        }

        do {
            player = try AVAudioPlayer(contentsOf: url)
            player?.prepareToPlay()
            player?.play()
        } catch {
            print("音声再生エラー: \(error)")
        }
    }
}
