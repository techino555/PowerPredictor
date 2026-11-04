import Foundation
internal import Combine
import UIKit

enum ScanState {
    case idle
    case detecting
    case analyzing
    case complete
}

@MainActor
final class ScanViewModel: ObservableObject {
    @Published var state: ScanState = .idle
    @Published var detectedPersonRect: CGRect? = nil
    @Published var effectImage: UIImage? = nil
    @Published var completeImage: UIImage? = nil
    
    func startScan() {
        state = .detecting
    }
    
    func setAnalyzing() {
        state = .analyzing
    }
    
    func finishScan() {
        state = .complete
    }
    
    func reset() {
        state = .idle
        effectImage = nil
        completeImage = nil
        detectedPersonRect = nil
    }
}
