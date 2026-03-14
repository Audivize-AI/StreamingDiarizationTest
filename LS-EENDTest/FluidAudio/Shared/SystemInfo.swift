import Foundation
import OSLog

/// Minimal SystemInfo stub for FluidAudio compatibility
public enum SystemInfo {
    @MainActor private static var hasLogged = false
    
    public static func logOnce(using logger: AppLogger) async {
        guard await !hasLogged else { return }
        Task { @MainActor in
            hasLogged = true
        }
        logger.info("SortformerTest running on \(ProcessInfo.processInfo.hostName)")
    }
    
    public static func summary() -> String {
        return "macOS"
    }
}
