import Foundation
import OSLog

/// Minimal SystemInfo stub for FluidAudio compatibility
public enum SystemInfo {
    private static var hasLogged = false
    
    public static func logOnce(using logger: AppLogger) async {
        guard !hasLogged else { return }
        hasLogged = true
        logger.info("SortformerTest running on \(ProcessInfo.processInfo.hostName)")
    }
    
    public static func summary() -> String {
        return "macOS"
    }
}
