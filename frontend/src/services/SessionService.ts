// Session management service for handling session lifecycle and storage
import { API_URL } from "./config";
interface SessionInfo {
  success: boolean;
  session_id: string;
  session_id_short: string;
  created_at: string;
  session_folder: string;
  file_count: number;
}

export class SessionService {
  private static readonly SESSION_KEY = "lizardmorph_session_id";
  private static readonly SESSION_TIMESTAMP_KEY =
    "lizardmorph_session_timestamp";
  private static readonly SESSION_CACHE_DURATION = 7 * 24 * 60 * 60 * 1000; // 7 Days in milliseconds
  private static sessionId: string | null = null;

  static {
    // Initialize sessionId from storage when the class is first loaded
    this.sessionId = sessionStorage.getItem(this.SESSION_KEY);
  }
  /**
   * Initialize session - reuse existing session if available, otherwise start new one
   */
  static async initializeSession(): Promise<string> {
    // Check if we have a cached session ID
    const cachedSessionId = sessionStorage.getItem(this.SESSION_KEY);
    const cachedTimestamp = sessionStorage.getItem(this.SESSION_TIMESTAMP_KEY);

    if (cachedSessionId && cachedTimestamp) {
      const timestamp = parseInt(cachedTimestamp, 10);
      const now = Date.now();

      // Check if cached session is still within the cache duration
      if (now - timestamp < this.SESSION_CACHE_DURATION) {
        this.sessionId = cachedSessionId;
        console.log(
          `Reusing cached session: ${cachedSessionId.substring(0, 8)}`
        );
        return cachedSessionId;
      }

      // Cache expired, validate with server
      try {
        const isValid = await this.validateSession(cachedSessionId);
        if (isValid) {
          this.sessionId = cachedSessionId;
          // Update timestamp
          sessionStorage.setItem(this.SESSION_TIMESTAMP_KEY, now.toString());
          console.log(
            `Reusing validated session: ${cachedSessionId.substring(0, 8)}`
          );
          return cachedSessionId;
        } else {
          // Session is invalid, clear it and start new one
          console.log("Cached session is invalid, starting new session");
          this.clearSession();
        }
      } catch (error) {
        console.warn(
          "Failed to validate cached session, starting new session:",
          error
        );
        this.clearSession();
      }
    }

    // No valid cached session, start a new one
    return await this.startNewSession();
  }

  /**
   * Start a new session
   */
  static async startNewSession(): Promise<string> {
    try {
      const response = await fetch(`${API_URL}/session/start`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to start session: ${response.status}`);
      }

      const data = await response.json();
      if (data.success && data.session_id) {
        this.sessionId = data.session_id;
        // Store in sessionStorage with timestamp
        sessionStorage.setItem(this.SESSION_KEY, data.session_id);
        sessionStorage.setItem(
          this.SESSION_TIMESTAMP_KEY,
          Date.now().toString()
        );
        console.log(`Started new session: ${data.session_id.substring(0, 8)}`);
        return data.session_id;
      } else {
        throw new Error("Invalid response from session start");
      }
    } catch (error) {
      console.error("Failed to start session:", error);
      throw error;
    }
  }
  /**
   * Get current session ID
   */
  static getSessionId(): string | null {
    return this.sessionId ?? sessionStorage.getItem(this.SESSION_KEY);
  }

  /**
   * Get session headers for API requests
   */
  static getSessionHeaders(): Record<string, string> {
    const sessionId = this.getSessionId();
    return sessionId ? { "X-Session-ID": sessionId } : {};
  }
  /**
   * Clear current session
   */
  static clearSession(): void {
    this.sessionId = null;
    sessionStorage.removeItem(this.SESSION_KEY);
    sessionStorage.removeItem(this.SESSION_TIMESTAMP_KEY);
  }
  /**
   * Get session information
   */
  static async getSessionInfo(): Promise<SessionInfo> {
    const sessionId = this.getSessionId();
    if (!sessionId) {
      throw new Error("No active session");
    }

    const response = await fetch(`${API_URL}/session/info`, {
      method: "GET",
      headers: {
        ...this.getSessionHeaders(),
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to get session info: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Check if session is active
   */
  static hasActiveSession(): boolean {
    return !!this.getSessionId();
  }

  /**
   * Validate if a session ID is still active on the server
   */
  static async validateSession(sessionId: string): Promise<boolean> {
    try {
      const response = await fetch(`${API_URL}/session/info`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          "X-Session-ID": sessionId,
        },
      });

      if (!response.ok) {
        return false;
      }

      const data = await response.json();
      return data.success === true;
    } catch (error) {
      console.error("Session validation failed:", error);
      return false;
    }
  }

  /**
   * Check if cached session is still fresh (within cache duration)
   */
  static isCachedSessionFresh(): boolean {
    const cachedTimestamp = sessionStorage.getItem(this.SESSION_TIMESTAMP_KEY);
    if (!cachedTimestamp) {
      return false;
    }

    const timestamp = parseInt(cachedTimestamp, 10);
    const now = Date.now();
    return now - timestamp < this.SESSION_CACHE_DURATION;
  }

  /**
   * Force create a new session (for testing or debugging)
   */
  static async forceNewSession(): Promise<string> {
    console.log("Forcing new session creation...");
    this.clearSession();
    return await this.startNewSession();
  }
}
