// Session management service for handling session lifecycle and storage

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
  private static sessionId: string | null = null;

  /**
   * Initialize session - either start a new one or restore existing
   */
  static async initializeSession(): Promise<string> {
    // Try to get existing session from localStorage
    const existingSessionId = localStorage.getItem(this.SESSION_KEY);

    if (existingSessionId) {
      // Verify the session still exists on the server
      try {
        const response = await fetch("/api/session/info", {
          method: "GET",
          headers: {
            "X-Session-ID": existingSessionId,
            "Content-Type": "application/json",
          },
        });

        if (response.ok) {
          const data = await response.json();
          if (data.success) {
            this.sessionId = existingSessionId;
            console.log(`Restored existing session: ${data.session_id_short}`);
            return existingSessionId;
          }
        }
      } catch (error) {
        console.warn("Failed to verify existing session:", error);
      }
    }

    // Create new session
    return await this.startNewSession();
  }

  /**
   * Start a new session
   */
  static async startNewSession(): Promise<string> {
    try {
      const response = await fetch("/api/session/start", {
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
        localStorage.setItem(this.SESSION_KEY, data.session_id);
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
    return this.sessionId || localStorage.getItem(this.SESSION_KEY);
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
    localStorage.removeItem(this.SESSION_KEY);
  }
  /**
   * Get session information
   */
  static async getSessionInfo(): Promise<SessionInfo> {
    const sessionId = this.getSessionId();
    if (!sessionId) {
      throw new Error("No active session");
    }

    const response = await fetch("/api/session/info", {
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
}
