import { Component } from "react";
import { ApiService } from "../services/ApiService";

interface SessionInfoState {
  sessionInfo: {
    session_id_short?: string;
    created_at?: string;
    file_count?: number;
  } | null;
  loading: boolean;
  error: string | null;
}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export class SessionInfo extends Component<{}, SessionInfoState> {
  state: SessionInfoState = {
    sessionInfo: null,
    loading: true,
    error: null,
  };

  componentDidMount() {
    this.loadSessionInfo();
  }

  private async loadSessionInfo() {
    try {
      this.setState({ loading: true, error: null });
      const info = await ApiService.getSessionInfo();
      this.setState({
        sessionInfo: info,
        loading: false,
      });
    } catch (error) {
      this.setState({
        error:
          error instanceof Error
            ? error.message
            : "Failed to load session info",
        loading: false,
      });
    }
  }

  private async handleNewSession() {
    try {
      this.setState({ loading: true, error: null });
      await ApiService.startNewSession();
      await this.loadSessionInfo();
      // Reload the page to refresh all data with new session
      window.location.reload();
    } catch (error) {
      this.setState({
        error:
          error instanceof Error
            ? error.message
            : "Failed to start new session",
        loading: false,
      });
    }
  }

  render() {
    const { sessionInfo, loading, error } = this.state;

    if (loading) {
      return (
        <div
          style={{
            padding: "8px 12px",
            fontSize: "12px",
            color: "#666",
            borderBottom: "1px solid #e0e0e0",
          }}
        >
          Loading session...
        </div>
      );
    }

    if (error) {
      return (
        <div
          style={{
            padding: "8px 12px",
            fontSize: "12px",
            color: "#d32f2f",
            borderBottom: "1px solid #e0e0e0",
          }}
        >
          Session error: {error}
          <button
            onClick={() => this.handleNewSession()}
            style={{
              marginLeft: "8px",
              padding: "2px 6px",
              fontSize: "10px",
              backgroundColor: "#1976d2",
              color: "white",
              border: "none",
              borderRadius: "3px",
              cursor: "pointer",
            }}
          >
            New Session
          </button>
        </div>
      );
    }

    return (
      <div
        style={{
          padding: "8px 12px",
          fontSize: "12px",
          color: "#666",
          borderBottom: "1px solid #e0e0e0",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span>
          Session: {sessionInfo?.session_id_short || "Unknown"} | Files:{" "}
          {sessionInfo?.file_count || 0} | Started:{" "}
          {sessionInfo?.created_at
            ? new Date(
                sessionInfo.created_at.replace(
                  /(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/,
                  "$1-$2-$3T$4:$5:$6"
                )
              ).toLocaleString()
            : "Unknown"}
        </span>
        <button
          onClick={() => this.handleNewSession()}
          style={{
            padding: "2px 6px",
            fontSize: "10px",
            backgroundColor: "#1976d2",
            color: "white",
            border: "none",
            borderRadius: "3px",
            cursor: "pointer",
          }}
          title="Start a new session"
        >
          New Session
        </button>
      </div>
    );
  }
}
