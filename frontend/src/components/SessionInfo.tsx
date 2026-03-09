import { Component } from "react";
import { ApiService } from "../services/ApiService";
import { SessionService } from "../services/SessionService";

interface SessionInfoState {
  sessionInfo: {
    session_id_short?: string;
    created_at?: string;
    file_count?: number;
  } | null;
  loading: boolean;
  error: string | null;
  isCached: boolean;
  storageType: "cookies" | "sessionStorage";
}

interface SessionInfoProps {
  onNavigateHome?: () => void;
}

export class SessionInfo extends Component<SessionInfoProps, SessionInfoState> {
  state: SessionInfoState = {
    sessionInfo: null,
    loading: true,
    error: null,
    isCached: false,
    storageType: SessionService.getStorageType(),
  };

  componentDidMount() {
    this.loadSessionInfo();
  }

  private async loadSessionInfo() {
    try {
      this.setState({ loading: true, error: null });

      // Check if session is cached
      const isCached =
        SessionService.hasActiveSession() &&
        SessionService.isCachedSessionFresh();

      const info = await ApiService.getSessionInfo();
      this.setState({
        sessionInfo: info,
        loading: false,
        isCached,
        storageType: SessionService.getStorageType(),
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
  render() {
    const { sessionInfo, loading, error, isCached, storageType } = this.state;

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
          alignItems: "center",
          gap: "8px",
        }}
      >
        <button
          onClick={this.props.onNavigateHome || (() => { window.location.href = "/"; })}
          style={{
            padding: "4px 8px",
            backgroundColor: "#ffffff",
            color: "#333",
            border: "1px solid #ccc",
            borderRadius: "4px",
            cursor: "pointer",
            fontWeight: "bold",
            fontSize: "12px",
            boxShadow: "0 1px 2px rgba(0,0,0,0.1)",
            display: "flex",
            alignItems: "center",
            marginRight: "8px",
          }}
          title="Return to home page"
        >
          &larr; Back to Home
        </button>
        <span>
          Session: {sessionInfo?.session_id_short ?? "Unknown"} | Files:{" "}
          {sessionInfo?.file_count ?? 0} | Started:{" "}
          {sessionInfo?.created_at
            ? new Date(
                sessionInfo.created_at.replace(
                  /(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/,
                  "$1-$2-$3T$4:$5:$6"
                )
              ).toLocaleString()
            : "Unknown"}{" "}
          | Storage: {storageType}
        </span>
        {isCached && (
          <span
            style={{
              backgroundColor: "#4caf50",
              color: "white",
              padding: "2px 6px",
              borderRadius: "3px",
              fontSize: "10px",
              fontWeight: "bold",
            }}
          >
            CACHED
          </span>
        )}
        {storageType === "cookies" && (
          <span
            style={{
              backgroundColor: "#2196f3",
              color: "white",
              padding: "2px 6px",
              borderRadius: "3px",
              fontSize: "10px",
              fontWeight: "bold",
            }}
          >
            PERSISTENT
          </span>
        )}
      </div>
    );
  }
}
