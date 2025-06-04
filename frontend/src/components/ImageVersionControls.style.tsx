import type { CSSProperties } from "react";

export class ImageVersionControlsStyles {
  static readonly imageVersionButtons: CSSProperties = {
    marginTop: "10px",
    display: "flex",
    gap: "10px",
    justifyContent: "center",
  };

  static readonly versionButton: CSSProperties = {
    padding: "8px 12px",
    backgroundColor: "#f0f0f0",
    color: "black",
    border: "none",
    borderRadius: "4px",
    fontSize: "14px",
    cursor: "pointer",
    opacity: 1,
  };

  static readonly versionButtonActive: CSSProperties = {
    backgroundColor: "#2196F3",
    color: "white",
  };

  static readonly versionButtonDisabled: CSSProperties = {
    cursor: "not-allowed",
    opacity: 0.7,
  };
}
