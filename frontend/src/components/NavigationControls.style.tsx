import type { CSSProperties } from "react";

export class NavigationControlsStyles {
  static readonly navigationControls: CSSProperties = {
    marginTop: "10px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "10px",
  };

  static readonly navButton: CSSProperties = {
    padding: "8px 12px",
    backgroundColor: "#f0f0f0",
    color: "black",
    border: "none",
    borderRadius: "4px",
    fontSize: "14px",
    cursor: "pointer",
  };

  static readonly navButtonDisabled: CSSProperties = {
    cursor: "not-allowed",
    opacity: 0.7,
  };

  static readonly imageCounter: CSSProperties = {
    fontSize: "14px",
    fontWeight: "bold",
    color: "#333",
  };
}
