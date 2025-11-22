import type { CSSProperties } from "react";

export class ScaleSettingsStyles {
  static readonly container: CSSProperties = {
    padding: "15px",
    borderBottom: "1px solid #ccc",
    backgroundColor: "#f9f9f9",
  };

  static readonly title: CSSProperties = {
    margin: "0 0 15px 0",
    fontSize: "18px",
    fontWeight: "bold",
    color: "#333",
  };

  static readonly formGroup: CSSProperties = {
    marginBottom: "12px",
    display: "flex",
    alignItems: "center",
    gap: "10px",
  };

  static readonly label: CSSProperties = {
    minWidth: "80px",
    fontWeight: "500",
    color: "#555",
  };

  static readonly select: CSSProperties = {
    flex: 1,
    padding: "6px 8px",
    border: "1px solid #ccc",
    borderRadius: "4px",
    fontSize: "14px",
    backgroundColor: "white",
  };

  static readonly input: CSSProperties = {
    flex: 1,
    padding: "6px 8px",
    border: "1px solid #ccc",
    borderRadius: "4px",
    fontSize: "14px",
  };

  static readonly infoText: CSSProperties = {
    fontSize: "12px",
    color: "#666",
    fontStyle: "italic",
    marginTop: "8px",
  };
}

