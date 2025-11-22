import type { CSSProperties } from "react";

export class MeasurementsPanelStyles {
  static readonly container: CSSProperties = {
    padding: "15px",
    backgroundColor: "#f9f9f9",
    flex: 1,
    overflowY: "auto",
  };

  static readonly title: CSSProperties = {
    margin: "0 0 15px 0",
    fontSize: "18px",
    fontWeight: "bold",
    color: "#333",
  };

  static readonly measurementItem: CSSProperties = {
    marginBottom: "20px",
    padding: "12px",
    border: "1px solid #ddd",
    borderRadius: "4px",
    backgroundColor: "white",
  };

  static readonly measurementHeader: CSSProperties = {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "10px",
  };

  static readonly measurementLabel: CSSProperties = {
    fontSize: "14px",
    fontWeight: "600",
    color: "#555",
  };

  static readonly deleteButton: CSSProperties = {
    padding: "4px 8px",
    backgroundColor: "#dc3545",
    color: "white",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
    fontSize: "12px",
  };

  static readonly formGroup: CSSProperties = {
    marginBottom: "10px",
    display: "flex",
    alignItems: "center",
    gap: "10px",
  };

  static readonly label: CSSProperties = {
    minWidth: "100px",
    fontWeight: "500",
    color: "#555",
    fontSize: "13px",
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

  static readonly distanceDisplay: CSSProperties = {
    marginTop: "8px",
    padding: "8px",
    backgroundColor: "#e7f3ff",
    borderRadius: "4px",
    fontSize: "14px",
    fontWeight: "500",
    color: "#0066cc",
  };

  static readonly addButton: CSSProperties = {
    padding: "10px 15px",
    backgroundColor: "#28a745",
    color: "white",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
    fontSize: "14px",
    fontWeight: "500",
    marginTop: "10px",
    width: "100%",
  };

  static readonly emptyState: CSSProperties = {
    textAlign: "center",
    color: "#666",
    padding: "20px",
    fontSize: "14px",
  };
}

