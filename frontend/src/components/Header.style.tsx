import type { CSSProperties } from "react";

export class HeaderStyles {
  static readonly header: CSSProperties = {
    padding: "15px",
    borderBottom: "1px solid #ccc",
    textAlign: "center",
    minHeight: "220px",
    position: "relative",
  };

  static readonly infoBox: CSSProperties = {
    position: "absolute",
    top: "10px",
    right: "15px",
    padding: "12px",
    backgroundColor: "#f8f9fa",
    borderRadius: "8px",
    boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
    border: "1px solid #dee2e6",
    textAlign: "right",
    minWidth: "300px",
  };

  static readonly infoBoxContent: CSSProperties = {
    fontSize: "0.9em",
    color: "#495057",
  };

  static readonly infoBoxParagraph: CSSProperties = {
    margin: "4px 0",
  };

  static readonly infoBoxItalic: CSSProperties = {
    margin: "4px 0",
    fontStyle: "italic",
  };

  static readonly infoBoxLink: CSSProperties = {
    color: "#0056b3",
    textDecoration: "none",
  };

  static readonly lizardCount: CSSProperties = {
    marginTop: "5px",
    color: "#0056b3",
  };

  static readonly mainContent: CSSProperties = {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    maxWidth: "1200px",
    margin: "15px auto 0",
    padding: "0 20px",
    marginRight: "340px",
    gap: "60px",
  };

  static readonly buttonContainer: CSSProperties = {
    display: "flex",
    flexDirection: "column",
    alignItems: "flex-start",
    gap: "15px",
    width: "220px",
    marginTop: "20px",
  };

  static readonly uploadButton: CSSProperties = {
    padding: "12px 20px",
    backgroundColor: "#2196F3",
    color: "white",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
    display: "inline-block",
    fontWeight: "bold",
    fontSize: "14px",
    textAlign: "center",
    boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
    width: "100%",
    boxSizing: "border-box",
  };

  static readonly uploadButtonDisabled: CSSProperties = {
    cursor: "not-allowed",
    opacity: 0.7,
  };

  static readonly exportButton: CSSProperties = {
    padding: "12px 20px",
    backgroundColor: "#4CAF50",
    color: "white",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
    width: "100%",
    boxSizing: "border-box",
    fontWeight: "bold",
    fontSize: "14px",
    boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
  };

  static readonly exportButtonDisabled: CSSProperties = {
    cursor: "not-allowed",
    opacity: 0.7,
  };

  static readonly clearHistoryButton: CSSProperties = {
    padding: "12px 20px",
    backgroundColor: "#f44336",
    color: "white",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
    width: "100%",
    boxSizing: "border-box",
    fontWeight: "bold",
    fontSize: "14px",
    boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
  };

  static readonly clearHistoryButtonDisabled: CSSProperties = {
    cursor: "not-allowed",
    opacity: 0.7,
  };

  static readonly titleContainer: CSSProperties = {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  };

  static readonly logo: CSSProperties = {
    height: "40px",
    marginRight: "12px",
  };

  static readonly title: CSSProperties = {
    margin: 0,
    whiteSpace: "nowrap",
  };

  static readonly viewType: CSSProperties = {
    margin: "8px 0 0 0",
    fontSize: "0.9em",
    color: "#666",
    fontWeight: "500",
    textTransform: "capitalize",
    letterSpacing: "0.5px",
  };

  static readonly rightSpacer: CSSProperties = {
    width: "220px",
  };

  static readonly errorMessage: CSSProperties = {
    color: "red",
  };
}
