import type { CSSProperties } from "react";

export class ScatterViewModelStyles {
  static readonly scatterView: CSSProperties = {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100vh",
    width: "100vw",
    padding: "20px",
    fontFamily: "Arial, sans-serif",
  };

  static readonly heading: CSSProperties = {
    fontSize: "1.5rem",
    fontWeight: "bold",
    color: "#333",
    marginBottom: "10px",
  };

  static readonly fileInput: CSSProperties = {
    marginBottom: "10px",
  };

  static readonly button: CSSProperties = {
    padding: "8px 16px",
    fontSize: "1rem",
    cursor: "pointer",
    backgroundColor: "#007BFF",
    color: "#fff",
    border: "none",
    borderRadius: "4px",
    margin: "10px 0",
    transition: "background-color 0.2s",
  };

  static readonly svgCanvas: CSSProperties = {
    marginTop: "20px",
    border: "1px solid #ccc",
    backgroundColor: "#f9f9f9",
  };
}
