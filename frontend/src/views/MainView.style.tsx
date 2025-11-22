import type { CSSProperties } from "react";

export class MainViewStyles {
  static readonly container: CSSProperties = {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    minWidth: "1400px",
  };

  static readonly mainContentArea: CSSProperties = {
    display: "flex",
    flex: 1,
    overflow: "hidden",
  };

  static readonly svgContainer: CSSProperties = {
    flex: 3,
    overflow: "auto",
    position: "relative",
  };

  static readonly measurementsPanel: CSSProperties = {
    width: "27vw",
    borderLeft: "1px solid #ccc",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  };

  static readonly placeholderMessage: CSSProperties = {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    textAlign: "center",
    color: "#666",
  };

  static readonly placeholderSubtext: CSSProperties = {
    fontSize: "0.9em",
  };

  static readonly svg: CSSProperties = {
    display: "block",
    margin: "0 auto",
    boxShadow: "0 0 5px rgba(0,0,0,0.2)",
  };

  static readonly svgWithData: CSSProperties = {
    backgroundColor: "#f9f9f9",
  };

  static readonly loadingOverlay: CSSProperties = {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    backgroundColor: "rgba(255,255,255,0.8)",
    padding: "15px",
    borderRadius: "5px",
    boxShadow: "0 0 10px rgba(0,0,0,0.1)",
  };

  static readonly errorOverlay: CSSProperties = {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    backgroundColor: "rgba(255,220,220,0.9)",
    padding: "15px",
    borderRadius: "5px",
    color: "red",
    boxShadow: "0 0 10px rgba(255,0,0,0.2)",
  };
}
