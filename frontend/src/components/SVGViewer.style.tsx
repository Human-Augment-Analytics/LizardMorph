// filepath: c:\src\Git\LizardMorph\frontend\src\components\SVGViewer.style.tsx
import type { CSSProperties } from "react";

export class SVGViewerStyles {
  public static readonly svgContainer: CSSProperties = {
    flex: 1,
    position: "relative",
    overflow: "hidden",
    backgroundColor: "#f0f0f0",
  };

  public static readonly placeholderMessage: CSSProperties = {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    textAlign: "center",
    color: "#666",
    fontSize: "18px",
    zIndex: 1,
  };

  public static readonly placeholderSubtext: CSSProperties = {
    fontSize: "14px",
    marginTop: "10px",
    color: "#999",
  };

  public static readonly svg: CSSProperties = {
    border: "2px solid #ddd",
    backgroundColor: "white",
    cursor: "crosshair",
  };

  public static readonly svgWithData: CSSProperties = {
    border: "2px solid #007acc",
  };

  public static readonly loadingOverlay: CSSProperties = {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    color: "white",
    padding: "20px",
    borderRadius: "5px",
    zIndex: 2,
  };

  public static readonly errorOverlay: CSSProperties = {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    backgroundColor: "rgba(255, 0, 0, 0.8)",
    color: "white",
    padding: "20px",
    borderRadius: "5px",
    zIndex: 2,
    textAlign: "center",
  };
}
