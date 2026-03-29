import type { CSSProperties } from "react";
import type { ResolvedTheme } from "../contexts/ThemeContext";
import { getTokens } from "../contexts/themeTokens";

export function getSVGViewerStyles(theme: ResolvedTheme) {
  const t = getTokens(theme);
  return {
    svgContainer: {
      flex: 1,
      position: "relative",
      overflow: "hidden",
      backgroundColor: t.bgTertiary,
    } as CSSProperties,

    placeholderMessage: {
      position: "absolute",
      top: "50%",
      left: "50%",
      transform: "translate(-50%, -50%)",
      textAlign: "center",
      color: t.textMuted,
      fontSize: "18px",
      zIndex: 1,
    } as CSSProperties,

    placeholderSubtext: {
      fontSize: "14px",
      marginTop: "10px",
      color: t.textMuted,
    } as CSSProperties,

    svg: {
      border: `2px solid ${t.borderLight}`,
      backgroundColor: theme === "dark" ? "#2a2a2a" : "white",
      cursor: "crosshair",
    } as CSSProperties,

    svgWithData: {
      border: "2px solid #007acc",
    } as CSSProperties,

    loadingOverlay: {
      position: "absolute",
      top: "50%",
      left: "50%",
      transform: "translate(-50%, -50%)",
      backgroundColor: "rgba(0, 0, 0, 0.7)",
      color: "white",
      padding: "20px",
      borderRadius: "5px",
      zIndex: 2,
    } as CSSProperties,

    errorOverlay: {
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
    } as CSSProperties,
  };
}
