import type { CSSProperties } from "react";
import type { ResolvedTheme } from "../contexts/ThemeContext";
import { getTokens } from "../contexts/themeTokens";

export function getMainViewStyles(theme: ResolvedTheme) {
  const t = getTokens(theme);
  return {
    container: {
      display: "flex",
      flexDirection: "column",
      height: "100vh",
      backgroundColor: t.bg,
      color: t.text,
    } as CSSProperties,

    mainContentArea: {
      display: "flex",
      flex: 1,
      overflow: "visible",
    } as CSSProperties,

    svgContainer: {
      flex: 1,
      overflow: "visible",
      position: "relative",
    } as CSSProperties,

    measurementsPanel: {
      width: "27vw",
      borderLeft: `1px solid ${t.border}`,
      display: "flex",
      flexDirection: "column",
      overflow: "hidden",
    } as CSSProperties,

    placeholderMessage: {
      position: "absolute",
      top: "50%",
      left: "50%",
      transform: "translate(-50%, -50%)",
      textAlign: "center",
      color: t.textMuted,
    } as CSSProperties,

    placeholderSubtext: {
      fontSize: "0.9em",
    } as CSSProperties,

    svg: {
      display: "block",
      margin: "0 auto",
      boxShadow: "0 0 5px rgba(0,0,0,0.2)",
    } as CSSProperties,

    svgWithData: {
      backgroundColor: t.bgSecondary,
    } as CSSProperties,

    loadingOverlay: {
      position: "absolute",
      top: "50%",
      left: "50%",
      transform: "translate(-50%, -50%)",
      backgroundColor: t.overlay,
      padding: "15px",
      borderRadius: "5px",
      boxShadow: "0 0 10px rgba(0,0,0,0.1)",
      color: t.text,
    } as CSSProperties,

    errorOverlay: {
      position: "absolute",
      top: "50%",
      left: "50%",
      transform: "translate(-50%, -50%)",
      backgroundColor: t.errorBg,
      padding: "15px",
      borderRadius: "5px",
      color: t.error,
      boxShadow: "0 0 10px rgba(255,0,0,0.2)",
    } as CSSProperties,
  };
}
