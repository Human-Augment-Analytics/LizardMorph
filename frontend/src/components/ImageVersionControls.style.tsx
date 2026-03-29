import type { CSSProperties } from "react";
import type { ResolvedTheme } from "../contexts/ThemeContext";
import { getTokens } from "../contexts/themeTokens";

export function getImageVersionControlsStyles(theme: ResolvedTheme) {
  const t = getTokens(theme);
  return {
    imageVersionButtons: {
      marginTop: "10px",
      display: "flex",
      gap: "12px",
      justifyContent: "center",
      padding: "5px",
      flexDirection: "row",
      alignItems: "center",
    } as CSSProperties,

    versionButton: {
      padding: "8px 12px",
      backgroundColor: t.bgTertiary,
      color: t.text,
      border: "none",
      borderRadius: "4px",
      fontSize: "14px",
      cursor: "pointer",
      opacity: 1,
    } as CSSProperties,

    versionButtonActive: {
      backgroundColor: "#2196F3",
      color: "white",
    } as CSSProperties,

    versionButtonDisabled: {
      cursor: "not-allowed",
      opacity: 0.7,
    } as CSSProperties,
  };
}
