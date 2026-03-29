import type { CSSProperties } from "react";
import type { ResolvedTheme } from "../contexts/ThemeContext";
import { getTokens } from "../contexts/themeTokens";

export function getNavigationControlsStyles(theme: ResolvedTheme) {
  const t = getTokens(theme);
  return {
    navigationControls: {
      marginTop: "10px",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      gap: "10px",
    } as CSSProperties,

    navButton: {
      padding: "8px 12px",
      backgroundColor: t.bgTertiary,
      color: t.text,
      border: "none",
      borderRadius: "4px",
      fontSize: "14px",
      cursor: "pointer",
    } as CSSProperties,

    navButtonDisabled: {
      cursor: "not-allowed",
      opacity: 0.7,
    } as CSSProperties,

    imageCounter: {
      fontSize: "14px",
      fontWeight: "bold",
      color: t.text,
    } as CSSProperties,
  };
}
