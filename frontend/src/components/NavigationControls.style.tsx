import type { CSSProperties } from "react";
import type { ResolvedTheme } from "../contexts/theme";
import { getTokens } from "../contexts/themeTokens";

export function getNavigationControlsStyles(theme: ResolvedTheme) {
  const t = getTokens(theme);
  const isDark = theme === "dark";

  return {
    navigationControls: {
      marginTop: "10px",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      gap: "10px",
      flexWrap: "wrap",
    } as CSSProperties,

    navButton: {
      padding: "8px 12px",
      backgroundColor: t.bgTertiary,
      color: t.text,
      border: "none",
      borderRadius: "4px",
      fontSize: "14px",
      cursor: "pointer",
      transition: "background-color 0.15s ease",
    } as CSSProperties,

    navButtonDisabled: {
      cursor: "not-allowed",
      opacity: 0.6,
    } as CSSProperties,

    imageCounter: {
      fontSize: "14px",
      fontWeight: "bold",
      color: t.text,
    } as CSSProperties,

    divider: {
      width: "1px",
      height: "22px",
      backgroundColor: t.borderLight,
      margin: "0 4px",
    } as CSSProperties,

    qcGroup: {
      display: "flex",
      alignItems: "center",
      gap: "8px",
    } as CSSProperties,

    statusSelector: {
      display: "inline-flex",
      alignItems: "center",
      gap: "6px",
      padding: "4px 8px",
      borderRadius: "4px",
      border: `1px solid ${t.borderLight}`,
      backgroundColor: isDark ? t.surface : t.bgSecondary,
      fontSize: "13px",
      color: t.text,
    } as CSSProperties,

    statusSelectDropdown: {
      backgroundColor: "transparent",
      color: t.text,
      border: "none",
      fontSize: "13px",
      fontWeight: "600",
      cursor: "pointer",
      outline: "none",
    } as CSSProperties,

    approveButton: {
      padding: "8px 14px",
      backgroundColor: isDark ? "#2e7d32" : "#388e3c",
      color: "#ffffff",
      border: "none",
      borderRadius: "4px",
      fontSize: "13px",
      fontWeight: "bold",
      cursor: "pointer",
      display: "inline-flex",
      alignItems: "center",
      gap: "6px",
      boxShadow: "0 1px 3px rgba(0,0,0,0.15)",
      transition: "all 0.15s ease",
    } as CSSProperties,

    approveButtonDisabled: {
      cursor: "not-allowed",
      opacity: 0.5,
      boxShadow: "none",
    } as CSSProperties,

    approveKbd: {
      backgroundColor: "rgba(255, 255, 255, 0.22)",
      color: "#ffffff",
      padding: "1px 5px",
      borderRadius: "3px",
      fontSize: "11px",
      fontWeight: "500",
      letterSpacing: "0.5px",
    } as CSSProperties,
  };
}
