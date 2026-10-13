import type { CSSProperties } from "react";
import type { ResolvedTheme } from "../contexts/theme";
import { getTokens } from "../contexts/themeTokens";

export function getToepadModalStyles(theme: ResolvedTheme, isEditMode: boolean = false) {
  const t = getTokens(theme);
  const isDark = theme === "dark";

  return {
    backdrop: {
      position: "fixed",
      inset: 0,
      backgroundColor: isDark ? "rgba(0, 0, 0, 0.80)" : "rgba(15, 23, 42, 0.70)",
      backdropFilter: "blur(6px)",
      WebkitBackdropFilter: "blur(6px)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      zIndex: 10000,
      padding: 16,
      boxSizing: "border-box",
    } as CSSProperties,

    dialog: {
      position: "relative",
      width: "min(90vw, 1100px)",
      height: "min(85vh, 850px)",
      maxHeight: "85vh",
      maxWidth: "90vw",
      display: "flex",
      flexDirection: "column",
      borderRadius: 14,
      overflow: "hidden",
      border: `1px solid ${isDark ? "rgba(255, 255, 255, 0.16)" : "rgba(0, 0, 0, 0.14)"}`,
      background: isDark ? "#17202e" : "#ffffff",
      boxShadow: isDark
        ? "0 25px 60px -15px rgba(0, 0, 0, 0.85), 0 0 0 1px rgba(255, 255, 255, 0.08)"
        : "0 25px 60px -15px rgba(0, 0, 0, 0.40), 0 0 0 1px rgba(0, 0, 0, 0.06)",
      color: t.text,
    } as CSSProperties,

    header: {
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      padding: "10px 16px",
      borderBottom: `1px solid ${t.borderLight}`,
      background: isDark ? "rgba(255, 255, 255, 0.03)" : "rgba(0, 0, 0, 0.02)",
      gap: 12,
      flexShrink: 0,
      flexWrap: "wrap",
    } as CSSProperties,

    headerLeft: {
      display: "flex",
      alignItems: "center",
      gap: 10,
      minWidth: 0,
    } as CSSProperties,

    title: {
      display: "flex",
      alignItems: "center",
      gap: 8,
      fontSize: 16,
      fontWeight: 800,
      color: isDark ? "#f0f0f0" : "#111827",
      margin: 0,
      whiteSpace: "nowrap",
    } as CSSProperties,

    swatch: {
      width: 10,
      height: 10,
      borderRadius: "50%",
      flexShrink: 0,
      boxShadow: "0 0 0 2px rgba(255,255,255,0.3)",
    } as CSSProperties,

    countChip: {
      flexShrink: 0,
      fontSize: 11,
      fontWeight: 700,
      padding: "2px 8px",
      borderRadius: 10,
      background: isDark ? "rgba(255, 255, 255, 0.10)" : "rgba(0, 0, 0, 0.06)",
      color: t.textMuted,
    } as CSSProperties,

    headerControls: {
      display: "flex",
      alignItems: "center",
      gap: 8,
      flexWrap: "wrap",
    } as CSSProperties,

    editButton: {
      display: "inline-flex",
      alignItems: "center",
      gap: 6,
      padding: "5px 12px",
      borderRadius: 6,
      fontSize: 12,
      fontWeight: 700,
      cursor: "pointer",
      border: isEditMode
        ? "1px solid #f59e0b"
        : `1px solid ${isDark ? "rgba(255, 255, 255, 0.18)" : "rgba(0, 0, 0, 0.18)"}`,
      background: isEditMode
        ? isDark
          ? "rgba(245, 158, 11, 0.22)"
          : "#fef3c7"
        : isDark
        ? "rgba(255, 255, 255, 0.06)"
        : "#ffffff",
      color: isEditMode ? (isDark ? "#fbbf24" : "#b45309") : t.text,
      transition: "all 0.15s ease",
    } as CSSProperties,

    undoButton: {
      display: "inline-flex",
      alignItems: "center",
      gap: 5,
      padding: "5px 12px",
      borderRadius: 6,
      fontSize: 12,
      fontWeight: 700,
      border: `1px solid ${isDark ? "rgba(255, 255, 255, 0.18)" : "rgba(0, 0, 0, 0.18)"}`,
      background: isDark ? "rgba(255, 255, 255, 0.06)" : "#ffffff",
      color: t.text,
      transition: "all 0.15s ease",
    } as CSSProperties,

    zoomButtonGroup: {
      display: "inline-flex",
      alignItems: "center",
      border: `1px solid ${isDark ? "rgba(255, 255, 255, 0.18)" : "rgba(0, 0, 0, 0.18)"}`,
      borderRadius: 6,
      overflow: "hidden",
      background: isDark ? "rgba(255, 255, 255, 0.04)" : "#ffffff",
    } as CSSProperties,

    zoomButton: {
      display: "inline-flex",
      alignItems: "center",
      justifyContent: "center",
      padding: "5px 10px",
      fontSize: 12,
      fontWeight: 700,
      cursor: "pointer",
      border: "none",
      background: "transparent",
      color: t.text,
      borderRight: `1px solid ${isDark ? "rgba(255, 255, 255, 0.14)" : "rgba(0, 0, 0, 0.12)"}`,
    } as CSSProperties,

    zoomBadge: {
      display: "inline-flex",
      alignItems: "center",
      padding: "5px 8px",
      fontSize: 11,
      fontWeight: 700,
      color: t.textMuted,
      userSelect: "none",
      borderRight: `1px solid ${isDark ? "rgba(255, 255, 255, 0.14)" : "rgba(0, 0, 0, 0.12)"}`,
      minWidth: 42,
      justifyContent: "center",
    } as CSSProperties,

    zoomButtonReset: {
      display: "inline-flex",
      alignItems: "center",
      justifyContent: "center",
      padding: "5px 10px",
      fontSize: 12,
      fontWeight: 700,
      cursor: "pointer",
      border: "none",
      background: "transparent",
      color: t.text,
    } as CSSProperties,

    closeButton: {
      display: "inline-flex",
      alignItems: "center",
      gap: 5,
      padding: "5px 12px",
      borderRadius: 6,
      fontSize: 12,
      fontWeight: 700,
      cursor: "pointer",
      border: `1px solid ${isDark ? "rgba(255, 255, 255, 0.18)" : "rgba(0, 0, 0, 0.18)"}`,
      background: isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(0, 0, 0, 0.05)",
      color: t.text,
      transition: "all 0.15s ease",
    } as CSSProperties,

    viewportContainer: {
      position: "relative",
      flex: 1,
      minHeight: 0,
      width: "100%",
      overflow: "hidden",
      backgroundColor: isDark ? "#0d131c" : "#080c12",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      userSelect: "none",
      containerType: "size",
    } as CSSProperties,

    viewportContent: {
      position: "relative",
      width: "100cqmin",
      height: "100cqmin",
      maxWidth: "100%",
      maxHeight: "100%",
      aspectRatio: "1 / 1",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    } as CSSProperties,

    cropImage: {
      position: "absolute",
      maxWidth: "none",
      objectFit: "fill",
      pointerEvents: "none",
      userSelect: "none",
    } as CSSProperties,

    overlay: {
      position: "absolute",
      inset: 0,
      width: "100%",
      height: "100%",
      display: "block",
      touchAction: "none",
    } as CSSProperties,

    emptyState: {
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      width: "100%",
      height: "100%",
      color: t.textMuted,
      fontSize: 14,
    } as CSSProperties,

    footer: {
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      padding: "8px 16px",
      fontSize: 11,
      fontWeight: 600,
      borderTop: `1px solid ${t.borderLight}`,
      background: isDark ? "rgba(0, 0, 0, 0.25)" : "rgba(0, 0, 0, 0.02)",
      color: isDark ? "rgba(255, 255, 255, 0.75)" : "rgba(15, 23, 42, 0.75)",
      flexShrink: 0,
      gap: 12,
      flexWrap: "wrap",
    } as CSSProperties,

    kbd: {
      display: "inline-block",
      padding: "1px 5px",
      fontSize: 10,
      fontWeight: 700,
      fontFamily: "monospace",
      borderRadius: 3,
      background: isDark ? "rgba(255, 255, 255, 0.12)" : "rgba(0, 0, 0, 0.08)",
      border: `1px solid ${isDark ? "rgba(255, 255, 255, 0.22)" : "rgba(0, 0, 0, 0.15)"}`,
      color: isDark ? "#fff" : "#111",
      whiteSpace: "nowrap",
    } as CSSProperties,
  };
}
