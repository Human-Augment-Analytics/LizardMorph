import type { CSSProperties } from "react";
import type { ResolvedTheme } from "../contexts/theme";
import { getTokens } from "../contexts/themeTokens";

export function getToepadPanelsStyles(theme: ResolvedTheme) {
  const t = getTokens(theme);
  const isDark = theme === "dark";
  return {
    container: {
      marginTop: 12,
      border: `1px solid ${isDark ? "rgba(255,255,255,0.10)" : "rgba(0,0,0,0.10)"}`,
      borderRadius: 12,
      overflow: "hidden",
      background: isDark ? "rgba(30,42,58,0.85)" : "rgba(255,255,255,0.85)",
    } as CSSProperties,

    header: {
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      gap: 10,
      padding: "10px 12px",
      background: isDark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.02)",
    } as CSSProperties,

    headerTitle: {
      display: "flex",
      alignItems: "center",
      gap: 8,
      fontWeight: 800,
      color: isDark ? "#e0e0e0" : "#111",
    } as CSSProperties,

    headerSubtitle: {
      fontWeight: 700,
      fontSize: 12,
      color: isDark ? "rgba(255,255,255,0.55)" : "rgba(0,0,0,0.55)",
    } as CSSProperties,

    headerActions: {
      display: "flex",
      alignItems: "center",
      gap: 8,
    } as CSSProperties,

    editButton: {
      padding: "6px 12px",
      color: "white",
      border: "none",
      borderRadius: "4px",
      cursor: "pointer",
      fontSize: "13px",
      fontWeight: "bold",
    } as CSSProperties,

    toggleButton: {
      background: "none",
      border: "none",
      cursor: "pointer",
      fontSize: 12,
      color: isDark ? "rgba(255,255,255,0.65)" : "rgba(0,0,0,0.65)",
    } as CSSProperties,

    grid: {
      display: "grid",
      gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
      gap: 10,
      padding: "0 12px 12px 12px",
    } as CSSProperties,

    card: {
      display: "flex",
      flexDirection: "column",
      minWidth: 0,
      borderRadius: 8,
      overflow: "hidden",
      border: `1px solid ${t.borderLight}`,
      background: t.surface,
    } as CSSProperties,

    cardHeader: {
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      gap: 6,
      padding: "6px 8px",
      fontSize: 12,
      fontWeight: 800,
      color: t.text,
    } as CSSProperties,

    cardTitle: {
      display: "flex",
      alignItems: "center",
      gap: 6,
      minWidth: 0,
      whiteSpace: "nowrap",
      overflow: "hidden",
      textOverflow: "ellipsis",
    } as CSSProperties,

    swatch: {
      width: 8,
      height: 8,
      borderRadius: "50%",
      flexShrink: 0,
    } as CSSProperties,

    countChip: {
      flexShrink: 0,
      fontSize: 11,
      fontWeight: 700,
      padding: "1px 6px",
      borderRadius: 4,
      background: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)",
      color: t.textMuted,
    } as CSSProperties,

    viewport: {
      position: "relative",
      width: "100%",
      aspectRatio: "1 / 1",
      overflow: "hidden",
      backgroundColor: isDark ? "#2a2a2a" : "#111",
      borderTop: `2px solid ${t.borderLight}`,
    } as CSSProperties,

    viewportEditing: {
      borderTop: "2px solid #ffc107",
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
      position: "absolute",
      inset: 0,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      textAlign: "center",
      padding: 8,
      fontSize: 12,
      color: t.textMuted,
      backgroundColor: t.bgTertiary,
    } as CSSProperties,
  };
}
