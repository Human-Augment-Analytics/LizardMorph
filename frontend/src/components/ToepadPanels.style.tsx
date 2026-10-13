import type { CSSProperties } from "react";
import type { ResolvedTheme } from "../contexts/theme";
import { getTokens } from "../contexts/themeTokens";

export function getToepadPanelsStyles(theme: ResolvedTheme) {
  const t = getTokens(theme);
  const isDark = theme === "dark";
  return {
    // Sized so the 2x2 grid of square close-ups fits below the toolbar on typical screens
    container: {
      width: "100%",
      maxWidth: "100%",
      margin: "0 auto",
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

    grid: {
      display: "grid",
      gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
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

    cardHeaderActions: {
      display: "flex",
      alignItems: "center",
      gap: 6,
    } as CSSProperties,

    zoomButton: {
      display: "inline-flex",
      alignItems: "center",
      gap: 4,
      padding: "2px 8px",
      fontSize: 11,
      fontWeight: 700,
      borderRadius: 4,
      border: `1px solid ${isDark ? "rgba(255,255,255,0.18)" : "rgba(0,0,0,0.18)"}`,
      background: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.05)",
      color: t.text,
      cursor: "pointer",
      transition: "background 0.15s, border-color 0.15s",
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

    legendContainer: {
      borderTop: `1px solid ${isDark ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.10)"}`,
      background: isDark ? "rgba(0,0,0,0.22)" : "rgba(0,0,0,0.02)",
      padding: "10px 12px 14px 12px",
    } as CSSProperties,

    legendHeaderButton: {
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      width: "100%",
      background: "none",
      border: "none",
      padding: 0,
      cursor: "pointer",
      color: isDark ? "#e0e0e0" : "#111",
      textAlign: "left",
    } as CSSProperties,

    legendTitle: {
      display: "flex",
      alignItems: "center",
      gap: 6,
      fontSize: 12,
      fontWeight: 800,
      color: isDark ? "#e0e0e0" : "#111",
    } as CSSProperties,

    legendSubtitle: {
      fontSize: 11,
      fontWeight: 700,
      color: isDark ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.55)",
    } as CSSProperties,

    legendSection: {
      marginTop: 10,
    } as CSSProperties,

    legendSectionTitle: {
      fontSize: 11,
      fontWeight: 800,
      textTransform: "uppercase",
      letterSpacing: "0.5px",
      color: isDark ? "rgba(255,255,255,0.75)" : "rgba(0,0,0,0.75)",
      marginBottom: 6,
      display: "flex",
      alignItems: "center",
      gap: 6,
    } as CSSProperties,

    legendTable: {
      width: "100%",
      borderCollapse: "collapse",
      fontSize: 11,
    } as CSSProperties,

    legendTh: {
      textAlign: "left",
      padding: "3px 6px",
      fontSize: 10,
      fontWeight: 700,
      color: t.textMuted,
      borderBottom: `1px solid ${t.borderLight}`,
      background: isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.03)",
    } as CSSProperties,

    legendTd: {
      padding: "4px 6px",
      borderBottom: `1px solid ${isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.05)"}`,
      verticalAlign: "middle",
      color: t.text,
      fontSize: 11,
    } as CSSProperties,

    kbd: {
      display: "inline-block",
      padding: "1px 5px",
      fontSize: 10,
      fontWeight: 700,
      fontFamily: "monospace",
      borderRadius: 3,
      background: isDark ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.08)",
      border: `1px solid ${isDark ? "rgba(255,255,255,0.22)" : "rgba(0,0,0,0.15)"}`,
      color: isDark ? "#fff" : "#111",
      whiteSpace: "nowrap",
    } as CSSProperties,

    modeBadgeView: {
      display: "inline-block",
      fontSize: 9,
      fontWeight: 700,
      padding: "1px 5px",
      borderRadius: 3,
      textTransform: "uppercase",
      letterSpacing: "0.3px",
      background: isDark ? "rgba(33,150,243,0.2)" : "#e3f2fd",
      color: isDark ? "#64b5f6" : "#1976d2",
    } as CSSProperties,

    modeBadgeEdit: {
      display: "inline-block",
      fontSize: 9,
      fontWeight: 700,
      padding: "1px 5px",
      borderRadius: 3,
      textTransform: "uppercase",
      letterSpacing: "0.3px",
      background: isDark ? "rgba(255,193,7,0.2)" : "#fff8e1",
      color: isDark ? "#ffd54f" : "#f57f17",
    } as CSSProperties,

    modeBadgeAny: {
      display: "inline-block",
      fontSize: 9,
      fontWeight: 700,
      padding: "1px 5px",
      borderRadius: 3,
      textTransform: "uppercase",
      letterSpacing: "0.3px",
      background: isDark ? "rgba(255,255,255,0.1)" : "#f0f0f0",
      color: isDark ? "#bbb" : "#666",
    } as CSSProperties,
  };
}
