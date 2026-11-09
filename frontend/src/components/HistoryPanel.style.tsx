import type { CSSProperties } from "react";
import type { ResolvedTheme } from "../contexts/theme";
import { getTokens } from "../contexts/themeTokens";

export function getHistoryPanelStyles(theme: ResolvedTheme) {
  const t = getTokens(theme);
  const isDark = theme === "dark";

  return {
    historyContainer: {
      width: "27vw",
      flexShrink: 0,
      boxSizing: "border-box",
      borderRight: `1px solid ${t.border}`,
      overflowY: "auto",
      padding: "10px",
      color: t.text,
      height: "100%",
    } as CSSProperties,

    historyHeader: {
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      marginBottom: "10px",
    } as CSSProperties,

    historyTitle: {
      margin: 0,
      fontSize: "1.17em",
      fontWeight: "bold",
      color: t.text,
    } as CSSProperties,

    collapseButton: {
      backgroundColor: isDark ? t.surface : t.bgSecondary,
      color: t.textMuted,
      border: `1px solid ${t.borderLight}`,
      borderRadius: "4px",
      padding: "4px 8px",
      fontSize: "12px",
      cursor: "pointer",
      display: "inline-flex",
      alignItems: "center",
      gap: "4px",
      fontWeight: "500",
      transition: "background-color 0.15s ease, color 0.15s ease, box-shadow 0.15s ease",
    } as CSSProperties,

    collapseButtonHovered: {
      backgroundColor: t.surfaceHover,
      color: t.text,
    } as CSSProperties,

    focusRing: {
      outline: `2px solid ${t.textLink}`,
      outlineOffset: "2px",
    } as CSSProperties,

    collapsedContainer: {
      width: "36px",
      minWidth: "36px",
      flex: "0 0 36px",
      borderRight: `1px solid ${t.border}`,
      backgroundColor: isDark ? t.surface : t.bgSecondary,
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      padding: "10px 0",
      color: t.text,
      userSelect: "none",
      cursor: "pointer",
      boxSizing: "border-box",
      height: "100%",
      transition: "background-color 0.15s ease",
    } as CSSProperties,

    collapsedContainerHovered: {
      backgroundColor: t.surfaceHover,
    } as CSSProperties,

    collapsedButton: {
      backgroundColor: "transparent",
      border: "none",
      color: t.text,
      cursor: "pointer",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: "8px",
      padding: "4px 0",
      width: "100%",
      borderRadius: "4px",
    } as CSSProperties,

    collapsedIcon: {
      fontSize: "12px",
      color: t.textMuted,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    } as CSSProperties,

    collapsedText: {
      writingMode: "vertical-rl",
      textOrientation: "mixed",
      whiteSpace: "nowrap",
      fontSize: "11px",
      fontWeight: 600,
      letterSpacing: "1px",
      color: t.textSecondary,
      marginTop: "6px",
    } as CSSProperties,

    filterToolbar: {
      display: "flex",
      flexWrap: "wrap",
      gap: "4px",
      marginBottom: "8px",
    } as CSSProperties,

    filterButton: {
      backgroundColor: isDark ? t.surface : t.bgSecondary,
      color: t.textMuted,
      border: `1px solid ${t.borderLight}`,
      borderRadius: "12px",
      padding: "2px 8px",
      fontSize: "11px",
      fontWeight: 500,
      cursor: "pointer",
      display: "inline-flex",
      alignItems: "center",
      gap: "4px",
      transition: "all 0.15s ease",
    } as CSSProperties,

    filterButtonActive: {
      backgroundColor: isDark ? "#1e3a5f" : "#e3f2fd",
      color: isDark ? "#90caf9" : "#1976d2",
      borderColor: isDark ? "#42a5f5" : "#90caf9",
      fontWeight: 600,
    } as CSSProperties,

    historyTableContainer: {
      maxHeight: "calc(100vh - 350px)",
      overflowY: "auto",
      border: `1px solid ${t.borderLight}`,
      borderRadius: "4px",
    } as CSSProperties,

    historyTable: {
      width: "100%",
      borderCollapse: "collapse",
    } as CSSProperties,

    historyTableHeader: {
      backgroundColor: t.bgTertiary,
    } as CSSProperties,

    historyTableHeaderCell: {
      padding: "8px",
      textAlign: "left",
      borderBottom: `1px solid ${t.borderLight}`,
      color: t.text,
      fontSize: "12px",
    } as CSSProperties,

    historyTableRow: {
      cursor: "pointer",
      transition: "background-color 0.15s ease",
    } as CSSProperties,

    historyTableRowSelected: {
      backgroundColor: t.surfaceHover,
    } as CSSProperties,

    historyTableCell: {
      padding: "8px",
      borderBottom: `1px solid ${t.borderLight}`,
      color: t.text,
    } as CSSProperties,

    historyTableCellSelected: {
      fontWeight: "bold",
      color: t.text,
    } as CSSProperties,

    historyItemNameRow: {
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      gap: "6px",
    } as CSSProperties,

    historyItemName: {
      overflow: "hidden",
      textOverflow: "ellipsis",
      whiteSpace: "nowrap",
      flex: 1,
      minWidth: 0,
      fontSize: "13px",
    } as CSSProperties,

    statusBadge: {
      display: "inline-flex",
      alignItems: "center",
      gap: "3px",
      fontSize: "11px",
      fontWeight: 600,
      padding: "1px 6px",
      borderRadius: "10px",
      whiteSpace: "nowrap",
      flexShrink: 0,
    } as CSSProperties,

    statusBadgeUnreviewed: {
      backgroundColor: isDark ? "#2a2d32" : "#f1f3f5",
      color: t.textMuted,
      border: `1px solid ${t.borderLight}`,
    } as CSSProperties,

    statusBadgeNeedsReview: {
      backgroundColor: isDark ? "#3e2700" : "#fff3cd",
      color: isDark ? "#ffc107" : "#856404",
      border: `1px solid ${isDark ? "#b28900" : "#ffeeba"}`,
    } as CSSProperties,

    statusBadgeVerified: {
      backgroundColor: isDark ? "#0d3315" : "#d4edda",
      color: isDark ? "#4caf50" : "#155724",
      border: `1px solid ${isDark ? "#1b5e20" : "#c3e6cb"}`,
    } as CSSProperties,

    historyItemTimestamp: {
      fontSize: "0.8em",
      color: t.textMuted,
      marginTop: "2px",
    } as CSSProperties,

    historyTableEmptyCell: {
      padding: "10px",
      textAlign: "center",
      color: t.textMuted,
    } as CSSProperties,

    progressContainer: {
      marginTop: "8px",
      display: "flex",
      alignItems: "center",
      gap: "8px",
    } as CSSProperties,

    progressBar: {
      flex: 1,
      height: "6px",
      backgroundColor: isDark ? t.border : t.borderLight,
      borderRadius: "3px",
      overflow: "hidden",
    } as CSSProperties,

    progressFill: {
      height: "100%",
      backgroundColor: "#4F7942",
      transition: "width 0.3s ease",
    } as CSSProperties,

    progressText: {
      fontSize: "0.7em",
      color: t.textMuted,
      minWidth: "30px",
      textAlign: "right",
    } as CSSProperties,

    progressError: {
      color: t.error,
      fontSize: "0.8em",
    } as CSSProperties,
  };
}
