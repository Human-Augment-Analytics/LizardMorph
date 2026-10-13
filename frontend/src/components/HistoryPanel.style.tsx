import type { CSSProperties } from "react";
import type { ResolvedTheme } from "../contexts/theme";
import { getTokens } from "../contexts/themeTokens";

export function getHistoryPanelStyles(theme: ResolvedTheme) {
  const t = getTokens(theme);
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
      backgroundColor: theme === "dark" ? t.surface : t.bgSecondary,
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
      backgroundColor: theme === "dark" ? t.surface : t.bgSecondary,
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
      whiteSpace: "nowrap",
      overflow: "hidden",
      textOverflow: "ellipsis",
      color: t.text,
    } as CSSProperties,

    historyTableCellSelected: {
      fontWeight: "bold",
      color: t.text,
    } as CSSProperties,

    historyItemTimestamp: {
      fontSize: "0.8em",
      color: t.textMuted,
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
      backgroundColor: theme === "dark" ? t.border : t.borderLight,
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
