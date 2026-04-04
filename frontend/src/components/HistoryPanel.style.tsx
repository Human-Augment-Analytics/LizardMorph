import type { CSSProperties } from "react";
import type { ResolvedTheme } from "../contexts/ThemeContext";
import { getTokens } from "../contexts/themeTokens";

export function getHistoryPanelStyles(theme: ResolvedTheme) {
  const t = getTokens(theme);
  return {
    historyContainer: {
      width: "27vw",
      borderRight: `1px solid ${t.border}`,
      overflowY: "auto",
      padding: "10px",
      color: t.text,
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
    } as CSSProperties,

    historyTableRowSelected: {
      backgroundColor: t.surfaceHover,
    } as CSSProperties,

    historyTableCell: {
      padding: "8px",
      borderBottom: `1px solid ${theme === "dark" ? t.borderLight : "#eee"}`,
      whiteSpace: "nowrap",
      overflow: "hidden",
      textOverflow: "ellipsis",
      color: t.text,
    } as CSSProperties,

    historyTableCellSelected: {
      fontWeight: "bold",
      color: t.text,
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
      backgroundColor: theme === "dark" ? "#3a4a5e" : "#e0e0e0",
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
  };
}
