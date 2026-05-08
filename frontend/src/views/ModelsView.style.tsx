import type { CSSProperties } from "react";
import type { ResolvedTheme } from "../contexts/ThemeContext";
import { getTokens } from "../contexts/themeTokens";

export function getModelsViewStyles(theme: ResolvedTheme) {
  const t = getTokens(theme);
  return {
    page: {
      minHeight: "100vh",
      backgroundColor: t.bg,
      color: t.text,
      padding: "24px",
      fontFamily: "system-ui, sans-serif",
    } as CSSProperties,

    header: {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      marginBottom: "24px",
    } as CSSProperties,

    title: {
      fontSize: "24px",
      fontWeight: 600,
      margin: 0,
    } as CSSProperties,

    backLink: {
      color: t.textLink,
      textDecoration: "none",
      fontSize: "14px",
    } as CSSProperties,

    section: {
      backgroundColor: t.surface,
      border: `1px solid ${t.borderLight}`,
      borderRadius: "8px",
      padding: "16px",
      marginBottom: "16px",
    } as CSSProperties,

    sectionTitle: {
      fontSize: "16px",
      fontWeight: 600,
      margin: "0 0 12px 0",
      color: t.textSecondary,
    } as CSSProperties,

    statusBadge: (ok: boolean): CSSProperties => ({
      display: "inline-block",
      padding: "4px 10px",
      borderRadius: "12px",
      fontSize: "12px",
      fontWeight: 500,
      backgroundColor: ok ? "rgba(40, 167, 69, 0.15)" : t.errorBg,
      color: ok ? "#28a745" : t.error,
      border: `1px solid ${ok ? "rgba(40, 167, 69, 0.3)" : t.error}`,
    }),

    grid: {
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
      gap: "12px",
    } as CSSProperties,

    card: {
      backgroundColor: t.bgSecondary,
      padding: "12px",
      borderRadius: "6px",
      border: `1px solid ${t.borderLight}`,
    } as CSSProperties,

    cardLabel: {
      fontSize: "11px",
      color: t.textMuted,
      textTransform: "uppercase",
      letterSpacing: "0.5px",
      marginBottom: "4px",
    } as CSSProperties,

    cardValue: {
      fontSize: "16px",
      fontWeight: 500,
      wordBreak: "break-word",
    } as CSSProperties,

    table: {
      width: "100%",
      borderCollapse: "collapse",
      fontSize: "13px",
    } as CSSProperties,

    th: {
      padding: "10px 8px",
      textAlign: "left",
      borderBottom: `2px solid ${t.borderLight}`,
      color: t.textSecondary,
      fontSize: "12px",
      fontWeight: 600,
      textTransform: "uppercase",
      letterSpacing: "0.5px",
    } as CSSProperties,

    td: {
      padding: "10px 8px",
      borderBottom: `1px solid ${t.borderLight}`,
    } as CSSProperties,

    link: {
      color: t.textLink,
      textDecoration: "none",
      fontSize: "13px",
    } as CSSProperties,

    downloadButton: {
      display: "inline-flex",
      alignItems: "center",
      gap: "6px",
      padding: "8px 14px",
      backgroundColor: t.textLink,
      color: "#ffffff",
      border: "none",
      borderRadius: "6px",
      fontSize: "13px",
      fontWeight: 500,
      textDecoration: "none",
      cursor: "pointer",
    } as CSSProperties,

    downloadRow: {
      display: "flex",
      flexWrap: "wrap",
      gap: "8px",
    } as CSSProperties,

    error: {
      color: t.error,
      padding: "12px",
      backgroundColor: t.errorBg,
      borderRadius: "6px",
      fontSize: "13px",
    } as CSSProperties,

    muted: {
      color: t.textMuted,
      fontSize: "13px",
    } as CSSProperties,

    code: {
      fontFamily: "monospace",
      fontSize: "12px",
      backgroundColor: t.bgTertiary,
      padding: "2px 6px",
      borderRadius: "3px",
    } as CSSProperties,
  };
}
