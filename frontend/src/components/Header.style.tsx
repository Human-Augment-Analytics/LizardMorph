import type { CSSProperties } from "react";
import type { ResolvedTheme } from "../contexts/ThemeContext";
import { getTokens } from "../contexts/themeTokens";

export function getHeaderStyles(theme: ResolvedTheme) {
  const t = getTokens(theme);
  return {
    header: {
      padding: "15px",
      borderBottom: `1px solid ${t.border}`,
      textAlign: "center",
      position: "relative",
      backgroundColor: t.bg,
      color: t.text,
    } as CSSProperties,

    infoBox: {
      position: "absolute",
      top: "10px",
      right: "15px",
      padding: "12px",
      backgroundColor: t.surface,
      borderRadius: "8px",
      boxShadow: `0 2px 4px ${theme === "dark" ? "rgba(0,0,0,0.3)" : "rgba(0,0,0,0.1)"}`,
      border: `1px solid ${t.borderLight}`,
      textAlign: "right",
      minWidth: "300px",
    } as CSSProperties,

    infoBoxContent: {
      fontSize: "0.9em",
      color: t.textSecondary,
    } as CSSProperties,

    infoBoxParagraph: {
      margin: "4px 0",
    } as CSSProperties,

    infoBoxItalic: {
      margin: "4px 0",
      fontStyle: "italic",
    } as CSSProperties,

    infoBoxLink: {
      color: t.textLink,
      textDecoration: "none",
    } as CSSProperties,

    infoBoxLinkRow: {
      display: "flex",
      alignItems: "baseline",
      justifyContent: "flex-end",
      flexWrap: "wrap",
      gap: "8px",
      rowGap: "2px",
    } as CSSProperties,

    appVersion: {
      fontSize: "0.8em",
      color: t.textMuted,
      letterSpacing: "0.02em",
      whiteSpace: "nowrap",
    } as CSSProperties,

    lizardCount: {
      marginTop: "5px",
      color: t.textLink,
    } as CSSProperties,

    mainContent: {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      maxWidth: "1200px",
      margin: "0 0 0 15px",
      padding: "0 20px",
      marginRight: "340px",
      gap: "60px",
    } as CSSProperties,

    buttonContainer: {
      display: "flex",
      flexDirection: "column",
      alignItems: "flex-start",
      gap: "15px",
      width: "220px",
    } as CSSProperties,

    dropdownContainer: {
      position: "relative",
      width: "100%",
    } as CSSProperties,

    dropdownButton: {
      padding: "12px 20px",
      backgroundColor: "#607D8B",
      color: "white",
      border: "none",
      borderRadius: "4px",
      cursor: "pointer",
      width: "100%",
      boxSizing: "border-box",
      fontWeight: "bold",
      fontSize: "14px",
      boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
    } as CSSProperties,

    dropdownContent: {
      position: "absolute",
      top: "100%",
      left: 0,
      backgroundColor: t.surface,
      minWidth: "220px",
      boxShadow: `0px 8px 16px 0px ${theme === "dark" ? "rgba(0,0,0,0.4)" : "rgba(0,0,0,0.2)"}`,
      zIndex: 1,
      display: "flex",
      flexDirection: "column",
      gap: "5px",
      padding: "5px",
      borderRadius: "4px",
    } as CSSProperties,

    uploadButton: {
      padding: "12px 20px",
      backgroundColor: "#2196F3",
      color: "white",
      border: "none",
      borderRadius: "4px",
      cursor: "pointer",
      display: "inline-block",
      fontWeight: "bold",
      fontSize: "14px",
      textAlign: "center",
      boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
      width: "100%",
      boxSizing: "border-box",
    } as CSSProperties,

    uploadButtonDisabled: {
      cursor: "not-allowed",
      opacity: 0.7,
    } as CSSProperties,

    exportButton: {
      padding: "12px 20px",
      backgroundColor: "#4F7942",
      color: "white",
      border: "none",
      borderRadius: "4px",
      cursor: "pointer",
      width: "100%",
      boxSizing: "border-box",
      fontWeight: "bold",
      fontSize: "14px",
      boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
    } as CSSProperties,

    exportButtonDisabled: {
      cursor: "not-allowed",
      opacity: 0.7,
    } as CSSProperties,

    clearHistoryButton: {
      padding: "12px 20px",
      backgroundColor: "#f44336",
      color: "white",
      border: "none",
      borderRadius: "4px",
      cursor: "pointer",
      width: "100%",
      boxSizing: "border-box",
      fontWeight: "bold",
      fontSize: "14px",
      boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
    } as CSSProperties,

    clearHistoryButtonDisabled: {
      cursor: "not-allowed",
      opacity: 0.7,
    } as CSSProperties,

    setScaleButton: {
      padding: "12px 20px",
      backgroundColor: "#FF9800",
      color: "white",
      border: "none",
      borderRadius: "4px",
      cursor: "pointer",
      width: "100%",
      boxSizing: "border-box",
      fontWeight: "bold",
      fontSize: "14px",
      boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
    } as CSSProperties,

    setScaleButtonDisabled: {
      cursor: "not-allowed",
      opacity: 0.7,
    } as CSSProperties,

    measurementsButton: {
      padding: "12px 20px",
      backgroundColor: "#9C27B0",
      color: "white",
      border: "none",
      borderRadius: "4px",
      cursor: "pointer",
      width: "100%",
      boxSizing: "border-box",
      fontWeight: "bold",
      fontSize: "14px",
      boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
    } as CSSProperties,

    measurementsButtonDisabled: {
      cursor: "not-allowed",
      opacity: 0.7,
    } as CSSProperties,

    titleContainer: {
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    } as CSSProperties,

    logo: {
      height: "40px",
      marginRight: "12px",
    } as CSSProperties,

    title: {
      margin: 0,
      whiteSpace: "nowrap",
      color: t.text,
    } as CSSProperties,

    viewType: {
      margin: "8px 0 0 0",
      fontSize: "0.9em",
      color: t.textMuted,
      fontWeight: "500",
      textTransform: "capitalize",
      letterSpacing: "0.5px",
    } as CSSProperties,

    rightSpacer: {
      width: "220px",
    } as CSSProperties,

    errorMessage: {
      color: t.error,
    } as CSSProperties,

    predictorSelector: {
      marginTop: "10px",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    } as CSSProperties,

    predictorLabel: {
      fontSize: "0.9em",
      color: t.textMuted,
      fontWeight: "500",
      display: "flex",
      alignItems: "center",
      gap: "8px",
    } as CSSProperties,

    predictorSelect: {
      padding: "6px 12px",
      borderRadius: "4px",
      border: `1px solid ${t.border}`,
      fontSize: "0.9em",
      backgroundColor: t.bg,
      color: t.text,
      cursor: "pointer",
    } as CSSProperties,

    themeToggle: {
      display: "flex",
      gap: "2px",
      marginTop: "8px",
      justifyContent: "flex-end",
      backgroundColor: theme === "dark" ? "#2a3a4e" : "#e8e8e8",
      borderRadius: "6px",
      padding: "2px",
    } as CSSProperties,

    themeToggleButton: {
      padding: "4px 10px",
      border: "none",
      borderRadius: "4px",
      cursor: "pointer",
      fontSize: "12px",
      backgroundColor: "transparent",
      color: t.textMuted,
      transition: "all 0.2s ease",
    } as CSSProperties,

    themeToggleButtonActive: {
      backgroundColor: t.bg,
      color: t.text,
      boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
    } as CSSProperties,
  };
}
