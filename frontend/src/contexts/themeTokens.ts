import type { ResolvedTheme } from "./ThemeContext";

export interface ThemeTokens {
  // Surfaces
  bg: string;
  bgSecondary: string;
  bgTertiary: string;
  surface: string;
  surfaceHover: string;
  overlay: string;

  // Borders
  border: string;
  borderLight: string;

  // Text
  text: string;
  textSecondary: string;
  textMuted: string;
  textLink: string;

  // Semantic
  error: string;
  errorBg: string;
}

const light: ThemeTokens = {
  bg: "#ffffff",
  bgSecondary: "#f5f5f5",
  bgTertiary: "#f0f0f0",
  surface: "#f8f9fa",
  surfaceHover: "#f0f0f0",
  overlay: "rgba(255,255,255,0.8)",

  border: "#ccc",
  borderLight: "#dee2e6",

  text: "#333",
  textSecondary: "#495057",
  textMuted: "#666",
  textLink: "#0056b3",

  error: "red",
  errorBg: "rgba(255,220,220,0.9)",
};

const dark: ThemeTokens = {
  bg: "#1a1a2e",
  bgSecondary: "#16213e",
  bgTertiary: "#0f3460",
  surface: "#1e2a3a",
  surfaceHover: "#2a3a4e",
  overlay: "rgba(0,0,0,0.7)",

  border: "#3a4a5e",
  borderLight: "#2a3a4e",

  text: "#e0e0e0",
  textSecondary: "#b0b8c4",
  textMuted: "#8899aa",
  textLink: "#64b5f6",

  error: "#ff6b6b",
  errorBg: "rgba(255,80,80,0.2)",
};

export function getTokens(theme: ResolvedTheme): ThemeTokens {
  return theme === "dark" ? dark : light;
}
