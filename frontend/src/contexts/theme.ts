import { createContext, useContext } from "react";

export type ThemePreference = "light" | "dark" | "auto";
export type ResolvedTheme = "light" | "dark";

export interface ThemeContextValue {
  preference: ThemePreference;
  resolved: ResolvedTheme;
  setPreference: (preference: ThemePreference) => void;
}

export const ThemeContext = createContext<ThemeContextValue>({
  preference: "auto",
  resolved: "light",
  setPreference: () => undefined,
});

export const useTheme = () => useContext(ThemeContext);
