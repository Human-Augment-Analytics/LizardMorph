import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useTheme } from "../contexts/ThemeContext";
import { getTokens } from "../contexts/themeTokens";
import lizard_logo from "../../public/lizard.svg";

export type LizardViewType = "dorsal" | "lateral" | "toepads" | "custom" | "free";

function getLandingPageStyles(isDark: boolean) {
  const t = getTokens(isDark ? "dark" : "light");
  return {
    container: {
      display: "flex",
      flexDirection: "column" as const,
      alignItems: "center",
      justifyContent: "center",
      minHeight: "100vh",
      width: "100vw",
      backgroundColor: t.bgSecondary,
      padding: "20px",
      boxSizing: "border-box" as const,
    },
    title: {
      fontSize: "3rem",
      fontWeight: "bold" as const,
      color: t.text,
      marginBottom: "1rem",
      textAlign: "center" as const,
    },
    subtitle: {
      fontSize: "1.3rem",
      color: t.textMuted,
      marginBottom: "4rem",
      textAlign: "center" as const,
    },
    optionsContainer: {
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
      gap: "2.5rem",
      maxWidth: "1200px",
      width: "100%",
      padding: "0 20px",
    },
    optionCard: {
      backgroundColor: t.bg,
      borderRadius: "16px",
      padding: "2.5rem",
      boxShadow: `0 6px 20px ${isDark ? "rgba(0,0,0,0.3)" : "rgba(0,0,0,0.1)"}`,
      cursor: "pointer",
      transition: "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
      border: "3px solid transparent",
      position: "relative" as const,
      overflow: "hidden" as const,
    },
    optionCardHover: {
      transform: "translateY(-8px) scale(1.02)",
      boxShadow: `0 20px 40px ${isDark ? "rgba(0,0,0,0.4)" : "rgba(0,0,0,0.2)"}`,
      borderColor: "#4CAF50",
    },
    optionCardActive: {
      transform: "translateY(-4px) scale(1.01)",
      boxShadow: `0 12px 30px ${isDark ? "rgba(0,0,0,0.35)" : "rgba(0,0,0,0.15)"}`,
      borderColor: "#45a049",
    },
    optionCardDisabled: {
      backgroundColor: isDark ? "#1e2a3a" : "#f8f8f8",
      cursor: "not-allowed",
      opacity: 0.7,
      transform: "none",
    },
    optionTitle: {
      fontSize: "1.8rem",
      fontWeight: "bold" as const,
      color: t.text,
      marginBottom: "0.8rem",
      transition: "color 0.3s ease",
    },
    optionTitleHover: {
      color: "#4CAF50",
    },
    optionDescription: {
      fontSize: "1.1rem",
      color: t.textMuted,
      marginBottom: "1rem",
      lineHeight: "1.6",
    },
    comingSoon: {
      fontSize: "1rem",
      color: "#ff9800",
      fontWeight: "bold" as const,
      textTransform: "uppercase" as const,
      letterSpacing: "1px",
    },
    icon: {
      fontSize: "4rem",
      height: "4rem",
      width: "4rem",
      margin: "0 auto 1.5rem auto",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      color: "#4F7942",
      transition: "all 0.3s ease",
      filter: "brightness(0) invert(41%) sepia(10%) saturate(2258%) hue-rotate(58deg) brightness(96%) contrast(84%)",
    },
    iconLogo: {
      fontSize: "4rem",
      height: "4rem",
      width: "4rem",
      margin: "0 auto 1.5rem auto",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      transition: "all 0.3s ease",
    },
    iconHover: {
      transform: "scale(1.1)",
      color: "#45a049",
    },
    iconDisabled: {
      color: isDark ? "#555" : "#ccc",
      transform: "none",
    },
    cardContent: {
      textAlign: "center" as const,
    },
    themeToggle: {
      position: "fixed" as const,
      top: "15px",
      right: "15px",
      display: "flex",
      gap: "2px",
      backgroundColor: isDark ? "#2a3a4e" : "#e8e8e8",
      borderRadius: "6px",
      padding: "2px",
      zIndex: 10,
    },
    themeToggleButton: {
      padding: "6px 12px",
      border: "none",
      borderRadius: "4px",
      cursor: "pointer",
      fontSize: "13px",
      backgroundColor: "transparent",
      color: t.textMuted,
      transition: "all 0.2s ease",
    },
    themeToggleButtonActive: {
      backgroundColor: t.bg,
      borderColor: "#4F7942",
      boxShadow: "0 8px 30px rgba(79, 121, 66, 0.15)",
    },
  };
}

export const LandingPage: React.FC = () => {
  const navigate = useNavigate();
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);
  const { resolved, preference, setPreference } = useTheme();
  const isDark = resolved === "dark";
  const LandingPageStyles = getLandingPageStyles(isDark);

  const handleOptionClick = (viewType: LizardViewType) => {
    if (viewType === "custom") {
      return; // Disabled
    }
    navigate(`/${viewType}`);
  };

  const handleMouseEnter = (viewType: string) => {
    setHoveredCard(viewType);
  };

  const handleMouseLeave = () => {
    setHoveredCard(null);
  };

  return (
    <div style={LandingPageStyles.container}>
      {/* Theme toggle */}
      <div style={LandingPageStyles.themeToggle}>
        <button
          onClick={() => setPreference("light")}
          style={{
            ...LandingPageStyles.themeToggleButton,
            ...(preference === "light" ? LandingPageStyles.themeToggleButtonActive : {}),
          }}
          title="Light mode"
        >
          ☀️
        </button>
        <button
          onClick={() => setPreference("dark")}
          style={{
            ...LandingPageStyles.themeToggleButton,
            ...(preference === "dark" ? LandingPageStyles.themeToggleButtonActive : {}),
          }}
          title="Dark mode"
        >
          🌙
        </button>
        <button
          onClick={() => setPreference("auto")}
          style={{
            ...LandingPageStyles.themeToggleButton,
            ...(preference === "auto" ? LandingPageStyles.themeToggleButtonActive : {}),
          }}
          title="Auto (follow system)"
        >
          Auto
        </button>
      </div>
      <h1 style={LandingPageStyles.title}>LizardMorph</h1>
      <p style={LandingPageStyles.subtitle}>
        Select the type of lizard x-ray images you want to analyze
      </p>
      
      <div style={LandingPageStyles.optionsContainer}>
        {/* Dorsal View */}
        <div
          style={{
            ...LandingPageStyles.optionCard,
            ...(hoveredCard === "dorsal" ? LandingPageStyles.optionCardHover : {}),
          }}
          onClick={() => handleOptionClick("dorsal")}
          onMouseEnter={() => handleMouseEnter("dorsal")}
          onMouseLeave={handleMouseLeave}
        >
          <div style={LandingPageStyles.cardContent}>
            <div style={{
              ...LandingPageStyles.iconLogo,
              ...(hoveredCard === "dorsal" ? LandingPageStyles.iconHover : {})
            }}>
              <img
                src={lizard_logo}
                alt="Dorsal View"
                style={{
                  height: "100%",
                  width: "100%",
                  objectFit: "contain",
                }}
              />
            </div>
            <h3 style={{
              ...LandingPageStyles.optionTitle,
              ...(hoveredCard === "dorsal" ? LandingPageStyles.optionTitleHover : {})
            }}>
              Dorsal View
            </h3>
            <p style={LandingPageStyles.optionDescription}>
              Analyze lizard x-ray images from the top view
            </p>
          </div>
        </div>

        {/* Lateral View */}
        <div
          style={{
            ...LandingPageStyles.optionCard,
            ...(hoveredCard === "lateral" ? LandingPageStyles.optionCardHover : {}),
          }}
          onClick={() => handleOptionClick("lateral")}
          onMouseEnter={() => handleMouseEnter("lateral")}
          onMouseLeave={handleMouseLeave}
        >
          <div style={LandingPageStyles.cardContent}>
            <div style={{
              ...LandingPageStyles.icon,
              ...(hoveredCard === "lateral" ? LandingPageStyles.iconHover : {})
            }}>
              🦖
            </div>
            <h3 style={{
              ...LandingPageStyles.optionTitle,
              ...(hoveredCard === "lateral" ? LandingPageStyles.optionTitleHover : {})
            }}>
              Lateral View
            </h3>
            <p style={LandingPageStyles.optionDescription}>
              Analyze lizard x-ray images from the side view
            </p>
          </div>
        </div>

        {/* Toepads View */}
        <div
          style={{
            ...LandingPageStyles.optionCard,
            ...(hoveredCard === "toepads" ? LandingPageStyles.optionCardHover : {}),
          }}
          onClick={() => handleOptionClick("toepads")}
          onMouseEnter={() => handleMouseEnter("toepads")}
          onMouseLeave={handleMouseLeave}
        >
          <div style={LandingPageStyles.cardContent}>
            <div style={{
              ...LandingPageStyles.icon,
              ...(hoveredCard === "toepads" ? LandingPageStyles.iconHover : {})
            }}>
              🦶
            </div>
            <h3 style={{
              ...LandingPageStyles.optionTitle,
              ...(hoveredCard === "toepads" ? LandingPageStyles.optionTitleHover : {})
            }}>
              Toepad View
            </h3>
            <p style={LandingPageStyles.optionDescription}>
              Analyze lizard toe pad structures using YOLO detection and landmark prediction
            </p>
          </div>
        </div>

        {/* Free Mode */}
        <div
          style={{
            ...LandingPageStyles.optionCard,
            ...(hoveredCard === "free" ? LandingPageStyles.optionCardHover : {}),
          }}
          onClick={() => handleOptionClick("free")}
          onMouseEnter={() => handleMouseEnter("free")}
          onMouseLeave={handleMouseLeave}
        >
          <div style={LandingPageStyles.cardContent}>
            <div style={{
              ...LandingPageStyles.icon,
              ...(hoveredCard === "free" ? LandingPageStyles.iconHover : {})
            }}>
              📌
            </div>
            <h3 style={{
              ...LandingPageStyles.optionTitle,
              ...(hoveredCard === "free" ? LandingPageStyles.optionTitleHover : {})
            }}>
              Free Mode
            </h3>
            <p style={LandingPageStyles.optionDescription}>
              Manually place landmarks on any image by clicking
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}; 