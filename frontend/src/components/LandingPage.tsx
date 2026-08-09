import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useTheme } from "../contexts/ThemeContext";
import { getTokens } from "../contexts/themeTokens";
import { ApiService } from "../services/ApiService";
import type { ModelVersionItem } from "../services/ApiService";
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
    secondaryContainer: {
      marginTop: "4rem",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      gap: "1.5rem",
      backgroundColor: t.bg,
      padding: "1.2rem 2.5rem",
      borderRadius: "16px",
      boxShadow: `0 8px 32px ${isDark ? "rgba(0,0,0,0.25)" : "rgba(0,0,0,0.06)"}`,
      border: `1px solid ${isDark ? "#2a3a4e" : "#e8e8e8"}`,
      maxWidth: "800px",
      width: "calc(100% - 40px)",
      boxSizing: "border-box" as const,
      flexWrap: "wrap" as const,
      textAlign: "center" as const,
    },
    secondaryText: {
      fontSize: "1.1rem",
      fontWeight: "500" as const,
      color: t.text,
    },
    secondaryButton: {
      backgroundColor: "transparent",
      color: "#4CAF50",
      border: "2px solid #4CAF50",
      padding: "0.7rem 1.8rem",
      borderRadius: "10px",
      fontSize: "1rem",
      fontWeight: "bold" as const,
      cursor: "pointer",
      display: "flex",
      alignItems: "center",
      gap: "0.5rem",
      transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
    },
    secondaryButtonHover: {
      backgroundColor: "#4CAF50",
      color: "#fff",
      boxShadow: "0 8px 24px rgba(76, 175, 80, 0.3)",
      transform: "translateY(-2px)",
    },
  };
}

export const LandingPage: React.FC = () => {
  const navigate = useNavigate();
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);
  const [isSecondaryHovered, setIsSecondaryHovered] = useState(false);
  const [models, setModels] = useState<ModelVersionItem[]>([]);
  const { resolved, preference, setPreference } = useTheme();
  const isDark = resolved === "dark";
  const LandingPageStyles = getLandingPageStyles(isDark);

  useEffect(() => {
    let isMounted = true;
    ApiService.getModels()
      .then((fetchedModels) => {
        if (isMounted && fetchedModels.length > 0) {
          setModels(fetchedModels);
        }
      })
      .catch(() => {
        // Fallback to built-in rendering if backend API fails
      });
    return () => {
      isMounted = false;
    };
  }, []);

  const handleOptionClick = (viewType: LizardViewType) => {
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
        Select a preinstalled lizard model or one of your custom built models to analyze
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
            <div
              style={{
                ...LandingPageStyles.iconLogo,
                ...(hoveredCard === "dorsal" ? LandingPageStyles.iconHover : {}),
              }}
            >
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
            <h3
              style={{
                ...LandingPageStyles.optionTitle,
                ...(hoveredCard === "dorsal" ? LandingPageStyles.optionTitleHover : {}),
              }}
            >
              Dorsal View
            </h3>
            <p style={LandingPageStyles.optionDescription}>
              Analyze lizard x-ray images from the top view (`lizard-dorsal-v1`)
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
            <div
              style={{
                ...LandingPageStyles.icon,
                ...(hoveredCard === "lateral" ? LandingPageStyles.iconHover : {}),
              }}
            >
              🦖
            </div>
            <h3
              style={{
                ...LandingPageStyles.optionTitle,
                ...(hoveredCard === "lateral" ? LandingPageStyles.optionTitleHover : {}),
              }}
            >
              Lateral View
            </h3>
            <p style={LandingPageStyles.optionDescription}>
              Analyze lizard x-ray images from the side view (`lizard-lateral-v1`)
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
            <div
              style={{
                ...LandingPageStyles.icon,
                ...(hoveredCard === "toepads" ? LandingPageStyles.iconHover : {}),
              }}
            >
              🦶
            </div>
            <h3
              style={{
                ...LandingPageStyles.optionTitle,
                ...(hoveredCard === "toepads" ? LandingPageStyles.optionTitleHover : {}),
              }}
            >
              Toepad View
            </h3>
            <p style={LandingPageStyles.optionDescription}>
              Analyze lizard toe pad structures using YOLO OBB detection & ML-Morph (`lizard-toepad-v1`)
            </p>
          </div>
        </div>

        {/* Custom User-Trained Models Dynamic Cards */}
        {models
          .filter(
            (m) =>
              !["lizard-dorsal-v1", "lizard-lateral-v1", "lizard-toepad-v1"].includes(m.id)
          )
          .map((m) => (
            <div
              key={m.id}
              style={{
                ...LandingPageStyles.optionCard,
                ...(hoveredCard === m.id ? LandingPageStyles.optionCardHover : {}),
              }}
              onClick={() => handleOptionClick("custom")}
              onMouseEnter={() => handleMouseEnter(m.id)}
              onMouseLeave={handleMouseLeave}
            >
              <div style={LandingPageStyles.cardContent}>
                <div
                  style={{
                    ...LandingPageStyles.icon,
                    ...(hoveredCard === m.id ? LandingPageStyles.iconHover : {}),
                  }}
                >
                  🧬
                </div>
                <h3
                  style={{
                    ...LandingPageStyles.optionTitle,
                    ...(hoveredCard === m.id ? LandingPageStyles.optionTitleHover : {}),
                  }}
                >
                  {m.name}
                </h3>
                <p style={LandingPageStyles.optionDescription}>
                  {m.manifest?.description || `Custom project model (${m.id})`}
                </p>
              </div>
            </div>
          ))}

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
            <div
              style={{
                ...LandingPageStyles.icon,
                ...(hoveredCard === "free" ? LandingPageStyles.iconHover : {}),
              }}
            >
              📌
            </div>
            <h3
              style={{
                ...LandingPageStyles.optionTitle,
                ...(hoveredCard === "free" ? LandingPageStyles.optionTitleHover : {}),
              }}
            >
              Free Mode
            </h3>
            <p style={LandingPageStyles.optionDescription}>
              Manually place landmarks on any image by clicking
            </p>
          </div>
        </div>
      </div>

      {/* Train Custom Model Segmented Section */}
      <div style={LandingPageStyles.secondaryContainer}>
        <span style={LandingPageStyles.secondaryText}>
          Want to create a custom project model for your species?
        </span>
        <button
          onClick={() => handleOptionClick("custom")}
          onMouseEnter={() => setIsSecondaryHovered(true)}
          onMouseLeave={() => setIsSecondaryHovered(false)}
          style={{
            ...LandingPageStyles.secondaryButton,
            ...(isSecondaryHovered ? LandingPageStyles.secondaryButtonHover : {}),
          }}
        >
          Train Custom Model Wizard
        </button>
      </div>
    </div>
  );
};