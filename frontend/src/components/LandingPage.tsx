import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

export type LizardViewType = "dorsal" | "lateral" | "toepads" | "custom";

const LandingPageStyles = {
  container: {
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
    justifyContent: "center",
    minHeight: "100vh",
    width: "100vw",
    backgroundColor: "#f5f5f5",
    padding: "20px",
    boxSizing: "border-box" as const,
  },
  title: {
    fontSize: "3rem",
    fontWeight: "bold" as const,
    color: "#333",
    marginBottom: "1rem",
    textAlign: "center" as const,
  },
  subtitle: {
    fontSize: "1.3rem",
    color: "#666",
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
    backgroundColor: "white",
    borderRadius: "16px",
    padding: "2.5rem",
    boxShadow: "0 6px 20px rgba(0, 0, 0, 0.1)",
    cursor: "pointer",
    transition: "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
    border: "3px solid transparent",
    position: "relative" as const,
    overflow: "hidden" as const,
  },
  optionCardHover: {
    transform: "translateY(-8px) scale(1.02)",
    boxShadow: "0 20px 40px rgba(0, 0, 0, 0.2)",
    borderColor: "#4CAF50",
  },
  optionCardActive: {
    transform: "translateY(-4px) scale(1.01)",
    boxShadow: "0 12px 30px rgba(0, 0, 0, 0.15)",
    borderColor: "#45a049",
  },
  optionCardDisabled: {
    backgroundColor: "#f8f8f8",
    cursor: "not-allowed",
    opacity: 0.7,
    transform: "none",
  },
  optionTitle: {
    fontSize: "1.8rem",
    fontWeight: "bold" as const,
    color: "#333",
    marginBottom: "0.8rem",
    transition: "color 0.3s ease",
  },
  optionTitleHover: {
    color: "#4CAF50",
  },
  optionDescription: {
    fontSize: "1.1rem",
    color: "#666",
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
    marginBottom: "1.5rem",
    color: "#4CAF50",
    transition: "all 0.3s ease",
  },
  iconHover: {
    transform: "scale(1.1)",
    color: "#45a049",
  },
  iconDisabled: {
    color: "#ccc",
    transform: "none",
  },
  cardContent: {
    textAlign: "center" as const,
  },
};

export const LandingPage: React.FC = () => {
  const navigate = useNavigate();
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);
  
  const handleOptionClick = (viewType: LizardViewType) => {
    if (viewType === "toepads" || viewType === "custom") {
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
              ...LandingPageStyles.icon,
              ...(hoveredCard === "dorsal" ? LandingPageStyles.iconHover : {})
            }}>
              🦎
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

        {/* Toepads View - Disabled */}
        <div
          style={{
            ...LandingPageStyles.optionCard,
            ...LandingPageStyles.optionCardDisabled,
          }}
        >
          <div style={LandingPageStyles.cardContent}>
            <div style={{ ...LandingPageStyles.icon, ...LandingPageStyles.iconDisabled }}>
              🦶
            </div>
            <h3 style={LandingPageStyles.optionTitle}>Toepads View</h3>
            <p style={LandingPageStyles.optionDescription}>
              Analyze lizard toe pad structures
            </p>
            <div style={LandingPageStyles.comingSoon}>Coming Soon</div>
          </div>
        </div>

        {/* Lizard Detection */}
        <div
          style={{
            ...LandingPageStyles.optionCard,
            ...(hoveredCard === "lizard-detection" ? LandingPageStyles.optionCardHover : {}),
          }}
          onClick={() => navigate('/lizard-detection')}
          onMouseEnter={() => handleMouseEnter("lizard-detection")}
          onMouseLeave={handleMouseLeave}
        >
          <div style={LandingPageStyles.cardContent}>
            <div style={{
              ...LandingPageStyles.icon,
              ...(hoveredCard === "lizard-detection" ? LandingPageStyles.iconHover : {})
            }}>
              🔍
            </div>
            <h3 style={LandingPageStyles.optionTitle}>Lizard Detection</h3>
            <p style={LandingPageStyles.optionDescription}>
              Detect lizard toepads with AI-powered bounding boxes
            </p>
          </div>
        </div>

        {/* Custom Model - Disabled */}
        <div
          style={{
            ...LandingPageStyles.optionCard,
            ...LandingPageStyles.optionCardDisabled,
          }}
        >
          <div style={LandingPageStyles.cardContent}>
            <div style={{ ...LandingPageStyles.icon, ...LandingPageStyles.iconDisabled }}>
              🤖
            </div>
            <h3 style={LandingPageStyles.optionTitle}>Custom Model</h3>
            <p style={LandingPageStyles.optionDescription}>
              Use your own trained model for analysis
            </p>
            <div style={LandingPageStyles.comingSoon}>Coming Soon</div>
          </div>
        </div>
      </div>
    </div>
  );
}; 