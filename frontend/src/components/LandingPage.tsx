import React from "react";
import { useNavigate } from "react-router-dom";

export type LizardViewType = "dorsal" | "lateral" | "toepads" | "custom";

const LandingPageStyles = {
  container: {
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
    justifyContent: "center",
    minHeight: "100vh",
    backgroundColor: "#f5f5f5",
    padding: "20px",
  },
  title: {
    fontSize: "2.5rem",
    fontWeight: "bold" as const,
    color: "#333",
    marginBottom: "1rem",
    textAlign: "center" as const,
  },
  subtitle: {
    fontSize: "1.2rem",
    color: "#666",
    marginBottom: "3rem",
    textAlign: "center" as const,
  },
  optionsContainer: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
    gap: "2rem",
    maxWidth: "800px",
    width: "100%",
  },
  optionCard: {
    backgroundColor: "white",
    borderRadius: "12px",
    padding: "2rem",
    boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
    cursor: "pointer",
    transition: "all 0.3s ease",
    border: "2px solid transparent",
  },
  optionCardHover: {
    transform: "translateY(-4px)",
    boxShadow: "0 8px 25px rgba(0, 0, 0, 0.15)",
    borderColor: "#4CAF50",
  },
  optionCardDisabled: {
    backgroundColor: "#f0f0f0",
    cursor: "not-allowed",
    opacity: 0.6,
  },
  optionTitle: {
    fontSize: "1.5rem",
    fontWeight: "bold" as const,
    color: "#333",
    marginBottom: "0.5rem",
  },
  optionDescription: {
    fontSize: "1rem",
    color: "#666",
    marginBottom: "1rem",
    lineHeight: "1.5",
  },
  comingSoon: {
    fontSize: "0.9rem",
    color: "#ff9800",
    fontWeight: "bold" as const,
    textTransform: "uppercase" as const,
  },
  icon: {
    fontSize: "3rem",
    marginBottom: "1rem",
    color: "#4CAF50",
  },
  iconDisabled: {
    color: "#ccc",
  },
};

export const LandingPage: React.FC = () => {
  const navigate = useNavigate();
  
  const handleOptionClick = (viewType: LizardViewType) => {
    if (viewType === "toepads" || viewType === "custom") {
      return; // Disabled
    }
    navigate(`/${viewType}`);
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
            ...LandingPageStyles.optionCardHover,
          }}
          onClick={() => handleOptionClick("dorsal")}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = "translateY(-4px)";
            e.currentTarget.style.boxShadow = "0 8px 25px rgba(0, 0, 0, 0.15)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = "translateY(0)";
            e.currentTarget.style.boxShadow = "0 4px 6px rgba(0, 0, 0, 0.1)";
          }}
        >
          <div style={LandingPageStyles.icon}>🦎</div>
          <h3 style={LandingPageStyles.optionTitle}>Dorsal View</h3>
        </div>

        {/* Lateral View */}
        <div
          style={{
            ...LandingPageStyles.optionCard,
            ...LandingPageStyles.optionCardHover,
          }}
          onClick={() => handleOptionClick("lateral")}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = "translateY(-4px)";
            e.currentTarget.style.boxShadow = "0 8px 25px rgba(0, 0, 0, 0.15)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = "translateY(0)";
            e.currentTarget.style.boxShadow = "0 4px 6px rgba(0, 0, 0, 0.1)";
          }}
        >
          <div style={LandingPageStyles.icon}>🦖</div>
          <h3 style={LandingPageStyles.optionTitle}>Lateral View</h3>
        </div>

        {/* Toepads View - Disabled */}
        <div
          style={{
            ...LandingPageStyles.optionCard,
            ...LandingPageStyles.optionCardDisabled,
          }}
        >
          <div style={{ ...LandingPageStyles.icon, ...LandingPageStyles.iconDisabled }}>
            🦶
          </div>
          <h3 style={LandingPageStyles.optionTitle}>Toepads View</h3>
          <div style={LandingPageStyles.comingSoon}>Coming Soon</div>
        </div>

        {/* Custom Model - Disabled */}
        <div
          style={{
            ...LandingPageStyles.optionCard,
            ...LandingPageStyles.optionCardDisabled,
          }}
        >
          <div style={{ ...LandingPageStyles.icon, ...LandingPageStyles.iconDisabled }}>
            🤖
          </div>
          <h3 style={LandingPageStyles.optionTitle}>Custom Model</h3>
          <div style={LandingPageStyles.comingSoon}>Coming Soon</div>
        </div>
      </div>
    </div>
  );
}; 