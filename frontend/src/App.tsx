import React from "react";
import { HashRouter as Router, Routes, Route, Navigate, useNavigate } from "react-router-dom";
import "./App.css";
import { MainView } from "./views/MainView";
import { LandingPage } from "./components/LandingPage";
import type { LizardViewType } from "./components/LandingPage";
import { ThemeProvider, useTheme } from "./contexts/ThemeContext";

// Wrapper component to provide navigation to MainView
const MainViewWrapper: React.FC<{ selectedViewType: LizardViewType }> = ({ selectedViewType }) => {
  const navigate = useNavigate();
  return <MainView selectedViewType={selectedViewType} onNavigateHome={() => navigate("/")} />;
};

// Version display component
const VersionInfo: React.FC = () => {
  const [showVersion, setShowVersion] = React.useState(false);
  const { resolved } = useTheme();
  const isDark = resolved === "dark";

  return (
    <div style={{
      position: 'fixed',
      bottom: '10px',
      right: '10px',
      fontSize: '10px',
      color: isDark ? '#8899aa' : '#666',
      cursor: 'pointer',
      zIndex: 9999
    }}
    onClick={() => setShowVersion(!showVersion)}
    title="Click to toggle version info"
    >
      {showVersion ? (
        <div style={{
          background: isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.8)',
          color: 'white',
          padding: '5px',
          borderRadius: '3px',
          fontSize: '9px'
        }}>
          <div>App: {__APP_VERSION__}</div>
          <div>Build: {__BUILD_VERSION__}</div>
        </div>
      ) : (
        __APP_VERSION__
      )}
    </div>
  );
};

function App() {
  return (
    <ThemeProvider>
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/dorsal" element={<MainViewWrapper selectedViewType="dorsal" />} />
        <Route path="/lateral" element={<MainViewWrapper selectedViewType="lateral" />} />
        <Route path="/toepads" element={<MainViewWrapper selectedViewType="toepads" />} />
        <Route path="/toepad" element={<MainViewWrapper selectedViewType="toepads" />} />
        <Route path="/free" element={<MainViewWrapper selectedViewType="free" />} />
        <Route path="/custom" element={<Navigate to="/" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <VersionInfo />
    </Router>
    </ThemeProvider>
  );
}

export default App;
