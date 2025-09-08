import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import "./App.css";
import { MainView } from "./views/MainView";
import { LandingPage } from "./components/LandingPage";
import type { LizardViewType } from "./components/LandingPage";

// Version information (injected by Vite)
declare const __APP_VERSION__: string;
declare const __BUILD_VERSION__: string;

// Wrapper component to provide navigation to MainView
const MainViewWrapper: React.FC<{ selectedViewType: LizardViewType }> = ({ selectedViewType }) => {
  return <MainView selectedViewType={selectedViewType} />;
};

// Version display component
const VersionInfo: React.FC = () => {
  const [showVersion, setShowVersion] = React.useState(false);
  
  return (
    <div style={{ 
      position: 'fixed', 
      bottom: '10px', 
      right: '10px', 
      fontSize: '10px', 
      color: '#666',
      cursor: 'pointer',
      zIndex: 9999
    }} 
    onClick={() => setShowVersion(!showVersion)}
    title="Click to toggle version info"
    >
      {showVersion ? (
        <div style={{ 
          background: 'rgba(0,0,0,0.8)', 
          color: 'white', 
          padding: '5px', 
          borderRadius: '3px',
          fontSize: '9px'
        }}>
          <div>App: {__APP_VERSION__}</div>
          <div>Build: {__BUILD_VERSION__}</div>
        </div>
      ) : (
        `v${__APP_VERSION__}`
      )}
    </div>
  );
};

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/dorsal" element={<MainViewWrapper selectedViewType="dorsal" />} />
        <Route path="/lateral" element={<MainViewWrapper selectedViewType="lateral" />} />
        <Route path="/toepads" element={<Navigate to="/" replace />} />
        <Route path="/custom" element={<Navigate to="/" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <VersionInfo />
    </Router>
  );
}

export default App;
