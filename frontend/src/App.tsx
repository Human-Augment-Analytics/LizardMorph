import React from "react";
import { HashRouter as Router, Routes, Route, Navigate, useNavigate } from "react-router-dom";
import "./App.css";
import { MainView } from "./views/MainView";
import { LandingPage } from "./components/LandingPage";
import type { LizardViewType } from "./components/LandingPage";

// Wrapper component to provide navigation to MainView
const MainViewWrapper: React.FC<{ selectedViewType: LizardViewType }> = ({ selectedViewType }) => {
  const navigate = useNavigate();
  return <MainView selectedViewType={selectedViewType} onNavigateHome={() => navigate("/")} />;
};

function App() {
  return (
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
    </Router>
  );
}

export default App;
