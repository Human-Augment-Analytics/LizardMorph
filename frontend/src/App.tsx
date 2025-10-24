import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import "./App.css";
import { MainView } from "./views/MainView";
import { LandingPage } from "./components/LandingPage";
import { YoloTestPage } from "./components/YoloTestPage";
import { LizardDetectionPage } from "./components/LizardDetectionPage";
import type { LizardViewType } from "./components/LandingPage";

// Wrapper component to provide navigation to MainView
const MainViewWrapper: React.FC<{ selectedViewType: LizardViewType }> = ({ selectedViewType }) => {
  return <MainView selectedViewType={selectedViewType} />;
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
        <Route path="/yolo-test" element={<YoloTestPage />} />
        <Route path="/lizard-detection" element={<LizardDetectionPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}

export default App;
