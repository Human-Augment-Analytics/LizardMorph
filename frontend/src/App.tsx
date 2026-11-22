import React from "react";
import { createHashRouter, RouterProvider, Outlet, Navigate, useNavigate, useLocation, useParams } from "react-router-dom";
import "./App.css";
import { MainView } from "./views/MainView";
import { LandingPage } from "./components/LandingPage";
import type { LizardViewType } from "./components/LandingPage";
import { ThemeProvider } from "./contexts/ThemeContext";
import { useTheme } from "./contexts/theme";
import { ModelsView } from "./views/ModelsView";
import { getModelApiUrl } from "./services/ModelsApiService";
import { TrainView } from "./views/TrainView";

// Wrapper component to provide navigation to MainView
const MainViewWrapper: React.FC<{ selectedViewType: LizardViewType }> = ({ selectedViewType }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const { modelId: routeModelId } = useParams<{ modelId?: string }>();
  const navigationState = location.state as { selectedModelId?: string } | null;
  const modelId = routeModelId ?? navigationState?.selectedModelId;
  return <MainView key={JSON.stringify([selectedViewType, modelId])} selectedViewType={selectedViewType} modelId={modelId} onNavigateHome={() => navigate("/")} />;
};

// Wrapper component for TrainView custom mode routing
const TrainViewWrapper: React.FC = () => {
  const navigate = useNavigate();
  return (
    <TrainView
      onNavigateHome={() => navigate("/")}
      onUseModel={(modelId) => navigate(`/custom-model/${encodeURIComponent(modelId)}`)}
    />
  );
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

const router = createHashRouter([
  {
    element: (
      <ThemeProvider>
        <Outlet />
        <VersionInfo />
      </ThemeProvider>
    ),
    children: [
      { path: "/", element: <LandingPage /> },
      { path: "/dorsal", element: <MainViewWrapper selectedViewType="dorsal" /> },
      { path: "/lateral", element: <MainViewWrapper selectedViewType="lateral" /> },
      { path: "/toepads", element: <MainViewWrapper selectedViewType="toepads" /> },
      { path: "/toepad", element: <MainViewWrapper selectedViewType="toepads" /> },
      { path: "/free", element: <MainViewWrapper selectedViewType="free" /> },
      { path: "/custom-model", element: <MainViewWrapper selectedViewType="custom" /> },
      { path: "/custom-model/:modelId", element: <MainViewWrapper selectedViewType="custom" /> },
      { path: "/custom", element: <TrainViewWrapper /> },
      ...(getModelApiUrl() ? [{ path: "/models", element: <ModelsView /> }] : []),
      { path: "*", element: <Navigate to="/" replace /> },
    ],
  },
]);

function App() {
  return <RouterProvider router={router} />;
}

export default App;
