import React from "react";

interface Props {
  onNavigateHome: () => void;
}

export const TrainView: React.FC<Props> = ({ onNavigateHome }) => {
  return (
    <div style={{ padding: 24, fontFamily: "sans-serif" }}>
      <button onClick={onNavigateHome}>← Back to Home</button>
      <h2>Train Custom Model</h2>
    </div>
  );
};
