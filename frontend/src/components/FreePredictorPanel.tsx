import type { PredictorMeta } from "../services/ApiService";
import React from "react";

interface Props {
  predictors: PredictorMeta[];
  selectedPredictorId: string | null;
  predictorsLoading: boolean;
  error: string | null;
  hasCurrentImage: boolean;
  onRefresh: () => void;
  onSelectPredictorId: (id: string | null) => void;
  onUploadPredictor: (file: File) => void;
  onDeleteSelected: () => void;
  onAutoplace: () => void;
}

export function FreePredictorPanel(props: Props) {
  const selected = props.selectedPredictorId;
  const canAutoplace = Boolean(selected) && props.hasCurrentImage && !props.predictorsLoading;
  const isBusy = props.predictorsLoading;

  const cardStyle: React.CSSProperties = {
    marginTop: 12,
    padding: "14px 14px",
    border: "1px solid rgba(0, 0, 0, 0.10)",
    borderRadius: 12,
    background: "rgba(255, 255, 255, 0.92)",
    boxShadow: "0 8px 24px rgba(0, 0, 0, 0.06)",
    backdropFilter: "blur(6px)",
  };

  const titleRowStyle: React.CSSProperties = {
    display: "flex",
    alignItems: "baseline",
    justifyContent: "space-between",
    gap: 12,
    marginBottom: 10,
    flexWrap: "wrap",
  };

  const titleStyle: React.CSSProperties = {
    fontWeight: 800,
    fontSize: 14,
    letterSpacing: "0.2px",
    color: "#111",
  };

  const subtitleStyle: React.CSSProperties = {
    marginTop: 4,
    fontSize: 12,
    color: "rgba(0, 0, 0, 0.65)",
    lineHeight: 1.35,
  };

  const gridStyle: React.CSSProperties = {
    display: "grid",
    gridTemplateColumns: "minmax(260px, 1.1fr) minmax(240px, 0.9fr) auto",
    gap: 12,
    alignItems: "end",
  };

  const groupStyle: React.CSSProperties = {
    display: "flex",
    flexDirection: "column",
    gap: 6,
    minWidth: 0,
  };

  const labelStyle: React.CSSProperties = {
    fontSize: 12,
    fontWeight: 700,
    color: "rgba(0, 0, 0, 0.75)",
  };

  const inputStyle: React.CSSProperties = {
    width: "100%",
    padding: "9px 10px",
    borderRadius: 10,
    border: "1px solid rgba(0, 0, 0, 0.14)",
    background: "white",
    outline: "none",
  };

  const buttonBase: React.CSSProperties = {
    padding: "9px 12px",
    borderRadius: 10,
    border: "1px solid rgba(0, 0, 0, 0.14)",
    background: "white",
    color: "#111",
    fontWeight: 700,
    cursor: "pointer",
    userSelect: "none",
  };

  const primaryButton: React.CSSProperties = {
    ...buttonBase,
    background: "#0d47a1",
    border: "1px solid rgba(13, 71, 161, 0.25)",
    color: "white",
  };

  const dangerButton: React.CSSProperties = {
    ...buttonBase,
    background: "#b71c1c",
    border: "1px solid rgba(183, 28, 28, 0.25)",
    color: "white",
  };

  const mutedButton: React.CSSProperties = {
    ...buttonBase,
    background: "rgba(0, 0, 0, 0.04)",
  };

  const actionsRowStyle: React.CSSProperties = {
    display: "flex",
    gap: 10,
    alignItems: "center",
    justifyContent: "flex-end",
    flexWrap: "wrap",
  };

  return (
    <div style={{ ...cardStyle, marginTop: 0, border: "none", boxShadow: "none", backdropFilter: "none" }}>
      <div style={titleRowStyle}>
        <div>
          <div style={titleStyle}>Predictor (Free mode)</div>
          <div style={subtitleStyle}>
            Select a global <code>.dat</code> model or upload a new one, then auto-place landmarks on the current image.
          </div>
        </div>

        <div style={actionsRowStyle}>
          <button onClick={props.onRefresh} disabled={isBusy} style={isBusy ? mutedButton : buttonBase}>
            {isBusy ? "Refreshing…" : "Refresh list"}
          </button>
          <button
            onClick={props.onAutoplace}
            disabled={!canAutoplace}
            style={!canAutoplace ? { ...primaryButton, opacity: 0.55, cursor: "not-allowed" } : primaryButton}
            title={!props.hasCurrentImage ? "Upload/select an image first" : !selected ? "Select a predictor first" : ""}
          >
            Auto-place landmarks
          </button>
        </div>
      </div>

      <div style={gridStyle}>
        <div style={groupStyle}>
          <div style={labelStyle}>Selected predictor</div>
          <select
            value={selected ?? ""}
            onChange={(e) => props.onSelectPredictorId(e.target.value || null)}
            disabled={isBusy}
            style={inputStyle}
          >
            <option value="">None (manual)</option>
            {props.predictors.map((p) => (
              <option key={p.id} value={p.id}>
                {p.display_name}
                {typeof p.num_parts === "number" ? ` (${p.num_parts} pts)` : ""}
              </option>
            ))}
          </select>
        </div>

        <div style={groupStyle}>
          <div style={labelStyle}>Upload new predictor</div>
          <input
            type="file"
            accept=".dat"
            disabled={isBusy}
            style={inputStyle}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) props.onUploadPredictor(file);
              e.currentTarget.value = "";
            }}
          />
        </div>

        <div style={groupStyle}>
          <div style={labelStyle}>&nbsp;</div>
          <button
            onClick={props.onDeleteSelected}
            disabled={!selected || isBusy}
            style={!selected || isBusy ? { ...dangerButton, opacity: 0.55, cursor: "not-allowed" } : dangerButton}
            title={!selected ? "Select a predictor to delete" : ""}
          >
            Delete
          </button>
        </div>
      </div>

      {props.error && (
        <div style={{ marginTop: 10, color: "#b71c1c", fontWeight: 700, fontSize: 12 }}>
          {props.error}
        </div>
      )}
    </div>
  );
}

