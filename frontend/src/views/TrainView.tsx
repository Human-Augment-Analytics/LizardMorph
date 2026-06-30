import React, { useState, useEffect, useRef, useMemo } from "react";
import { ApiService } from "../services/ApiService";
import type { PredictorMeta } from "../services/ApiService";
import { useTheme } from "../contexts/ThemeContext";
import { getTokens } from "../contexts/themeTokens";

interface Props {
  onNavigateHome: () => void;
}

export const TrainView: React.FC<Props> = ({ onNavigateHome }) => {
  const { resolved } = useTheme();
  const isDark = resolved === "dark";
  const t = getTokens(resolved);

  const [predictors, setPredictors] = useState<PredictorMeta[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Training form state
  const [modelName, setModelName] = useState("Custom Predictor");
  const [zipFile, setZipFile] = useState<File | null>(null);
  
  // Training execution state
  const [isTraining, setIsTraining] = useState(false);
  const [progressText, setProgressText] = useState<string | null>(null);
  const [trainingJobId, setTrainingJobId] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  const pollIntervalRef = useRef<number | null>(null);

  // Fetch current custom predictors list
  const fetchPredictors = async () => {
    setLoading(true);
    try {
      const res = await ApiService.listPredictors();
      setPredictors(res);
    } catch (err: any) {
      setError(`Failed to load predictors: ${err.message || err}`);
    } finally {
      setLoading(false);
    }
  };

  const stopPolling = () => {
    if (pollIntervalRef.current !== null) {
      window.clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  };

  useEffect(() => {
    fetchPredictors();
    return () => stopPolling();
  }, []);

  const handleTrainModel = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!zipFile || !modelName.trim()) return;

    setIsTraining(true);
    setProgressText("Uploading dataset ZIP file...");
    setError(null);

    try {
      const res = await ApiService.trainPredictor(modelName.trim(), zipFile);
      if (!res.success || !res.job_id) {
        throw new Error((res as any).error || res.message || "Failed to start training (no job ID returned)");
      }

      setTrainingJobId(res.job_id);
      setProgressText("Training shape predictor... (this may take 10-40s)");

      pollIntervalRef.current = window.setInterval(async () => {
        try {
          const statusRes = await ApiService.getTrainStatus(res.job_id);
          if (statusRes.success) {
            if (statusRes.status === "completed" && statusRes.predictor) {
              stopPolling();
              setIsTraining(false);
              setProgressText(null);
              setTrainingJobId(null);
              setZipFile(null);
              setModelName("Custom Predictor");
              // Refresh list
              await fetchPredictors();
            } else if (statusRes.status === "failed") {
              stopPolling();
              setIsTraining(false);
              setProgressText(null);
              setTrainingJobId(null);
              setError(`Training failed: ${statusRes.error || "Unknown server error"}`);
            } else if (statusRes.status === "training") {
              setProgressText("Training model: learning shape predictor cascades...");
            }
          } else {
            stopPolling();
            setIsTraining(false);
            setProgressText(null);
            setTrainingJobId(null);
            setError(`Training status check failed: ${statusRes.error || "Unknown server error"}`);
          }
        } catch (pollErr: any) {
          stopPolling();
          setIsTraining(false);
          setProgressText(null);
          setTrainingJobId(null);
          setError(`Status polling check failed: ${pollErr.message || pollErr}`);
        }
      }, 2000);
    } catch (err: any) {
      setIsTraining(false);
      setProgressText(null);
      setError(`Failed to initiate training: ${err.message || err}`);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (isTraining) return;
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (isTraining) return;
    const file = e.dataTransfer.files?.[0];
    if (file) {
      if (file.name.toLowerCase().endsWith(".zip")) {
        setZipFile(file);
        setError(null);
      } else {
        setError("Please drop a valid .zip dataset file.");
      }
    }
  };

  const handleDeletePredictor = async (id: string) => {
    if (!window.confirm("Are you sure you want to delete this custom model?")) return;
    try {
      await ApiService.deletePredictor(id);
      await fetchPredictors();
    } catch (err: any) {
      setError(`Failed to delete model: ${err.message || err}`);
    }
  };

  const styles = useMemo(() => `
    .train-view-container {
      max-width: 1100px;
      margin: 0 auto;
      padding: 40px 20px;
      color: ${t.text};
      font-family: 'Outfit', 'Inter', sans-serif;
      min-height: 100vh;
    }
    .train-view-container .btn-back {
      background: none;
      border: none;
      color: ${isDark ? "#81c784" : "#2e7d32"};
      cursor: pointer;
      font-weight: 700;
      font-size: 14px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 16px;
      border-radius: 8px;
      transition: all 0.2s ease;
      margin-bottom: 20px;
    }
    .train-view-container .btn-back:hover {
      background: ${isDark ? "rgba(129, 199, 132, 0.15)" : "rgba(46, 125, 50, 0.08)"};
      transform: translateX(-2px);
    }
    .train-view-container .train-grid {
      display: grid;
      grid-template-columns: 1fr 1.2fr;
      gap: 32px;
      margin-top: 30px;
    }
    @media (max-width: 768px) {
      .train-view-container .train-grid {
        grid-template-columns: 1fr;
      }
    }
    .train-view-container .train-card {
      background: ${isDark ? "rgba(30, 42, 58, 0.65)" : "rgba(255, 255, 255, 0.75)"};
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid ${t.borderLight};
      border-radius: 18px;
      padding: 30px;
      box-shadow: 0 8px 32px ${isDark ? "rgba(0,0,0,0.25)" : "rgba(0,0,0,0.06)"};
      transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .train-view-container .train-card:hover {
      box-shadow: 0 12px 40px ${isDark ? "rgba(0,0,0,0.35)" : "rgba(0,0,0,0.1)"};
    }
    .train-view-container .form-group {
      margin-bottom: 20px;
    }
    .train-view-container .form-label {
      display: block;
      font-size: 13px;
      font-weight: 600;
      color: ${t.textSecondary};
      margin-bottom: 8px;
      letter-spacing: 0.5px;
    }
    .train-view-container .form-input {
      width: 100%;
      padding: 12px 14px;
      border-radius: 10px;
      border: 1px solid ${t.border};
      background: ${isDark ? "#2a3a4e" : "white"};
      color: ${t.text};
      outline: none;
      box-sizing: border-box;
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
      font-size: 14px;
    }
    .train-view-container .form-input:focus {
      border-color: #4CAF50;
      box-shadow: 0 0 0 3px ${isDark ? "rgba(76, 175, 80, 0.25)" : "rgba(76, 175, 80, 0.15)"};
    }
    .train-view-container .btn-primary {
      padding: 12px 24px;
      border-radius: 10px;
      border: none;
      background: linear-gradient(135deg, #4F7942 0%, #3F6932 100%);
      color: white;
      font-weight: 700;
      cursor: pointer;
      width: 100%;
      font-size: 15px;
      box-shadow: 0 4px 15px ${isDark ? "rgba(79, 121, 66, 0.3)" : "rgba(79, 121, 66, 0.2)"};
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 8px;
    }
    .train-view-container .btn-primary:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px ${isDark ? "rgba(79, 121, 66, 0.4)" : "rgba(79, 121, 66, 0.3)"};
      background: linear-gradient(135deg, #5c8c4f 0%, #467339 100%);
    }
    .train-view-container .btn-primary:active:not(:disabled) {
      transform: translateY(0);
    }
    .train-view-container .btn-primary:disabled {
      background: ${isDark ? "#2c3e50" : "#bdc3c7"};
      color: ${t.textMuted};
      box-shadow: none;
      cursor: not-allowed;
      opacity: 0.6;
    }
    .train-view-container .model-list {
      display: flex;
      flex-direction: column;
      gap: 14px;
      max-height: 480px;
      overflow-y: auto;
      padding-right: 4px;
    }
    .train-view-container .model-list::-webkit-scrollbar {
      width: 6px;
    }
    .train-view-container .model-list::-webkit-scrollbar-track {
      background: transparent;
    }
    .train-view-container .model-list::-webkit-scrollbar-thumb {
      background: ${isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"};
      border-radius: 3px;
    }
    .train-view-container .model-list-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 20px;
      background: ${isDark ? "rgba(255,255,255,0.02)" : "rgba(0, 0, 0, 0.02)"};
      border: 1px solid ${t.borderLight};
      border-radius: 12px;
      transition: all 0.2s ease;
    }
    .train-view-container .model-list-item:hover {
      background: ${isDark ? "rgba(255,255,255,0.04)" : "rgba(0, 0, 0, 0.04)"};
      border-color: ${isDark ? "rgba(255,255,255,0.12)" : "rgba(0, 0, 0, 0.12)"};
      transform: translateX(2px);
    }
    .train-view-container .btn-delete {
      background: ${isDark ? "rgba(239, 83, 80, 0.1)" : "rgba(211, 47, 47, 0.06)"};
      border: 1px solid ${isDark ? "rgba(239, 83, 80, 0.2)" : "rgba(211, 47, 47, 0.15)"};
      color: #ef5350;
      padding: 8px 14px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 700;
      transition: all 0.2s ease;
    }
    .train-view-container .btn-delete:hover {
      background: #ef5350;
      color: white;
      border-color: #ef5350;
      box-shadow: 0 4px 12px rgba(239, 83, 80, 0.25);
    }
    .train-view-container .status-alert {
      margin-top: 20px;
      padding: 14px 18px;
      border-radius: 12px;
      font-size: 14px;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .train-view-container .status-error {
      background: ${isDark ? "rgba(239, 83, 80, 0.12)" : "rgba(211, 47, 47, 0.08)"};
      border: 1px solid ${isDark ? "rgba(239, 83, 80, 0.2)" : "rgba(211, 47, 47, 0.15)"};
      color: ${t.error};
    }
    .train-view-container .status-progress {
      background: ${isDark ? "rgba(129, 199, 132, 0.12)" : "rgba(46, 125, 50, 0.08)"};
      border: 1px solid ${isDark ? "rgba(129, 199, 132, 0.2)" : "rgba(46, 125, 50, 0.15)"};
      color: ${isDark ? "#81c784" : "#2e7d32"};
    }
    .train-view-container .spinner {
      display: inline-block;
      width: 16px;
      height: 16px;
      border: 2.5px solid currentColor;
      border-right-color: transparent;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  `, [isDark, t]);

  return (
    <div className="train-view-container">
      <button onClick={onNavigateHome} className="btn-back">
        ← Back to Overview
      </button>

      <h1 style={{ marginTop: "12px", fontSize: "32px", fontWeight: 800 }}>🧠 Custom Model Training</h1>
      <p style={{ opacity: 0.7, fontSize: "14px", marginTop: "6px", marginBottom: "30px" }}>
        Train new shape predictors using your own annotated datasets on your local CPU.
      </p>

      {error && (
        <div className="status-alert status-error" style={{ marginBottom: "20px" }}>
          ⚠️ {error}
        </div>
      )}

      <div className="train-grid">
        {/* Left Side: Training Form */}
        <div className="train-card">
          <h2 style={{ fontSize: "20px", fontWeight: 700, marginBottom: "20px" }}>🛠️ Train New Predictor</h2>
          <form onSubmit={handleTrainModel}>
            <div className="form-group">
              <label className="form-label">Model Display Name</label>
              <input
                type="text"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                disabled={isTraining}
                required
                className="form-input"
              />
            </div>

            <div className="form-group">
              <span className="form-label">Dataset ZIP File</span>
              <label 
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  padding: "28px",
                  borderRadius: "12px",
                  border: `2px dashed ${
                    isDragging || isHovered
                      ? "#4CAF50"
                      : zipFile
                      ? "#4CAF50"
                      : t.border
                  }`,
                  background:
                    isDragging || isHovered
                      ? isDark
                        ? "rgba(255, 255, 255, 0.08)"
                        : "rgba(0, 0, 0, 0.04)"
                      : isDark
                      ? "rgba(255, 255, 255, 0.02)"
                      : "rgba(0, 0, 0, 0.01)",
                  cursor: isTraining ? "not-allowed" : "pointer",
                  transition: "all 0.2s ease",
                  textAlign: "center",
                  gap: "8px"
                }}
                className="file-dropzone"
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onMouseEnter={() => !isTraining && setIsHovered(true)}
                onMouseLeave={() => setIsHovered(false)}
              >
                <input
                  type="file"
                  accept=".zip"
                  onChange={(e) => setZipFile(e.target.files?.[0] || null)}
                  disabled={isTraining}
                  style={{ display: "none" }}
                  required
                />
                <span style={{ fontSize: "28px" }}>{zipFile ? "📦" : "📤"}</span>
                <span style={{ fontSize: "14px", fontWeight: 600, color: t.text }}>
                  {zipFile ? zipFile.name : "Choose dataset ZIP file"}
                </span>
                <span style={{ fontSize: "11px", color: t.textMuted }}>
                  {zipFile ? `Size: ${(zipFile.size / (1024 * 1024)).toFixed(2)} MB` : "Must contain images and TPS/XML metadata"}
                </span>
              </label>
            </div>

            <button
              type="submit"
              disabled={isTraining || !zipFile || !modelName.trim()}
              className="btn-primary"
            >
              {isTraining ? (
                <>
                  <span className="spinner" />
                  <span>Training...</span>
                </>
              ) : (
                "Start Local Training"
              )}
            </button>
          </form>

          {isTraining && progressText && (
            <div className="status-alert status-progress" style={{ flexDirection: "column", alignItems: "flex-start", gap: "6px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <span className="spinner" />
                <span>{progressText}</span>
              </div>
              {trainingJobId && (
                <div style={{ fontSize: "11px", opacity: 0.7, paddingLeft: "26px" }}>
                  Job ID: <code style={{ fontFamily: "monospace" }}>{trainingJobId}</code>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Side: Predictors List */}
        <div className="train-card">
          <h2 style={{ fontSize: "20px", fontWeight: 700, marginBottom: "20px" }}>📚 Available Custom Models</h2>
          {loading && <p style={{ fontSize: "14px", opacity: 0.7 }}>Loading models...</p>}
          {!loading && predictors.length === 0 && (
            <p style={{ fontSize: "14px", opacity: 0.6, fontStyle: "italic" }}>
              No custom models trained yet. Upload a dataset to train your first model!
            </p>
          )}
          {!loading && predictors.length > 0 && (
            <div className="model-list">
              {predictors.map((p) => (
                <div key={p.id} className="model-list-item">
                  <div>
                    <div style={{ fontWeight: 700, fontSize: "15px" }}>{p.display_name}</div>
                    <div style={{ fontSize: "12px", opacity: 0.6, marginTop: "4px" }}>
                      Points: {p.num_parts ?? "Unknown"} | Size: {p.size_bytes ? `${(p.size_bytes / (1024 * 1024)).toFixed(2)} MB` : "Unknown size"}
                    </div>
                  </div>
                  <button
                    onClick={() => handleDeletePredictor(p.id)}
                    className="btn-delete"
                  >
                    Delete
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      <style>{styles}</style>
    </div>
  );
};
