import React, { useState, useEffect, useRef, useMemo } from "react";
import JSZip from "jszip";
import { ApiService } from "../services/ApiService";
import type { ModelVersionItem, ProjectItem, DeriveBoxesResult, TrainJobStatusResult } from "../services/ApiService";
import { useTheme } from "../contexts/theme";
import { getTokens } from "../contexts/themeTokens";

interface Props {
  onNavigateHome: () => void;
  onUseModel: (modelId: string) => void;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export const TrainView: React.FC<Props> = ({ onNavigateHome, onUseModel }) => {
  const { resolved } = useTheme();
  const isDark = resolved === "dark";
  const t = getTokens(resolved);

  // Active Wizard Step (1 to 7)
  const [currentStep, setCurrentStep] = useState<number>(1);

  // Step 1: Create Project State
  const [projectName, setProjectName] = useState("Anolis Morphometrics");
  const [organism, setOrganism] = useState("Anolis carolinensis");
  const [createdProject, setCreatedProject] = useState<ProjectItem | null>(null);

  // Step 2: Add Data State
  const [datasetFiles, setDatasetFiles] = useState<File[]>([]);
  const [padding, setPadding] = useState<number>(0.2);
  const [isDragging, setIsDragging] = useState(false);

  // Step 3: Check Data & Visual Preview State
  const [derivedBoxes, setDerivedBoxes] = useState<DeriveBoxesResult | null>(null);
  const [isCheckingData, setIsCheckingData] = useState(false);
  const [previewImageUrls, setPreviewImageUrls] = useState<Record<string, string>>({});
  const [selectedImageIndex, setSelectedImageIndex] = useState<number>(0);
  const [showPreviewLandmarks, setShowPreviewLandmarks] = useState<boolean>(true);
  const [showPreviewBoxes, setShowPreviewBoxes] = useState<boolean>(true);
  const [showPreviewLabels, setShowPreviewLabels] = useState<boolean>(true);

  // Step 4: Train Setup State
  const [trainingPreset, setTrainingPreset] = useState<"fast" | "standard" | "accurate">("standard");
  const [showExpertPanel, setShowExpertPanel] = useState(false);
  const [nu, setNu] = useState<number>(0.1);
  const [treeDepth, setTreeDepth] = useState<number>(4);
  const [cascadeDepth, setCascadeDepth] = useState<number>(15);
  const [oversamplingAmount, setOversamplingAmount] = useState<number>(5);
  const [featurePoolSize, setFeaturePoolSize] = useState<number>(400);
  const [numTestSplits, setNumTestSplits] = useState<number>(20);
  const [testSplit, setTestSplit] = useState<number>(0.2);

  // Step 5 & 6: Follow Progress & Review Results State
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<TrainJobStatusResult | null>(null);
  const [isTraining, setIsTraining] = useState(false);

  // Step 7: Publish & Use State
  const [publishedModelId, setPublishedModelId] = useState<string | null>(null);

  // General state
  const [registeredModels, setRegisteredModels] = useState<ModelVersionItem[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);

  const pollIntervalRef = useRef<number | null>(null);
  const pollFailureCountRef = useRef(0);
  const interruptedFailureCountRef = useRef(0);

  const handleCancelJob = async () => {
    if (!activeJobId) return;
    setIsCancelling(true);
    try {
      await ApiService.cancelTrainJob(activeJobId);
      stopPolling();
      setIsTraining(false);
      setError("Training job cancelled by user.");
    } catch (err: unknown) {
      setError(`Failed to cancel training job: ${errorMessage(err)}`);
    } finally {
      setIsCancelling(false);
    }
  };

  const fetchModels = async () => {
    setLoadingModels(true);
    try {
      const models = await ApiService.getModels();
      setRegisteredModels(models.filter((model) => model.project_id !== "built-in"));
    } catch {
      setRegisteredModels([]);
    } finally {
      setLoadingModels(false);
    }
  };

  const stopPolling = () => {
    if (pollIntervalRef.current !== null) {
      window.clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  };

  useEffect(() => {
    void fetchModels();
    return () => stopPolling();
  }, []);

  // Step 1 Handler: Create Project
  const handleCreateProject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!projectName.trim()) return;
    setError(null);
    try {
      const proj = await ApiService.createProject(projectName.trim(), organism.trim());
      setCreatedProject(proj);
      setCurrentStep(2);
    } catch (err: unknown) {
      setError(`Failed to create project: ${errorMessage(err)}`);
    }
  };

  // Step 2 Handler: Read Dataset File
  const handleFileUpload = (files: File[]) => {
    setDatasetFiles(files);
    setDerivedBoxes(null);
    setError(null);
  };

  // Step 3 Handler: Check Data (Derive Boxes)
  const handleCheckData = async () => {
    if (datasetFiles.length === 0) {
      setError("Select a ZIP dataset, or select one TPS/XML annotation file together with all referenced images.");
      return;
    }
    setIsCheckingData(true);
    setError(null);
    try {
      const result = await ApiService.deriveBoxes(datasetFiles, padding);
      if (!result.training_ready) {
        throw new Error(result.validation_errors[0] || "Dataset is not ready for training.");
      }
      setDerivedBoxes(result);
      setSelectedImageIndex(0);

      // Unpack / generate preview URLs for the images in the dataset
      const urls: Record<string, string> = {};
      for (const file of datasetFiles) {
        if (file.name.toLowerCase().endsWith(".zip")) {
          try {
            const zip = await JSZip.loadAsync(file);
            for (const [relativePath, zipEntry] of Object.entries(zip.files)) {
              if (!zipEntry.dir && /\.(jpe?g|png|bmp|webp|tif?f)$/i.test(relativePath)) {
                const blob = await zipEntry.async("blob");
                const objectUrl = URL.createObjectURL(blob);
                const baseName = relativePath.split("/").pop() || relativePath;
                urls[baseName.toLowerCase()] = objectUrl;
                urls[relativePath.toLowerCase()] = objectUrl;
              }
            }
          } catch (e) {
            console.warn("Failed to unpack images for preview:", e);
          }
        } else if (/\.(jpe?g|png|bmp|webp|tif?f)$/i.test(file.name)) {
          const objectUrl = URL.createObjectURL(file);
          urls[file.name.toLowerCase()] = objectUrl;
        }
      }
      setPreviewImageUrls(urls);
      setCurrentStep(3);
    } catch (err: unknown) {
      setError(`Dataset validation error: ${errorMessage(err)}`);
    } finally {
      setIsCheckingData(false);
    }
  };

  // Step 4 Handler: Start Training
  const handleStartTraining = async () => {
    if (!derivedBoxes || datasetFiles.length === 0) {
      setError("Validate the complete dataset before starting training.");
      setCurrentStep(2);
      return;
    }
    setError(null);
    setIsTraining(true);
    setCurrentStep(5);

    let epochs = 50;
    if (trainingPreset === "fast") epochs = 10;
    if (trainingPreset === "accurate") epochs = 100;

    const config = {
      model_name: createdProject ? createdProject.name : projectName,
      preset: trainingPreset,
      epochs,
      padding,
      nu,
      tree_depth: treeDepth,
      cascade_depth: cascadeDepth,
      oversampling_amount: oversamplingAmount,
      feature_pool_size: featurePoolSize,
      num_test_splits: numTestSplits,
      test_split: testSplit,
    };

    try {
      const projId = createdProject ? createdProject.id : "default";
      const res = await ApiService.submitTrain(projId, datasetFiles, config);
      setActiveJobId(res.job_id);
      setPublishedModelId(res.job_id.replace(/^job_/, ""));
      pollFailureCountRef.current = 0;

      // Start polling status
      pollIntervalRef.current = window.setInterval(async () => {
        try {
          const statusRes: TrainJobStatusResult = await ApiService.getTrainJobStatus(res.job_id);

          setJobStatus(statusRes);
          pollFailureCountRef.current = 0;

          if (statusRes.status === "completed") {
            interruptedFailureCountRef.current = 0;
            stopPolling();
            setIsTraining(false);
            setCurrentStep(6);
            void fetchModels();
          } else if (statusRes.status === "failed") {
            const detail = statusRes.error || statusRes.stage || "Training process failed";
            if (/interrupted.*backend|backend.*restart/i.test(detail)) {
              interruptedFailureCountRef.current += 1;
              // A frozen training worker used to expose a brief interrupted
              // status while it initialized. Confirm that state for 30
              // seconds before treating it as terminal.
              if (interruptedFailureCountRef.current < 15) return;
            }
            stopPolling();
            setIsTraining(false);
            setError(`Training job failed: ${detail}`);
          } else if (statusRes.status === "cancelled") {
            interruptedFailureCountRef.current = 0;
            stopPolling();
            setIsTraining(false);
            setError("Training job cancelled by user.");
          } else {
            interruptedFailureCountRef.current = 0;
            setError(null);
          }
        } catch (pollError: unknown) {
          pollFailureCountRef.current += 1;
          if (pollFailureCountRef.current >= 3) {
            stopPolling();
            setIsTraining(false);
            setError(`Lost contact with the training job: ${errorMessage(pollError)}`);
          }
        }
      }, 2000);
    } catch (err: unknown) {
      setIsTraining(false);
      setError(`Failed to start training: ${errorMessage(err)}`);
    }
  };

  const steps = [
    { num: 1, label: "Create Project" },
    { num: 2, label: "Add Data" },
    { num: 3, label: "Check Data" },
    { num: 4, label: "Train Model" },
    { num: 5, label: "Follow Progress" },
    { num: 6, label: "Review Results" },
    { num: 7, label: "Publish & Use" },
  ];

  const styles = useMemo(() => `
    .wizard-container {
      max-width: 1100px;
      margin: 0 auto;
      padding: 30px 20px;
      color: ${t.text};
      font-family: 'Outfit', 'Inter', sans-serif;
      min-height: 100vh;
    }
    .btn-back-nav {
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
      margin-bottom: 16px;
    }
    .btn-back-nav:hover {
      background: ${isDark ? "rgba(129, 199, 132, 0.15)" : "rgba(46, 125, 50, 0.08)"};
    }
    .wizard-stepper {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin: 24px 0 36px 0;
      padding: 16px;
      background: ${isDark ? "rgba(30, 42, 58, 0.5)" : "rgba(255, 255, 255, 0.8)"};
      backdrop-filter: blur(12px);
      border-radius: 16px;
      border: 1px solid ${t.borderLight};
      overflow-x: auto;
    }
    .step-item {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 6px;
      flex: 1;
      min-width: 90px;
      cursor: pointer;
      position: relative;
      transition: all 0.2s ease;
    }
    .step-bubble {
      width: 38px;
      height: 38px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 14px;
      background: ${isDark ? "#2a3a4e" : "#e0e0e0"};
      color: ${t.textMuted};
      border: 2px solid transparent;
      transition: all 0.3s ease;
    }
    .step-item.active .step-bubble {
      background: linear-gradient(135deg, #4F7942 0%, #3F6932 100%);
      color: white;
      box-shadow: 0 4px 14px ${isDark ? "rgba(79,121,66,0.4)" : "rgba(79,121,66,0.25)"};
      transform: scale(1.1);
    }
    .step-item.completed .step-bubble {
      background: #4CAF50;
      color: white;
    }
    .step-label {
      font-size: 11px;
      font-weight: 600;
      color: ${t.textMuted};
      text-align: center;
      white-space: nowrap;
    }
    .step-item.active .step-label {
      color: ${t.text};
      font-weight: 700;
    }
    .card-panel {
      background: ${isDark ? "rgba(30, 42, 58, 0.65)" : "rgba(255, 255, 255, 0.85)"};
      backdrop-filter: blur(12px);
      border: 1px solid ${t.borderLight};
      border-radius: 18px;
      padding: 32px;
      box-shadow: 0 8px 32px ${isDark ? "rgba(0,0,0,0.25)" : "rgba(0,0,0,0.06)"};
      margin-bottom: 24px;
    }
    .form-control {
      width: 100%;
      padding: 12px 14px;
      border-radius: 10px;
      border: 1px solid ${t.border};
      background: ${isDark ? "#2a3a4e" : "white"};
      color: ${t.text};
      font-size: 14px;
      box-sizing: border-box;
      outline: none;
      margin-top: 6px;
    }
    .form-control:focus {
      border-color: #4CAF50;
    }
    .btn-action {
      padding: 12px 24px;
      border-radius: 10px;
      border: none;
      background: linear-gradient(135deg, #4F7942 0%, #3F6932 100%);
      color: white;
      font-weight: 700;
      font-size: 15px;
      cursor: pointer;
      transition: all 0.3s ease;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .btn-action:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(79, 121, 66, 0.35);
    }
    .btn-action:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .btn-secondary {
      background: ${isDark ? "#2a3a4e" : "#e0e0e0"};
      color: ${t.text};
      border: 1px solid ${t.border};
    }
    .preset-card {
      flex: 1;
      padding: 20px;
      border-radius: 14px;
      border: 2px solid ${t.border};
      background: ${isDark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.01)"};
      cursor: pointer;
      transition: all 0.2s ease;
      text-align: center;
    }
    .preset-card.selected {
      border-color: #4CAF50;
      background: ${isDark ? "rgba(76, 175, 80, 0.12)" : "rgba(76, 175, 80, 0.06)"};
      box-shadow: 0 4px 16px rgba(76, 175, 80, 0.15);
    }
    .progress-bar-bg {
      width: 100%;
      height: 16px;
      background: ${isDark ? "#1e2a3a" : "#e0e0e0"};
      border-radius: 8px;
      overflow: hidden;
      margin: 16px 0;
    }
    .progress-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, #4CAF50 0%, #81c784 100%);
      transition: width 0.4s ease;
    }
    .alert-banner {
      padding: 14px 18px;
      border-radius: 12px;
      margin-bottom: 20px;
      font-weight: 600;
      font-size: 14px;
    }
    .alert-error {
      background: ${isDark ? "rgba(239, 83, 80, 0.15)" : "rgba(211, 47, 47, 0.08)"};
      border: 1px solid ${isDark ? "rgba(239, 83, 80, 0.3)" : "rgba(211, 47, 47, 0.2)"};
      color: ${t.error};
    }
  `, [isDark, t]);

  return (
    <div className="wizard-container">
      <button onClick={onNavigateHome} className="btn-back-nav">
        ← Back to Overview
      </button>

      <h1 style={{ fontSize: "32px", fontWeight: 800, margin: "8px 0 4px 0" }}>
        Model Training Wizard
      </h1>
      <p style={{ opacity: 0.7, fontSize: "14px", margin: 0 }}>
        7-step non-technical wizard to prepare data, configure, and train YOLO-OBB + ML-Morph models.
      </p>

      {error && (
        <div className="alert-banner alert-error" style={{ marginTop: "16px" }}>
          {error}
        </div>
      )}

      {/* 7-Step Horizontal Stepper */}
      <div className="wizard-stepper">
        {steps.map((s) => {
          const isActive = currentStep === s.num;
          const isCompleted = currentStep > s.num;
          return (
            <div
              key={s.num}
              className={`step-item ${isActive ? "active" : ""} ${isCompleted ? "completed" : ""}`}
              onClick={() => {
                if (s.num < currentStep || isCompleted) {
                  setCurrentStep(s.num);
                }
              }}
            >
              <div className="step-bubble">
                {isCompleted ? "✓" : s.num}
              </div>
              <span className="step-label">
                {s.label}
              </span>
            </div>
          );
        })}
      </div>

      {/* Wizard Content Panels */}
      <div className="card-panel">
        {/* STEP 1: Create Project */}
        {currentStep === 1 && (
          <div>
            <h2 style={{ fontSize: "22px", fontWeight: 700, marginBottom: "8px" }}>
              Step 1: Create or Select Project
            </h2>
            <p style={{ fontSize: "14px", opacity: 0.7, marginBottom: "24px" }}>
              Define the target organism and metadata for your morphometric model repository.
            </p>

            <form onSubmit={handleCreateProject}>
              <div style={{ marginBottom: "20px" }}>
                <label style={{ fontWeight: 600, fontSize: "13px" }}>Project Name</label>
                <input
                  type="text"
                  className="form-control"
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                  placeholder="e.g. Anolis Toepad Morphometrics"
                  required
                />
              </div>

              <div style={{ marginBottom: "28px" }}>
                <label style={{ fontWeight: 600, fontSize: "13px" }}>Target Organism / Taxon</label>
                <input
                  type="text"
                  className="form-control"
                  value={organism}
                  onChange={(e) => setOrganism(e.target.value)}
                  placeholder="e.g. Anolis carolinensis"
                  required
                />
              </div>

              <div style={{ display: "flex", justifyContent: "flex-end" }}>
                <button type="submit" className="btn-action">
                  Continue to Add Data →
                </button>
              </div>
            </form>
          </div>
        )}

        {/* STEP 2: Add Data */}
        {currentStep === 2 && (
          <div>
            <h2 style={{ fontSize: "22px", fontWeight: 700, marginBottom: "8px" }}>
              Step 2: Add Dataset & Configure Padding
            </h2>
            <p style={{ fontSize: "14px", opacity: 0.7, marginBottom: "24px" }}>
              Upload your annotated TPS, XML, or ZIP dataset, and set crop padding for bounding boxes.
            </p>

            <div style={{ marginBottom: "24px" }}>
              <label style={{ fontWeight: 600, fontSize: "13px", display: "block", marginBottom: "8px" }}>
                Upload Dataset File (TPS, XML, or ZIP)
              </label>
              <div
                style={{
                  padding: "36px",
                  borderRadius: "14px",
                  border: `2px dashed ${isDragging || datasetFiles.length > 0 ? "#4CAF50" : t.border}`,
                  background: isDark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.01)",
                  textAlign: "center",
                  cursor: "pointer",
                }}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDragging(false);
                  const files = Array.from(e.dataTransfer.files || []);
                  if (files.length > 0) handleFileUpload(files);
                }}
              >
                <input
                  type="file"
                  accept=".tps,.xml,.zip,.bmp,.jpeg,.jpg,.png,.tif,.tiff,.webp"
                  multiple
                  style={{ display: "none" }}
                  id="dataset-file-input"
                  onChange={(e) => {
                    const files = Array.from(e.target.files || []);
                    if (files.length > 0) handleFileUpload(files);
                  }}
                />
                <label htmlFor="dataset-file-input" style={{ cursor: "pointer" }}>
                  <div style={{ fontWeight: 700, fontSize: "15px" }}>
                    {datasetFiles.length > 0
                      ? `${datasetFiles.length} file${datasetFiles.length === 1 ? "" : "s"} selected`
                      : "Click or Drag & Drop Dataset Files"}
                  </div>
                  <div style={{ fontSize: "12px", opacity: 0.6, marginTop: "4px" }}>
                    Select a ZIP archive, or select TPS/XML and every referenced image together
                  </div>
                </label>
              </div>
            </div>

            <div style={{ marginBottom: "28px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
                <label style={{ fontWeight: 600, fontSize: "13px" }}>Crop Padding Extent</label>
                <code style={{ fontSize: "13px", fontWeight: 700, color: "#4CAF50" }}>
                  {(padding * 100).toFixed(0)}%
                </code>
              </div>
              <input
                type="range"
                min="0.0"
                max="0.5"
                step="0.05"
                value={padding}
                onChange={(e) => setPadding(parseFloat(e.target.value))}
                style={{ width: "100%", accentColor: "#4CAF50" }}
              />
              <span style={{ fontSize: "12px", opacity: 0.6 }}>
                Fractional padding added around derived bounding boxes for crop alignment.
              </span>
            </div>

            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <button onClick={() => setCurrentStep(1)} className="btn-action btn-secondary">
                ← Back
              </button>
              <button onClick={handleCheckData} className="btn-action" disabled={isCheckingData}>
                {isCheckingData ? "Checking Data..." : "Check & Validate Data →"}
              </button>
            </div>
          </div>
        )}

        {/* STEP 3: Check Data */}
        {currentStep === 3 && (
          <div>
            <h2 style={{ fontSize: "22px", fontWeight: 700, marginBottom: "8px" }}>
              Step 3: Check Data & Preview Bounding Boxes
            </h2>
            <p style={{ fontSize: "14px", opacity: 0.7, marginBottom: "24px" }}>
              Inspect data health, image counts, and preview derived bounding box extents and landmark points.
            </p>

            <div style={{
              padding: "20px",
              borderRadius: "14px",
              background: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.02)",
              border: `1px solid ${t.borderLight}`,
              marginBottom: "24px",
            }}>
              <h3 style={{ fontSize: "16px", fontWeight: 700, marginBottom: "12px", color: "#4CAF50" }}>
                Dataset Health Check Passed
              </h3>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "16px", marginTop: "16px" }}>
                <div style={{ background: isDark ? "#2a3a4e" : "#f5f5f5", padding: "16px", borderRadius: "10px", textAlign: "center" }}>
                  <div style={{ fontSize: "24px", fontWeight: 800 }}>
                    {derivedBoxes ? derivedBoxes.total_images : "0"}
                  </div>
                  <div style={{ fontSize: "12px", opacity: 0.7 }}>Images Registered</div>
                </div>

                <div style={{ background: isDark ? "#2a3a4e" : "#f5f5f5", padding: "16px", borderRadius: "10px", textAlign: "center" }}>
                  <div style={{ fontSize: "24px", fontWeight: 800 }}>
                    {derivedBoxes ? derivedBoxes.total_objects : "0"}
                  </div>
                  <div style={{ fontSize: "12px", opacity: 0.7 }}>Specimens / Objects</div>
                </div>

                <div style={{ background: isDark ? "#2a3a4e" : "#f5f5f5", padding: "16px", borderRadius: "10px", textAlign: "center" }}>
                  <div style={{ fontSize: "24px", fontWeight: 800, color: "#4CAF50" }}>
                    {(padding * 100).toFixed(0)}%
                  </div>
                  <div style={{ fontSize: "12px", opacity: 0.7 }}>Crop Padding Applied</div>
                </div>
              </div>
            </div>

            {/* Interactive Visual Dataset & Landmark Inspector */}
            {derivedBoxes && derivedBoxes.images && derivedBoxes.images.length > 0 && (
              <div style={{
                marginBottom: "24px",
                borderRadius: "14px",
                border: `1px solid ${t.borderLight}`,
                background: isDark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.01)",
                padding: "20px",
              }}>
                {/* Controls Header */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px", flexWrap: "wrap", gap: "12px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                    <span style={{ fontSize: "14px", fontWeight: 700 }}>Inspect Specimen Image:</span>
                    <select
                      value={selectedImageIndex}
                      onChange={(e) => setSelectedImageIndex(parseInt(e.target.value))}
                      style={{
                        padding: "8px 12px",
                        borderRadius: "8px",
                        border: `1px solid ${t.border}`,
                        background: isDark ? "#1e293b" : "#ffffff",
                        color: t.text,
                        fontSize: "13px",
                        fontWeight: 600,
                      }}
                    >
                      {derivedBoxes.images.map((img, idx) => (
                        <option key={img.file_path || idx} value={idx}>
                          {idx + 1}. {img.file_path.split("/").pop()} ({img.objects?.length || 0} objects)
                        </option>
                      ))}
                    </select>
                    <div style={{ display: "flex", gap: "4px" }}>
                      <button
                        type="button"
                        disabled={selectedImageIndex <= 0}
                        onClick={() => setSelectedImageIndex(prev => Math.max(0, prev - 1))}
                        className="btn-action btn-secondary"
                        style={{ padding: "6px 12px", fontSize: "12px" }}
                      >
                        ← Prev
                      </button>
                      <button
                        type="button"
                        disabled={selectedImageIndex >= derivedBoxes.images.length - 1}
                        onClick={() => setSelectedImageIndex(prev => Math.min(derivedBoxes.images.length - 1, prev + 1))}
                        className="btn-action btn-secondary"
                        style={{ padding: "6px 12px", fontSize: "12px" }}
                      >
                        Next →
                      </button>
                    </div>
                  </div>

                  <div style={{ display: "flex", gap: "16px", alignItems: "center" }}>
                    <label style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "13px", cursor: "pointer", fontWeight: 600 }}>
                      <input
                        type="checkbox"
                        checked={showPreviewLandmarks}
                        onChange={(e) => setShowPreviewLandmarks(e.target.checked)}
                        style={{ accentColor: "#00E5FF" }}
                      />
                      Show Landmarks (Points)
                    </label>
                    <label style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "13px", cursor: "pointer", fontWeight: 600 }}>
                      <input
                        type="checkbox"
                        checked={showPreviewBoxes}
                        onChange={(e) => setShowPreviewBoxes(e.target.checked)}
                        style={{ accentColor: "#4CAF50" }}
                      />
                      Show Bounding Boxes
                    </label>
                    <label style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "13px", cursor: "pointer", fontWeight: 600 }}>
                      <input
                        type="checkbox"
                        checked={showPreviewLabels}
                        onChange={(e) => setShowPreviewLabels(e.target.checked)}
                        style={{ accentColor: "#FFD700" }}
                      />
                      Show Point IDs
                    </label>
                  </div>
                </div>

                {/* Canvas / SVG Image Display */}
                {(() => {
                  const curImg = derivedBoxes.images[selectedImageIndex];
                  if (!curImg) return null;
                  const baseKey = curImg.file_path.split("/").pop()?.toLowerCase() || "";
                  const imgSrc = previewImageUrls[baseKey] || previewImageUrls[curImg.file_path.toLowerCase()];

                  const getObbPolygonPoints = (obb: number[]): string => {
                    if (!obb || obb.length < 4) return "";
                    const [cx, cy, w, h, angle = 0] = obb;
                    const rad = (angle * Math.PI) / 180;
                    const cos = Math.cos(rad);
                    const sin = Math.sin(rad);
                    const hw = w / 2;
                    const hh = h / 2;
                    const corners = [
                      [-hw, -hh],
                      [hw, -hh],
                      [hw, hh],
                      [-hw, hh],
                    ];
                    return corners
                      .map(([dx, dy]) => `${cx + dx * cos - dy * sin},${cy + dx * sin + dy * cos}`)
                      .join(" ");
                  };

                  return (
                    <div style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 280px",
                      gap: "20px",
                      alignItems: "start",
                    }}>
                      {/* Main Visual SVG */}
                      <div style={{
                        position: "relative",
                        width: "100%",
                        borderRadius: "10px",
                        overflow: "hidden",
                        background: "#111",
                        border: `1px solid ${t.border}`,
                        boxShadow: "0 4px 20px rgba(0,0,0,0.15)",
                      }}>
                        <svg
                          viewBox={`0 0 ${curImg.width || 800} ${curImg.height || 600}`}
                          style={{ width: "100%", height: "auto", display: "block" }}
                        >
                          {imgSrc ? (
                            <image
                              href={imgSrc}
                              width={curImg.width || 800}
                              height={curImg.height || 600}
                            />
                          ) : (
                            <rect width={curImg.width || 800} height={curImg.height || 600} fill="#222" />
                          )}

                          {/* Bounding Boxes & Landmarks */}
                          {curImg.objects?.map((obj, oIdx) => {
                            const polyPoints = getObbPolygonPoints(obj.obb);
                            const classColors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0"];
                            const boxColor = classColors[oIdx % classColors.length];
                            const cx = obj.obb?.[0] || 0;
                            const cy = obj.obb?.[1] || 0;

                            return (
                              <g key={`obj-${obj.object_id || oIdx}`}>
                                {/* Bounding Box */}
                                {showPreviewBoxes && polyPoints && (
                                  <>
                                    <polygon
                                      points={polyPoints}
                                      fill={boxColor}
                                      fillOpacity="0.12"
                                      stroke={boxColor}
                                      strokeWidth="3"
                                      strokeDasharray="4 2"
                                    />
                                    <rect
                                      x={cx - 38}
                                      y={cy - 12}
                                      width="76"
                                      height="20"
                                      rx="4"
                                      fill={boxColor}
                                      fillOpacity="0.85"
                                    />
                                    <text
                                      x={cx}
                                      y={cy + 2}
                                      fill="#ffffff"
                                      fontSize="11"
                                      fontWeight="700"
                                      textAnchor="middle"
                                    >
                                      {obj.class_name}
                                    </text>
                                  </>
                                )}

                                {/* Landmarks */}
                                {showPreviewLandmarks && obj.landmarks?.map((pt, pIdx) => (
                                  <g key={`lm-${obj.object_id}-${pt.name}-${pIdx}`}>
                                    <circle
                                      cx={pt.x}
                                      cy={pt.y}
                                      r="5"
                                      fill="#00E5FF"
                                      stroke="#000000"
                                      strokeWidth="1.5"
                                    />
                                    {showPreviewLabels && (
                                      <text
                                        x={pt.x + 6}
                                        y={pt.y - 6}
                                        fill="#FFD700"
                                        stroke="#000000"
                                        strokeWidth="0.6"
                                        fontSize="13"
                                        fontWeight="800"
                                      >
                                        {pt.name}
                                      </text>
                                    )}
                                  </g>
                                ))}
                              </g>
                            );
                          })}
                        </svg>
                      </div>

                      {/* Sidebar: Object and Landmark Breakdown */}
                      <div style={{
                        maxHeight: "520px",
                        overflowY: "auto",
                        background: isDark ? "#1e293b" : "#ffffff",
                        padding: "16px",
                        borderRadius: "10px",
                        border: `1px solid ${t.border}`,
                      }}>
                        <h4 style={{ fontSize: "14px", fontWeight: 700, marginBottom: "12px" }}>
                          Detected Objects ({curImg.objects?.length || 0})
                        </h4>
                        {curImg.objects?.map((obj, oIdx) => (
                          <div
                            key={obj.object_id || oIdx}
                            style={{
                              marginBottom: "12px",
                              padding: "10px",
                              borderRadius: "8px",
                              background: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.02)",
                              border: `1px solid ${t.borderLight}`,
                            }}
                          >
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
                              <span style={{ fontWeight: 700, fontSize: "13px", color: "#4CAF50" }}>
                                {obj.class_name}
                              </span>
                              <span style={{ fontSize: "11px", opacity: 0.7 }}>
                                {obj.landmarks?.length || 0} points
                              </span>
                            </div>
                            <div style={{ fontSize: "11px", opacity: 0.8, lineHeight: "1.4" }}>
                              <div>Center: ({obj.obb?.[0]?.toFixed(1)}, {obj.obb?.[1]?.toFixed(1)})</div>
                              <div>Dim: {obj.obb?.[2]?.toFixed(1)} × {obj.obb?.[3]?.toFixed(1)} px</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}

            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <button onClick={() => setCurrentStep(2)} className="btn-action btn-secondary">
                ← Back
              </button>
              <button onClick={() => setCurrentStep(4)} className="btn-action">
                Configure Training →
              </button>
            </div>
          </div>
        )}

        {/* STEP 4: Train Setup */}
        {currentStep === 4 && (
          <div>
            <h2 style={{ fontSize: "22px", fontWeight: 700, marginBottom: "8px" }}>
              Step 4: Configure Training Presets & Expert Options
            </h2>
            <p style={{ fontSize: "14px", opacity: 0.7, marginBottom: "24px" }}>
              Select a training preset or customize advanced hyperparameter controls in the Expert Panel.
            </p>

            {/* Presets */}
            <div style={{ display: "flex", gap: "16px", marginBottom: "24px" }}>
              <div
                className={`preset-card ${trainingPreset === "fast" ? "selected" : ""}`}
                onClick={() => setTrainingPreset("fast")}
              >
                <div style={{ fontWeight: 700, fontSize: "16px" }}>Fast Training</div>
                <div style={{ fontSize: "12px", opacity: 0.7, marginTop: "4px" }}>
                  ~10 Epochs • Quick Preview
                </div>
              </div>

              <div
                className={`preset-card ${trainingPreset === "standard" ? "selected" : ""}`}
                onClick={() => setTrainingPreset("standard")}
              >
                <div style={{ fontWeight: 700, fontSize: "16px" }}>Standard</div>
                <div style={{ fontSize: "12px", opacity: 0.7, marginTop: "4px" }}>
                  ~50 Epochs • Balanced Performance
                </div>
              </div>

              <div
                className={`preset-card ${trainingPreset === "accurate" ? "selected" : ""}`}
                onClick={() => setTrainingPreset("accurate")}
              >
                <div style={{ fontWeight: 700, fontSize: "16px" }}>High Accuracy</div>
                <div style={{ fontSize: "12px", opacity: 0.7, marginTop: "4px" }}>
                  ~100 Epochs • Maximum Precision
                </div>
              </div>
            </div>

            {/* Collapsible Expert Panel */}
            <div style={{ marginBottom: "28px" }}>
              <button
                type="button"
                onClick={() => setShowExpertPanel(!showExpertPanel)}
                style={{
                  background: "none",
                  border: "none",
                  color: "#4CAF50",
                  fontWeight: 700,
                  fontSize: "14px",
                  cursor: "pointer",
                  padding: 0,
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                }}
              >
                {showExpertPanel ? "▼ Hide Expert Panel" : "▶ Show Expert Panel (Advanced Hyperparameters)"}
              </button>

              {showExpertPanel && (
                <div style={{
                  marginTop: "14px",
                  padding: "20px",
                  borderRadius: "12px",
                  background: isDark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.02)",
                  border: `1px solid ${t.border}`,
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "16px 24px",
                }}>
                  <div>
                    <label style={{ fontSize: "12px", fontWeight: 600, display: "flex", justifyContent: "space-between" }}>
                      <span>Regularization (nu):</span> <code>{nu}</code>
                    </label>
                    <input type="range" min="0.01" max="1.0" step="0.01" value={nu} onChange={(e) => setNu(parseFloat(e.target.value))} style={{ width: "100%", accentColor: "#4CAF50" }} />
                  </div>

                  <div>
                    <label style={{ fontSize: "12px", fontWeight: 600, display: "flex", justifyContent: "space-between" }}>
                      <span>Tree Depth:</span> <code>{treeDepth}</code>
                    </label>
                    <input type="range" min="2" max="8" step="1" value={treeDepth} onChange={(e) => setTreeDepth(parseInt(e.target.value))} style={{ width: "100%", accentColor: "#4CAF50" }} />
                  </div>

                  <div>
                    <label style={{ fontSize: "12px", fontWeight: 600, display: "flex", justifyContent: "space-between" }}>
                      <span>Cascade Depth:</span> <code>{cascadeDepth}</code>
                    </label>
                    <input type="range" min="1" max="60" step="1" value={cascadeDepth} onChange={(e) => setCascadeDepth(parseInt(e.target.value))} style={{ width: "100%", accentColor: "#4CAF50" }} />
                  </div>

                  <div>
                    <label style={{ fontSize: "12px", fontWeight: 600, display: "flex", justifyContent: "space-between" }}>
                      <span>Oversampling Amount:</span> <code>{oversamplingAmount}</code>
                    </label>
                    <input type="range" min="0" max="50" step="1" value={oversamplingAmount} onChange={(e) => setOversamplingAmount(parseInt(e.target.value))} style={{ width: "100%", accentColor: "#4CAF50" }} />
                  </div>

                  <div>
                    <label style={{ fontSize: "12px", fontWeight: 600, display: "flex", justifyContent: "space-between" }}>
                      <span>Feature Pool Size:</span> <code>{featurePoolSize}</code>
                    </label>
                    <input type="range" min="50" max="2000" step="50" value={featurePoolSize} onChange={(e) => setFeaturePoolSize(parseInt(e.target.value))} style={{ width: "100%", accentColor: "#4CAF50" }} />
                  </div>

                  <div>
                    <label style={{ fontSize: "12px", fontWeight: 600, display: "flex", justifyContent: "space-between" }}>
                      <span>Number of Test Splits:</span> <code>{numTestSplits}</code>
                    </label>
                    <input type="range" min="5" max="100" step="5" value={numTestSplits} onChange={(e) => setNumTestSplits(parseInt(e.target.value))} style={{ width: "100%", accentColor: "#4CAF50" }} />
                  </div>

                  <div>
                    <label style={{ fontSize: "12px", fontWeight: 600, display: "flex", justifyContent: "space-between" }}>
                      <span>Test Split Ratio:</span> <code>{Math.round(testSplit * 100)}%</code>
                    </label>
                    <input type="range" min="0.0" max="0.5" step="0.05" value={testSplit} onChange={(e) => setTestSplit(parseFloat(e.target.value))} style={{ width: "100%", accentColor: "#4CAF50" }} />
                  </div>
                </div>
              )}
            </div>

            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <button onClick={() => setCurrentStep(3)} className="btn-action btn-secondary">
                ← Back
              </button>
              <button onClick={handleStartTraining} className="btn-action">
                Start Training Job →
              </button>
            </div>
          </div>
        )}

        {/* STEP 5: Follow Progress */}
        {currentStep === 5 && (() => {
          const rawProg = jobStatus?.progress ?? 0;
          const displayProg = rawProg <= 1.0 ? rawProg * 100 : rawProg;

          return (
            <div style={{ textAlign: "center", padding: "20px 0" }}>
              <h2 style={{ fontSize: "22px", fontWeight: 700, marginBottom: "8px" }}>
                Step 5: Training Progress & Live Monitor
              </h2>
              <p style={{ fontSize: "14px", opacity: 0.7, marginBottom: "24px" }}>
                {jobStatus?.stage || "Initializing training execution engine..."}
              </p>

              <div className="progress-bar-bg">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${Math.min(100, displayProg)}%` }}
                />
              </div>

              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "13px", fontWeight: 700, marginBottom: "24px" }}>
                <span>Stage: {jobStatus?.stage || "Checking data"}</span>
                <span>{displayProg.toFixed(1)}%</span>
              </div>

              {activeJobId && (
                <div style={{ fontSize: "12px", opacity: 0.6, marginBottom: "24px" }}>
                  Job Identifier: <code style={{ fontFamily: "monospace" }}>{activeJobId}</code>
                </div>
              )}

              <div style={{ display: "flex", justifyContent: "center", gap: "12px" }}>
                {isTraining && displayProg < 100 && (
                  <button
                    type="button"
                    onClick={handleCancelJob}
                    className="btn-action btn-secondary"
                    disabled={isCancelling}
                    style={{ background: isDark ? "#c62828" : "#d32f2f", color: "white", border: "none" }}
                  >
                    {isCancelling ? "Cancelling..." : "Cancel Training Job"}
                  </button>
                )}
                <button
                  onClick={() => setCurrentStep(6)}
                  className="btn-action"
                  disabled={jobStatus?.status !== "completed"}
                >
                  Review Results →
                </button>
              </div>
            </div>
          );
        })()}

        {/* STEP 6: Review Results */}
        {currentStep === 6 && (
          <div>
            <h2 style={{ fontSize: "22px", fontWeight: 700, marginBottom: "8px" }}>
              Step 6: Review Model Metrics & Evaluation
            </h2>
            <p style={{ fontSize: "14px", opacity: 0.7, marginBottom: "24px" }}>
              Evaluate validation accuracy, bounding box IoU, and landmark error metrics.
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "16px", marginBottom: "28px" }}>
              <div style={{ background: isDark ? "#2a3a4e" : "#f5f5f5", padding: "20px", borderRadius: "12px", textAlign: "center" }}>
                <div style={{ fontSize: "28px", fontWeight: 800, color: "#4CAF50" }}>
                    {typeof jobStatus?.metrics?.mAP50 === "number"
                      ? `${(jobStatus.metrics.mAP50 * 100).toFixed(1)}%`
                      : "Not reported"}
                </div>
                <div style={{ fontSize: "12px", opacity: 0.7, marginTop: "4px" }}>Detector mAP50</div>
              </div>

              <div style={{ background: isDark ? "#2a3a4e" : "#f5f5f5", padding: "20px", borderRadius: "12px", textAlign: "center" }}>
                <div style={{ fontSize: "28px", fontWeight: 800, color: "#4CAF50" }}>
                    {typeof jobStatus?.metrics?.test_error === "number"
                      ? `${jobStatus.metrics.test_error.toFixed(2)} px`
                      : "Not reported"}
                </div>
                <div style={{ fontSize: "12px", opacity: 0.7, marginTop: "4px" }}>Landmark Test Error</div>
              </div>

              <div style={{ background: isDark ? "#2a3a4e" : "#f5f5f5", padding: "20px", borderRadius: "12px", textAlign: "center" }}>
                <div style={{ fontSize: "28px", fontWeight: 800, color: "#4CAF50" }}>
                  {jobStatus?.status === "completed" ? "Passed" : "Not complete"}
                </div>
                <div style={{ fontSize: "12px", opacity: 0.7, marginTop: "4px" }}>Validation Status</div>
              </div>
            </div>

            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <button onClick={() => setCurrentStep(5)} className="btn-action btn-secondary">
                ← Back to Progress
              </button>
              <button
                onClick={() => {
                  setCurrentStep(7);
                }}
                className="btn-action"
                disabled={jobStatus?.status !== "completed" || !publishedModelId}
              >
                Proceed to Publish →
              </button>
            </div>
          </div>
        )}

        {/* STEP 7: Publish & Use */}
        {currentStep === 7 && (
          <div style={{ textAlign: "center", padding: "20px 0" }}>
            <h2 style={{ fontSize: "24px", fontWeight: 800, marginBottom: "8px", color: "#4CAF50" }}>
              Model Ready & Registered!
            </h2>
            <p style={{ fontSize: "14px", opacity: 0.7, maxWidth: "500px", margin: "0 auto 24px auto" }}>
              Your custom morphometric model has been published to the model registry and is ready for inference.
            </p>

            <div style={{
              background: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.02)",
              padding: "16px 24px",
              borderRadius: "12px",
              display: "inline-block",
              marginBottom: "32px",
              border: `1px solid ${t.borderLight}`,
            }}>
              <div style={{ fontSize: "12px", opacity: 0.6 }}>Registered Model Identifier</div>
              <code style={{ fontSize: "16px", fontWeight: 700, color: "#4CAF50" }}>
                {publishedModelId || "lizard-custom-v1"}
              </code>
            </div>

            <div style={{ display: "flex", justifyContent: "center", gap: "16px" }}>
              <button onClick={onNavigateHome} className="btn-action btn-secondary">
                Back to Landing Page
              </button>
              <button
                onClick={() => publishedModelId && onUseModel(publishedModelId)}
                className="btn-action"
                disabled={!publishedModelId}
              >
                Start Analyzing Images →
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Available Published Custom Models Card */}
      <div className="card-panel">
        <h3 style={{ fontSize: "18px", fontWeight: 700, marginBottom: "16px" }}>
          Registered Models in Registry
        </h3>
        {loadingModels && <p style={{ fontSize: "14px", opacity: 0.7 }}>Loading registered models...</p>}
        {!loadingModels && registeredModels.length === 0 && (
          <p style={{ fontSize: "14px", opacity: 0.6, fontStyle: "italic" }}>
            No custom model bundles registered yet. Use the wizard above to train your first model!
          </p>
        )}
        {!loadingModels && registeredModels.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {registeredModels.map((model) => (
              <div key={model.id} style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "12px 16px",
                borderRadius: "10px",
                background: isDark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.02)",
                border: `1px solid ${t.borderLight}`,
              }}>
                <div>
                  <div style={{ fontWeight: 700, fontSize: "14px" }}>{model.name}</div>
                  <div style={{ fontSize: "12px", opacity: 0.6 }}>
                    ID: {model.id} · {model.manifest.classes.length} class{model.manifest.classes.length === 1 ? "" : "es"}
                  </div>
                </div>
                <button
                  className="btn-action btn-secondary"
                  style={{ padding: "7px 12px", fontSize: "12px" }}
                  onClick={() => onUseModel(model.id)}
                >
                  Use Model
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <style>{styles}</style>
    </div>
  );
};
